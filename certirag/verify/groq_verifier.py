"""
Groq Verifier — Fast LLM-as-Judge via Groq Inference API
==========================================================

Uses Groq's free-tier inference API to run open-source LLMs
(Llama 3.3 70B, Mixtral 8x7B, Gemma 2 9B, etc.) at extremely
high speed for claim verification.

Groq offers free API access with generous rate limits, making it
ideal for evaluation. The API is OpenAI-compatible.

Usage
-----
    export GROQ_API_KEY="gsk_..."
    from certirag.verify.groq_verifier import GroqVerifier
    verifier = GroqVerifier()
    scores = verifier.verify_batch([("The sky is blue", "Blue sky today")])

Free tier: https://console.groq.com/docs/rate-limits
Models:    https://console.groq.com/docs/models
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from certirag.schemas.verification import EvidenceScore, VerificationLabel
from certirag.verify.verifier import BaseVerifier

logger = logging.getLogger("certirag.verify.groq_verifier")


JUDGE_PROMPT = """You are an expert fact verification system. Given a CLAIM and EVIDENCE passage, determine whether the evidence supports, contradicts, or is insufficient to verify the claim.

Here are examples of correct classifications:

Example 1:
CLAIM: "Apple was founded in 1976."
EVIDENCE: "Apple Inc. was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne."
Analysis: The evidence directly states Apple was founded in 1976. All parts of the claim are supported.
Answer: {{"label": "entailed", "score": 0.98, "reasoning": "Evidence explicitly confirms founding year 1976."}}

Example 2:
CLAIM: "The population of Tokyo is 5 million."
EVIDENCE: "Tokyo has an estimated population of approximately 14 million people in the city proper."
Analysis: The evidence says 14 million, but the claim says 5 million. These numbers directly conflict.
Answer: {{"label": "contradicted", "score": 0.95, "reasoning": "Evidence states 14 million, contradicting the claimed 5 million."}}

Example 3:
CLAIM: "Einstein won the Nobel Prize for his work on relativity."
EVIDENCE: "Albert Einstein was a renowned theoretical physicist who developed the theory of relativity."
Analysis: The evidence confirms Einstein worked on relativity but says nothing about the Nobel Prize.
Answer: {{"label": "not_enough_info", "score": 0.85, "reasoning": "Evidence discusses relativity but does not mention the Nobel Prize."}}

Now verify this claim:

CLAIM: {claim}

EVIDENCE: {evidence}

Analyze step by step:
1. Identify the core assertion in the claim.
2. Check if the evidence directly addresses this assertion.
3. Determine if the evidence confirms, denies, or is silent on the claim.

Classification rules:
- "entailed": The evidence clearly and directly supports ALL parts of the claim. Partial support is NOT entailment.
- "contradicted": The evidence explicitly states something that conflicts with the claim (e.g. different numbers, dates, names, or negation of the claim).
- "not_enough_info": The evidence does not address the claim, is only tangentially related, or provides insufficient information to confirm or deny it.

Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{{"label": "<entailed|contradicted|not_enough_info>", "score": <0.0-1.0>, "reasoning": "<one sentence>"}}"""


BATCH_PROMPT = """You are an expert fact verification system. For each numbered CLAIM-EVIDENCE pair below, determine if the evidence supports, contradicts, or is insufficient to verify the claim.

{pairs}

Classification rules:
- "entailed": Evidence clearly and directly supports ALL parts of the claim.
- "contradicted": Evidence explicitly conflicts with the claim.
- "not_enough_info": Evidence doesn't address, only tangentially relates, or is insufficient.

Respond with ONLY a JSON array — one object per pair, in order:
[{{"id": 1, "label": "<entailed|contradicted|not_enough_info>", "score": <0.0-1.0>, "reasoning": "<brief>"}}, ...]"""


# ── Available Groq models (free tier) ──────────────────────────
GROQ_MODELS = {
    "llama-3.3-70b":   "llama-3.3-70b-versatile",
    "llama-3.1-8b":    "llama-3.1-8b-instant",
    "mixtral-8x7b":    "mixtral-8x7b-32768",
    "gemma2-9b":       "gemma2-9b-it",
    "llama-guard-3-8b": "llama-guard-3-8b",
}


class GroqVerifier(BaseVerifier):
    """
    Fact verification using Groq's inference API (free tier).

    Runs open-source LLMs (Llama 3, Mixtral, Gemma 2) at extremely
    high token throughput (~750 t/s) for NLI-style claim verification.

    Args:
        api_key: Groq API key. Falls back to GROQ_API_KEY env var.
        model: Model identifier. Use short name or full model ID.
        max_pairs: Maximum pairs to verify per call.
        batch_size: Number of pairs per API call (batched in prompt).
        requests_per_minute: Rate limit (free tier: 30 RPM for most models).
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "llama-3.3-70b",
        max_pairs: int = 500,
        batch_size: int = 5,
        requests_per_minute: int = 30,
    ):
        resolved_model = GROQ_MODELS.get(model, model)
        super().__init__(model_name=f"Groq({resolved_model})", max_pairs=max_pairs)
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = resolved_model
        self.batch_size = batch_size
        self.rpm = requests_per_minute
        self._client = None
        self._last_call_time = 0.0

    # ── Client ─────────────────────────────────────────────────

    def _get_client(self):
        """Lazy-initialize OpenAI-compatible client for Groq."""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "Groq API key required. Set GROQ_API_KEY env var or pass api_key=. "
                    "Get free key at: https://console.groq.com/"
                )
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError(
                    "openai package required for Groq verifier. "
                    "Install with: pip install openai"
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        return self._client

    # ── Rate limiting ──────────────────────────────────────────

    def _rate_limit(self) -> None:
        """Simple token-bucket rate limiter."""
        if self.rpm <= 0:
            return
        min_interval = 60.0 / self.rpm
        elapsed = time.time() - self._last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_time = time.time()

    # ── Response parsing ───────────────────────────────────────

    def _parse_response(self, text: str) -> list[dict]:
        """Parse JSON from LLM response, handling markdown wrapping."""
        text = text.strip()

        # Strip markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            ).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return [parsed]
            return parsed
        except json.JSONDecodeError:
            # Try extracting JSON substring
            start = text.find("[")
            if start == -1:
                start = text.find("{")
            end = max(text.rfind("]"), text.rfind("}")) + 1
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start:end])
                    if isinstance(parsed, dict):
                        return [parsed]
                    return parsed
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Failed to parse Groq response: {text[:200]}")
            return []

    def _label_from_str(self, label_str: str) -> VerificationLabel:
        """Convert string label to VerificationLabel enum."""
        label_lower = label_str.lower().strip()
        if label_lower in ("entailed", "entailment", "supported", "supports"):
            return VerificationLabel.ENTAILED
        elif label_lower in ("contradicted", "contradiction", "refuted", "refutes"):
            return VerificationLabel.CONTRADICTED
        else:
            return VerificationLabel.NOT_ENOUGH_INFO

    def _make_evidence_score(
        self, label: VerificationLabel, score: float
    ) -> EvidenceScore:
        """Build EvidenceScore with appropriate raw_score for AUROC."""
        score = max(0.0, min(1.0, score))

        # raw_score = entailment probability (for AUROC computation)
        if label == VerificationLabel.ENTAILED:
            raw_score = score
        elif label == VerificationLabel.CONTRADICTED:
            raw_score = 1.0 - score
        else:
            raw_score = 1.0 - score  # low entailment confidence

        return EvidenceScore(
            chunk_id="",
            span_id="",
            label=label,
            score=score,
            raw_score=raw_score,
        )

    def _default_score(self) -> EvidenceScore:
        return EvidenceScore(
            chunk_id="",
            span_id="",
            label=VerificationLabel.NOT_ENOUGH_INFO,
            score=0.3,
            raw_score=0.3,
        )

    # ── API calls ──────────────────────────────────────────────

    def _call_single(self, claim: str, evidence: str) -> EvidenceScore:
        """Verify a single pair with one API call."""
        client = self._get_client()
        self._rate_limit()

        prompt = JUDGE_PROMPT.format(claim=claim, evidence=evidence)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise fact verification judge. Respond only with JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            text = response.choices[0].message.content or ""
            results = self._parse_response(text)

            if results:
                r = results[0]
                label = self._label_from_str(r.get("label", "not_enough_info"))
                score = float(r.get("score", 0.5))
                return self._make_evidence_score(label, score)

        except Exception as e:
            logger.error(f"Groq API call failed: {e}")

        return self._default_score()

    def _call_batch(self, pairs: list[tuple[str, str]]) -> list[EvidenceScore]:
        """Verify multiple pairs in a single batched prompt."""
        client = self._get_client()
        self._rate_limit()

        pairs_text = ""
        for i, (claim, evidence) in enumerate(pairs, 1):
            pairs_text += f"\n--- Pair {i} ---\nCLAIM: {claim}\nEVIDENCE: {evidence}\n"

        prompt = BATCH_PROMPT.format(pairs=pairs_text)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise fact verification judge. Respond only with a JSON array.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=150 * len(pairs),
            )
            text = response.choices[0].message.content or ""
            results = self._parse_response(text)

            scores = []
            for i in range(len(pairs)):
                if i < len(results):
                    r = results[i]
                    label = self._label_from_str(r.get("label", "not_enough_info"))
                    score = float(r.get("score", 0.5))
                    scores.append(self._make_evidence_score(label, score))
                else:
                    scores.append(self._default_score())
            return scores

        except Exception as e:
            logger.error(f"Groq batch API call failed: {e}")
            return [self._default_score() for _ in pairs]

    # ── Public interface ───────────────────────────────────────

    def verify_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[EvidenceScore]:
        """
        Verify claim-evidence pairs using Groq API.

        Batches pairs into prompt groups for efficiency.
        Llama 3.3 70B on Groq processes ~750 tokens/sec.
        """
        all_scores: list[EvidenceScore] = []

        for batch_start in range(0, len(pairs), self.batch_size):
            batch = pairs[batch_start : batch_start + self.batch_size]
            if len(batch) == 1:
                all_scores.append(self._call_single(batch[0][0], batch[0][1]))
            else:
                all_scores.extend(self._call_batch(batch))

        return all_scores

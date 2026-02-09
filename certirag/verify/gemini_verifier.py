"""
Gemini Verifier — Free LLM-as-Judge Verification
==================================================

Uses Google's Gemini API (free tier) as a fact verification judge.
Gemini 2.0 Flash provides 15 RPM free, sufficient for development
and end-to-end testing with real LLM verification.

Integration:
    - Follows the BaseVerifier contract (verify_batch)
    - Uses the same JUDGE_PROMPT pattern as LLMJudgeVerifier
    - Parses JSON responses into EvidenceScore objects
    - Falls back gracefully on malformed responses
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from certirag.schemas.verification import EvidenceScore, VerificationLabel
from certirag.verify.verifier import BaseVerifier

logger = logging.getLogger("certirag.verify.gemini_verifier")


GEMINI_JUDGE_PROMPT = """You are a precise fact verification judge. Given a CLAIM and EVIDENCE, determine if the evidence supports, contradicts, or is insufficient to verify the claim.

CLAIM: {claim}

EVIDENCE: {evidence}

Instructions:
1. Carefully compare the claim against the evidence text.
2. If the evidence clearly and directly supports the claim → label: "entailed", score: 0.9-1.0
3. If the evidence clearly contradicts the claim → label: "contradicted", score: 0.9-1.0
4. If the evidence is unrelated, vague, or insufficient → label: "not_enough_info", score: 0.3-0.6
5. Be precise: partial matches or tangential evidence should be "not_enough_info"

Respond with ONLY a valid JSON object, no other text:
{{"label": "entailed", "score": 0.95, "reasoning": "brief explanation"}}"""


class GeminiVerifier(BaseVerifier):
    """
    Fact verification using Google Gemini API (free tier).

    Uses Gemini 2.0 Flash for NLI-style claim verification.
    Each (claim, evidence) pair gets one API call.

    Usage:
        verifier = GeminiVerifier(api_key="AIza...")
        scores = verifier.verify_batch([("claim text", "evidence text")])

    Args:
        api_key: Google AI API key (get free at aistudio.google.com).
        model: Gemini model name (default: gemini-2.0-flash).
        max_pairs: Cost/rate cap on pairs per query.
        requests_per_minute: Rate limit (free tier: 15 RPM).
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-2.0-flash",
        max_pairs: int = 200,
        requests_per_minute: int = 15,
    ):
        super().__init__(model_name=model, max_pairs=max_pairs)
        self.api_key = api_key
        self.model = model
        self.rpm = requests_per_minute
        self._client = None
        self._last_call_time = 0.0

    def _get_client(self):
        """Lazy-initialize the Gemini client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "google-genai package required for Gemini verifier. "
                    "Install with: pip install google-genai"
                )
        return self._client

    def _rate_limit(self) -> None:
        """Simple rate limiter to stay within free tier."""
        if self.rpm <= 0:
            return
        min_interval = 60.0 / self.rpm
        elapsed = time.time() - self._last_call_time
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_call_time = time.time()

    def _parse_response(self, text: str) -> dict:
        """Parse JSON from Gemini response, handling common issues."""
        text = text.strip()

        # Strip markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

        logger.warning(f"Failed to parse Gemini response: {text[:200]}")
        return {"label": "not_enough_info", "score": 0.5, "reasoning": "parse_error"}

    def verify_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[EvidenceScore]:
        """
        Verify each (claim, evidence) pair using Gemini API.

        Makes one API call per pair with rate limiting.

        Args:
            pairs: List of (claim_text, evidence_text) tuples.

        Returns:
            List of EvidenceScore objects with verification results.
        """
        client = self._get_client()
        all_scores: list[EvidenceScore] = []

        label_map = {
            "entailed": VerificationLabel.ENTAILED,
            "contradicted": VerificationLabel.CONTRADICTED,
            "not_enough_info": VerificationLabel.NOT_ENOUGH_INFO,
        }

        for i, (claim, evidence) in enumerate(pairs):
            try:
                self._rate_limit()
                prompt = GEMINI_JUDGE_PROMPT.format(
                    claim=claim, evidence=evidence
                )

                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "temperature": 0.0,
                        "max_output_tokens": 200,
                    },
                )

                result = self._parse_response(response.text)
                label_str = result.get("label", "not_enough_info")
                score = float(result.get("score", 0.5))
                label = label_map.get(label_str, VerificationLabel.NOT_ENOUGH_INFO)

                all_scores.append(EvidenceScore(
                    chunk_id="",
                    span_id="",
                    label=label,
                    score=score,
                    raw_score=score,
                ))

                logger.debug(
                    f"Pair {i+1}/{len(pairs)}: {label_str} ({score:.2f}) — "
                    f"{result.get('reasoning', '')[:60]}"
                )

            except Exception as e:
                logger.warning(f"Gemini API call failed for pair {i}: {e}")
                all_scores.append(EvidenceScore(
                    chunk_id="",
                    span_id="",
                    label=VerificationLabel.NOT_ENOUGH_INFO,
                    score=0.5,
                    raw_score=0.5,
                ))

        return all_scores

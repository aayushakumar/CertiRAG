"""
LLM-as-Judge Verifier (Ablation Ceiling)
==========================================

Uses a large language model (GPT-4o) as a fact verification judge.
This serves as an ablation ceiling — it's expensive but should yield
the highest accuracy, validating that smaller models (MiniCheck) are
competitive.

This verifier operates in LITE mode only (API calls).
It is NOT used in production; only for ablation tables.
"""

from __future__ import annotations

import json
import logging

from certirag.schemas.verification import EvidenceScore, VerificationLabel
from certirag.verify.verifier import BaseVerifier

logger = logging.getLogger("certirag.verify.llm_judge")


JUDGE_PROMPT = """You are a precise fact verification judge. Given a CLAIM and EVIDENCE, determine if the evidence supports, contradicts, or is insufficient to verify the claim.

CLAIM: {claim}

EVIDENCE: {evidence}

Instructions:
1. If the evidence clearly supports the claim → label: "entailed", score: 0.9-1.0
2. If the evidence clearly contradicts the claim → label: "contradicted", score: 0.9-1.0
3. If the evidence is insufficient → label: "not_enough_info", score: 0.5-0.8

Respond with ONLY a JSON object:
{{"label": "entailed|contradicted|not_enough_info", "score": 0.0-1.0, "reasoning": "brief explanation"}}"""


class LLMJudgeVerifier(BaseVerifier):
    """
    LLM-as-judge verifier using GPT-4o or similar.

    Serves as the accuracy ceiling for ablation experiments.
    Much more expensive than MiniCheck but should produce
    the highest-quality verification results.

    Usage:
        verifier = LLMJudgeVerifier(api_key="sk-...", model="gpt-4o")
        scores = verifier.verify_batch([("claim", "evidence")])

    Args:
        api_key: OpenAI API key.
        model: OpenAI model name.
        max_pairs: Cost cap (important — each pair = 1 API call).
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        max_pairs: int = 100,  # Lower default due to cost
    ):
        super().__init__(model_name=model, max_pairs=max_pairs)
        self.api_key = api_key
        self.model = model
        self._client = None

    def verify_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[EvidenceScore]:
        """
        Verify each pair using LLM API call.

        Note: This makes one API call per pair (expensive).
        Use only for ablation experiments, not production.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package required for LLM judge")

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        all_scores: list[EvidenceScore] = []

        for claim, evidence in pairs:
            try:
                prompt = JUDGE_PROMPT.format(claim=claim, evidence=evidence)
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=200,
                )

                result = json.loads(response.choices[0].message.content)
                label_str = result.get("label", "not_enough_info")
                score = float(result.get("score", 0.5))

                label_map = {
                    "entailed": VerificationLabel.ENTAILED,
                    "contradicted": VerificationLabel.CONTRADICTED,
                    "not_enough_info": VerificationLabel.NOT_ENOUGH_INFO,
                }
                label = label_map.get(label_str, VerificationLabel.NOT_ENOUGH_INFO)

                all_scores.append(EvidenceScore(
                    chunk_id="",
                    span_id="",
                    label=label,
                    score=score,
                    raw_score=score,
                ))

            except Exception as e:
                logger.error(f"LLM judge failed for pair: {e}")
                all_scores.append(EvidenceScore(
                    chunk_id="",
                    span_id="",
                    label=VerificationLabel.NOT_ENOUGH_INFO,
                    score=0.0,
                    raw_score=0.0,
                ))

        return all_scores

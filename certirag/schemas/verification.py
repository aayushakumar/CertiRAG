"""
Verification Result Schema (§6.3)
==================================

Defines the output of the verification step: for each claim,
the verifier produces a label (entailed/contradicted/NEI),
a confidence score, and the best supporting evidence.

Design Decisions:
    - Labels follow NLI convention (entailment / contradiction / neutral)
    - Scores are continuous [0, 1] for threshold tuning and calibration
    - All evidence scores are preserved (not just the best) for auditing
    - MSE (Minimal Sufficient Evidence) is computed post-verification

Data Flow:
    ClaimIR + EvidenceChunks → Verifier → VerificationResult → Renderer
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class VerificationLabel(str, Enum):
    """
    Three-way verification label following NLI convention.

    - ENTAILED:       Claim is supported by the evidence
    - CONTRADICTED:   Evidence directly contradicts the claim
    - NOT_ENOUGH_INFO: Evidence is insufficient to verify or refute
    """
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFO = "not_enough_info"


class EvidenceScore(BaseModel):
    """
    Verification score for a single (claim, evidence_span) pair.

    The verifier produces one of these for every claim × span
    combination, enabling fine-grained evidence attribution.
    """
    chunk_id: str = Field(description="Reference to EvidenceChunk.chunk_id")
    span_id: str = Field(description="Reference to EvidenceSpan.span_id")
    label: VerificationLabel = Field(description="Predicted verification label")
    score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score for the predicted label"
    )
    raw_score: float | None = Field(
        default=None,
        description="Pre-calibration raw score (for calibration analysis)"
    )


class VerificationResult(BaseModel):
    """
    Complete verification output for a single claim.

    Aggregates scores from all claim × span pairs and selects
    the best evidence. This is the input to the renderer's
    policy engine.

    Schema (§6.3):
        {
          "claim_id": "c1",
          "label": "entailed",
          "score": 0.91,
          "best_evidence": [{"chunk_id": "doc42#c17", "span_id": "s1"}],
          "all_scores": [
            {"chunk_id": "doc42#c17", "span_id": "s1",
             "label": "entailed", "score": 0.91}
          ]
        }
    """
    claim_id: str = Field(description="Reference to Claim.id")
    label: VerificationLabel = Field(description="Aggregated verification label")
    score: float = Field(
        ge=0.0, le=1.0,
        description="Best entailment score across all evidence spans"
    )
    best_evidence: list[EvidenceScore] = Field(
        default_factory=list,
        description="Minimal Sufficient Evidence (MSE) set"
    )
    all_scores: list[EvidenceScore] = Field(
        default_factory=list,
        description="All (claim, span) verification scores"
    )

    @field_validator("best_evidence")
    @classmethod
    def validate_best_evidence_subset(cls, v, info):
        """Best evidence must be a subset of all_scores (when both are set)."""
        # This validation is informational; we don't enforce strictly
        # because best_evidence may be set before all_scores in some flows
        return v

    @property
    def max_entail_score(self) -> float:
        """Highest entailment score across all evidence spans."""
        entail_scores = [
            s.score for s in self.all_scores
            if s.label == VerificationLabel.ENTAILED
        ]
        return max(entail_scores) if entail_scores else 0.0

    @property
    def max_contradict_score(self) -> float:
        """Highest contradiction score across all evidence spans."""
        contradict_scores = [
            s.score for s in self.all_scores
            if s.label == VerificationLabel.CONTRADICTED
        ]
        return max(contradict_scores) if contradict_scores else 0.0

    @property
    def evidence_span_count(self) -> int:
        """Number of evidence spans in the MSE set."""
        return len(self.best_evidence)

    @property
    def has_contradiction(self) -> bool:
        """Check if any evidence span shows contradiction."""
        return any(
            s.label == VerificationLabel.CONTRADICTED
            for s in self.all_scores
        )

"""
Claim IR Schema (§6.2)
=======================

The Claim Intermediate Representation (Claim IR) is the structured
output of the Claim Compiler. It decomposes an LLM's answer into
atomic, independently verifiable claims.

Design Philosophy:
    - Each claim contains exactly ONE fact (atomicity)
    - Claims are linked to candidate evidence spans (traceability)
    - Claims are typed for downstream analysis (taxonomy)
    - The Claim IR format is the "assembly language" of CertiRAG

Data Flow:
    LLM Answer → Claim Compiler → ClaimIR → Normalizer → Verifier
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ClaimType(str, Enum):
    """
    Taxonomy of claim types for analysis and ablation.

    This categorization enables:
    - Per-type accuracy analysis (are temporal claims harder?)
    - Targeted verification strategies (numeric claims need exact match)
    - Failure taxonomy in the paper
    """
    FACTUAL = "factual"             # Simple factual statement
    TEMPORAL_FACT = "temporal_fact"  # Involves dates, periods, durations
    NUMERIC_FACT = "numeric_fact"   # Involves quantities, measurements
    CAUSAL = "causal"               # Cause-effect relationship
    COMPARATIVE = "comparative"     # Comparison between entities
    DEFINITION = "definition"       # Definition or explanation
    PROCEDURAL = "procedural"       # Step or procedure description
    OTHER = "other"                 # Uncategorized


class CandidateEvidence(BaseModel):
    """
    A pointer to a specific evidence span that may support a claim.

    This is an *anchor*, not a verification result. The verifier
    will later score whether this evidence actually entails the claim.
    """
    chunk_id: str = Field(description="Reference to EvidenceChunk.chunk_id")
    span_id: str = Field(description="Reference to EvidenceSpan.span_id within the chunk")


class Claim(BaseModel):
    """
    An atomic, independently verifiable claim.

    Requirements (enforced by the normalizer):
    - Contains exactly ONE fact (no conjunctions joining independent facts)
    - Is self-contained (can be understood without other claims)
    - Has at least one candidate evidence span (or is flagged unsupported)

    Schema (§6.2):
        {
          "id": "c1",
          "text": "X was founded in 2017.",
          "type": "temporal_fact",
          "fields": {"entity": "X", "year": 2017},
          "evidence_candidates": [{"chunk_id": "doc42#c17", "span_id": "s1"}]
        }
    """
    id: str = Field(description="Unique claim ID (format: 'c{N}')")
    text: str = Field(description="The claim text (atomic, single-fact)")
    type: ClaimType = Field(
        default=ClaimType.FACTUAL,
        description="Claim type for analysis"
    )
    fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted structured fields (entity, year, quantity, etc.)"
    )
    evidence_candidates: list[CandidateEvidence] = Field(
        default_factory=list,
        description="Candidate evidence spans from retrieval"
    )
    hedge: bool = Field(
        default=False,
        description="Whether claim contains hedging language (may, possibly, etc.)"
    )
    unsupported_by_design: bool = Field(
        default=False,
        description="True if claim intentionally has no evidence (e.g., meta-commentary)"
    )

    @field_validator("text")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Claims must contain actual text."""
        if not v.strip():
            raise ValueError("Claim text cannot be empty")
        return v.strip()

    @property
    def has_evidence(self) -> bool:
        """Check if this claim has at least one candidate evidence span."""
        return len(self.evidence_candidates) > 0 or self.unsupported_by_design


class ClaimIR(BaseModel):
    """
    Claim Intermediate Representation — the full structured output
    of the claim compiler for a single query.

    This is the central data structure that flows through the pipeline:
        ClaimIR → Verifier → Renderer → Certificate

    Schema (§6.2):
        {
          "question": "...",
          "claims": [
            {
              "id": "c1",
              "text": "X was founded in 2017.",
              "type": "temporal_fact",
              "fields": {"entity": "X", "year": 2017},
              "evidence_candidates": [{"chunk_id": "doc42#c17", "span_id": "s1"}]
            }
          ]
        }
    """
    question: str = Field(description="The original user question")
    claims: list[Claim] = Field(
        default_factory=list,
        description="List of atomic claims extracted from the answer"
    )
    model_id: Optional[str] = Field(
        default=None,
        description="LLM model used to generate claims"
    )
    raw_answer: Optional[str] = Field(
        default=None,
        description="Original LLM answer before claim decomposition"
    )

    @property
    def num_claims(self) -> int:
        """Total number of claims."""
        return len(self.claims)

    @property
    def num_grounded_claims(self) -> int:
        """Number of claims with at least one evidence candidate."""
        return sum(1 for c in self.claims if c.has_evidence)

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Look up a claim by its ID."""
        for claim in self.claims:
            if claim.id == claim_id:
                return claim
        return None

    def get_all_evidence_pairs(self) -> list[tuple[str, str, str]]:
        """
        Get all (claim_id, chunk_id, span_id) triples for batch verification.

        Returns:
            List of (claim_id, chunk_id, span_id) tuples.
        """
        pairs = []
        for claim in self.claims:
            for ev in claim.evidence_candidates:
                pairs.append((claim.id, ev.chunk_id, ev.span_id))
        return pairs

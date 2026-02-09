"""
Certificate / Renderer Decision Schema (§6.4)
===============================================

Defines the output of the renderer's policy engine: for each claim,
a deterministic decision about whether to display it as Verified,
Unverified, or Blocked.

The AuditCertificate is the complete audit trail for a single query,
containing everything needed to reproduce and verify the result.

Design Philosophy:
    The certificate is the machine-readable "proof" that the renderer
    correctly applied the verification policy. It enables:
    - Post-hoc auditing ("was this claim correctly verified?")
    - Reproducibility ("same config + data → same certificate")
    - Non-spoofability ("only the renderer assigns Verified badges")

    This is inspired by Proof-Carrying Numbers (PCN), where verification
    is enforced at the presentation layer.

Data Flow:
    VerificationResult + Policy → RendererDecision → AuditCertificate
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from certirag.utils import compute_content_hash


class RenderState(str, Enum):
    """
    Rendering state assigned by the policy engine.

    - VERIFIED:   Claim passed all checks; displayed with ✅ badge
    - UNVERIFIED: Insufficient evidence; hidden (strict) or warned (mixed)
    - BLOCKED:    Contradicted by evidence; never displayed in strict/mixed

    The renderer is the ONLY component that can assign these states.
    The LLM's output cannot influence the rendering state.
    """
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    BLOCKED = "BLOCKED"


class PolicySnapshot(BaseModel):
    """
    Snapshot of the verification policy used for this decision.

    Stored in every certificate for reproducibility.
    Two decisions made with the same PolicySnapshot and same
    VerificationResult will always produce the same RenderState.
    """
    tau_entail: float = Field(description="Entailment threshold")
    tau_contradict: float = Field(description="Contradiction blocking threshold")
    min_evidence_spans: int = Field(
        default=1,
        description="Minimum evidence spans required for Verified"
    )
    policy_version: str = Field(
        default="v1.0",
        description="Policy version identifier"
    )


class RendererDecision(BaseModel):
    """
    The renderer's decision for a single claim.

    This is deterministic: given the same VerificationResult and
    PolicySnapshot, the same RendererDecision is always produced.

    Schema (§6.4):
        {
          "claim_id": "c1",
          "render_state": "VERIFIED",
          "policy": {"tau_entail": 0.85, "tau_contradict": 0.70},
          "reason": "entailed>=tau_entail and no contradiction>=tau_contradict"
        }
    """
    claim_id: str = Field(description="Reference to Claim.id")
    render_state: RenderState = Field(description="Rendering decision")
    policy: PolicySnapshot = Field(description="Policy used for this decision")
    reason: str = Field(description="Human-readable explanation of the decision")
    entail_score: float = Field(
        default=0.0,
        description="Max entailment score (for debugging)"
    )
    contradict_score: float = Field(
        default=0.0,
        description="Max contradiction score (for debugging)"
    )
    evidence_count: int = Field(
        default=0,
        description="Number of evidence spans available"
    )


class AuditCertificate(BaseModel):
    """
    Complete audit trail for a single CertiRAG query.

    Contains everything needed to:
    1. Reproduce the result (config hash, model versions)
    2. Verify the result (all verification scores, decisions)
    3. Audit the result (evidence chains, policy justifications)

    This is the "certificate" in CertiRAG's name. It makes the
    system auditable and inspectable, unlike traditional RAG.

    The certificate includes an integrity hash computed over its
    content (excluding the hash field itself), enabling tamper
    detection.
    """
    # ── Query & Response ───────────────────────────────────────────
    query_id: str = Field(description="Unique query identifier")
    question: str = Field(description="Original user question")
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        description="ISO 8601 timestamp"
    )

    # ── Retrieval ──────────────────────────────────────────────────
    retrieval_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of retrieved chunks (count, top scores, etc.)"
    )

    # ── Claims & Verification ──────────────────────────────────────
    claims: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Serialized ClaimIR claims"
    )
    verification_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Serialized VerificationResults"
    )
    renderer_decisions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Serialized RendererDecisions"
    )

    # ── Provenance ─────────────────────────────────────────────────
    config_hash: str = Field(
        default="",
        description="SHA-256 hash of the configuration used"
    )
    model_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Model name → version/hash mapping"
    )
    policy: Optional[PolicySnapshot] = Field(
        default=None,
        description="Policy snapshot used for rendering"
    )

    # ── Summary Statistics ─────────────────────────────────────────
    stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics (counts, latencies, etc.)"
    )

    # ── Integrity ──────────────────────────────────────────────────
    integrity_hash: str = Field(
        default="",
        description="SHA-256 of certificate content (tamper detection)"
    )

    def compute_integrity_hash(self) -> str:
        """
        Compute the integrity hash over all certificate content
        except the integrity_hash field itself.

        Returns:
            SHA-256 hex digest of the certificate content.
        """
        content = self.model_dump(exclude={"integrity_hash"})
        return compute_content_hash(content)

    def seal(self) -> "AuditCertificate":
        """
        Seal the certificate by computing and storing the integrity hash.

        Call this after all fields are populated. The resulting certificate
        can be verified by recomputing the hash and comparing.

        Returns:
            Self, with integrity_hash populated.
        """
        self.integrity_hash = self.compute_integrity_hash()
        return self

    def verify_integrity(self) -> bool:
        """
        Verify that the certificate has not been tampered with.

        Returns:
            True if the integrity hash matches the content.
        """
        if not self.integrity_hash:
            return False
        return self.integrity_hash == self.compute_integrity_hash()

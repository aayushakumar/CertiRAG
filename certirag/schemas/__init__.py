"""
CertiRAG Data Schemas
======================

Pydantic v2 models implementing the four core data contracts
defined in the CertiRAG specification (§6 of README):

1. EvidenceChunk  — Retrieved evidence with provenance + spans
2. ClaimIR        — Structured claim intermediate representation
3. VerificationResult — Verifier output per claim
4. RendererDecision   — Renderer policy decision (the certificate)

All schemas support:
- Runtime validation with Pydantic
- JSON Schema export for interoperability
- Serialization/deserialization for audit trails
"""

from certirag.schemas.evidence import (
    EvidenceChunk,
    EvidenceSpan,
    RetrievalScores,
    SourceInfo,
)
from certirag.schemas.claim_ir import (
    Claim,
    ClaimIR,
    ClaimType,
    CandidateEvidence,
)
from certirag.schemas.verification import (
    EvidenceScore,
    VerificationLabel,
    VerificationResult,
)
from certirag.schemas.certificate import (
    AuditCertificate,
    PolicySnapshot,
    RenderState,
    RendererDecision,
)

__all__ = [
    # Evidence
    "EvidenceChunk",
    "EvidenceSpan",
    "RetrievalScores",
    "SourceInfo",
    # Claim IR
    "Claim",
    "ClaimIR",
    "ClaimType",
    "CandidateEvidence",
    # Verification
    "EvidenceScore",
    "VerificationLabel",
    "VerificationResult",
    # Certificate
    "AuditCertificate",
    "PolicySnapshot",
    "RenderState",
    "RendererDecision",
]

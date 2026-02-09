"""
Certificate Generator
======================

Creates and manages AuditCertificates — the machine-readable
proof that CertiRAG correctly applied its verification policy.

A certificate contains the complete audit trail:
    - Question + retrieval summary
    - All claims and verification results
    - All renderer decisions with policy justifications
    - Configuration hash for reproducibility
    - Integrity hash for tamper detection

Usage:
    builder = CertificateBuilder(config)
    cert = builder.build(
        question="...",
        claim_ir=claim_ir,
        evidence_chunks=chunks,
        verification_results=results,
        renderer_decisions=decisions,
    )
    cert_json = builder.export_json(cert, "output/cert.json")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from certirag.config import CertiRAGConfig
from certirag.schemas.certificate import AuditCertificate, PolicySnapshot
from certirag.schemas.claim_ir import ClaimIR
from certirag.schemas.evidence import EvidenceChunk
from certirag.schemas.certificate import RendererDecision
from certirag.schemas.verification import VerificationResult
from certirag.utils import compute_content_hash, generate_run_id

logger = logging.getLogger("certirag.render.certificate")


class CertificateBuilder:
    """
    Builds complete AuditCertificates from pipeline outputs.

    The certificate is the "receipt" proving correct policy application.
    It enables post-hoc auditing and reproducibility.

    Args:
        config: CertiRAG configuration.
    """

    def __init__(self, config: CertiRAGConfig):
        self.config = config

    def build(
        self,
        question: str,
        claim_ir: ClaimIR,
        evidence_chunks: list[EvidenceChunk],
        verification_results: list[VerificationResult],
        renderer_decisions: list[RendererDecision],
        policy: Optional[PolicySnapshot] = None,
        latency_ms: Optional[float] = None,
        extra_metadata: Optional[dict] = None,
    ) -> AuditCertificate:
        """
        Build a complete AuditCertificate.

        Args:
            question: Original user question.
            claim_ir: Compiled ClaimIR.
            evidence_chunks: Retrieved evidence chunks.
            verification_results: Verifier outputs.
            renderer_decisions: Renderer decisions.
            policy: Policy snapshot used.
            latency_ms: Total pipeline latency in milliseconds.
            extra_metadata: Additional metadata to include.

        Returns:
            Sealed AuditCertificate with integrity hash.
        """
        from certirag.schemas.certificate import RenderState

        # Retrieval summary (compact, not full chunk text)
        retrieval_summary = {
            "num_chunks": len(evidence_chunks),
            "chunk_ids": [c.chunk_id for c in evidence_chunks],
            "total_spans": sum(len(c.spans) for c in evidence_chunks),
            "top_scores": {
                c.chunk_id: {
                    "bm25": c.retrieval.bm25,
                    "dense": c.retrieval.dense,
                    "rrf": c.retrieval.rrf,
                    "rerank": c.retrieval.rerank,
                }
                for c in evidence_chunks[:5]
            },
        }

        # Compute stats
        stats = {
            "num_claims": claim_ir.num_claims,
            "num_grounded_claims": claim_ir.num_grounded_claims,
            "num_verified": sum(
                1 for d in renderer_decisions
                if d.render_state == RenderState.VERIFIED
            ),
            "num_unverified": sum(
                1 for d in renderer_decisions
                if d.render_state == RenderState.UNVERIFIED
            ),
            "num_blocked": sum(
                1 for d in renderer_decisions
                if d.render_state == RenderState.BLOCKED
            ),
            "num_verifier_pairs": sum(
                len(r.all_scores) for r in verification_results
            ),
        }

        if latency_ms is not None:
            stats["latency_ms"] = latency_ms

        # Model versions
        model_versions = {
            "claim_compiler": self.config.claim.llm_model,
            "verifier": self.config.verification.verifier_model,
            "embedding": self.config.retrieval.embedding_model,
        }

        if self.config.is_full:
            model_versions["reranker"] = self.config.retrieval.reranker_model

        certificate = AuditCertificate(
            query_id=generate_run_id(),
            question=question,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            retrieval_summary=retrieval_summary,
            claims=[c.model_dump() for c in claim_ir.claims],
            verification_results=[r.model_dump() for r in verification_results],
            renderer_decisions=[d.model_dump() for d in renderer_decisions],
            config_hash=self.config.config_hash(),
            model_versions=model_versions,
            policy=policy,
            stats=stats,
        )

        if extra_metadata:
            certificate.stats.update(extra_metadata)

        # Seal the certificate (compute integrity hash)
        certificate.seal()

        logger.info(
            f"Certificate built: {stats['num_verified']} verified, "
            f"{stats['num_unverified']} unverified, "
            f"{stats['num_blocked']} blocked"
        )

        return certificate

    def export_json(
        self,
        certificate: AuditCertificate,
        path: str | Path,
    ) -> Path:
        """
        Export certificate as JSON file.

        Args:
            certificate: The audit certificate to export.
            path: Output file path.

        Returns:
            Path to the exported file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                certificate.model_dump(),
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        logger.info(f"Certificate exported to {path}")
        return path

    def verify_certificate(self, path: str | Path) -> dict[str, Any]:
        """
        Load and verify a certificate's integrity.

        Args:
            path: Path to certificate JSON file.

        Returns:
            Dict with 'valid' (bool), 'certificate' (AuditCertificate),
            and 'errors' (list of issues found).
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cert = AuditCertificate.model_validate(data)
        errors = []

        if not cert.verify_integrity():
            errors.append("Integrity hash mismatch — certificate may have been tampered with")

        if not cert.config_hash:
            errors.append("Missing config hash — reproducibility not guaranteed")

        return {
            "valid": len(errors) == 0,
            "certificate": cert,
            "errors": errors,
        }

"""
CertiRAG UI — Pipeline Runner
================================

Handles pipeline execution for the demo UI.
Uses a BM25 + word-overlap heuristic for verification
(no GPU/API required) so the demo works everywhere.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from certirag.claim_ir.normalizer import ClaimNormalizer
from certirag.ingest.chunker import DocumentChunker
from certirag.ingest.indexer import BM25Index, ChunkStore
from certirag.render.policy import RendererPolicy
from certirag.schemas.certificate import (
    AuditCertificate,
    RenderState,
)
from certirag.schemas.claim_ir import (
    CandidateEvidence,
    Claim,
    ClaimIR,
    ClaimType,
)
from certirag.schemas.evidence import EvidenceChunk
from certirag.schemas.verification import (
    EvidenceScore,
    VerificationLabel,
    VerificationResult,
)

# ── Public API ──────────────────────────────────────────────────

def run_pipeline(
    question: str,
    documents: str,
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Execute the CertiRAG demo pipeline.

    Returns a dict with all pipeline outputs, or None on error.
    The demo uses heuristic scoring (word overlap) instead of
    an NLI model so it runs without GPU/API.
    """
    t0 = time.time()

    # ─── Step 1: Ingest ───────────────────────────────────────
    docs = _parse_documents(documents)
    chunker = DocumentChunker(chunk_size=200, overlap=0, use_spacy=False)
    chunks = chunker.chunk_documents(docs)

    # ─── Step 2: Index & Retrieve ─────────────────────────────
    index = BM25Index()
    index.build(chunks)
    results = index.query(question, top_k=settings.get("top_k", 5))

    store = ChunkStore()
    store.add_many(chunks)
    evidence: list[EvidenceChunk] = [
        store.get(cid)
        for cid, _ in results
        if store.get(cid) is not None
    ]

    # Fallback: if BM25 returns nothing (e.g. single doc), use all chunks
    if not evidence:
        evidence = chunks[: settings.get("top_k", 5)]

    # ─── Step 3: Claim Extraction (heuristic) ─────────────────
    claims = _extract_claims(question, evidence)
    claim_ir = ClaimIR(question=question, claims=claims)

    # ─── Step 4: Normalize ────────────────────────────────────
    normalizer = ClaimNormalizer(split_conjunctions=True)
    claim_ir = normalizer.normalize(claim_ir)

    # ─── Step 5: Verification (word-overlap heuristic) ────────
    verification_results = _heuristic_verify(claim_ir, evidence)

    # ─── Step 6: Render ───────────────────────────────────────
    policy = RendererPolicy(
        tau_entail=settings.get("tau_entail", 0.5),
        tau_contradict=settings.get("tau_contradict", 0.7),
    )
    decisions = policy.render(verification_results)
    display = policy.filter_for_display(
        claim_ir, decisions, settings.get("display_mode", "mixed"),
    )

    # ─── Step 7: Audit Certificate ────────────────────────────
    elapsed = time.time() - t0
    certificate = _build_certificate(
        question, claim_ir, verification_results, decisions, policy, elapsed,
    )

    return {
        "claim_ir": claim_ir,
        "evidence": evidence,
        "chunks": chunks,
        "verification_results": verification_results,
        "decisions": decisions,
        "display": display,
        "certificate": certificate,
        "elapsed": elapsed,
        "pipeline_steps": [
            ("Ingest", len(chunks)),
            ("BM25 Retrieve", len(evidence)),
            ("Claim Extract", len(claim_ir.claims)),
            ("Normalize", len(claim_ir.claims)),
            ("Verify", len(verification_results)),
            ("Render", len(decisions)),
            ("Certify", 1),
        ],
    }


# ── Internal Helpers ────────────────────────────────────────────

def _parse_documents(documents: str) -> list[dict[str, str]]:
    """Parse document text into a list of {doc_id, text} dicts."""
    paragraphs = [p.strip() for p in documents.split("\n") if p.strip()]
    if not paragraphs:
        return [{"doc_id": "doc_0", "text": documents}]
    return [
        {"doc_id": f"doc_{i}", "text": para}
        for i, para in enumerate(paragraphs)
    ]


def _extract_claims(
    question: str,
    evidence: list[EvidenceChunk],
) -> list[Claim]:
    """
    Extract claims from the question using a simple heuristic.

    In production, this would use an LLM. For the demo, we split
    on question marks and sentence boundaries.
    """
    # Split by question-mark or period
    parts = []
    for seg in question.replace("?", "?\n").replace(".", ".\n").split("\n"):
        seg = seg.strip()
        if seg and len(seg) > 5:
            parts.append(seg)
    if not parts:
        parts = [question]

    claims = []
    for i, text in enumerate(parts):
        # Build candidate evidence from top-k retrieved chunks
        candidates = [
            CandidateEvidence(
                chunk_id=ev.chunk_id,
                span_id=ev.spans[0].span_id if ev.spans else f"s{j}",
            )
            for j, ev in enumerate(evidence[:3])
        ]

        claims.append(Claim(
            id=f"c{i}",
            text=text,
            type=ClaimType.FACTUAL,
            evidence_candidates=candidates,
        ))

    return claims


def _heuristic_verify(
    claim_ir: ClaimIR,
    evidence: list[EvidenceChunk],
) -> list[VerificationResult]:
    """
    Verify claims using word-overlap heuristic.

    In production, this would use a cross-encoder NLI model.
    The heuristic computes Jaccard-like overlap between claim
    words and evidence words to produce entailment/contradiction
    scores.
    """
    results = []
    for claim in claim_ir.claims:
        all_scores: list[EvidenceScore] = []
        claim_words = set(claim.text.lower().split())

        for ev in evidence[:5]:
            ev_words = set(ev.text.lower().split())
            overlap = len(claim_words & ev_words) / max(len(claim_words | ev_words), 1)

            # Heuristic: high overlap → entailed, low → not enough info
            entail_score = min(overlap * 2.0, 1.0)
            contradict_score = max(0.0, 0.05 - overlap * 0.1)

            # Determine label based on scores
            if entail_score >= 0.5:
                label = VerificationLabel.ENTAILED
                score = entail_score
            elif contradict_score >= 0.5:
                label = VerificationLabel.CONTRADICTED
                score = contradict_score
            else:
                label = VerificationLabel.NOT_ENOUGH_INFO
                score = 1.0 - entail_score

            span_id = ev.spans[0].span_id if ev.spans else "s0"
            all_scores.append(EvidenceScore(
                chunk_id=ev.chunk_id,
                span_id=span_id,
                label=label,
                score=score,
            ))

        # Overall label: pick the best evidence
        entail_scores = [s for s in all_scores if s.label == VerificationLabel.ENTAILED]
        contradict_scores = [s for s in all_scores if s.label == VerificationLabel.CONTRADICTED]

        if entail_scores:
            best = max(entail_scores, key=lambda s: s.score)
            overall_label = VerificationLabel.ENTAILED
            overall_score = best.score
        elif contradict_scores:
            best = max(contradict_scores, key=lambda s: s.score)
            overall_label = VerificationLabel.CONTRADICTED
            overall_score = best.score
        else:
            overall_label = VerificationLabel.NOT_ENOUGH_INFO
            overall_score = 0.3

        results.append(VerificationResult(
            claim_id=claim.id,
            label=overall_label,
            score=overall_score,
            best_evidence=all_scores[:3],
            all_scores=all_scores,
        ))

    return results


def _build_certificate(
    question: str,
    claim_ir: ClaimIR,
    verification_results: list[VerificationResult],
    decisions: list,
    policy: RendererPolicy,
    elapsed: float,
) -> AuditCertificate:
    """Build an audit certificate for the pipeline run."""
    query_id = hashlib.sha256(
        f"{question}:{time.time()}".encode()
    ).hexdigest()[:16]

    verified_count = sum(
        1 for d in decisions if d.render_state == RenderState.VERIFIED
    )
    blocked_count = sum(
        1 for d in decisions if d.render_state == RenderState.BLOCKED
    )

    cert = AuditCertificate(
        query_id=query_id,
        question=question,
        claims=[c.model_dump() for c in claim_ir.claims],
        verification_results=[v.model_dump() for v in verification_results],
        renderer_decisions=[d.model_dump() for d in decisions],
        policy=policy.policy_snapshot,
        stats={
            "total_claims": len(claim_ir.claims),
            "verified": verified_count,
            "blocked": blocked_count,
            "unverified": len(decisions) - verified_count - blocked_count,
            "latency_ms": round(elapsed * 1000, 1),
        },
        model_versions={
            "chunker": "regex-v1",
            "retriever": "bm25-demo",
            "verifier": "heuristic-overlap-v1",
            "renderer": f"policy-{policy.policy_version}",
        },
    )

    return cert.seal()

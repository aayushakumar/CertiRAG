"""
Verifier Interface
===================

Abstract base class for all CertiRAG verifiers. Every verifier
(MiniCheck, NLI, LLM-judge) must implement this interface.

The interface ensures:
- Consistent input/output format across verifier backends
- Easy swapping for ablation experiments
- Batch processing support for efficiency

Data Flow:
    [(claim_text, evidence_text)] → Verifier → [VerificationResult]
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from certirag.schemas.claim_ir import ClaimIR
from certirag.schemas.evidence import EvidenceChunk
from certirag.schemas.verification import (
    EvidenceScore,
    VerificationLabel,
    VerificationResult,
)

logger = logging.getLogger("certirag.verify.verifier")


class BaseVerifier(ABC):
    """
    Abstract base class for claim verifiers.

    All verifiers implement the same interface:
        verify_batch(pairs) → list[EvidenceScore]

    Where each pair is (claim_text, evidence_text).

    The higher-level `verify_claims()` method orchestrates:
    1. Collecting all (claim, evidence) pairs from ClaimIR + chunks
    2. Applying the cost cap (max_pairs)
    3. Calling verify_batch() for scoring
    4. Aggregating scores into VerificationResult objects
    """

    def __init__(self, model_name: str = "base", max_pairs: int = 240):
        self.model_name = model_name
        self.max_pairs = max_pairs

    @abstractmethod
    def verify_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[EvidenceScore]:
        """
        Score a batch of (claim, evidence) pairs.

        Args:
            pairs: List of (claim_text, evidence_text) tuples.

        Returns:
            List of EvidenceScore objects (same length as pairs).
            Each score has label, score, and evidence references.

        Note:
            Implementations should handle batching internally
            for GPU efficiency.
        """
        ...

    def verify_claims(
        self,
        claim_ir: ClaimIR,
        evidence_chunks: list[EvidenceChunk],
        max_pairs: Optional[int] = None,
    ) -> list[VerificationResult]:
        """
        Verify all claims against their candidate evidence spans.

        Pipeline:
            1. Collect all (claim, span) pairs from ClaimIR
            2. Look up evidence text from chunks
            3. Apply cost cap (truncate if needed)
            4. Call verify_batch() for scoring
            5. Aggregate into VerificationResult per claim

        Args:
            claim_ir: ClaimIR with claims and evidence candidates.
            evidence_chunks: Retrieved evidence chunks (for text lookup).
            max_pairs: Override max pairs cap.

        Returns:
            List of VerificationResult objects (one per claim).
        """
        max_pairs = max_pairs or self.max_pairs

        # Build chunk/span lookup
        chunk_map: dict[str, EvidenceChunk] = {c.chunk_id: c for c in evidence_chunks}
        span_text_map: dict[tuple[str, str], str] = {}
        for chunk in evidence_chunks:
            for span in chunk.spans:
                span_text_map[(chunk.chunk_id, span.span_id)] = span.sentence

        # Collect all (claim, evidence) pairs
        all_pairs: list[tuple[str, str, str, str]] = []  # (claim_id, chunk_id, span_id, ...)
        pair_texts: list[tuple[str, str]] = []

        for claim in claim_ir.claims:
            for ec in claim.evidence_candidates:
                evidence_text = span_text_map.get((ec.chunk_id, ec.span_id))
                if evidence_text:
                    all_pairs.append((claim.id, ec.chunk_id, ec.span_id, evidence_text))
                    pair_texts.append((claim.text, evidence_text))

        # Apply cost cap (FR-V1: cap total verifier pairs)
        if len(pair_texts) > max_pairs:
            logger.warning(
                f"Cost cap: truncating {len(pair_texts)} pairs to {max_pairs} "
                f"(dropping lowest-ranked evidence first)"
            )
            pair_texts = pair_texts[:max_pairs]
            all_pairs = all_pairs[:max_pairs]

        logger.info(f"Verifying {len(pair_texts)} (claim, evidence) pairs")
        start_time = time.time()

        # Score all pairs
        if pair_texts:
            raw_scores = self.verify_batch(pair_texts)
        else:
            raw_scores = []

        elapsed = time.time() - start_time
        logger.info(f"Verification completed in {elapsed:.2f}s")

        # Enrich scores with chunk/span references
        enriched_scores: list[EvidenceScore] = []
        for i, score in enumerate(raw_scores):
            claim_id, chunk_id, span_id, _ = all_pairs[i]
            enriched_scores.append(EvidenceScore(
                chunk_id=chunk_id,
                span_id=span_id,
                label=score.label,
                score=score.score,
                raw_score=score.raw_score,
            ))

        # Aggregate scores by claim
        claim_scores: dict[str, list[EvidenceScore]] = {}
        for i, (claim_id, _, _, _) in enumerate(all_pairs):
            if claim_id not in claim_scores:
                claim_scores[claim_id] = []
            if i < len(enriched_scores):
                claim_scores[claim_id].append(enriched_scores[i])

        # Build VerificationResults
        results: list[VerificationResult] = []
        for claim in claim_ir.claims:
            scores = claim_scores.get(claim.id, [])
            result = self._aggregate_scores(claim.id, scores)
            results.append(result)

        return results

    def _aggregate_scores(
        self, claim_id: str, scores: list[EvidenceScore]
    ) -> VerificationResult:
        """
        Aggregate per-span scores into a single VerificationResult.

        Aggregation rule:
        - Label is determined by the highest-scoring span
        - If best entailment > best contradiction → entailed
        - If best contradiction > best entailment → contradicted
        - Otherwise → not_enough_info
        """
        if not scores:
            return VerificationResult(
                claim_id=claim_id,
                label=VerificationLabel.NOT_ENOUGH_INFO,
                score=0.0,
                best_evidence=[],
                all_scores=[],
            )

        # Find best scores per label
        max_entail = max(
            (s.score for s in scores if s.label == VerificationLabel.ENTAILED),
            default=0.0,
        )
        max_contradict = max(
            (s.score for s in scores if s.label == VerificationLabel.CONTRADICTED),
            default=0.0,
        )

        # Determine overall label
        if max_entail >= max_contradict and max_entail > 0:
            label = VerificationLabel.ENTAILED
            score = max_entail
        elif max_contradict > max_entail:
            label = VerificationLabel.CONTRADICTED
            score = max_contradict
        else:
            label = VerificationLabel.NOT_ENOUGH_INFO
            score = max(s.score for s in scores) if scores else 0.0

        # Best evidence = entailing spans sorted by score
        best_evidence = sorted(
            [s for s in scores if s.label == VerificationLabel.ENTAILED],
            key=lambda s: s.score,
            reverse=True,
        )

        return VerificationResult(
            claim_id=claim_id,
            label=label,
            score=score,
            best_evidence=best_evidence[:5],  # top-5 for MSE
            all_scores=scores,
        )

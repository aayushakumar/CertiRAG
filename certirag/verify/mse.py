"""
Minimal Sufficient Evidence (MSE) Selection
=============================================

For each verified claim, selects the smallest set of evidence spans
that achieves the entailment threshold (§7.2 of README).

Purpose:
    - Reduces certificate size (fewer spans to audit)
    - Highlights only the most relevant evidence in the UI
    - Enables precise ablation: top-1 vs top-k vs MSE

Algorithm (greedy):
    1. Sort spans by entailment score (descending)
    2. Add spans until cumulative best score >= τ_entail
    3. Return the selected spans

Data Flow:
    VerificationResult → MSE Selection → trimmed VerificationResult
"""

from __future__ import annotations

import logging

from certirag.schemas.verification import (
    EvidenceScore,
    VerificationLabel,
    VerificationResult,
)

logger = logging.getLogger("certirag.verify.mse")


class MSESelector:
    """
    Minimal Sufficient Evidence selector.

    Given a VerificationResult with all span scores, selects the
    smallest subset of entailing spans that achieves the threshold.

    Usage:
        selector = MSESelector(tau_entail=0.85)
        trimmed = selector.select(verification_result)
        # trimmed.best_evidence contains only MSE spans

    Args:
        tau_entail: Entailment threshold to achieve.
        max_spans: Maximum spans in MSE set (hard cap).
    """

    def __init__(self, tau_entail: float = 0.85, max_spans: int = 5):
        self.tau_entail = tau_entail
        self.max_spans = max_spans

    def select(self, result: VerificationResult) -> VerificationResult:
        """
        Select Minimal Sufficient Evidence for a single claim.

        Greedy algorithm:
            1. Filter to entailing spans only
            2. Sort by entailment score descending
            3. Select greedily until threshold is met

        Args:
            result: VerificationResult with all_scores populated.

        Returns:
            Modified VerificationResult with best_evidence = MSE set.
        """
        # Get all entailing spans sorted by score
        entailing_spans = sorted(
            [s for s in result.all_scores if s.label == VerificationLabel.ENTAILED],
            key=lambda s: s.score,
            reverse=True,
        )

        if not entailing_spans:
            result.best_evidence = []
            return result

        # Greedy selection: add spans until threshold is met
        selected: list[EvidenceScore] = []
        best_score = 0.0

        for span in entailing_spans:
            if len(selected) >= self.max_spans:
                break

            selected.append(span)
            best_score = max(best_score, span.score)

            # Once we have a span meeting threshold, we can stop
            if best_score >= self.tau_entail:
                break

        result.best_evidence = selected
        logger.debug(
            f"MSE for {result.claim_id}: {len(selected)} spans "
            f"(best score: {best_score:.3f})"
        )
        return result

    def select_batch(
        self, results: list[VerificationResult]
    ) -> list[VerificationResult]:
        """Apply MSE selection to a batch of VerificationResults."""
        return [self.select(r) for r in results]

    def compute_stats(
        self, results: list[VerificationResult]
    ) -> dict[str, float]:
        """
        Compute MSE statistics for reporting.

        Returns:
            Dict with avg_spans, max_spans, total_spans, pct_single_span.
        """
        span_counts = [len(r.best_evidence) for r in results]
        if not span_counts:
            return {"avg_spans": 0, "max_spans": 0, "total_spans": 0, "pct_single_span": 0}

        return {
            "avg_spans": sum(span_counts) / len(span_counts),
            "max_spans": max(span_counts),
            "total_spans": sum(span_counts),
            "pct_single_span": sum(1 for c in span_counts if c == 1) / len(span_counts),
        }

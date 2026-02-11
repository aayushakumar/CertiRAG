"""
Threshold & Weight Optimizer for SOTA Fact Verification
=========================================================

Implements optimal threshold selection, ensemble weight tuning,
and cross-validation to squeeze the best performance from NLI models.

Key techniques:
    1. **Per-class threshold optimization** — Instead of argmax over
       [entail, neutral, contradict], find optimal thresholds that
       maximize macro-F1 on calibration data.

    2. **Ensemble weight tuning** — Grid search over model weights
       to maximize AUROC / accuracy on calibration data.

    3. **Stratified split** — Uses stratified sampling so calibration
       and test sets have balanced label distributions.

Usage
-----
    from eval.optimizer import ThresholdOptimizer, optimize_ensemble_weights

    opt = ThresholdOptimizer()
    opt.fit(cal_predictions)               # learn thresholds
    adjusted = opt.apply(test_predictions)  # apply to test set
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from typing import Optional

import numpy as np
from eval.scoring import (
    BenchmarkPrediction,
    BenchmarkScores,
    score_predictions,
    _safe_div,
)

logger = logging.getLogger("certirag.eval.optimizer")


@dataclass
class OptimizationResult:
    """Result from threshold or weight optimization."""

    best_params: dict
    best_metric: float
    metric_name: str
    cal_scores: Optional[BenchmarkScores] = None
    test_scores: Optional[BenchmarkScores] = None


class ThresholdOptimizer:
    """
    Finds optimal decision thresholds for NLI classification.

    Instead of argmax over the probability distribution, this optimizer
    finds per-class thresholds that maximize a target metric (macro-F1
    or 3-class accuracy) on calibration data.

    The key insight: NLI models have miscalibrated probability distributions
    where the argmax label isn't always the best choice. A higher threshold
    for entailment can reduce false positives and improve precision.

    Strategy:
        1. Use entailment_score (raw_score from verifier) as primary signal
        2. Use pred_score (confidence in predicted class) as secondary
        3. Grid search over entailment threshold: if entailment_score > thresh → entailed
        4. For remaining examples, use the model's predicted label

    This is much more effective than just moving argmax boundaries.
    """

    def __init__(self, metric: str = "macro_f1", n_steps: int = 50):
        """
        Args:
            metric: Target metric to optimize ("macro_f1", "accuracy_3class",
                    "accuracy_binary").
            n_steps: Number of threshold values to try per dimension.
        """
        self.metric = metric
        self.n_steps = n_steps
        self.entail_threshold: float = 0.5
        self.contradict_threshold: float = 0.5
        self._fitted = False

    def fit(self, predictions: list[BenchmarkPrediction]) -> OptimizationResult:
        """
        Find optimal thresholds using grid search.

        Searches over entailment_threshold:
            if entailment_score >= entail_threshold → "entailed"
            elif pred_label == "contradicted" and pred_score >= contradict_threshold → "contradicted"
            else → the model's pred_label (or "not_enough_info")

        Args:
            predictions: Calibration predictions with entailment_score.

        Returns:
            OptimizationResult with best thresholds and achieved metric.
        """
        if not predictions:
            raise ValueError("Need calibration predictions")

        # Collect scores
        entail_scores = np.array([p.entailment_score for p in predictions])
        pred_scores = np.array([p.pred_score for p in predictions])

        # Grid search
        best_metric_val = -1.0
        best_et = 0.5
        best_ct = 0.5

        # Entailment thresholds: search around the distribution
        et_min = max(0.1, float(np.percentile(entail_scores, 10)))
        et_max = min(0.95, float(np.percentile(entail_scores, 90)))
        et_range = np.linspace(et_min, et_max, self.n_steps)

        # Contradiction thresholds
        ct_range = np.linspace(0.3, 0.95, max(self.n_steps // 3, 10))

        for et in et_range:
            for ct in ct_range:
                adjusted = self._apply_thresholds(predictions, et, ct)
                scores = score_predictions(adjusted, dataset="cal", model_name="opt")
                val = getattr(scores, self.metric, 0.0)
                if val > best_metric_val:
                    best_metric_val = val
                    best_et = et
                    best_ct = ct

        self.entail_threshold = best_et
        self.contradict_threshold = best_ct
        self._fitted = True

        logger.info(
            f"Optimal thresholds: entail={best_et:.4f}, "
            f"contradict={best_ct:.4f}, {self.metric}={best_metric_val:.4f}"
        )

        # Compute cal scores with best thresholds
        cal_adjusted = self._apply_thresholds(predictions, best_et, best_ct)
        cal_scores = score_predictions(cal_adjusted, dataset="cal", model_name="optimized")

        return OptimizationResult(
            best_params={
                "entail_threshold": best_et,
                "contradict_threshold": best_ct,
            },
            best_metric=best_metric_val,
            metric_name=self.metric,
            cal_scores=cal_scores,
        )

    def _apply_thresholds(
        self,
        predictions: list[BenchmarkPrediction],
        entail_thresh: float,
        contradict_thresh: float,
    ) -> list[BenchmarkPrediction]:
        """Apply given thresholds to produce adjusted predictions."""
        adjusted = []
        for p in predictions:
            if p.entailment_score >= entail_thresh:
                new_label = "entailed"
            elif p.pred_label == "contradicted" and p.pred_score >= contradict_thresh:
                new_label = "contradicted"
            elif p.pred_label == "contradicted":
                # Model says contradicted but low confidence → keep as contradicted
                # only if entailment score is very low
                if p.entailment_score < 0.3:
                    new_label = "contradicted"
                else:
                    new_label = "not_enough_info"
            else:
                new_label = p.pred_label

            adjusted.append(BenchmarkPrediction(
                example_id=p.example_id,
                gold_label=p.gold_label,
                pred_label=new_label,
                pred_score=p.pred_score,
                entailment_score=p.entailment_score,
                latency_ms=p.latency_ms,
            ))
        return adjusted

    def apply(self, predictions: list[BenchmarkPrediction]) -> list[BenchmarkPrediction]:
        """Apply fitted thresholds to test predictions."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        return self._apply_thresholds(
            predictions, self.entail_threshold, self.contradict_threshold
        )


def stratified_split(
    predictions: list[BenchmarkPrediction],
    cal_fraction: float = 0.3,
    seed: int = 42,
) -> tuple[list[BenchmarkPrediction], list[BenchmarkPrediction]]:
    """
    Split predictions into calibration and test sets with stratified sampling.

    Ensures both sets have the same label distribution.

    Args:
        predictions: All predictions to split.
        cal_fraction: Fraction for calibration set.
        seed: Random seed for reproducibility.

    Returns:
        (cal_predictions, test_predictions)
    """
    rng = np.random.RandomState(seed)

    # Group by gold label
    by_label: dict[str, list[BenchmarkPrediction]] = {}
    for p in predictions:
        by_label.setdefault(p.gold_label, []).append(p)

    cal_preds = []
    test_preds = []

    for label, preds in by_label.items():
        indices = rng.permutation(len(preds))
        n_cal = max(1, int(len(preds) * cal_fraction))
        for i in indices[:n_cal]:
            cal_preds.append(preds[i])
        for i in indices[n_cal:]:
            test_preds.append(preds[i])

    return cal_preds, test_preds


def optimize_and_evaluate(
    all_predictions: list[BenchmarkPrediction],
    dataset: str = "unknown",
    model_name: str = "unknown",
    cal_fraction: float = 0.3,
    metric: str = "macro_f1",
) -> tuple[BenchmarkScores, BenchmarkScores, OptimizationResult]:
    """
    End-to-end: split data, optimize thresholds on cal set, evaluate on test set.

    Args:
        all_predictions: Complete set of predictions.
        dataset: Dataset name for reporting.
        model_name: Model name for reporting.
        cal_fraction: Fraction for calibration.
        metric: Metric to optimize.

    Returns:
        (unoptimized_scores, optimized_scores, optimization_result)
        unoptimized_scores: Scores before threshold optimization (on test set).
        optimized_scores: Scores after threshold optimization (on test set).
        optimization_result: Details of the optimization.
    """
    cal_preds, test_preds = stratified_split(all_predictions, cal_fraction)

    logger.info(
        f"Split: {len(cal_preds)} calibration, {len(test_preds)} test "
        f"({cal_fraction:.0%}/{1-cal_fraction:.0%})"
    )

    # Baseline scores on full test set (no optimization)
    baseline_scores = score_predictions(
        test_preds, dataset=dataset, model_name=model_name
    )

    # Optimize thresholds on calibration set
    optimizer = ThresholdOptimizer(metric=metric)
    opt_result = optimizer.fit(cal_preds)

    # Apply to test set
    adjusted_test = optimizer.apply(test_preds)
    optimized_scores = score_predictions(
        adjusted_test, dataset=dataset, model_name=f"{model_name}+opt"
    )

    opt_result.test_scores = optimized_scores

    logger.info(
        f"Baseline {metric}: {getattr(baseline_scores, metric, 0):.4f} → "
        f"Optimized: {getattr(optimized_scores, metric, 0):.4f}"
    )

    return baseline_scores, optimized_scores, opt_result

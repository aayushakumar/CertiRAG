"""
Evaluation Metrics
===================

Shared metric functions used across all benchmarks.
Categorized into:
    - Faithfulness metrics: claim-level accuracy, faithfulness rate
    - Ranking metrics: AUROC for score calibration
    - Calibration metrics: ECE, reliability curves
    - Coverage metrics: verification yield, MSE efficiency
    - Latency metrics: per-component and end-to-end timing
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("certirag.eval.metrics")


# ───────────────────── Faithfulness Metrics ─────────────────────

def claim_level_accuracy(
    predictions: list[str],
    gold_labels: list[str],
) -> float:
    """
    Exact-match claim-level accuracy.

    Args:
        predictions: Predicted labels (ENTAILED/CONTRADICTED/NOT_ENOUGH_INFO).
        gold_labels: Ground-truth labels.

    Returns:
        Accuracy in [0, 1].
    """
    if len(predictions) != len(gold_labels):
        raise ValueError("Predictions and gold labels must have same length")
    if not predictions:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, gold_labels))
    return correct / len(predictions)


def faithfulness_rate(
    predictions: list[str],
    gold_labels: list[str],
) -> dict[str, float]:
    """
    Compute per-class precision, recall, F1 for faithfulness evaluation.

    Following ALCE (Gao et al., 2023) and RAGTruth (Wu et al., 2024),
    the key metric is the F1 on ENTAILED claims (i.e., claims supported
    by evidence).

    Returns:
        Dict with keys: entailed_precision, entailed_recall, entailed_f1,
        contradicted_precision, contradicted_recall, contradicted_f1,
        macro_f1.
    """
    labels = ["ENTAILED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
    result = {}

    f1s = []
    for label in labels:
        tp = sum(1 for p, g in zip(predictions, gold_labels) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, gold_labels) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, gold_labels) if p != label and g == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        prefix = label.lower()
        result[f"{prefix}_precision"] = precision
        result[f"{prefix}_recall"] = recall
        result[f"{prefix}_f1"] = f1
        f1s.append(f1)

    result["macro_f1"] = np.mean(f1s).item()
    return result


def sentence_level_faithfulness(
    claim_labels: list[str],
    claim_to_sentence: list[int],
) -> dict[str, float]:
    """
    Sentence-level faithfulness (Table 1 in ALCE).

    A sentence is faithful iff ALL claims extracted from it are ENTAILED.

    Args:
        claim_labels: Labels for each claim.
        claim_to_sentence: Mapping from claim index → sentence index.

    Returns:
        Dict with sentence_faithful_rate, num_sentences.
    """
    from collections import defaultdict
    sentence_claims: dict[int, list[str]] = defaultdict(list)
    for i, (label, sent_idx) in enumerate(zip(claim_labels, claim_to_sentence)):
        sentence_claims[sent_idx].append(label)

    total = len(sentence_claims)
    if total == 0:
        return {"sentence_faithful_rate": 0.0, "num_sentences": 0}

    faithful = sum(
        1 for labels in sentence_claims.values()
        if all(l == "ENTAILED" for l in labels)
    )
    return {
        "sentence_faithful_rate": faithful / total,
        "num_sentences": total,
    }


# ───────────────────── Ranking Metrics ──────────────────────────

def compute_auroc(
    entail_scores: list[float],
    gold_labels: list[str],
    positive_label: str = "ENTAILED",
) -> float:
    """
    Area Under ROC curve for verifier score quality.

    Measures how well the entailment score separates ENTAILED from
    non-ENTAILED claims. Higher = better calibrated verifier.

    Args:
        entail_scores: Model-assigned entailment probabilities.
        gold_labels: True labels.
        positive_label: Which label counts as positive.

    Returns:
        AUROC in [0, 1].
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        logger.warning("scikit-learn not installed; returning 0.0 for AUROC")
        return 0.0

    binary = [1 if g == positive_label else 0 for g in gold_labels]
    if len(set(binary)) < 2:
        logger.warning("Only one class present; AUROC undefined → returning 0.5")
        return 0.5

    return roc_auc_score(binary, entail_scores)


# ───────────────────── Calibration Metrics ──────────────────────

def compute_ece(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (Naeini et al., 2015).

    Measures how well verifier confidence aligns with actual accuracy.
    Key metric for Theorem 1 guarantees.

    ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|

    Args:
        confidences: Model confidence for each prediction.
        correct: Whether each prediction was correct.
        n_bins: Number of bins.

    Returns:
        ECE in [0, 1]. Lower = better calibrated.
    """
    if not confidences:
        return 0.0

    confs = np.array(confidences)
    accs = np.array(correct, dtype=float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accs[mask].mean()
        bin_conf = confs[mask].mean()
        ece += (mask.sum() / len(confs)) * abs(bin_acc - bin_conf)

    return float(ece)


def reliability_diagram_data(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 15,
) -> dict[str, list[float]]:
    """
    Data for plotting a reliability diagram (Figure in paper).

    Returns:
        Dict with bin_centers, bin_accuracies, bin_counts, bin_confidences.
    """
    confs = np.array(confidences)
    accs = np.array(correct, dtype=float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    centers, accuracies, counts, avg_confs = [], [], [], []
    for i in range(n_bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        center = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
        centers.append(float(center))
        counts.append(int(mask.sum()))
        if mask.sum() > 0:
            accuracies.append(float(accs[mask].mean()))
            avg_confs.append(float(confs[mask].mean()))
        else:
            accuracies.append(0.0)
            avg_confs.append(0.0)

    return {
        "bin_centers": centers,
        "bin_accuracies": accuracies,
        "bin_counts": counts,
        "bin_confidences": avg_confs,
    }


# ───────────────────── Coverage Metrics ──────────────────────

def verification_yield(
    predictions: list[str],
    total_claims: int,
) -> dict[str, float]:
    """
    Verification yield and coverage stats.

    Args:
        predictions: Labels for verified claims.
        total_claims: Total claims before filtering.

    Returns:
        Dict with yield, entailed_rate, blocked_rate.
    """
    n = len(predictions)
    if total_claims == 0:
        return {"yield": 0.0, "entailed_rate": 0.0, "blocked_rate": 1.0}

    entailed = sum(1 for p in predictions if p == "ENTAILED")
    yield_rate = n / total_claims
    entailed_rate = entailed / n if n > 0 else 0.0
    blocked_rate = 1.0 - yield_rate

    return {
        "yield": yield_rate,
        "entailed_rate": entailed_rate,
        "blocked_rate": blocked_rate,
    }


def mse_efficiency(
    original_evidence_counts: list[int],
    mse_evidence_counts: list[int],
) -> dict[str, float]:
    """
    MSE selection efficiency: how much evidence was pruned.

    Args:
        original_evidence_counts: Number of evidence per claim before MSE.
        mse_evidence_counts: Number of evidence per claim after MSE.

    Returns:
        Dict with mean_reduction, median_reduction.
    """
    if not original_evidence_counts:
        return {"mean_reduction": 0.0, "median_reduction": 0.0}

    reductions = []
    for orig, mse in zip(original_evidence_counts, mse_evidence_counts):
        if orig > 0:
            reductions.append(1.0 - mse / orig)
        else:
            reductions.append(0.0)

    return {
        "mean_reduction": float(np.mean(reductions)),
        "median_reduction": float(np.median(reductions)),
    }


# ───────────────────── Latency Metrics ──────────────────────

def latency_stats(
    timings: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Aggregate latency statistics across runs.

    Args:
        timings: List of timing dicts (from PipelineResult.timings).

    Returns:
        Dict mapping component → {mean, p50, p95, p99}.
    """
    if not timings:
        return {}

    # Collect all keys
    keys = set()
    for t in timings:
        keys.update(t.keys())

    result = {}
    for key in sorted(keys):
        values = [t[key] for t in timings if key in t]
        if values:
            arr = np.array(values)
            result[key] = {
                "mean": float(arr.mean()),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }
    return result

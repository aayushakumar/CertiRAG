"""
Scoring & Metrics for CertiRAG Evaluation
==========================================

Comprehensive metrics framework for evaluating claim verification systems.
Computes accuracy, per-class precision/recall/F1, macro-F1, binary metrics,
AUROC, confusion matrix, ECE, and timing statistics.

All functions operate on the uniform ``BenchmarkPrediction`` format
produced by the benchmark harness.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BenchmarkPrediction:
    """A single prediction with gold label."""

    example_id: str
    gold_label: str       # "entailed" | "contradicted" | "not_enough_info"
    pred_label: str       # same domain as gold_label
    pred_score: float     # model confidence in predicted class (0-1)
    entailment_score: float = 0.0  # entailment probability (for AUROC)
    latency_ms: float = 0.0     # wall-clock time for this pair


@dataclass
class ClassMetrics:
    """Precision / Recall / F1 for a single class."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int  # number of gold examples
    predicted: int  # number of predictions


@dataclass
class BenchmarkScores:
    """Complete scoring report for one evaluation run."""

    dataset: str
    model_name: str
    n_examples: int

    # Overall
    accuracy_3class: float
    accuracy_binary: float  # entailed vs not-entailed

    # Per-class
    class_metrics: list[ClassMetrics]
    macro_f1: float
    binary_entail_f1: float

    # AUROC (entailment score vs binary gold)
    auroc: Optional[float]
    auroc_error: Optional[str]

    # Calibration
    ece: Optional[float]

    # Confusion matrix (row=gold, col=pred)
    confusion_matrix: dict[str, dict[str, int]]

    # Timing
    total_time_ms: float
    avg_latency_ms: float
    p95_latency_ms: float

    # Label distribution
    gold_distribution: dict[str, int]
    pred_distribution: dict[str, int]

    # Per-example details (optional)
    predictions: list[BenchmarkPrediction] = field(default_factory=list)

    def summary_table(self) -> str:
        """Format as a readable text table."""
        lines = [
            "=" * 64,
            f"  BENCHMARK: {self.dataset}  |  MODEL: {self.model_name}",
            "=" * 64,
            f"  Examples:          {self.n_examples}",
            f"  3-class accuracy:  {self.accuracy_3class:.1%}",
            f"  Binary accuracy:   {self.accuracy_binary:.1%}",
            f"  Macro F1:          {self.macro_f1:.1%}",
            f"  Binary Entail F1:  {self.binary_entail_f1:.1%}",
            f"  AUROC:             {self.auroc:.4f}" if self.auroc is not None else f"  AUROC:             N/A ({self.auroc_error})",
            f"  ECE:               {self.ece:.4f}" if self.ece is not None else "  ECE:               N/A",
            "",
            "  ── Per-Class Metrics ──",
        ]

        for cm in self.class_metrics:
            lines.append(
                f"  {cm.label:<20s}  P={cm.precision:.3f}  R={cm.recall:.3f}  "
                f"F1={cm.f1:.3f}  (n={cm.support}, pred={cm.predicted})"
            )

        lines.extend([
            "",
            "  ── Confusion Matrix (rows=gold, cols=pred) ──",
        ])
        all_labels = sorted(set(list(self.confusion_matrix.keys())))
        header = "  " + f"{'':20s}" + "".join(f"{l:>16s}" for l in all_labels)
        lines.append(header)
        for gold_label in all_labels:
            row = self.confusion_matrix.get(gold_label, {})
            vals = "".join(f"{row.get(p, 0):>16d}" for p in all_labels)
            lines.append(f"  {gold_label:20s}{vals}")

        lines.extend([
            "",
            f"  ── Timing ──",
            f"  Total:  {self.total_time_ms:.0f}ms  |  "
            f"Avg: {self.avg_latency_ms:.1f}ms  |  "
            f"P95: {self.p95_latency_ms:.1f}ms",
            "=" * 64,
        ])

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dict (JSON-friendly)."""
        return {
            "dataset": self.dataset,
            "model_name": self.model_name,
            "n_examples": self.n_examples,
            "accuracy_3class": round(self.accuracy_3class, 4),
            "accuracy_binary": round(self.accuracy_binary, 4),
            "macro_f1": round(self.macro_f1, 4),
            "binary_entail_f1": round(self.binary_entail_f1, 4),
            "auroc": round(self.auroc, 4) if self.auroc is not None else None,
            "ece": round(self.ece, 4) if self.ece is not None else None,
            "per_class": [
                {
                    "label": cm.label,
                    "precision": round(cm.precision, 4),
                    "recall": round(cm.recall, 4),
                    "f1": round(cm.f1, 4),
                    "support": cm.support,
                    "predicted": cm.predicted,
                }
                for cm in self.class_metrics
            ],
            "confusion_matrix": self.confusion_matrix,
            "timing": {
                "total_ms": round(self.total_time_ms, 1),
                "avg_ms": round(self.avg_latency_ms, 1),
                "p95_ms": round(self.p95_latency_ms, 1),
            },
            "gold_distribution": self.gold_distribution,
            "pred_distribution": self.pred_distribution,
        }


# ── Scoring functions ──────────────────────────────────────────────

def _safe_div(a: float, b: float) -> float:
    """Safe division, returns 0.0 on division by zero."""
    return a / b if b > 0 else 0.0


def compute_class_metrics(
    predictions: list[BenchmarkPrediction],
    label: str,
) -> ClassMetrics:
    """Compute P/R/F1 for a single class."""
    tp = sum(1 for p in predictions if p.gold_label == label and p.pred_label == label)
    fp = sum(1 for p in predictions if p.gold_label != label and p.pred_label == label)
    fn = sum(1 for p in predictions if p.gold_label == label and p.pred_label != label)
    support = sum(1 for p in predictions if p.gold_label == label)
    predicted = sum(1 for p in predictions if p.pred_label == label)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return ClassMetrics(
        label=label,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        predicted=predicted,
    )


def compute_binary_metrics(
    predictions: list[BenchmarkPrediction],
) -> tuple[float, float]:
    """
    Compute binary accuracy and F1 (entailed vs not-entailed).
    Maps "contradicted" and "not_enough_info" → "not_entailed".
    Returns (binary_accuracy, entailment_f1).
    """
    def binarize(label: str) -> str:
        return "entailed" if label == "entailed" else "not_entailed"

    binary_preds = [
        BenchmarkPrediction(
            example_id=p.example_id,
            gold_label=binarize(p.gold_label),
            pred_label=binarize(p.pred_label),
            pred_score=p.pred_score,
            latency_ms=p.latency_ms,
        )
        for p in predictions
    ]

    accuracy = _safe_div(
        sum(1 for p in binary_preds if p.gold_label == p.pred_label),
        len(binary_preds),
    )

    entail_metrics = compute_class_metrics(binary_preds, "entailed")
    return accuracy, entail_metrics.f1


def compute_confusion_matrix(
    predictions: list[BenchmarkPrediction],
) -> dict[str, dict[str, int]]:
    """Build confusion matrix as nested dict: matrix[gold][pred] = count."""
    all_labels = sorted(
        set(p.gold_label for p in predictions) | set(p.pred_label for p in predictions)
    )
    matrix: dict[str, dict[str, int]] = {g: {p: 0 for p in all_labels} for g in all_labels}
    for p in predictions:
        matrix[p.gold_label][p.pred_label] += 1
    return matrix


def compute_auroc(predictions: list[BenchmarkPrediction]) -> tuple[Optional[float], Optional[str]]:
    """
    Compute Area Under ROC Curve for entailment detection.
    Uses pred_score as the confidence score, and binary gold label.
    Returns (auroc, error_msg). error_msg is None on success.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None, "scikit-learn not installed"

    gold_binary = [1 if p.gold_label == "entailed" else 0 for p in predictions]
    # Use entailment_score (raw entailment probability) for AUROC,
    # NOT pred_score which is confidence in the predicted class
    scores = [p.entailment_score for p in predictions]

    # Need both classes present
    if len(set(gold_binary)) < 2:
        return None, "only one class in gold labels"

    try:
        auroc = roc_auc_score(gold_binary, scores)
        return auroc, None
    except Exception as e:
        return None, str(e)


def compute_ece(
    predictions: list[BenchmarkPrediction],
    n_bins: int = 10,
) -> Optional[float]:
    """
    Compute Expected Calibration Error.
    Uses the predicted score as confidence and checks if gold == pred.
    """
    if not predictions:
        return None

    confidences = np.array([p.pred_score for p in predictions])
    correct = np.array([1.0 if p.gold_label == p.pred_label else 0.0 for p in predictions])

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / len(predictions) * abs(bin_acc - bin_conf)

    return float(ece)


def score_predictions(
    predictions: list[BenchmarkPrediction],
    dataset: str = "unknown",
    model_name: str = "unknown",
) -> BenchmarkScores:
    """
    Compute all metrics from a list of predictions.

    This is the main entry point for scoring. It computes:
    - 3-class and binary accuracy
    - Per-class P/R/F1
    - Macro F1
    - AUROC
    - ECE
    - Confusion matrix
    - Timing statistics
    """
    if not predictions:
        return BenchmarkScores(
            dataset=dataset,
            model_name=model_name,
            n_examples=0,
            accuracy_3class=0.0,
            accuracy_binary=0.0,
            class_metrics=[],
            macro_f1=0.0,
            binary_entail_f1=0.0,
            auroc=None,
            auroc_error="no predictions",
            ece=None,
            confusion_matrix={},
            total_time_ms=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            gold_distribution={},
            pred_distribution={},
            predictions=[],
        )

    # 3-class accuracy
    correct_3 = sum(1 for p in predictions if p.gold_label == p.pred_label)
    accuracy_3class = correct_3 / len(predictions)

    # Binary accuracy and entailment F1
    accuracy_binary, binary_entail_f1 = compute_binary_metrics(predictions)

    # Per-class metrics
    all_labels = sorted(set(p.gold_label for p in predictions) | set(p.pred_label for p in predictions))
    class_metrics = [compute_class_metrics(predictions, label) for label in all_labels]
    macro_f1 = _safe_div(
        sum(cm.f1 for cm in class_metrics),
        len(class_metrics),
    )

    # AUROC
    auroc, auroc_error = compute_auroc(predictions)

    # ECE
    ece = compute_ece(predictions)

    # Confusion matrix
    confusion_matrix = compute_confusion_matrix(predictions)

    # Timing
    latencies = [p.latency_ms for p in predictions]
    total_time_ms = sum(latencies)
    avg_latency_ms = total_time_ms / len(latencies) if latencies else 0.0
    p95_latency_ms = float(np.percentile(latencies, 95)) if latencies else 0.0

    # Label distributions
    gold_distribution = dict(Counter(p.gold_label for p in predictions))
    pred_distribution = dict(Counter(p.pred_label for p in predictions))

    return BenchmarkScores(
        dataset=dataset,
        model_name=model_name,
        n_examples=len(predictions),
        accuracy_3class=accuracy_3class,
        accuracy_binary=accuracy_binary,
        class_metrics=class_metrics,
        macro_f1=macro_f1,
        binary_entail_f1=binary_entail_f1,
        auroc=auroc,
        auroc_error=auroc_error,
        ece=ece,
        confusion_matrix=confusion_matrix,
        total_time_ms=total_time_ms,
        avg_latency_ms=avg_latency_ms,
        p95_latency_ms=p95_latency_ms,
        gold_distribution=gold_distribution,
        pred_distribution=pred_distribution,
        predictions=predictions,
    )


def compare_models(
    results: list[BenchmarkScores],
) -> str:
    """
    Format a comparison table across multiple model/dataset runs.
    Returns a formatted string table.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        # Fallback: simple text
        lines = []
        for r in results:
            lines.append(
                f"{r.model_name:30s} | {r.dataset:15s} | "
                f"Acc3={r.accuracy_3class:.1%} | AccBin={r.accuracy_binary:.1%} | "
                f"MacroF1={r.macro_f1:.1%} | EntF1={r.binary_entail_f1:.1%}"
            )
        return "\n".join(lines)

    headers = [
        "Model", "Dataset", "N", "Acc(3)", "Acc(bin)",
        "MacroF1", "EntF1", "AUROC", "ECE", "Avg(ms)",
    ]
    rows = []
    for r in results:
        rows.append([
            r.model_name,
            r.dataset,
            r.n_examples,
            f"{r.accuracy_3class:.1%}",
            f"{r.accuracy_binary:.1%}",
            f"{r.macro_f1:.1%}",
            f"{r.binary_entail_f1:.1%}",
            f"{r.auroc:.4f}" if r.auroc is not None else "N/A",
            f"{r.ece:.4f}" if r.ece is not None else "N/A",
            f"{r.avg_latency_ms:.1f}",
        ])

    return tabulate(rows, headers=headers, tablefmt="grid")


# ── Bootstrap Confidence Intervals ────────────────────────────────


def bootstrap_confidence_intervals(
    predictions: list[BenchmarkPrediction],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Compute bootstrap confidence intervals for key metrics.

    Returns dict mapping metric name → {"mean", "lower", "upper", "std"}.
    Uses the percentile method with ``n_bootstrap`` resamples.

    Example output::

        {
            "accuracy_3class": {"mean": 0.735, "lower": 0.680, "upper": 0.785, "std": 0.027},
            "binary_accuracy": {"mean": 0.845, "lower": 0.800, "upper": 0.885, "std": 0.022},
            ...
        }
    """
    rng = np.random.RandomState(seed)
    n = len(predictions)

    if n == 0:
        return {}

    alpha = 1.0 - confidence
    lo_pct = 100 * (alpha / 2)
    hi_pct = 100 * (1 - alpha / 2)

    # Pre-extract arrays for speed
    gold = np.array([p.gold_label for p in predictions])
    pred = np.array([p.pred_label for p in predictions])
    ent_scores = np.array([p.entailment_score for p in predictions])

    # Binary gold: entailed=1, else=0
    gold_binary = (gold == "entailed").astype(int)
    pred_binary = (pred == "entailed").astype(int)

    boot_acc3 = []
    boot_acc_bin = []
    boot_macro_f1 = []
    boot_ent_f1 = []
    boot_auroc = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)

        b_gold = gold[idx]
        b_pred = pred[idx]
        b_gold_bin = gold_binary[idx]
        b_pred_bin = pred_binary[idx]
        b_ent = ent_scores[idx]

        # 3-class accuracy
        boot_acc3.append(np.mean(b_gold == b_pred))

        # Binary accuracy
        boot_acc_bin.append(np.mean(b_gold_bin == b_pred_bin))

        # Entailment F1 (binary)
        tp = np.sum((b_pred_bin == 1) & (b_gold_bin == 1))
        fp = np.sum((b_pred_bin == 1) & (b_gold_bin == 0))
        fn = np.sum((b_pred_bin == 0) & (b_gold_bin == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        boot_ent_f1.append(f1)

        # Macro F1 (3-class)
        labels = ["entailed", "contradicted", "not_enough_info"]
        f1s = []
        for lbl in labels:
            tp_l = np.sum((b_pred == lbl) & (b_gold == lbl))
            fp_l = np.sum((b_pred == lbl) & (b_gold != lbl))
            fn_l = np.sum((b_pred != lbl) & (b_gold == lbl))
            p_l = tp_l / (tp_l + fp_l) if (tp_l + fp_l) > 0 else 0.0
            r_l = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0.0
            f1_l = 2 * p_l * r_l / (p_l + r_l) if (p_l + r_l) > 0 else 0.0
            f1s.append(f1_l)
        boot_macro_f1.append(np.mean(f1s))

        # AUROC (may fail if only one class in resample)
        if len(np.unique(b_gold_bin)) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                boot_auroc.append(roc_auc_score(b_gold_bin, b_ent))
            except Exception:
                pass

    def _ci(values):
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "lower": float(np.percentile(arr, lo_pct)),
            "upper": float(np.percentile(arr, hi_pct)),
            "std": float(np.std(arr)),
        }

    result = {
        "accuracy_3class": _ci(boot_acc3),
        "binary_accuracy": _ci(boot_acc_bin),
        "macro_f1": _ci(boot_macro_f1),
        "entailment_f1": _ci(boot_ent_f1),
    }

    if boot_auroc:
        result["auroc"] = _ci(boot_auroc)

    return result

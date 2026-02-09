"""
Publication-Quality Plots
===========================

Generates figures for the paper using matplotlib/seaborn:
    - Figure 3: Reliability diagram (calibration)
    - Figure 4: AUROC curves per verifier
    - Figure 5: Latency breakdown bar chart
    - Figure 6: MSE efficiency vs. faithfulness trade-off
    - Figure 7: Threshold sensitivity (τ_entail sweep)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("certirag.eval.plots")

# Reproducible style
STYLE_PARAMS = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (5.5, 4.0),  # ACL column width
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

# Color palette (colorblind-friendly)
COLORS = {
    "certirag": "#2196F3",      # Blue
    "standard_rag": "#FF9800",   # Orange
    "posthoc": "#4CAF50",        # Green
    "multi_query": "#9C27B0",    # Purple
    "nli": "#F44336",            # Red
    "llm_judge": "#795548",      # Brown
}


def setup_style():
    """Apply publication style to matplotlib."""
    try:
        import matplotlib.pyplot as plt
        plt.rcParams.update(STYLE_PARAMS)
    except ImportError:
        logger.warning("matplotlib not available; plots disabled")


def plot_reliability_diagram(
    reliability_data: dict[str, list[float]],
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram",
) -> None:
    """
    Plot reliability diagram (Figure 3).

    Shows calibration quality: perfect calibration = diagonal line.

    Args:
        reliability_data: Output from metrics.reliability_diagram_data().
        save_path: Where to save the figure (PNG/PDF).
        title: Figure title.
    """
    import matplotlib.pyplot as plt

    setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.5),
                                     gridspec_kw={"height_ratios": [3, 1]})

    centers = reliability_data["bin_centers"]
    accuracies = reliability_data["bin_accuracies"]
    counts = reliability_data["bin_counts"]
    confidences = reliability_data["bin_confidences"]

    # Main reliability plot
    ax1.bar(centers, accuracies, width=1/len(centers), alpha=0.7,
            color=COLORS["certirag"], edgecolor="white", label="CertiRAG")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy")
    ax1.set_title(title)
    ax1.legend(loc="upper left")

    # Histogram of confidence distribution
    ax2.bar(centers, counts, width=1/len(centers), alpha=0.7,
            color=COLORS["certirag"], edgecolor="white")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        logger.info(f"Reliability diagram saved to {save_path}")
    plt.close()


def plot_auroc_curves(
    systems: dict[str, tuple[list[float], list[str]]],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot AUROC curves for multiple systems (Figure 4).

    Args:
        systems: Dict mapping system name → (scores, gold_labels).
        save_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    setup_style()
    fig, ax = plt.subplots()

    for name, (scores, labels) in systems.items():
        binary = [1 if l == "ENTAILED" else 0 for l in labels]
        fpr, tpr, _ = roc_curve(binary, scores)
        roc_auc = auc(fpr, tpr)

        color = COLORS.get(name, "#333333")
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Verifier ROC Curves")
    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        logger.info(f"AUROC plot saved to {save_path}")
    plt.close()


def plot_latency_breakdown(
    timings: dict[str, float],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot latency breakdown bar chart (Figure 5).

    Shows wall-clock time per pipeline component.

    Args:
        timings: Dict mapping component → mean latency (ms).
        save_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax = plt.subplots(figsize=(6, 3))

    # Order by pipeline stage
    stage_order = [
        "retrieve_ms", "compile_ms", "normalize_ms",
        "verify_ms", "mse_ms", "render_ms", "certificate_ms",
    ]
    components = [s for s in stage_order if s in timings]
    values = [timings[c] for c in components]
    labels = [c.replace("_ms", "").title() for c in components]

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(components)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}ms", va="center", fontsize=8)

    ax.set_xlabel("Latency (ms)")
    ax.set_title("CertiRAG Pipeline Latency Breakdown")
    ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        logger.info(f"Latency plot saved to {save_path}")
    plt.close()


def plot_threshold_sweep(
    thresholds: list[float],
    accuracies: list[float],
    yields: list[float],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot threshold sensitivity analysis (Figure 7).

    Shows trade-off between accuracy and yield as τ_entail varies.

    Args:
        thresholds: τ_entail values tested.
        accuracies: Accuracy at each threshold.
        yields: Verification yield at each threshold.
        save_path: Where to save the figure.
    """
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax1 = plt.subplots()

    # Accuracy on left y-axis
    color1 = COLORS["certirag"]
    ax1.plot(thresholds, accuracies, "o-", color=color1, lw=2, label="Accuracy")
    ax1.set_xlabel(r"$\tau_{\mathrm{entail}}$")
    ax1.set_ylabel("Accuracy", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Yield on right y-axis
    ax2 = ax1.twinx()
    color2 = COLORS["standard_rag"]
    ax2.plot(thresholds, yields, "s--", color=color2, lw=2, label="Yield")
    ax2.set_ylabel("Yield", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title(r"Accuracy vs. Yield ($\tau_{\mathrm{entail}}$ Sensitivity)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        logger.info(f"Threshold sweep plot saved to {save_path}")
    plt.close()

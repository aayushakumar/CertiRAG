"""
Verifier Score Calibration
============================

Calibrates verifier confidence scores so they reflect true probabilities.
This is critical for threshold selection (§7.3 of README).

An uncalibrated verifier might output 0.85 entailment score but actually
be correct only 70% of the time at that confidence. Calibration fixes this.

Methods:
    - Temperature scaling (single-parameter, fast)
    - Isotonic regression (non-parametric, more flexible)

Outputs:
    - Calibrated score mapping function
    - Reliability diagrams for the paper
    - Expected Calibration Error (ECE) before/after

Data Flow:
    raw_scores → Calibrator.fit(calibration_set) → calibrated_scores
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("certirag.verify.calibrator")


class VerifierCalibrator:
    """
    Calibrates verifier scores to reflect true entailment probabilities.

    Two methods:
        - "temperature": Platt scaling (learns a single temperature T)
        - "isotonic": Isotonic regression (non-parametric)

    Usage:
        calibrator = VerifierCalibrator(method="isotonic")
        calibrator.fit(raw_scores, true_labels)
        calibrated = calibrator.calibrate(new_raw_scores)
        ece = calibrator.compute_ece(raw_scores, true_labels)

    Args:
        method: Calibration method ("temperature", "isotonic", "none").
        n_bins: Number of bins for ECE computation.
    """

    def __init__(self, method: str = "isotonic", n_bins: int = 10):
        self.method = method
        self.n_bins = n_bins
        self._calibrator = None
        self._temperature: float = 1.0
        self._is_fitted: bool = False

    def fit(
        self,
        raw_scores: np.ndarray,
        true_labels: np.ndarray,
    ) -> None:
        """
        Fit the calibration model on a held-out calibration set.

        Args:
            raw_scores: Array of raw verifier scores [0, 1].
            true_labels: Array of true binary labels (1=entailed, 0=not).

        Note:
            The calibration set should be separate from the test set
            used for final evaluation.
        """
        raw_scores = np.asarray(raw_scores, dtype=np.float64)
        true_labels = np.asarray(true_labels, dtype=np.float64)

        if len(raw_scores) != len(true_labels):
            raise ValueError("raw_scores and true_labels must have same length")

        if self.method == "temperature":
            self._fit_temperature(raw_scores, true_labels)
        elif self.method == "isotonic":
            self._fit_isotonic(raw_scores, true_labels)
        elif self.method == "none":
            pass  # no calibration
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self._is_fitted = True
        logger.info(
            f"Calibrator fitted using {self.method} method "
            f"on {len(raw_scores)} samples"
        )

    def _fit_temperature(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """
        Platt/temperature scaling.

        Learns T such that calibrated_score = sigmoid(logit(score) / T).
        Optimized by minimizing negative log-likelihood (cross-entropy).
        """
        from scipy.optimize import minimize_scalar

        # Convert probabilities to logits
        eps = 1e-7
        scores_clipped = np.clip(scores, eps, 1 - eps)
        logits = np.log(scores_clipped / (1 - scores_clipped))

        def nll(T):
            """Negative log-likelihood for temperature T."""
            scaled_logits = logits / max(T, eps)
            calibrated = 1 / (1 + np.exp(-scaled_logits))
            calibrated = np.clip(calibrated, eps, 1 - eps)
            loss = -(labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated))
            return np.mean(loss)

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self._temperature = result.x
        logger.info(f"Temperature scaling: T = {self._temperature:.4f}")

    def _fit_isotonic(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Isotonic regression calibration."""
        from sklearn.isotonic import IsotonicRegression

        self._calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        self._calibrator.fit(scores, labels)

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw scores.

        Args:
            raw_scores: Array of raw verifier scores [0, 1].

        Returns:
            Array of calibrated scores [0, 1].
        """
        raw_scores = np.asarray(raw_scores, dtype=np.float64)

        if self.method == "none" or not self._is_fitted:
            return raw_scores

        if self.method == "temperature":
            eps = 1e-7
            clipped = np.clip(raw_scores, eps, 1 - eps)
            logits = np.log(clipped / (1 - clipped))
            scaled_logits = logits / self._temperature
            return 1 / (1 + np.exp(-scaled_logits))

        elif self.method == "isotonic":
            return self._calibrator.predict(raw_scores)

        return raw_scores

    def calibrate_single(self, raw_score: float) -> float:
        """Calibrate a single score."""
        return float(self.calibrate(np.array([raw_score]))[0])

    def compute_ece(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        n_bins: Optional[int] = None,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE measures how well calibrated the scores are:
        ECE = Σ (|B_k|/N) * |accuracy(B_k) - confidence(B_k)|

        Lower is better. A perfectly calibrated model has ECE = 0.

        Args:
            scores: Predicted probabilities [0, 1].
            labels: True binary labels (0 or 1).
            n_bins: Number of bins (default: self.n_bins).

        Returns:
            ECE value (lower is better).
        """
        n_bins = n_bins or self.n_bins
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            mask = (scores > bin_lower) & (scores <= bin_upper)
            bin_count = mask.sum()

            if bin_count > 0:
                bin_accuracy = labels[mask].mean()
                bin_confidence = scores[mask].mean()
                ece += (bin_count / len(scores)) * abs(bin_accuracy - bin_confidence)

        return ece

    def get_reliability_data(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        n_bins: Optional[int] = None,
    ) -> dict:
        """
        Compute data for reliability diagrams (paper figure).

        Returns:
            Dict with 'bin_centers', 'bin_accuracies', 'bin_confidences',
            'bin_counts', 'ece'.
        """
        n_bins = n_bins or self.n_bins
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            mask = (scores > bin_lower) & (scores <= bin_upper)
            bin_count = int(mask.sum())

            if bin_count > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(float(labels[mask].mean()))
                bin_confidences.append(float(scores[mask].mean()))
                bin_counts.append(bin_count)

        return {
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
            "ece": self.compute_ece(scores, labels, n_bins),
        }

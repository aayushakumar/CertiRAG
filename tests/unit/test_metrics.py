"""
Metrics Tests
==============

Tests for the evaluation metrics module.
"""

from __future__ import annotations

import pytest

from eval.metrics import (
    claim_level_accuracy,
    compute_auroc,
    compute_ece,
    faithfulness_rate,
    latency_stats,
    mse_efficiency,
    reliability_diagram_data,
    sentence_level_faithfulness,
    verification_yield,
)


class TestClaimLevelAccuracy:
    def test_perfect(self):
        assert claim_level_accuracy(["ENTAILED", "CONTRADICTED"], ["ENTAILED", "CONTRADICTED"]) == 1.0

    def test_zero(self):
        assert claim_level_accuracy(["ENTAILED"], ["CONTRADICTED"]) == 0.0

    def test_half(self):
        assert claim_level_accuracy(["ENTAILED", "CONTRADICTED"], ["ENTAILED", "ENTAILED"]) == 0.5

    def test_empty(self):
        assert claim_level_accuracy([], []) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            claim_level_accuracy(["ENTAILED"], ["ENTAILED", "CONTRADICTED"])


class TestFaithfulnessRate:
    def test_perfect_entailment(self):
        preds = ["ENTAILED", "ENTAILED"]
        golds = ["ENTAILED", "ENTAILED"]
        result = faithfulness_rate(preds, golds)
        assert result["entailed_f1"] == 1.0

    def test_no_match(self):
        preds = ["CONTRADICTED", "CONTRADICTED"]
        golds = ["ENTAILED", "ENTAILED"]
        result = faithfulness_rate(preds, golds)
        assert result["entailed_f1"] == 0.0

    def test_mixed(self):
        preds = ["ENTAILED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
        golds = ["ENTAILED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
        result = faithfulness_rate(preds, golds)
        assert result["macro_f1"] == 1.0


class TestECE:
    def test_perfect_calibration(self):
        # Confidence matches accuracy perfectly
        ece = compute_ece([0.9, 0.9, 0.9], [True, True, True])
        assert ece < 0.2  # Should be very low

    def test_empty(self):
        assert compute_ece([], []) == 0.0

    def test_worst_case(self):
        # High confidence but all wrong
        ece = compute_ece([0.95, 0.95, 0.95], [False, False, False])
        assert ece > 0.5


class TestAUROC:
    def test_perfect_separation(self):
        scores = [0.9, 0.8, 0.1, 0.2]
        labels = ["ENTAILED", "ENTAILED", "CONTRADICTED", "CONTRADICTED"]
        auroc = compute_auroc(scores, labels)
        assert auroc == 1.0

    def test_random(self):
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = ["ENTAILED", "CONTRADICTED", "ENTAILED", "CONTRADICTED"]
        auroc = compute_auroc(scores, labels)
        assert auroc == pytest.approx(0.5)


class TestVerificationYield:
    def test_full_yield(self):
        result = verification_yield(["ENTAILED", "ENTAILED"], 2)
        assert result["yield"] == 1.0

    def test_partial_yield(self):
        result = verification_yield(["ENTAILED"], 2)
        assert result["yield"] == 0.5


class TestMSEEfficiency:
    def test_no_reduction(self):
        result = mse_efficiency([3, 3], [3, 3])
        assert result["mean_reduction"] == 0.0

    def test_full_reduction(self):
        result = mse_efficiency([3, 3], [0, 0])
        assert result["mean_reduction"] == 1.0

    def test_half_reduction(self):
        result = mse_efficiency([4, 4], [2, 2])
        assert result["mean_reduction"] == pytest.approx(0.5)


class TestLatencyStats:
    def test_basic(self):
        timings = [
            {"total_ms": 100, "verify_ms": 50},
            {"total_ms": 200, "verify_ms": 100},
        ]
        result = latency_stats(timings)
        assert "total_ms" in result
        assert result["total_ms"]["mean"] == pytest.approx(150.0)

    def test_empty(self):
        assert latency_stats([]) == {}

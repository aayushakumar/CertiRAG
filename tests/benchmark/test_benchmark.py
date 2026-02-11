"""
CertiRAG Benchmark Test Suite
==============================

Pytest-based evaluation suite that downloads real NLI datasets,
runs verifiers against them, and reports comprehensive metrics.

Usage
-----
    # Quick smoke test (50 examples per dataset)
    pytest tests/benchmark/ -v -k "smoke"

    # Full benchmark (200 examples per dataset)
    pytest tests/benchmark/ -v -k "not smoke" --benchmark-max=200

    # Single dataset
    pytest tests/benchmark/ -v -k "vitaminc"

    # Compare models
    pytest tests/benchmark/ -v -k "compare"

Environment
-----------
    BENCHMARK_MAX_EXAMPLES: Override max examples (default: 200)
    BENCHMARK_DATASETS: Comma-separated dataset list (default: all)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

import pytest

# ── Configure logging ──────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Fixtures ───────────────────────────────────────────────────────

def _max_examples() -> int:
    """Get max examples from env or default."""
    return int(os.environ.get("BENCHMARK_MAX_EXAMPLES", "200"))


def _requested_datasets() -> list[str]:
    """Get dataset list from env or return all."""
    env = os.environ.get("BENCHMARK_DATASETS", "")
    if env:
        return [d.strip() for d in env.split(",") if d.strip()]
    return ["vitaminc", "anli_r1", "fever_nli"]


@pytest.fixture(scope="module")
def hf_nli_verifier():
    """Load the HF NLI verifier once for the module."""
    try:
        from certirag.verify.hf_nli_verifier import HFNLIVerifier
        verifier = HFNLIVerifier(device="cpu", batch_size=32)
        return verifier
    except (ImportError, OSError) as e:
        pytest.skip(f"Model not available: {e}")


@pytest.fixture(scope="module")
def minicheck_verifier():
    """Load MiniCheck lite verifier."""
    from certirag.verify.minicheck import MiniCheckVerifier
    return MiniCheckVerifier(mode="lite")


@pytest.fixture(scope="module")
def benchmark_runner(hf_nli_verifier):
    """Create a BenchmarkRunner with the HF NLI verifier."""
    from eval.benchmark import BenchmarkRunner
    return BenchmarkRunner(
        verifier=hf_nli_verifier,
        output_dir="eval_results",
    )


# ── Dataset loading tests ─────────────────────────────────────────

class TestDatasetLoading:
    """Verify datasets download and parse correctly."""

    @pytest.mark.benchmark
    def test_load_vitaminc(self):
        """VitaminC loads with correct format."""
        from eval.datasets.loaders import load_vitaminc
        examples = load_vitaminc(max_examples=20)
        assert len(examples) > 0
        ex = examples[0]
        assert ex.claim
        assert ex.evidence
        assert ex.gold_label in ("entailed", "contradicted", "not_enough_info")
        assert ex.dataset == "vitaminc"

        # Check label distribution
        labels = Counter(e.gold_label for e in examples)
        print(f"\n  VitaminC (n={len(examples)}): {dict(labels)}")

    @pytest.mark.benchmark
    def test_load_anli_r1(self):
        """ANLI R1 loads with correct format."""
        from eval.datasets.loaders import load_anli
        examples = load_anli(round="r1", max_examples=20)
        assert len(examples) > 0
        ex = examples[0]
        assert ex.claim
        assert ex.evidence
        assert ex.gold_label in ("entailed", "contradicted", "not_enough_info")
        assert ex.dataset == "anli_r1"

        labels = Counter(e.gold_label for e in examples)
        print(f"\n  ANLI R1 (n={len(examples)}): {dict(labels)}")

    @pytest.mark.benchmark
    def test_load_fever_nli(self):
        """FEVER-NLI loads with correct format."""
        from eval.datasets.loaders import load_fever_nli
        examples = load_fever_nli(max_examples=20)
        assert len(examples) > 0
        ex = examples[0]
        assert ex.claim
        assert ex.evidence
        assert ex.gold_label in ("entailed", "contradicted", "not_enough_info")
        assert ex.dataset == "fever_nli"

        labels = Counter(e.gold_label for e in examples)
        print(f"\n  FEVER-NLI (n={len(examples)}): {dict(labels)}")

    @pytest.mark.benchmark
    def test_label_filtering(self):
        """Label filters work correctly."""
        from eval.datasets.loaders import load_vitaminc
        examples = load_vitaminc(
            max_examples=100,
            label_filter=["entailed", "contradicted"],
        )
        labels = set(e.gold_label for e in examples)
        assert "not_enough_info" not in labels
        assert len(examples) > 0
        print(f"\n  Filtered: {len(examples)} examples, labels: {labels}")


# ── Smoke tests (fast, small samples) ─────────────────────────────

class TestBenchmarkSmoke:
    """Fast smoke tests with tiny samples to verify pipeline works."""

    @pytest.mark.benchmark
    @pytest.mark.smoke
    def test_smoke_vitaminc(self, hf_nli_verifier):
        """Quick sanity check: 50 VitaminC examples."""
        from eval.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(verifier=hf_nli_verifier, output_dir="eval_results")
        scores = runner.run("vitaminc", max_examples=50, save_results=False)

        print(f"\n{scores.summary_table()}")

        assert scores.n_examples > 0
        assert scores.accuracy_binary > 0.5  # better than random
        assert scores.macro_f1 > 0.0

    @pytest.mark.benchmark
    @pytest.mark.smoke
    def test_smoke_anli(self, hf_nli_verifier):
        """Quick sanity check: 50 ANLI R1 examples."""
        from eval.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(verifier=hf_nli_verifier, output_dir="eval_results")
        scores = runner.run("anli_r1", max_examples=50, save_results=False)

        print(f"\n{scores.summary_table()}")

        assert scores.n_examples > 0
        assert scores.accuracy_binary > 0.3  # ANLI is hard

    @pytest.mark.benchmark
    @pytest.mark.smoke
    def test_smoke_fever(self, hf_nli_verifier):
        """Quick sanity check: 50 FEVER-NLI examples."""
        from eval.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(verifier=hf_nli_verifier, output_dir="eval_results")
        scores = runner.run("fever_nli", max_examples=50, save_results=False)

        print(f"\n{scores.summary_table()}")

        assert scores.n_examples > 0
        assert scores.accuracy_binary > 0.5


# ── Full benchmarks (per dataset) ─────────────────────────────────

class TestVitaminCBenchmark:
    """Full VitaminC evaluation."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_vitaminc_full(self, benchmark_runner):
        """Evaluate on VitaminC test set."""
        max_n = _max_examples()
        scores = benchmark_runner.run("vitaminc", max_examples=max_n)

        print(f"\n{scores.summary_table()}")

        # Quality gates for VitaminC
        assert scores.accuracy_binary >= 0.70, (
            f"Binary accuracy {scores.accuracy_binary:.1%} below 70% threshold"
        )
        assert scores.binary_entail_f1 >= 0.60, (
            f"Entailment F1 {scores.binary_entail_f1:.1%} below 60% threshold"
        )

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_vitaminc_entailment_only(self, hf_nli_verifier):
        """Evaluate entailment detection on VitaminC SUPPORTS examples."""
        from eval.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(verifier=hf_nli_verifier, output_dir="eval_results")
        scores = runner.run(
            "vitaminc",
            max_examples=_max_examples(),
            label_filter=["entailed"],
            save_results=False,
        )

        print(f"\n  Entailment-only: {scores.accuracy_3class:.1%} accuracy on {scores.n_examples} examples")

        # On only-entailed examples, accuracy = recall of entailment class
        assert scores.accuracy_3class >= 0.60, (
            f"Entailment recall {scores.accuracy_3class:.1%} below 60%"
        )


class TestANLIBenchmark:
    """Full ANLI evaluation (adversarial, expect lower scores)."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_anli_r1(self, benchmark_runner):
        """ANLI Round 1 (easiest adversarial round)."""
        scores = benchmark_runner.run("anli_r1", max_examples=_max_examples())
        print(f"\n{scores.summary_table()}")

        # ANLI is adversarial — lower thresholds
        assert scores.accuracy_binary >= 0.50, (
            f"Binary accuracy {scores.accuracy_binary:.1%} below 50%"
        )

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_anli_r2(self, hf_nli_verifier):
        """ANLI Round 2 (harder)."""
        from eval.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(verifier=hf_nli_verifier, output_dir="eval_results")
        scores = runner.run("anli_r2", max_examples=_max_examples())
        print(f"\n{scores.summary_table()}")

        # R2 is harder
        assert scores.accuracy_binary >= 0.45

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_anli_r3(self, hf_nli_verifier):
        """ANLI Round 3 (hardest)."""
        from eval.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(verifier=hf_nli_verifier, output_dir="eval_results")
        scores = runner.run("anli_r3", max_examples=_max_examples())
        print(f"\n{scores.summary_table()}")

        assert scores.accuracy_binary >= 0.40


class TestFEVERBenchmark:
    """Full FEVER-NLI evaluation."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_fever_nli_full(self, benchmark_runner):
        """Full FEVER-NLI dev set evaluation."""
        scores = benchmark_runner.run("fever_nli", max_examples=_max_examples())
        print(f"\n{scores.summary_table()}")

        assert scores.accuracy_binary >= 0.65, (
            f"Binary accuracy {scores.accuracy_binary:.1%} below 65%"
        )
        assert scores.binary_entail_f1 >= 0.55, (
            f"Entailment F1 {scores.binary_entail_f1:.1%} below 55%"
        )


# ── Model comparison tests ─────────────────────────────────────────

class TestModelComparison:
    """Compare multiple verifiers side-by-side."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_compare_hf_vs_minicheck(self, hf_nli_verifier, minicheck_verifier):
        """Compare HF NLI vs MiniCheck lite on VitaminC."""
        from eval.benchmark import BenchmarkRunner
        from eval.scoring import compare_models

        max_n = min(_max_examples(), 100)

        # HF NLI
        runner1 = BenchmarkRunner(
            verifier=hf_nli_verifier,
            model_name="DeBERTa-NLI",
            output_dir="eval_results",
        )
        scores1 = runner1.run("vitaminc", max_examples=max_n, save_results=False)

        # MiniCheck lite
        runner2 = BenchmarkRunner(
            verifier=minicheck_verifier,
            model_name="MiniCheck-lite",
            output_dir="eval_results",
        )
        scores2 = runner2.run("vitaminc", max_examples=max_n, save_results=False)

        # Print comparison
        table = compare_models([scores1, scores2])
        print(f"\n{table}")

        # HF NLI should outperform MiniCheck lite (word overlap)
        assert scores1.accuracy_binary >= scores2.accuracy_binary, (
            f"DeBERTa ({scores1.accuracy_binary:.1%}) should beat "
            f"MiniCheck-lite ({scores2.accuracy_binary:.1%})"
        )


# ── Scoring & metrics tests ───────────────────────────────────────

class TestScoringFramework:
    """Test the scoring functions themselves."""

    @pytest.mark.benchmark
    def test_score_predictions(self):
        """Score computation works on synthetic data."""
        from eval.scoring import BenchmarkPrediction, score_predictions

        preds = [
            BenchmarkPrediction("1", "entailed", "entailed", 0.95, 10.0),
            BenchmarkPrediction("2", "entailed", "not_enough_info", 0.4, 10.0),
            BenchmarkPrediction("3", "contradicted", "contradicted", 0.9, 10.0),
            BenchmarkPrediction("4", "not_enough_info", "not_enough_info", 0.8, 10.0),
            BenchmarkPrediction("5", "contradicted", "entailed", 0.7, 10.0),
        ]

        scores = score_predictions(preds, dataset="synthetic", model_name="test")
        assert scores.n_examples == 5
        assert scores.accuracy_3class == 3 / 5  # 3 correct
        assert scores.accuracy_binary > 0.0
        assert len(scores.class_metrics) == 3
        assert scores.macro_f1 > 0.0

        print(f"\n{scores.summary_table()}")

    @pytest.mark.benchmark
    def test_confusion_matrix(self):
        """Confusion matrix is computed correctly."""
        from eval.scoring import BenchmarkPrediction, compute_confusion_matrix

        preds = [
            BenchmarkPrediction("1", "entailed", "entailed", 0.9, 1.0),
            BenchmarkPrediction("2", "entailed", "contradicted", 0.6, 1.0),
            BenchmarkPrediction("3", "contradicted", "entailed", 0.7, 1.0),
            BenchmarkPrediction("4", "contradicted", "contradicted", 0.8, 1.0),
        ]
        cm = compute_confusion_matrix(preds)
        assert cm["entailed"]["entailed"] == 1
        assert cm["entailed"]["contradicted"] == 1
        assert cm["contradicted"]["entailed"] == 1
        assert cm["contradicted"]["contradicted"] == 1

    @pytest.mark.benchmark
    def test_auroc(self):
        """AUROC computation works."""
        from eval.scoring import BenchmarkPrediction, compute_auroc

        # Perfect separation — entailment_score high for entailed, low for others
        preds = [
            BenchmarkPrediction("1", "entailed", "entailed", 0.95, entailment_score=0.95),
            BenchmarkPrediction("2", "entailed", "entailed", 0.90, entailment_score=0.90),
            BenchmarkPrediction("3", "contradicted", "contradicted", 0.1, entailment_score=0.1),
            BenchmarkPrediction("4", "not_enough_info", "not_enough_info", 0.2, entailment_score=0.2),
        ]
        auroc, err = compute_auroc(preds)
        assert err is None
        assert auroc is not None
        assert auroc == 1.0  # perfect separation

    @pytest.mark.benchmark
    def test_ece(self):
        """ECE computation works."""
        from eval.scoring import BenchmarkPrediction, compute_ece

        # Perfectly calibrated: high confidence → correct
        preds = [
            BenchmarkPrediction("1", "entailed", "entailed", 0.95, entailment_score=0.95),
            BenchmarkPrediction("2", "contradicted", "contradicted", 0.90, entailment_score=0.1),
        ]
        ece = compute_ece(preds, n_bins=5)
        assert ece is not None
        assert ece < 0.2  # should be well calibrated

    @pytest.mark.benchmark
    def test_compare_models_format(self):
        """Model comparison table formats correctly."""
        from eval.scoring import BenchmarkScores, compare_models

        scores1 = BenchmarkScores(
            dataset="test", model_name="A", n_examples=100,
            accuracy_3class=0.8, accuracy_binary=0.9,
            class_metrics=[], macro_f1=0.75, binary_entail_f1=0.85,
            auroc=0.92, auroc_error=None, ece=0.05,
            confusion_matrix={}, total_time_ms=1000,
            avg_latency_ms=10, p95_latency_ms=20,
            gold_distribution={}, pred_distribution={},
        )
        scores2 = BenchmarkScores(
            dataset="test", model_name="B", n_examples=100,
            accuracy_3class=0.6, accuracy_binary=0.7,
            class_metrics=[], macro_f1=0.55, binary_entail_f1=0.65,
            auroc=0.78, auroc_error=None, ece=0.15,
            confusion_matrix={}, total_time_ms=500,
            avg_latency_ms=5, p95_latency_ms=10,
            gold_distribution={}, pred_distribution={},
        )
        table = compare_models([scores1, scores2])
        assert "A" in table
        assert "B" in table
        print(f"\n{table}")


# ── Full benchmark suite runner ─────────────────────────────────────

class TestFullBenchmarkSuite:
    """Run the complete benchmark suite and produce a report."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_full_suite(self, hf_nli_verifier):
        """
        Complete benchmark: all datasets, full metrics, JSON output.
        This is the main entry point for evaluating CertiRAG.
        """
        from eval.benchmark import BenchmarkRunner
        from eval.scoring import compare_models

        max_n = _max_examples()
        runner = BenchmarkRunner(
            verifier=hf_nli_verifier,
            output_dir="eval_results",
        )

        datasets = _requested_datasets()
        all_scores = []

        for ds_name in datasets:
            try:
                scores = runner.run(ds_name, max_examples=max_n)
                all_scores.append(scores)
                print(f"\n{scores.summary_table()}")
            except Exception as e:
                logger.error(f"Failed {ds_name}: {e}")

        # Print comparison table
        if len(all_scores) > 1:
            table = compare_models(all_scores)
            print(f"\n{'=' * 64}")
            print("  CROSS-DATASET COMPARISON")
            print(f"{'=' * 64}")
            print(table)

        # Save combined report
        report = {
            "model": runner.model_name,
            "max_examples_per_dataset": max_n,
            "datasets": {s.dataset: s.to_dict() for s in all_scores},
        }

        report_path = Path("eval_results") / "benchmark_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Full report saved to: {report_path}")

        # At least one dataset should have reasonable performance
        best_binary = max(s.accuracy_binary for s in all_scores) if all_scores else 0
        assert best_binary > 0.5, (
            f"Best binary accuracy across all datasets is {best_binary:.1%} — "
            f"model is not performing above random"
        )

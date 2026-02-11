"""
CertiRAG Benchmark Harness
===========================

Pluggable evaluation framework for benchmarking claim verification models.
Supports any verifier that implements the ``BaseVerifier.verify_batch()``
interface, and runs against standardised NLI/fact-verification datasets.

Quick start
-----------
    from eval.benchmark import BenchmarkRunner
    from certirag.verify.hf_nli_verifier import HFNLIVerifier

    runner = BenchmarkRunner(verifier=HFNLIVerifier())
    scores = runner.run("vitaminc", max_examples=200)
    print(scores.summary_table())

Plug in a different model
-------------------------
    from certirag.verify.minicheck import MiniCheckVerifier
    runner = BenchmarkRunner(verifier=MiniCheckVerifier(mode="lite"))
    scores = runner.run("anli_r1", max_examples=500)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

# ── Fix sys.path when running as `python eval/benchmark.py` ────────
# Python adds the script's directory (eval/) to sys.path[0], which
# causes `import datasets` to resolve to eval/datasets/ instead of
# the HuggingFace datasets package.  Replace it with the repo root.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_this_dir)
if sys.path and os.path.samefile(sys.path[0], _this_dir):
    sys.path[0] = _repo_root

from eval.datasets.loaders import (
    BenchmarkExample,
    load_dataset_by_name,
    load_all_datasets,
    DATASET_REGISTRY,
)
from eval.scoring import (
    BenchmarkPrediction,
    BenchmarkScores,
    compare_models,
    score_predictions,
)

logger = logging.getLogger(__name__)


# ── Verifier protocol (duck-typing for pluggability) ───────────────

@runtime_checkable
class VerifierProtocol(Protocol):
    """
    Any object with a ``verify_batch`` method can be used as a verifier.
    This matches ``BaseVerifier.verify_batch()`` from certirag.
    """

    def verify_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list:
        """
        Verify (claim, evidence) pairs.
        Returns list of objects with ``.label`` and ``.score`` attributes.
        """
        ...


# ── Label mapping from verifier output to benchmark labels ─────────

VERIFIER_LABEL_MAP = {
    "entailed": "entailed",
    "contradicted": "contradicted",
    "not_enough_info": "not_enough_info",
    # Handle VerificationLabel enum .value
    "VerificationLabel.ENTAILED": "entailed",
    "VerificationLabel.CONTRADICTED": "contradicted",
    "VerificationLabel.NOT_ENOUGH_INFO": "not_enough_info",
}


def _normalize_pred_label(label) -> str:
    """Convert verifier output label to benchmark label string."""
    # Handle enum
    if hasattr(label, "value"):
        label_str = str(label.value)
    else:
        label_str = str(label)
    return VERIFIER_LABEL_MAP.get(label_str, label_str)


# ── Benchmark runner ───────────────────────────────────────────────

class BenchmarkRunner:
    """
    Runs a verifier against benchmark datasets and computes metrics.

    Args:
        verifier: Any object implementing ``verify_batch(pairs) → scores``.
        model_name: Display name (auto-detected if None).
        batch_size: How many pairs to verify at once.
        output_dir: Directory to save JSON results.
    """

    def __init__(
        self,
        verifier,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        output_dir: str = "eval_results",
    ):
        self.verifier = verifier
        self.model_name = model_name or self._detect_model_name(verifier)
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)

    @staticmethod
    def _detect_model_name(verifier) -> str:
        """Try to auto-detect model name from verifier."""
        if hasattr(verifier, "model_name"):
            name = verifier.model_name
            # Shorten HF paths
            if "/" in name:
                name = name.split("/")[-1]
            return name
        return type(verifier).__name__

    def run(
        self,
        dataset_name: str,
        max_examples: Optional[int] = None,
        label_filter: Optional[list[str]] = None,
        save_results: bool = True,
        **dataset_kwargs,
    ) -> BenchmarkScores:
        """
        Run benchmark on a single dataset.

        Args:
            dataset_name: Name from DATASET_REGISTRY (e.g. "vitaminc", "anli_r1").
            max_examples: Cap on examples to evaluate.
            label_filter: Only include certain gold labels.
            save_results: Write JSON results to output_dir.

        Returns:
            BenchmarkScores with all metrics.
        """
        logger.info(f"Loading dataset: {dataset_name}")
        examples = load_dataset_by_name(
            dataset_name,
            max_examples=max_examples,
            label_filter=label_filter,
            **dataset_kwargs,
        )

        if not examples:
            logger.warning(f"No examples loaded for {dataset_name}")
            return score_predictions([], dataset=dataset_name, model_name=self.model_name)

        logger.info(
            f"Evaluating {len(examples)} examples from {dataset_name} "
            f"with {self.model_name}"
        )

        predictions = self._evaluate_examples(examples)
        scores = score_predictions(
            predictions, dataset=dataset_name, model_name=self.model_name
        )

        if save_results:
            self._save_results(scores)

        return scores

    def run_all(
        self,
        max_per_dataset: int = 200,
        label_filter: Optional[list[str]] = None,
        save_results: bool = True,
    ) -> list[BenchmarkScores]:
        """Run benchmark on all registered datasets."""
        results = []
        for name in DATASET_REGISTRY:
            try:
                scores = self.run(
                    name,
                    max_examples=max_per_dataset,
                    label_filter=label_filter,
                    save_results=save_results,
                )
                results.append(scores)
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
        return results

    def _evaluate_examples(
        self, examples: list[BenchmarkExample]
    ) -> list[BenchmarkPrediction]:
        """Run verifier on examples and collect predictions."""
        predictions: list[BenchmarkPrediction] = []

        # Process in batches
        for batch_start in range(0, len(examples), self.batch_size):
            batch = examples[batch_start : batch_start + self.batch_size]
            pairs = [(ex.claim, ex.evidence) for ex in batch]

            start_time = time.time()
            try:
                scores = self.verifier.verify_batch(pairs)
            except Exception as e:
                logger.error(f"Verifier failed on batch {batch_start}: {e}")
                # Fill with NOT_ENOUGH_INFO on error
                for ex in batch:
                    predictions.append(
                        BenchmarkPrediction(
                            example_id=ex.example_id,
                            gold_label=ex.gold_label,
                            pred_label="not_enough_info",
                            pred_score=0.0,
                            latency_ms=0.0,
                        )
                    )
                continue

            elapsed_ms = (time.time() - start_time) * 1000
            per_pair_ms = elapsed_ms / len(batch) if batch else 0

            for ex, score in zip(batch, scores):
                pred_label = _normalize_pred_label(score.label)
                # raw_score = entailment probability (for AUROC)
                # score = confidence in predicted class
                entailment_score = float(
                    getattr(score, 'raw_score', score.score)
                )
                predictions.append(
                    BenchmarkPrediction(
                        example_id=ex.example_id,
                        gold_label=ex.gold_label,
                        pred_label=pred_label,
                        pred_score=float(score.score),
                        entailment_score=entailment_score,
                        latency_ms=per_pair_ms,
                    )
                )

            if (batch_start + self.batch_size) % (self.batch_size * 10) == 0:
                logger.info(
                    f"  Progress: {min(batch_start + self.batch_size, len(examples))}"
                    f"/{len(examples)}"
                )

        return predictions

    def _save_results(self, scores: BenchmarkScores) -> None:
        """Save results as JSON, including bootstrap confidence intervals."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{scores.model_name}_{scores.dataset}_{scores.n_examples}.json"
        filepath = self.output_dir / filename

        result_dict = scores.to_dict()

        # Compute bootstrap confidence intervals
        if scores.predictions and len(scores.predictions) >= 20:
            try:
                from eval.scoring import bootstrap_confidence_intervals
                ci = bootstrap_confidence_intervals(scores.predictions)
                result_dict["confidence_intervals_95"] = ci
                logger.info("Bootstrap 95%% CIs computed for %s", filename)
            except Exception as e:
                logger.warning("Could not compute bootstrap CIs: %s", e)

        with open(filepath, "w") as f:
            json.dump(result_dict, f, indent=2)

        logger.info(f"Results saved to {filepath}")


# ── CLI entry point ────────────────────────────────────────────────

def main():
    """Run benchmarks from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="CertiRAG Benchmark Runner")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()) + ["all"],
        default="all",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--max-examples", type=int, default=200,
        help="Maximum examples per dataset",
    )
    parser.add_argument(
        "--model",
        choices=[
            "hf_nli", "minicheck_lite", "gemini",
            "groq", "groq_mixtral", "groq_gemma",
            "ensemble_accurate", "ensemble_sota", "ensemble_large",
            "cascade",
        ],
        default="hf_nli",
        help="Verifier model to use",
    )
    parser.add_argument(
        "--output-dir", default="eval_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for verification",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Enable isotonic-regression calibration (uses first --cal-size examples)",
    )
    parser.add_argument(
        "--cal-size", type=int, default=50,
        help="Number of examples for calibration fitting",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # Create verifier
    if args.model == "hf_nli":
        from certirag.verify.hf_nli_verifier import HFNLIVerifier
        verifier = HFNLIVerifier(device="cpu", batch_size=args.batch_size)
    elif args.model == "minicheck_lite":
        from certirag.verify.minicheck import MiniCheckVerifier
        verifier = MiniCheckVerifier(mode="lite")
    elif args.model == "gemini":
        from certirag.verify.gemini_verifier import GeminiVerifier
        api_key = os.environ.get("CERTIRAG_GEMINI_API_KEY")
        if not api_key:
            print("ERROR: Set CERTIRAG_GEMINI_API_KEY environment variable")
            return
        verifier = GeminiVerifier(api_key=api_key)
    elif args.model.startswith("groq"):
        from certirag.verify.groq_verifier import GroqVerifier
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            print("ERROR: Set GROQ_API_KEY environment variable")
            print("Get free key at: https://console.groq.com/")
            return
        model_map = {
            "groq": "llama-3.3-70b",
            "groq_mixtral": "mixtral-8x7b",
            "groq_gemma": "gemma2-9b",
        }
        groq_model = model_map.get(args.model, "llama-3.3-70b")
        verifier = GroqVerifier(api_key=api_key, model=groq_model, batch_size=5)
    elif args.model.startswith("ensemble_"):
        from certirag.verify.ensemble_verifier import EnsembleNLIVerifier
        preset = args.model.replace("ensemble_", "")  # accurate | sota | large
        verifier = EnsembleNLIVerifier(preset=preset, device="cpu")
    elif args.model == "cascade":
        from certirag.verify.cascade_verifier import CascadeVerifier
        verifier = CascadeVerifier(device="cpu")
    else:
        print(f"Unknown model: {args.model}")
        return

    runner = BenchmarkRunner(
        verifier=verifier,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    # Optional calibration
    if args.calibrate and hasattr(verifier, "fit_calibrator"):
        from eval.datasets.loaders import load_vitaminc
        print(f"Fitting calibrator on {args.cal_size} VitaminC examples...")
        cal_examples = load_vitaminc(max_examples=args.cal_size)
        pairs = [(ex.claim, ex.evidence) for ex in cal_examples]
        labels = [ex.gold_label for ex in cal_examples]  # string labels: entailed/contradicted/not_enough_info
        ece = verifier.fit_calibrator(pairs, labels)
        print(f"Post-calibration ECE: {ece:.4f}")

    # Run
    if args.dataset == "all":
        results = runner.run_all(max_per_dataset=args.max_examples)
    else:
        results = [runner.run(args.dataset, max_examples=args.max_examples)]

    # Print results
    for r in results:
        print(r.summary_table())
        print()

    if len(results) > 1:
        print("\n" + compare_models(results))


if __name__ == "__main__":
    main()

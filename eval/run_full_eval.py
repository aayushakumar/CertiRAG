#!/usr/bin/env python3
"""
CertiRAG Comprehensive Evaluation Suite
==========================================

Runs all models across all datasets with threshold optimization,
calibration, and comparison. Produces a complete leaderboard.

Usage
-----
    # Full sweep (all models, all datasets, 200 examples each)
    python eval/run_full_eval.py

    # Quick test (50 examples)
    python eval/run_full_eval.py --max-examples 50

    # Specific dataset only
    python eval/run_full_eval.py --dataset vitaminc

    # Skip heavy models
    python eval/run_full_eval.py --skip-large
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Fix sys.path: When running as `python eval/run_full_eval.py`, Python
# adds eval/ to sys.path[0], which shadows the HuggingFace 'datasets'
# package with our local eval/datasets/. Remove that entry and ensure
# the project root is available instead.
_script_dir = str(Path(__file__).resolve().parent)
_project_root = str(Path(__file__).resolve().parent.parent)
if sys.path and sys.path[0] == _script_dir:
    sys.path[0] = _project_root
elif _project_root not in sys.path:
    sys.path.append(_project_root)

from eval.benchmark import BenchmarkRunner
from eval.datasets.loaders import (
    BenchmarkExample,
    load_dataset_by_name,
    DATASET_REGISTRY,
)
from eval.optimizer import (
    ThresholdOptimizer,
    optimize_and_evaluate,
    stratified_split,
)
from eval.scoring import (
    BenchmarkPrediction,
    BenchmarkScores,
    compare_models,
    score_predictions,
)

logger = logging.getLogger("certirag.eval.full")

# ── Model registry ──────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Configuration for a benchmark model."""
    name: str
    factory: str  # function name to create verifier
    supports_calibration: bool = False
    supports_optimization: bool = True
    is_heavy: bool = False
    needs_api_key: bool = False
    api_key_env: str = ""


MODEL_CONFIGS = {
    "deberta_2class": ModelConfig(
        name="DeBERTa-NLI-2class",
        factory="hf_nli",
        supports_optimization=True,
    ),
    "deberta_3class": ModelConfig(
        name="DeBERTa-NLI-3class",
        factory="ensemble_accurate",
        supports_calibration=True,
        supports_optimization=True,
    ),
    "ensemble_sota": ModelConfig(
        name="Ensemble-SOTA",
        factory="ensemble_sota",
        supports_calibration=True,
        supports_optimization=True,
    ),
    "deberta_large": ModelConfig(
        name="DeBERTa-Large",
        factory="ensemble_large",
        supports_calibration=True,
        supports_optimization=True,
        is_heavy=True,
    ),
    "groq_llama70b": ModelConfig(
        name="Groq-Llama3.3-70B",
        factory="groq",
        supports_optimization=True,
        needs_api_key=True,
        api_key_env="GROQ_API_KEY",
    ),
    "groq_mixtral": ModelConfig(
        name="Groq-Mixtral-8x7B",
        factory="groq_mixtral",
        supports_optimization=True,
        needs_api_key=True,
        api_key_env="GROQ_API_KEY",
    ),
}


def create_verifier(factory: str, batch_size: int = 32):
    """Create a verifier by factory name."""
    if factory == "hf_nli":
        from certirag.verify.hf_nli_verifier import HFNLIVerifier
        return HFNLIVerifier(device="cpu", batch_size=batch_size)

    elif factory == "ensemble_accurate":
        from certirag.verify.ensemble_verifier import EnsembleNLIVerifier
        return EnsembleNLIVerifier(preset="accurate", device="cpu")

    elif factory == "ensemble_sota":
        from certirag.verify.ensemble_verifier import EnsembleNLIVerifier
        return EnsembleNLIVerifier(preset="sota", device="cpu")

    elif factory == "ensemble_large":
        from certirag.verify.ensemble_verifier import EnsembleNLIVerifier
        return EnsembleNLIVerifier(preset="large", device="cpu")

    elif factory.startswith("groq"):
        from certirag.verify.groq_verifier import GroqVerifier
        model_map = {
            "groq": "llama-3.3-70b",
            "groq_mixtral": "mixtral-8x7b",
            "groq_gemma": "gemma2-9b",
        }
        groq_model = model_map.get(factory, "llama-3.3-70b")
        return GroqVerifier(model=groq_model, batch_size=5)

    else:
        raise ValueError(f"Unknown verifier factory: {factory}")


def run_single_evaluation(
    model_key: str,
    config: ModelConfig,
    dataset_name: str,
    max_examples: int = 200,
    batch_size: int = 32,
    output_dir: str = "eval_results",
    optimize_thresholds: bool = True,
    run_calibration: bool = True,
) -> dict:
    """
    Run a single model on a single dataset with all optimizations.

    Returns dict with keys: baseline, optimized, calibrated.
    """
    result = {
        "model": config.name,
        "dataset": dataset_name,
        "baseline": None,
        "optimized": None,
        "calibrated": None,
    }

    try:
        verifier = create_verifier(config.factory, batch_size)
    except Exception as e:
        logger.error(f"Failed to create {config.name}: {e}")
        return result

    runner = BenchmarkRunner(
        verifier=verifier,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    # Run baseline
    print(f"\n{'='*60}")
    print(f"  {config.name} on {dataset_name} ({max_examples} examples)")
    print(f"{'='*60}")

    try:
        baseline = runner.run(
            dataset_name,
            max_examples=max_examples,
            save_results=True,
        )
        result["baseline"] = baseline
        print(f"  Baseline: Acc3={baseline.accuracy_3class:.1%}  "
              f"MacF1={baseline.macro_f1:.1%}  AUROC={baseline.auroc:.4f}")
    except Exception as e:
        logger.error(f"Baseline failed for {config.name}/{dataset_name}: {e}")
        return result

    # Threshold optimization (if we have per-example predictions)
    if optimize_thresholds and config.supports_optimization and baseline.predictions:
        try:
            base_scores, opt_scores, opt_result = optimize_and_evaluate(
                baseline.predictions,
                dataset=dataset_name,
                model_name=config.name,
                cal_fraction=0.3,
                metric="macro_f1",
            )
            result["optimized"] = opt_scores

            delta = opt_scores.macro_f1 - baseline.macro_f1
            print(f"  + Threshold opt: Acc3={opt_scores.accuracy_3class:.1%}  "
                  f"MacF1={opt_scores.macro_f1:.1%}  "
                  f"(Δ={delta:+.1%})")
            print(f"    Thresholds: entail={opt_result.best_params['entail_threshold']:.3f}  "
                  f"contradict={opt_result.best_params['contradict_threshold']:.3f}")

            # Save optimized results
            out_path = Path(output_dir) / f"{config.name}+opt_{dataset_name}_{opt_scores.n_examples}.json"
            with open(out_path, "w") as f:
                json.dump(opt_scores.to_dict(), f, indent=2)

        except Exception as e:
            logger.error(f"Threshold optimization failed: {e}")

    # Calibration (for ensemble models)
    if run_calibration and config.supports_calibration and hasattr(verifier, "fit_calibrator"):
        try:
            from eval.datasets.loaders import load_vitaminc

            # Fit calibrator on separate data
            cal_examples = load_vitaminc(max_examples=100)
            pairs = [(ex.claim, ex.evidence) for ex in cal_examples]
            labels = [ex.gold_label for ex in cal_examples]
            ece = verifier.fit_calibrator(pairs, labels)
            print(f"  + Calibration fitted: ECE={ece:.4f}")

            # Re-run with calibration
            cal_scores = runner.run(
                dataset_name,
                max_examples=max_examples,
                save_results=False,
            )
            cal_scores = BenchmarkScores(
                dataset=cal_scores.dataset,
                model_name=f"{config.name}+cal",
                n_examples=cal_scores.n_examples,
                accuracy_3class=cal_scores.accuracy_3class,
                accuracy_binary=cal_scores.accuracy_binary,
                class_metrics=cal_scores.class_metrics,
                macro_f1=cal_scores.macro_f1,
                binary_entail_f1=cal_scores.binary_entail_f1,
                auroc=cal_scores.auroc,
                auroc_error=cal_scores.auroc_error,
                ece=cal_scores.ece,
                confusion_matrix=cal_scores.confusion_matrix,
                total_time_ms=cal_scores.total_time_ms,
                avg_latency_ms=cal_scores.avg_latency_ms,
                p95_latency_ms=cal_scores.p95_latency_ms,
                gold_distribution=cal_scores.gold_distribution,
                pred_distribution=cal_scores.pred_distribution,
                predictions=cal_scores.predictions,
            )
            result["calibrated"] = cal_scores

            delta = cal_scores.macro_f1 - baseline.macro_f1
            print(f"  + Calibrated:    Acc3={cal_scores.accuracy_3class:.1%}  "
                  f"MacF1={cal_scores.macro_f1:.1%}  "
                  f"(Δ={delta:+.1%})")

            out_path = Path(output_dir) / f"{config.name}+cal_{dataset_name}_{cal_scores.n_examples}.json"
            with open(out_path, "w") as f:
                json.dump(cal_scores.to_dict(), f, indent=2)

        except Exception as e:
            logger.error(f"Calibration failed: {e}")

    return result


def print_leaderboard(all_results: list[dict]) -> str:
    """Print a comprehensive leaderboard from all results."""
    rows = []
    for r in all_results:
        dataset = r["dataset"]
        model = r["model"]

        for variant_key, suffix in [("baseline", ""), ("optimized", "+opt"), ("calibrated", "+cal")]:
            scores = r.get(variant_key)
            if scores is None:
                continue

            name = f"{model}{suffix}" if suffix else model
            rows.append({
                "Model": name,
                "Dataset": dataset,
                "N": scores.n_examples,
                "Acc(3)": f"{scores.accuracy_3class:.1%}",
                "Acc(bin)": f"{scores.accuracy_binary:.1%}",
                "MacroF1": f"{scores.macro_f1:.1%}",
                "EntF1": f"{scores.binary_entail_f1:.1%}",
                "AUROC": f"{scores.auroc:.4f}" if scores.auroc else "N/A",
                "ECE": f"{scores.ece:.4f}" if scores.ece else "N/A",
            })

    if not rows:
        return "No results to display"

    try:
        from tabulate import tabulate
        headers = list(rows[0].keys())
        table_rows = [[r[h] for h in headers] for r in rows]
        return tabulate(table_rows, headers=headers, tablefmt="grid")
    except ImportError:
        lines = []
        for r in rows:
            lines.append(
                f"{r['Model']:30s} | {r['Dataset']:12s} | "
                f"Acc3={r['Acc(3)']:>6s} MacF1={r['MacroF1']:>6s} AUROC={r['AUROC']:>6s}"
            )
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="CertiRAG Full Evaluation Suite — run all models with optimizations"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default=["all"],
    )
    parser.add_argument("--skip-large", action="store_true",
                       help="Skip DeBERTa-large (slow, high memory)")
    parser.add_argument("--skip-api", action="store_true",
                       help="Skip models requiring API keys")
    parser.add_argument("--no-optimize", action="store_true",
                       help="Skip threshold optimization")
    parser.add_argument("--no-calibrate", action="store_true",
                       help="Skip calibration")
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Select models
    if "all" in args.models:
        model_keys = list(MODEL_CONFIGS.keys())
    else:
        model_keys = args.models

    # Select datasets
    if args.dataset == "all":
        datasets = list(DATASET_REGISTRY.keys())
    else:
        datasets = [args.dataset]

    # Filter models
    active_models = {}
    for key in model_keys:
        config = MODEL_CONFIGS[key]
        if args.skip_large and config.is_heavy:
            print(f"Skipping {config.name} (--skip-large)")
            continue
        if args.skip_api and config.needs_api_key:
            print(f"Skipping {config.name} (--skip-api)")
            continue
        if config.needs_api_key and not os.environ.get(config.api_key_env):
            print(f"Skipping {config.name} (no {config.api_key_env} set)")
            continue
        active_models[key] = config

    print(f"\n{'#'*60}")
    print(f"  CertiRAG Full Evaluation Suite")
    print(f"  Models:   {', '.join(c.name for c in active_models.values())}")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Max examples: {args.max_examples}")
    print(f"  Optimize: {not args.no_optimize}  Calibrate: {not args.no_calibrate}")
    print(f"{'#'*60}\n")

    # Run all evaluations
    all_results = []
    start = time.time()

    for model_key, config in active_models.items():
        for dataset_name in datasets:
            result = run_single_evaluation(
                model_key=model_key,
                config=config,
                dataset_name=dataset_name,
                max_examples=args.max_examples,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                optimize_thresholds=not args.no_optimize,
                run_calibration=not args.no_calibrate and config.supports_calibration,
            )
            all_results.append(result)

    elapsed = time.time() - start

    # Print leaderboard
    print(f"\n\n{'#'*60}")
    print(f"  LEADERBOARD (total time: {elapsed:.0f}s)")
    print(f"{'#'*60}\n")
    print(print_leaderboard(all_results))

    # Save full report
    report_path = Path(args.output_dir) / "full_eval_report.json"
    report = {
        "total_time_s": round(elapsed, 1),
        "models": list(active_models.keys()),
        "datasets": datasets,
        "max_examples": args.max_examples,
        "results": [],
    }

    for r in all_results:
        entry = {"model": r["model"], "dataset": r["dataset"]}
        for variant in ["baseline", "optimized", "calibrated"]:
            if r.get(variant) is not None:
                entry[variant] = r[variant].to_dict()
        report["results"].append(entry)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()

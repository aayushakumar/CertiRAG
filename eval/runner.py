"""
Evaluation Runner
==================

Orchestrates benchmark evaluation: loads data, runs pipeline,
computes metrics, and exports results for paper tables.

Supports:
    - Single-benchmark evaluation
    - Cross-benchmark evaluation (Table 1)
    - Ablation sweeps (Table 2)
    - Latency profiling (Table 3)
    - Calibration analysis (Figure 4)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from eval.metrics import (
    claim_level_accuracy,
    compute_auroc,
    compute_ece,
    faithfulness_rate,
    latency_stats,
    mse_efficiency,
    reliability_diagram_data,
    verification_yield,
)

logger = logging.getLogger("certirag.eval.runner")


@dataclass
class EvalResult:
    """Container for evaluation results."""
    benchmark: str
    subset: str
    num_examples: int
    accuracy: float
    faithfulness: dict[str, float]
    auroc: float
    ece: float
    yield_stats: dict[str, float]
    latency: dict[str, dict[str, float]]
    mse_stats: dict[str, float]
    per_example_results: list[dict] = field(default_factory=list)
    config_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "subset": self.subset,
            "num_examples": self.num_examples,
            "accuracy": self.accuracy,
            **self.faithfulness,
            "auroc": self.auroc,
            "ece": self.ece,
            **{f"yield_{k}": v for k, v in self.yield_stats.items()},
            **{f"latency_{k}": v for k, v in self.latency.get("total_ms", {}).items()},
            **{f"mse_{k}": v for k, v in self.mse_stats.items()},
            "config_hash": self.config_hash,
        }

    def to_latex_row(self) -> str:
        """Format as a LaTeX table row for the paper."""
        return (
            f"{self.subset} & "
            f"{self.accuracy:.1%} & "
            f"{self.faithfulness.get('entailed_f1', 0):.1%} & "
            f"{self.faithfulness.get('macro_f1', 0):.1%} & "
            f"{self.auroc:.3f} & "
            f"{self.ece:.3f} & "
            f"{self.latency.get('total_ms', {}).get('p50', 0):.0f} \\\\"
        )


class EvalRunner:
    """
    Run CertiRAG evaluation on benchmarks.

    Usage:
        from certirag.pipeline import CertiRAGPipeline
        from eval.runner import EvalRunner

        pipeline = CertiRAGPipeline.from_config()
        runner = EvalRunner(pipeline)

        result = runner.evaluate_alce("data/alce", subset="asqa")
        print(result.to_latex_row())
    """

    def __init__(self, pipeline, output_dir: str = "eval_results"):
        """
        Args:
            pipeline: Initialized CertiRAGPipeline.
            output_dir: Directory for saving results.
        """
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_alce(
        self,
        data_dir: str,
        subset: str = "asqa",
        max_examples: Optional[int] = None,
    ) -> EvalResult:
        """Run evaluation on ALCE benchmark."""
        from eval.datasets.alce import load_alce

        examples = load_alce(data_dir, subset=subset, max_examples=max_examples)
        return self._evaluate_examples(examples, benchmark="alce", subset=subset)

    def evaluate_ragtruth(
        self,
        data_dir: str,
        task: Optional[str] = None,
        max_examples: Optional[int] = None,
    ) -> EvalResult:
        """Run evaluation on RAGTruth benchmark."""
        from eval.datasets.ragtruth import load_ragtruth

        examples = load_ragtruth(data_dir, task=task, max_examples=max_examples)
        subset = task or "all"
        return self._evaluate_examples(examples, benchmark="ragtruth", subset=subset)

    def evaluate_aggrefact(
        self,
        data_dir: str,
        subset: Optional[str] = None,
        max_examples: Optional[int] = None,
    ) -> EvalResult:
        """Run evaluation on AggreFact benchmark."""
        from eval.datasets.aggrefact import load_aggrefact

        examples = load_aggrefact(data_dir, subset=subset, max_examples=max_examples)
        return self._evaluate_examples(
            examples, benchmark="aggrefact", subset=subset or "all"
        )

    def _evaluate_examples(
        self,
        examples: list[dict],
        benchmark: str,
        subset: str,
    ) -> EvalResult:
        """
        Core evaluation loop.

        For each example:
            1. Ingest documents
            2. Run pipeline
            3. Match predictions to gold labels
            4. Accumulate metrics
        """
        all_preds = []
        all_golds = []
        all_scores = []
        all_correct = []
        all_timings = []
        per_example = []
        orig_counts = []
        mse_counts = []

        for i, ex in enumerate(examples):
            logger.info(f"Evaluating {benchmark}/{subset} [{i+1}/{len(examples)}]")

            try:
                # Ingest this example's documents
                self.pipeline.ingest(ex["documents"])

                # Run pipeline on question
                question = ex["question"] or ex.get("answer", "")
                result = self.pipeline.run(question)

                # Extract predictions and match to gold
                pred_labels, pred_scores = self._extract_predictions(result)
                gold_labels = [c["label"] for c in ex["gold_claims"]]

                # Align predictions to gold (greedy matching)
                matched_preds, matched_golds, matched_scores = self._align_predictions(
                    pred_labels, pred_scores,
                    gold_labels, ex["gold_claims"]
                )

                all_preds.extend(matched_preds)
                all_golds.extend(matched_golds)
                all_scores.extend(matched_scores)
                all_correct.extend(
                    p == g for p, g in zip(matched_preds, matched_golds)
                )
                all_timings.append(result.timings)

                # MSE counts
                for vr in result.verification_results:
                    orig_counts.append(len(vr.evidence_scores))
                    mse = sum(
                        1 for es in vr.evidence_scores if es.in_mse
                    ) if hasattr(vr.evidence_scores[0], "in_mse") else len(vr.evidence_scores)
                    mse_counts.append(mse)

                per_example.append({
                    "example_id": ex["example_id"],
                    "num_claims": len(result.claim_ir.claims),
                    "num_pred": len(matched_preds),
                    "num_gold": len(gold_labels),
                    "accuracy": claim_level_accuracy(matched_preds, matched_golds),
                    "timings": result.timings,
                })

            except Exception as e:
                logger.warning(f"Error on example {ex['example_id']}: {e}")
                per_example.append({
                    "example_id": ex["example_id"],
                    "error": str(e),
                })
                continue

        # Aggregate metrics
        accuracy = claim_level_accuracy(all_preds, all_golds)
        faith = faithfulness_rate(all_preds, all_golds)
        auroc = compute_auroc(all_scores, all_golds) if all_scores else 0.0
        ece = compute_ece(all_scores, all_correct) if all_scores else 0.0
        yield_s = verification_yield(all_preds, len(all_golds))
        lat = latency_stats(all_timings)
        mse_s = mse_efficiency(orig_counts, mse_counts)

        result = EvalResult(
            benchmark=benchmark,
            subset=subset,
            num_examples=len(examples),
            accuracy=accuracy,
            faithfulness=faith,
            auroc=auroc,
            ece=ece,
            yield_stats=yield_s,
            latency=lat,
            mse_stats=mse_s,
            per_example_results=per_example,
            config_hash=self.pipeline.config.config_hash(),
        )

        # Save results
        self._save_result(result)
        return result

    def _extract_predictions(self, result) -> tuple[list[str], list[float]]:
        """Extract labels and scores from pipeline result."""
        labels = []
        scores = []
        for vr in result.verification_results:
            labels.append(vr.label.value)
            scores.append(vr.max_entail_score)
        return labels, scores

    def _align_predictions(
        self,
        pred_labels: list[str],
        pred_scores: list[float],
        gold_labels: list[str],
        gold_claims: list[dict],
    ) -> tuple[list[str], list[str], list[float]]:
        """
        Align predicted claims to gold claims.

        Uses simple positional matching (first N predictions → first N golds).
        For more sophisticated matching, override with text-based alignment.
        """
        min_len = min(len(pred_labels), len(gold_labels))
        return (
            pred_labels[:min_len],
            gold_labels[:min_len],
            pred_scores[:min_len],
        )

    def _save_result(self, result: EvalResult) -> None:
        """Save evaluation result to file."""
        out_path = self.output_dir / f"{result.benchmark}_{result.subset}.json"
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {out_path}")

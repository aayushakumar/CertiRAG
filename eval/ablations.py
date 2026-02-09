"""
Ablation Configurations
========================

Defines ablation variants for Table 2 of the paper.
Each ablation disables or modifies one component to
measure its marginal contribution.

Ablations:
    A1. No claim decomposition (whole-answer check)
    A2. No MSE selection (all evidence kept)
    A3. BM25-only retrieval (no dense)
    A4. Dense-only retrieval (no BM25)
    A5. No reranking
    A6. NLI verifier instead of MiniCheck
    A7. LLM-judge instead of MiniCheck
    A8. No calibration
    A9. Lowered τ_entail threshold
    A10. Raised τ_entail threshold
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

from certirag.config import CertiRAGConfig, get_config


def get_ablation_configs(
    base_config: Optional[CertiRAGConfig] = None,
) -> dict[str, CertiRAGConfig]:
    """
    Generate all ablation configurations.

    Each variant modifies exactly one aspect of the base config
    to isolate individual component contributions.

    Args:
        base_config: Base configuration (default settings if None).

    Returns:
        Dict mapping ablation name → modified config.
    """
    base = base_config or get_config()
    ablations = {}

    # Full system (no ablation)
    ablations["full_system"] = deepcopy(base)

    # A1: No claim decomposition → treat entire answer as one claim
    a1 = deepcopy(base)
    a1.claim.split_conjunctions = False
    a1.claim.max_claim_tokens = 512  # Allow large single claims
    ablations["no_decomposition"] = a1

    # A2: No MSE selection → keep all evidence
    a2 = deepcopy(base)
    a2.verification.use_mse = False
    ablations["no_mse"] = a2

    # A3: BM25-only retrieval
    a3 = deepcopy(base)
    a3.retrieval.bm25_weight = 1.0
    a3.retrieval.dense_weight = 0.0
    ablations["bm25_only"] = a3

    # A4: Dense-only retrieval
    a4 = deepcopy(base)
    a4.retrieval.bm25_weight = 0.0
    a4.retrieval.dense_weight = 1.0
    ablations["dense_only"] = a4

    # A5: No reranking
    a5 = deepcopy(base)
    a5.retrieval.use_reranker = False
    ablations["no_reranker"] = a5

    # A6: NLI verifier instead of MiniCheck
    a6 = deepcopy(base)
    a6.verification.verifier_model = "cross-encoder/nli-deberta-v3-base"
    ablations["nli_verifier"] = a6

    # A7: LLM-judge verifier (API-based)
    a7 = deepcopy(base)
    a7.verification.verifier_model = "gpt-4o"
    ablations["llm_judge"] = a7

    # A8: No calibration
    a8 = deepcopy(base)
    a8.verification.calibration_method = "none"
    ablations["no_calibration"] = a8

    # A9: Lower τ_entail (more permissive)
    a9 = deepcopy(base)
    a9.verification.tau_entail = 0.3
    ablations["low_threshold"] = a9

    # A10: Higher τ_entail (more conservative)
    a10 = deepcopy(base)
    a10.verification.tau_entail = 0.9
    ablations["high_threshold"] = a10

    return ablations


def format_ablation_table(results: dict[str, dict]) -> str:
    """
    Format ablation results as a LaTeX table for the paper.

    Args:
        results: Dict mapping ablation name → metric dict.

    Returns:
        LaTeX table string.
    """
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Ablation study results (ALCE-ASQA).}\n"
        "\\label{tab:ablation}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "System & Acc & F1$_E$ & AUROC & ECE \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for name, metrics in results.items():
        display_name = name.replace("_", " ").title()
        row = (
            f"{display_name} & "
            f"{metrics.get('accuracy', 0):.1%} & "
            f"{metrics.get('entailed_f1', 0):.1%} & "
            f"{metrics.get('auroc', 0):.3f} & "
            f"{metrics.get('ece', 0):.3f} \\\\"
        )
        rows.append(row)

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )

    return header + "\n".join(rows) + "\n" + footer

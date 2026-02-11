"""
Dataset Loaders for CertiRAG Evaluation
========================================

Downloads and normalises public NLI / fact-verification datasets into
a uniform format for benchmarking verifier accuracy.

Datasets
--------
1. **VitaminC** (Schuster et al., 2021)
   - Real-world fact verification with Wikipedia evidence
   - Labels: SUPPORTS, REFUTES, NOT ENOUGH INFO
   - 55K test examples

2. **ANLI** (Nie et al., 2020)
   - Adversarial NLI: human-authored hard examples
   - Labels: 0=entailment, 1=neutral, 2=contradiction
   - 3 rounds (R1/R2/R3) of increasing difficulty

3. **FEVER-NLI** (Thorne et al., 2018; Nie et al., 2019)
   - FEVER claims re-formatted as NLI premise/hypothesis pairs
   - Labels: 0=entailment, 1=neutral, 2=contradiction
   - 20K dev / 20K test examples

Uniform output format
---------------------
Each loader returns ``list[BenchmarkExample]`` with:
    - claim: str (hypothesis)
    - evidence: str (premise)
    - gold_label: str ("entailed" | "contradicted" | "not_enough_info")
    - dataset: str
    - example_id: str
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _hf_load_dataset(*args, **kwargs):
    """
    Load a HuggingFace dataset, working around the local eval/datasets/ shadow.

    When running ``python eval/benchmark.py``, Python adds ``eval/`` to
    sys.path[0], so ``import datasets`` resolves to ``eval/datasets/``
    instead of the HuggingFace package.  We fix this by:
      1. Temporarily removing path entries that cause shadowing
      2. Clearing cached module entries for the shadow
      3. Importing the real HuggingFace ``datasets`` package
      4. Restoring everything afterwards
    """
    # Fast path: already have the real HF datasets module
    ds_mod = sys.modules.get("datasets")
    if ds_mod and hasattr(ds_mod, "load_dataset"):
        return ds_mod.load_dataset(*args, **kwargs)

    import importlib

    # Save and remove all 'datasets*' entries from sys.modules
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == "datasets" or key.startswith("datasets."):
            saved_modules[key] = sys.modules.pop(key)

    # Remove path entries that cause shadowing (eval/ dir contains datasets/)
    saved_paths = []
    for i, p in enumerate(list(sys.path)):
        import os
        datasets_dir = os.path.join(p, "datasets")
        if os.path.isdir(datasets_dir) and os.path.exists(
            os.path.join(datasets_dir, "__init__.py")
        ):
            # Check if this is our local eval/datasets, not the HF one
            if "site-packages" not in p:
                saved_paths.append((i, p))

    # Remove shadow paths (reverse order to preserve indices)
    for _, p in reversed(saved_paths):
        sys.path.remove(p)

    # Also clear the importlib finder caches
    importlib.invalidate_caches()

    try:
        import datasets as _real_datasets

        if not hasattr(_real_datasets, "load_dataset"):
            raise ImportError(
                f"Imported 'datasets' from {getattr(_real_datasets, '__file__', '?')} "
                "but it doesn't have load_dataset — still shadowed!"
            )
        result = _real_datasets.load_dataset(*args, **kwargs)
        return result
    finally:
        # Restore shadow paths at their original positions
        for idx, p in saved_paths:
            if p not in sys.path:
                sys.path.insert(idx, p)
        # Keep the real HF datasets in sys.modules (don't restore shadow)
        # Only restore non-datasets modules that were accidentally removed
        for key, mod in saved_modules.items():
            if key not in sys.modules:
                # Don't restore the shadow — keep the real HF module
                pass


@dataclass
class BenchmarkExample:
    """A single claim-evidence pair with gold label."""

    claim: str
    evidence: str
    gold_label: str  # "entailed" | "contradicted" | "not_enough_info"
    dataset: str
    example_id: str
    metadata: dict = field(default_factory=dict)


# ── Label normalisation maps ───────────────────────────────────────

VITAMINC_LABEL_MAP = {
    "SUPPORTS": "entailed",
    "REFUTES": "contradicted",
    "NOT ENOUGH INFO": "not_enough_info",
}

ANLI_LABEL_MAP = {
    0: "entailed",
    1: "not_enough_info",
    2: "contradicted",
}

FEVER_NLI_LABEL_MAP = {
    0: "entailed",
    1: "not_enough_info",
    2: "contradicted",
}


def load_vitaminc(
    split: str = "test",
    max_examples: Optional[int] = None,
    label_filter: Optional[list[str]] = None,
) -> list[BenchmarkExample]:
    """
    Load VitaminC fact-verification dataset.

    Args:
        split: Dataset split ("train", "validation", "test").
        max_examples: Cap total examples returned.
        label_filter: Only include these gold labels (e.g. ["entailed", "contradicted"]).

    Returns:
        List of BenchmarkExample.
    """
    slice_spec = f"{split}[:{max_examples}]" if max_examples else split
    ds = _hf_load_dataset("tals/vitaminc", split=slice_spec)
    logger.info(f"Loaded VitaminC {split}: {len(ds)} examples")

    examples = []
    for row in ds:
        gold = VITAMINC_LABEL_MAP.get(row["label"])
        if gold is None:
            continue
        if label_filter and gold not in label_filter:
            continue
        examples.append(
            BenchmarkExample(
                claim=row["claim"],
                evidence=row["evidence"],
                gold_label=gold,
                dataset="vitaminc",
                example_id=str(row.get("unique_id", "")),
                metadata={
                    "page": row.get("page", ""),
                    "revision_type": row.get("revision_type", ""),
                },
            )
        )
    logger.info(f"VitaminC: {len(examples)} examples after filtering")
    return examples


def load_anli(
    round: str = "r1",
    split: str = "test",
    max_examples: Optional[int] = None,
    label_filter: Optional[list[str]] = None,
) -> list[BenchmarkExample]:
    """
    Load Adversarial NLI dataset (ANLI).

    Args:
        round: ANLI round ("r1", "r2", "r3").
        split: Dataset split for the round (e.g. "test_r1").
        max_examples: Cap total examples.
        label_filter: Only include these gold labels.

    Returns:
        List of BenchmarkExample.
    """
    split_name = f"{split}_{round}"
    slice_spec = f"{split_name}[:{max_examples}]" if max_examples else split_name
    ds = _hf_load_dataset("facebook/anli", split=slice_spec)
    logger.info(f"Loaded ANLI {round} {split}: {len(ds)} examples")

    examples = []
    for row in ds:
        gold = ANLI_LABEL_MAP.get(row["label"])
        if gold is None:
            continue
        if label_filter and gold not in label_filter:
            continue
        examples.append(
            BenchmarkExample(
                claim=row["hypothesis"],
                evidence=row["premise"],
                gold_label=gold,
                dataset=f"anli_{round}",
                example_id=str(row.get("uid", "")),
                metadata={"reason": row.get("reason", "")},
            )
        )
    logger.info(f"ANLI {round}: {len(examples)} examples after filtering")
    return examples


def load_fever_nli(
    split: str = "dev",
    max_examples: Optional[int] = None,
    label_filter: Optional[list[str]] = None,
) -> list[BenchmarkExample]:
    """
    Load FEVER-NLI dataset (FEVER reformulated as NLI).

    Args:
        split: Dataset split ("train", "dev", "test").
        max_examples: Cap total examples.
        label_filter: Only include these gold labels.

    Returns:
        List of BenchmarkExample.
    """
    slice_spec = f"{split}[:{max_examples}]" if max_examples else split
    ds = _hf_load_dataset("pietrolesci/nli_fever", split=slice_spec)
    logger.info(f"Loaded FEVER-NLI {split}: {len(ds)} examples")

    examples = []
    for row in ds:
        label_int = row.get("label", -1)
        gold = FEVER_NLI_LABEL_MAP.get(label_int)
        if gold is None:
            continue

        hypothesis = row.get("hypothesis", "")
        premise = row.get("premise", "")
        if not hypothesis or not premise:
            continue

        if label_filter and gold not in label_filter:
            continue

        examples.append(
            BenchmarkExample(
                claim=hypothesis,
                evidence=premise,
                gold_label=gold,
                dataset="fever_nli",
                example_id=f"fever_{row.get('cid', '')}",
                metadata={
                    "verifiable": row.get("verifiable", ""),
                    "fever_gold_label": row.get("fever_gold_label", ""),
                },
            )
        )
    logger.info(f"FEVER-NLI: {len(examples)} examples after filtering")
    return examples


# ── Convenience: load all datasets ─────────────────────────────────

DATASET_REGISTRY = {
    "vitaminc": load_vitaminc,
    "anli_r1": lambda **kw: load_anli(round="r1", **kw),
    "anli_r2": lambda **kw: load_anli(round="r2", **kw),
    "anli_r3": lambda **kw: load_anli(round="r3", **kw),
    "fever_nli": load_fever_nli,
}


def load_dataset_by_name(
    name: str,
    max_examples: Optional[int] = None,
    label_filter: Optional[list[str]] = None,
    **kwargs,
) -> list[BenchmarkExample]:
    """Load a dataset by registry name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name](
        max_examples=max_examples, label_filter=label_filter, **kwargs
    )


def load_all_datasets(
    max_per_dataset: int = 200,
    label_filter: Optional[list[str]] = None,
) -> dict[str, list[BenchmarkExample]]:
    """Load all registered datasets with a per-dataset sample cap."""
    results = {}
    for name in DATASET_REGISTRY:
        try:
            results[name] = load_dataset_by_name(
                name, max_examples=max_per_dataset, label_filter=label_filter
            )
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
            results[name] = []
    return results

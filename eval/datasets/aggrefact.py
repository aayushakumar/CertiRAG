"""
AggreFact Benchmark Loader
============================

Loads the AggreFact benchmark (Tang et al., 2022) for summarization
factuality. This is a transfer evaluation — CertiRAG is designed for
RAG but should generalize to abstractive summarization factuality.

AggreFact aggregates multiple factuality datasets:
    - XSumFaith, Polytope, FactCC, SummEval, FRANK, etc.

We map sentence-level factuality labels to claim-level for evaluation.

Dataset: https://github.com/Liyan06/AggreFact
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("certirag.eval.datasets.aggrefact")


def load_aggrefact(
    data_dir: str | Path,
    subset: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> list[dict]:
    """
    Load AggreFact dataset into uniform format.

    Args:
        data_dir: Directory containing AggreFact data.
        subset: Filter by source dataset (e.g. 'xsumfaith', 'polytope').
        max_examples: Cap on examples.

    Returns:
        List of dicts with keys:
            - example_id: str
            - question: str (empty for summarization)
            - documents: list[dict] with doc_id, text (source article)
            - answer: str (summary)
            - gold_claims: list[dict] with text, label
            - source_dataset: str
    """
    data_dir = Path(data_dir)

    # Try standard file locations
    data_file = None
    for candidate in [
        data_dir / "aggrefact_data.jsonl",
        data_dir / "test.jsonl",
        data_dir / "aggrefact.json",
    ]:
        if candidate.exists():
            data_file = candidate
            break

    if data_file is None:
        raise FileNotFoundError(
            f"AggreFact data not found in {data_dir}. "
            f"Download from https://github.com/Liyan06/AggreFact"
        )

    logger.info(f"Loading AggreFact from {data_file}")

    examples = []
    with open(data_file) as f:
        if data_file.suffix == ".json":
            raw_data = json.load(f)
            items = raw_data if isinstance(raw_data, list) else [raw_data]
        else:
            items = [json.loads(line) for line in f if line.strip()]

    for i, item in enumerate(items):
        if max_examples and len(examples) >= max_examples:
            break

        source = item.get("dataset", item.get("source", "unknown"))
        if subset and source.lower() != subset.lower():
            continue

        # Source article = "document"
        article = item.get("article", item.get("source", ""))
        documents = [{"doc_id": f"aggre_{i}_src", "text": article}]

        # Summary = "answer" in our uniform format
        summary = item.get("summary", item.get("claim", ""))

        # Labels
        label = _normalize_aggrefact_label(item.get("label", "unknown"))
        gold_claims = [{
            "text": summary,
            "label": label,
        }]

        examples.append({
            "example_id": item.get("id", f"aggrefact_{i}"),
            "question": "",  # No question for summarization
            "documents": documents,
            "answer": summary,
            "gold_claims": gold_claims,
            "source_dataset": source,
        })

    logger.info(f"Loaded {len(examples)} examples from AggreFact")
    return examples


def _normalize_aggrefact_label(label) -> str:
    """Map AggreFact labels to CertiRAG's 3-way scheme."""
    if isinstance(label, (int, float)):
        # Binary labels: 1 = consistent, 0 = inconsistent
        return "ENTAILED" if label >= 0.5 else "CONTRADICTED"

    label = str(label).lower().strip()
    if label in ("consistent", "factual", "1", "true"):
        return "ENTAILED"
    elif label in ("inconsistent", "non-factual", "0", "false"):
        return "CONTRADICTED"
    else:
        return "NOT_ENOUGH_INFO"

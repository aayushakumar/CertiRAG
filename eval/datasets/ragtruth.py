"""
RAGTruth Benchmark Loader
===========================

Loads the RAGTruth dataset (Wu et al., 2024) for RAG-specific
hallucination detection. RAGTruth provides:
    - Real RAG outputs from multiple LLMs
    - Word-level hallucination annotations
    - Task categories: QA, data-to-text, summarization

We evaluate CertiRAG's claim-level verification against RAGTruth's
word-level annotations by mapping word spans to claims.

Dataset: https://github.com/ParaGonTHero/RAGTruth
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("certirag.eval.datasets.ragtruth")


def load_ragtruth(
    data_dir: str | Path,
    task: Optional[str] = None,
    split: str = "test",
    max_examples: Optional[int] = None,
) -> list[dict]:
    """
    Load RAGTruth dataset into uniform format.

    Args:
        data_dir: Directory containing RAGTruth data files.
        task: Filter by task type: 'qa', 'data2text', 'summary', or None for all.
        split: Data split.
        max_examples: Cap on examples.

    Returns:
        List of dicts with keys:
            - example_id: str
            - question: str (or prompt)
            - documents: list[dict] with doc_id, text
            - answer: str (model-generated response)
            - gold_claims: list[dict] with text, label, span_start, span_end
            - task_type: str
            - model_name: str
    """
    data_dir = Path(data_dir)

    # RAGTruth uses JSONL format
    data_file = data_dir / f"{split}.jsonl"
    if not data_file.exists():
        # Try alternate naming
        data_file = data_dir / f"ragtruth_{split}.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(
            f"RAGTruth data not found at {data_file}. "
            f"Download from https://github.com/ParaGonTHero/RAGTruth"
        )

    logger.info(f"Loading RAGTruth from {data_file}")

    examples = []
    with open(data_file) as f:
        for line_num, line in enumerate(f):
            if max_examples and len(examples) >= max_examples:
                break

            item = json.loads(line.strip())
            task_type = item.get("task_type", "qa")

            # Filter by task if specified
            if task and task_type != task:
                continue

            # Extract documents (source passages)
            documents = []
            for j, doc in enumerate(item.get("source", item.get("documents", []))):
                if isinstance(doc, str):
                    documents.append({"doc_id": f"rt_{line_num}_doc{j}", "text": doc})
                else:
                    documents.append({
                        "doc_id": doc.get("id", f"rt_{line_num}_doc{j}"),
                        "text": doc.get("text", ""),
                    })

            # Extract hallucination annotations
            gold_claims = []
            for ann in item.get("annotations", item.get("labels", [])):
                label = _normalize_ragtruth_label(
                    ann.get("label", ann.get("type", "unknown"))
                )
                gold_claims.append({
                    "text": ann.get("text", ann.get("span", "")),
                    "label": label,
                    "span_start": ann.get("start", -1),
                    "span_end": ann.get("end", -1),
                    "error_type": ann.get("error_type", ""),
                })

            examples.append({
                "example_id": item.get("id", f"ragtruth_{line_num}"),
                "question": item.get("question", item.get("prompt", "")),
                "documents": documents,
                "answer": item.get("response", item.get("output", "")),
                "gold_claims": gold_claims,
                "task_type": task_type,
                "model_name": item.get("model", "unknown"),
            })

    logger.info(f"Loaded {len(examples)} examples from RAGTruth")
    return examples


def _normalize_ragtruth_label(label: str) -> str:
    """Map RAGTruth labels to CertiRAG's 3-way scheme."""
    label = label.lower().strip()
    if label in ("faithful", "supported", "accurate"):
        return "ENTAILED"
    elif label in ("hallucination", "fabrication", "contradiction", "inaccurate"):
        return "CONTRADICTED"
    else:
        return "NOT_ENOUGH_INFO"

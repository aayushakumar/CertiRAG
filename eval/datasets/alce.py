"""
ALCE Benchmark Loader
======================

Loads the ALCE (Attributed Language Model Evaluation) dataset
from Gao et al., 2023. ALCE provides:
    - Questions with reference documents
    - Model-generated answers with citations
    - Human annotations for citation support

We evaluate whether CertiRAG's verifier agrees with ALCE's
human judgments on whether claims are supported by cited passages.

Dataset: https://github.com/princeton-nlp/ALCE

Splits:
    - ASQA (Stelmakh et al., 2022): ambiguous factoid QA
    - ELI5 (Fan et al., 2019): long-form explanatory answers
    - QAMPARI (Amouyal et al., 2023): multi-answer list QA
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("certirag.eval.datasets.alce")


def load_alce(
    data_dir: str | Path,
    subset: str = "asqa",
    split: str = "test",
    max_examples: Optional[int] = None,
) -> list[dict]:
    """
    Load ALCE dataset into uniform format.

    Expected directory structure:
        data_dir/
            asqa_eval_gtr_top100.json
            eli5_eval_bm25_top100.json
            qampari_eval_gtr_top100.json

    Args:
        data_dir: Directory containing ALCE JSON files.
        subset: One of 'asqa', 'eli5', 'qampari'.
        split: Data split (ALCE only has eval/test).
        max_examples: Cap on number of examples to load.

    Returns:
        List of dicts with keys:
            - example_id: str
            - question: str
            - documents: list[dict] with doc_id, title, text
            - answer: str (model-generated)
            - gold_claims: list[dict] with text, label, cited_docs
    """
    data_dir = Path(data_dir)

    # Find the correct file
    patterns = {
        "asqa": "asqa*.json",
        "eli5": "eli5*.json",
        "qampari": "qampari*.json",
    }
    if subset not in patterns:
        raise ValueError(f"Unknown ALCE subset: {subset}. Choose from {list(patterns)}")

    files = list(data_dir.glob(patterns[subset]))
    if not files:
        raise FileNotFoundError(
            f"No ALCE {subset} files found in {data_dir}. "
            f"Download from https://github.com/princeton-nlp/ALCE"
        )

    data_path = files[0]
    logger.info(f"Loading ALCE {subset} from {data_path}")

    with open(data_path) as f:
        raw_data = json.load(f)

    examples = []
    for i, item in enumerate(raw_data):
        if max_examples and i >= max_examples:
            break

        # Extract documents (passages)
        documents = []
        for j, doc in enumerate(item.get("docs", item.get("ctxs", []))):
            documents.append({
                "doc_id": doc.get("id", f"{subset}_{i}_doc{j}"),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            })

        # Extract gold claims (if available)
        gold_claims = []
        for claim_data in item.get("claims", []):
            gold_claims.append({
                "text": claim_data.get("claim", claim_data.get("text", "")),
                "label": _normalize_alce_label(claim_data.get("label", "unknown")),
                "cited_docs": claim_data.get("cited_docs", []),
            })

        examples.append({
            "example_id": item.get("qa_id", f"{subset}_{i}"),
            "question": item.get("question", ""),
            "documents": documents,
            "answer": item.get("output", item.get("answer", "")),
            "gold_claims": gold_claims,
            "subset": subset,
        })

    logger.info(f"Loaded {len(examples)} examples from ALCE {subset}")
    return examples


def _normalize_alce_label(label: str) -> str:
    """Map ALCE labels to CertiRAG's 3-way scheme."""
    label = label.lower().strip()
    if label in ("supported", "entailed", "fully_supported", "1", "true"):
        return "ENTAILED"
    elif label in ("contradicted", "refuted", "not_supported", "0", "false"):
        return "CONTRADICTED"
    else:
        return "NOT_ENOUGH_INFO"

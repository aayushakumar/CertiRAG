"""
HuggingFace NLI Verifier — Local CPU Fact Verification
=======================================================

Uses a DeBERTa-v3-based NLI model to verify claims against evidence
using Natural Language Inference. Runs entirely on CPU with no API keys.

Two model options:
    - 2-class (entailment vs not): MoritzLaurer/deberta-v3-base-zeroshot-v2.0
      Very high accuracy for fact verification (recommended for CertiRAG)
    - 3-class (entailment/contradiction/neutral): cross-encoder/nli-deberta-v3-xsmall
      More granular but lower accuracy on NEI cases

For CertiRAG's fail-closed design, entailment detection accuracy is
the primary metric. The 2-class model excels at this.

Speed: ~50ms per pair on CPU | No API keys needed
"""

from __future__ import annotations

import logging
from typing import Optional

from certirag.schemas.verification import EvidenceScore, VerificationLabel
from certirag.verify.verifier import BaseVerifier

logger = logging.getLogger("certirag.verify.hf_nli_verifier")

# 2-class model: best for fact verification (entailment vs not)
DEFAULT_NLI_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"


class HFNLIVerifier(BaseVerifier):
    """
    Local NLI verifier using HuggingFace transformers (CPU-only).

    Uses DeBERTa-v3-based cross-encoder for NLI classification.
    Supports both 2-class (entailment/not) and 3-class
    (entailment/contradiction/neutral) models.

    No API keys needed. Runs entirely on CPU.

    Args:
        model_name: HuggingFace model ID for NLI cross-encoder.
        batch_size: Inference batch size.
        max_pairs: Hard cap on pairs per query.
        device: "cpu" or "cuda" (auto-detected if not specified).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        batch_size: int = 32,
        max_pairs: int = 240,
        device: Optional[str] = None,
    ):
        super().__init__(model_name=model_name, max_pairs=max_pairs)
        self._model_name = model_name
        self.batch_size = batch_size
        self._device = device
        self._model = None
        self._tokenizer = None
        self._label_map: dict[int, str] = {}
        self._is_2class = False

    def _load_model(self) -> None:
        """Lazy-load the NLI model and tokenizer."""
        if self._model is not None:
            return

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
            import torch
        except ImportError:
            raise RuntimeError(
                "torch and transformers required for HF NLI verifier. "
                "Install with: pip install torch transformers"
            )

        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading NLI model: {self._model_name} on {device}")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name
        )
        self._model.to(device)
        self._model.eval()
        self._device = device

        # Read label mapping from model config
        self._label_map = self._model.config.id2label
        num_labels = len(self._label_map)
        self._is_2class = num_labels == 2

        logger.info(
            f"NLI model loaded. Labels: {self._label_map}, "
            f"{'2-class' if self._is_2class else '3-class'}, device: {device}"
        )

    def _find_label_indices(self) -> dict[str, int]:
        """Find indices for entailment, contradiction, neutral labels."""
        indices = {}
        for idx, label in self._label_map.items():
            label_lower = str(label).lower()
            if label_lower == "entailment" or (
                "entail" in label_lower and "not" not in label_lower
            ):
                indices["entailment"] = int(idx)
            elif "contra" in label_lower:
                indices["contradiction"] = int(idx)
            elif "neutral" in label_lower or "not_entail" in label_lower or "not entail" in label_lower:
                indices["neutral"] = int(idx)
        return indices

    def verify_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[EvidenceScore]:
        """
        Verify (claim, evidence) pairs using local NLI model.

        NLI format: premise=evidence, hypothesis=claim.
        For 2-class models: maps to entailed or not_enough_info.
        For 3-class models: maps to full 3-way classification.

        Args:
            pairs: List of (claim_text, evidence_text) tuples.

        Returns:
            List of EvidenceScore with NLI-based verification results.
        """
        import torch

        self._load_model()
        all_scores: list[EvidenceScore] = []
        label_indices = self._find_label_indices()

        # Process in batches
        for batch_start in range(0, len(pairs), self.batch_size):
            batch = pairs[batch_start: batch_start + self.batch_size]

            # NLI: premise (evidence) → hypothesis (claim)
            # BaseVerifier passes pairs as (claim, evidence)
            premises = [evidence for _, evidence in batch]
            hypotheses = [claim for claim, _ in batch]

            inputs = self._tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

            for i in range(len(batch)):
                prob_vec = probs[i].cpu().numpy()
                entail_idx = label_indices.get("entailment")
                entail_prob = float(prob_vec[entail_idx]) if entail_idx is not None else 0.0

                if self._is_2class:
                    # 2-class: entailment vs not_entailment
                    if entail_prob >= 0.5:
                        label = VerificationLabel.ENTAILED
                        score = entail_prob
                    else:
                        label = VerificationLabel.NOT_ENOUGH_INFO
                        score = 1.0 - entail_prob
                else:
                    # 3-class: full NLI classification
                    pred_idx = int(prob_vec.argmax())
                    pred_label_str = str(self._label_map[pred_idx]).lower()

                    if "entail" in pred_label_str:
                        label = VerificationLabel.ENTAILED
                        score = entail_prob
                    elif "contra" in pred_label_str:
                        label = VerificationLabel.CONTRADICTED
                        contra_idx = label_indices.get("contradiction")
                        score = float(prob_vec[contra_idx]) if contra_idx is not None else float(prob_vec[pred_idx])
                    else:
                        label = VerificationLabel.NOT_ENOUGH_INFO
                        score = float(prob_vec[pred_idx])

                all_scores.append(EvidenceScore(
                    chunk_id="",
                    span_id="",
                    label=label,
                    score=score,
                    raw_score=entail_prob,
                ))

            logger.debug(
                f"Batch {batch_start // self.batch_size + 1}: "
                f"processed {len(batch)} pairs"
            )

        return all_scores

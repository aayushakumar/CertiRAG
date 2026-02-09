"""
NLI Verifier (Ablation Baseline)
==================================

Uses a standard NLI model (DeBERTa-v3-large-MNLI) for claim
verification. This serves as an ablation baseline to compare
against the specialized MiniCheck verifier.

NLI models are trained on generic textual entailment, not specifically
for document-grounded fact verification. This comparison shows
why specialized verifiers (MiniCheck) are preferred.

Reference Models:
    - microsoft/deberta-v3-large (fine-tuned on MNLI)
    - cross-encoder/nli-deberta-v3-base (smaller, faster)
"""

from __future__ import annotations

import logging

from certirag.schemas.verification import EvidenceScore, VerificationLabel
from certirag.verify.verifier import BaseVerifier

logger = logging.getLogger("certirag.verify.nli_verifier")


class NLIVerifier(BaseVerifier):
    """
    NLI-based verifier using DeBERTa-v3 or similar MNLI model.

    Maps NLI labels to verification labels:
        entailment → ENTAILED
        contradiction → CONTRADICTED
        neutral → NOT_ENOUGH_INFO

    This is an ablation baseline — for production, use MiniCheck.

    Args:
        model_name: HuggingFace NLI model ID.
        device: PyTorch device.
        batch_size: Inference batch size.
        max_pairs: Cost cap on total pairs.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: str = "auto",
        batch_size: int = 32,
        max_pairs: int = 240,
    ):
        super().__init__(model_name=model_name, max_pairs=max_pairs)
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy-load the NLI model."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading NLI model: {self.model_name} on {device}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
            ).to(device)
            self._model.eval()
            self.device = device

        except ImportError:
            raise RuntimeError(
                "transformers + torch required for NLI verifier. "
                "Install with: pip install 'certirag[full]'"
            )

    def verify_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[EvidenceScore]:
        """
        Score pairs using NLI model.

        NLI format: premise = evidence, hypothesis = claim
        """
        import torch

        self._load_model()
        all_scores: list[EvidenceScore] = []

        # NLI label mapping (model-dependent)
        # Most NLI models: {0: contradiction, 1: neutral, 2: entailment}
        label_map = {
            0: VerificationLabel.CONTRADICTED,
            1: VerificationLabel.NOT_ENOUGH_INFO,
            2: VerificationLabel.ENTAILED,
        }

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]

            # NLI convention: premise=evidence, hypothesis=claim
            inputs = self._tokenizer(
                [evidence for _, evidence in batch],
                [claim for claim, _ in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for j in range(len(batch)):
                pred_class = int(probs[j].argmax())
                label = label_map.get(pred_class, VerificationLabel.NOT_ENOUGH_INFO)
                score = float(probs[j][pred_class])

                all_scores.append(EvidenceScore(
                    chunk_id="",
                    span_id="",
                    label=label,
                    score=score,
                    raw_score=score,
                ))

        return all_scores

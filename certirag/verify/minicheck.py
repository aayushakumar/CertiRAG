"""
MiniCheck Verifier Adapter
============================

Primary verifier for CertiRAG using MiniCheck — a small, efficient
fact-checking model specifically designed for grounding verification.

MiniCheck (Tang et al., 2024) is trained for document-grounded
verification and achieves competitive quality with LLM judges at
a fraction of the cost (~400M parameters).

Reference: https://arxiv.org/abs/2404.10774

Two Modes:
    - FULL mode: Loads MiniCheck model locally on GPU
    - LITE mode: Uses a simple NLI-like heuristic as placeholder

Data Flow:
    (claim_text, evidence_text) → MiniCheck → {label, score}
"""

from __future__ import annotations

import logging
from typing import Optional

from certirag.schemas.verification import EvidenceScore, VerificationLabel
from certirag.verify.verifier import BaseVerifier

logger = logging.getLogger("certirag.verify.minicheck")


class MiniCheckVerifier(BaseVerifier):
    """
    MiniCheck-based grounding verifier.

    Loads the MiniCheck model (RoBERTa-large backbone) for efficient
    document-grounded fact verification. Each (claim, evidence) pair
    is scored for entailment, with scores calibrated to [0, 1].

    FULL mode usage:
        verifier = MiniCheckVerifier(
            model_name="lytang/MiniCheck-RoBERTa-large",
            device="cuda",
        )
        scores = verifier.verify_batch([("Paris is in France.", "France's capital is Paris.")])

    LITE mode usage:
        verifier = MiniCheckVerifier(mode="lite")
        scores = verifier.verify_batch([...])  # uses word-overlap heuristic

    Args:
        model_name: HuggingFace model ID for MiniCheck.
        mode: "full" (GPU model) or "lite" (heuristic fallback).
        device: PyTorch device ("cuda", "cpu", "auto").
        batch_size: Batch size for GPU inference.
        max_pairs: Maximum (claim, evidence) pairs per query.
    """

    def __init__(
        self,
        model_name: str = "lytang/MiniCheck-RoBERTa-large",
        mode: str = "full",
        device: str = "auto",
        batch_size: int = 32,
        max_pairs: int = 240,
    ):
        super().__init__(model_name=model_name, max_pairs=max_pairs)
        self.mode = mode
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy-load MiniCheck model and tokenizer."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading MiniCheck model: {self.model_name} on {device}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
            )
            self._model = self._model.to(device)
            self._model.eval()
            self.device = device
            logger.info("MiniCheck model loaded successfully")

        except ImportError:
            raise RuntimeError(
                "transformers + torch required for MiniCheck FULL mode. "
                "Install with: pip install 'certirag[full]'"
            )
        except Exception as e:
            logger.error(f"Failed to load MiniCheck model: {e}")
            raise

    def verify_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[EvidenceScore]:
        """
        Score a batch of (claim, evidence) pairs using MiniCheck.

        Args:
            pairs: List of (claim_text, evidence_text) tuples.

        Returns:
            List of EvidenceScore objects with labels and scores.
        """
        if not pairs:
            return []

        if self.mode == "lite":
            return self._verify_lite(pairs)
        else:
            return self._verify_full(pairs)

    def _verify_full(
        self, pairs: list[tuple[str, str]]
    ) -> list[EvidenceScore]:
        """GPU-based verification using MiniCheck model."""
        import torch

        self._load_model()
        all_scores: list[EvidenceScore] = []

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]

            # MiniCheck expects: (evidence, claim) or (document, claim)
            # Format depends on model training; check model card
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
                logits = outputs.logits

                # MiniCheck typically outputs 2 classes: [not_supported, supported]
                # or 3 classes: [contradiction, neutral, entailment]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for j, (claim, evidence) in enumerate(batch):
                prob = probs[j]

                if len(prob) == 2:
                    # Binary: [not_supported, supported]
                    entail_score = float(prob[1])
                    if entail_score >= 0.5:
                        label = VerificationLabel.ENTAILED
                        score = entail_score
                    else:
                        label = VerificationLabel.NOT_ENOUGH_INFO
                        score = entail_score

                elif len(prob) >= 3:
                    # 3-class: [contradiction, neutral, entailment]
                    contradict_score = float(prob[0])
                    neutral_score = float(prob[1])
                    entail_score = float(prob[2])

                    if entail_score >= contradict_score and entail_score >= neutral_score:
                        label = VerificationLabel.ENTAILED
                        score = entail_score
                    elif contradict_score >= entail_score and contradict_score >= neutral_score:
                        label = VerificationLabel.CONTRADICTED
                        score = contradict_score
                    else:
                        label = VerificationLabel.NOT_ENOUGH_INFO
                        score = neutral_score
                else:
                    # Single logit — treat as entailment probability
                    entail_score = float(torch.sigmoid(torch.tensor(prob[0])))
                    label = VerificationLabel.ENTAILED if entail_score >= 0.5 else VerificationLabel.NOT_ENOUGH_INFO
                    score = entail_score

                all_scores.append(EvidenceScore(
                    chunk_id="",  # filled by verify_claims
                    span_id="",
                    label=label,
                    score=score,
                    raw_score=score,
                ))

        return all_scores

    def _verify_lite(
        self, pairs: list[tuple[str, str]]
    ) -> list[EvidenceScore]:
        """
        Lightweight heuristic verifier for LITE/CPU mode.

        Uses word overlap + simple negation detection as a placeholder
        when the full MiniCheck model is not available.

        This is NOT suitable for paper results — only for development
        and testing in Codespaces.
        """
        all_scores: list[EvidenceScore] = []

        for claim, evidence in pairs:
            claim_words = set(claim.lower().split())
            evidence_words = set(evidence.lower().split())

            # Simple word overlap score
            if not claim_words:
                overlap = 0.0
            else:
                overlap = len(claim_words & evidence_words) / len(claim_words)

            # Negation detection (very basic)
            negation_words = {"not", "no", "never", "neither", "nor", "isn't", "wasn't", "doesn't", "don't"}
            claim_has_negation = bool(claim_words & negation_words)
            evidence_has_negation = bool(evidence_words & negation_words)
            negation_mismatch = claim_has_negation != evidence_has_negation

            if negation_mismatch and overlap > 0.3:
                label = VerificationLabel.CONTRADICTED
                score = min(overlap * 0.8, 0.95)
            elif overlap >= 0.5:
                label = VerificationLabel.ENTAILED
                score = min(overlap, 0.95)
            elif overlap >= 0.2:
                label = VerificationLabel.NOT_ENOUGH_INFO
                score = overlap
            else:
                label = VerificationLabel.NOT_ENOUGH_INFO
                score = overlap

            all_scores.append(EvidenceScore(
                chunk_id="",
                span_id="",
                label=label,
                score=score,
                raw_score=score,
            ))

        return all_scores

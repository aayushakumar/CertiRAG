"""
Cascade Verifier — Two-Stage Verification Pipeline
=====================================================

A novel verification architecture that chains a fast binary classifier
with a precise 3-class model in a cascade:

    **Stage 1 — Entailment Gate (2-class DeBERTa)**
        High-confidence entailment detection using the zero-shot model.
        If P(entail) > ``gate_threshold`` (default 0.92), immediately
        return ENTAILED without running Stage 2.

    **Stage 2 — Contradiction Resolver (3-class DeBERTa)**
        For all remaining pairs, run the 3-class model to distinguish
        CONTRADICTED from NOT_ENOUGH_INFO — something the 2-class
        model fundamentally cannot do.

Rationale
---------
Our evaluation (EVALUATION.md) revealed that:

1. The 2-class model has excellent entailment detection (96.7% binary
   accuracy on FEVER-NLI, AUROC 0.991) but **cannot detect
   contradictions at all** (P/R/F1 = 0.000).

2. The 3-class model detects contradictions well (F1 = 0.748 on ANLI R1)
   but is more conservative on entailment.

The cascade combines both strengths:
- Stage 1 catches clear entailments quickly and confidently.
- Stage 2 handles the harder cases with full 3-class discrimination.
- Overall: higher 3-class accuracy than either model alone, with
  contradiction detection intact.

This is strictly superior to simple probability averaging (as in
Ensemble(sota)) because it **routes** rather than **blends** — the
2-class model's inability to produce contradiction probabilities
cannot dilute the 3-class signal.

Usage
-----
    from certirag.verify.cascade_verifier import CascadeVerifier

    verifier = CascadeVerifier()  # defaults
    verifier = CascadeVerifier(gate_threshold=0.95)  # stricter gate
    scores = verifier.verify_batch([("claim", "evidence")])
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from certirag.schemas.verification import EvidenceScore, VerificationLabel
from certirag.verify.verifier import BaseVerifier

logger = logging.getLogger("certirag.verify.cascade")

# ── Model catalogue ────────────────────────────────────────────────

_GATE_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"       # 2-class
_RESOLVER_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"  # 3-class


class CascadeVerifier(BaseVerifier):
    """
    Two-stage cascade: fast entailment gate → full 3-class resolver.

    Args:
        gate_threshold: Minimum entailment probability from the 2-class
            model to accept as ENTAILED without running Stage 2.
            Higher = more conservative gate → more pairs go to Stage 2.
        gate_model: HuggingFace model ID for the binary gate.
        resolver_model: HuggingFace model ID for the 3-class resolver.
        batch_size: Inference batch size per model.
        device: "cpu" or "cuda".
    """

    def __init__(
        self,
        gate_threshold: float = 0.92,
        gate_model: str = _GATE_MODEL,
        resolver_model: str = _RESOLVER_MODEL,
        batch_size: int = 32,
        max_pairs: int = 240,
        device: Optional[str] = None,
    ):
        super().__init__(model_name="Cascade(gate+resolver)", max_pairs=max_pairs)
        self.gate_threshold = gate_threshold
        self._gate_model_name = gate_model
        self._resolver_model_name = resolver_model
        self.batch_size = batch_size
        self._device = device

        # Loaded at first use
        self._gate = None
        self._gate_tok = None
        self._gate_labels = {}

        self._resolver = None
        self._resolver_tok = None
        self._resolver_labels = {}

        self._loaded = False

    # ── Model loading ──────────────────────────────────────────────

    def _load_models(self) -> None:
        """Lazy-load both models on first call."""
        if self._loaded:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading gate model: %s (device=%s)", self._gate_model_name, dev)

        self._gate_tok = AutoTokenizer.from_pretrained(self._gate_model_name)
        self._gate = AutoModelForSequenceClassification.from_pretrained(
            self._gate_model_name
        ).to(dev).eval()
        self._gate_labels = self._find_label_indices(self._gate)

        logger.info("Loading resolver model: %s", self._resolver_model_name)
        self._resolver_tok = AutoTokenizer.from_pretrained(self._resolver_model_name)
        self._resolver = AutoModelForSequenceClassification.from_pretrained(
            self._resolver_model_name
        ).to(dev).eval()
        self._resolver_labels = self._find_label_indices(self._resolver)

        self._loaded = True
        logger.info(
            "Cascade loaded: gate_labels=%s  resolver_labels=%s",
            self._gate_labels, self._resolver_labels,
        )

    @staticmethod
    def _find_label_indices(model) -> dict[str, int]:
        """Map model label names → {entail, contradict, neutral} indices."""
        id2label = model.config.id2label
        mapping = {}

        entail_keys = {"entailment", "entail", "yes", "true"}
        contra_keys = {"contradiction", "contradict", "no", "false"}
        neutral_keys = {"neutral", "neither", "not_entailment", "maybe"}

        for idx, name in id2label.items():
            low = name.lower().strip()
            if low in entail_keys:
                mapping["entail"] = int(idx)
            elif low in contra_keys:
                mapping["contradict"] = int(idx)
            elif low in neutral_keys:
                mapping["neutral"] = int(idx)

        return mapping

    # ── Inference helpers ──────────────────────────────────────────

    def _infer(
        self, model, tokenizer, premises: list[str], hypotheses: list[str]
    ) -> np.ndarray:
        """
        Run NLI inference, return (N, num_classes) softmax probabilities.
        """
        import torch

        all_probs = []
        for start in range(0, len(premises), self.batch_size):
            end = start + self.batch_size
            batch_p = premises[start:end]
            batch_h = hypotheses[start:end]

            inputs = tokenizer(
                batch_p, batch_h,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

            with torch.no_grad():
                logits = model(**inputs).logits

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    # ── Main verify_batch ──────────────────────────────────────────

    def verify_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[EvidenceScore]:
        """
        Two-stage cascade verification.

        Stage 1: Run 2-class gate model on ALL pairs.
                 High-confidence entailments (P(entail) > gate_threshold)
                 are immediately accepted.

        Stage 2: Run 3-class resolver model on REMAINING pairs.
                 Returns full NLI label with contradiction detection.
        """
        if not pairs:
            return []

        self._load_models()

        claims = [p[0] for p in pairs]
        evidences = [p[1] for p in pairs]

        # ── Stage 1: Binary entailment gate ────────────────────────
        gate_probs = self._infer(
            self._gate, self._gate_tok, evidences, claims
        )

        entail_idx = self._gate_labels.get("entail", 0)
        gate_entail_scores = gate_probs[:, entail_idx]

        # Identify high-confidence entailments
        gate_pass = gate_entail_scores >= self.gate_threshold
        gate_pass_indices = set(np.where(gate_pass)[0])

        logger.info(
            "Cascade Stage 1: %d/%d pairs passed gate (threshold=%.2f)",
            len(gate_pass_indices), len(pairs), self.gate_threshold,
        )

        # ── Stage 2: 3-class resolver on remaining ─────────────────
        remaining_indices = [i for i in range(len(pairs)) if i not in gate_pass_indices]

        resolver_probs = {}
        if remaining_indices:
            rem_claims = [claims[i] for i in remaining_indices]
            rem_evidence = [evidences[i] for i in remaining_indices]

            probs_3class = self._infer(
                self._resolver, self._resolver_tok, rem_evidence, rem_claims
            )

            for j, orig_idx in enumerate(remaining_indices):
                resolver_probs[orig_idx] = probs_3class[j]

            logger.info(
                "Cascade Stage 2: resolved %d remaining pairs with 3-class model",
                len(remaining_indices),
            )

        # ── Combine results ────────────────────────────────────────
        results: list[EvidenceScore] = []

        entail_r = self._resolver_labels.get("entail", 0)
        contra_r = self._resolver_labels.get("contradict", 2)
        neutral_r = self._resolver_labels.get("neutral", 1)

        for i in range(len(pairs)):
            if i in gate_pass_indices:
                # Stage 1 accepted: high-confidence entailment
                score = float(gate_entail_scores[i])
                results.append(EvidenceScore(
                    chunk_id=f"pair_{i}",
                    span_id=f"span_{i}",
                    label=VerificationLabel.ENTAILED,
                    score=score,
                    raw_score=score,
                ))
            else:
                # Stage 2 resolved: full 3-class
                probs = resolver_probs[i]
                p_entail = float(probs[entail_r])
                p_contra = float(probs[contra_r])
                p_neutral = float(probs[neutral_r])

                if p_contra >= p_entail and p_contra >= p_neutral:
                    label = VerificationLabel.CONTRADICTED
                    score = p_contra
                elif p_entail >= p_contra and p_entail >= p_neutral:
                    label = VerificationLabel.ENTAILED
                    score = p_entail
                else:
                    label = VerificationLabel.NOT_ENOUGH_INFO
                    score = p_neutral

                # raw_score = entailment probability (for AUROC)
                results.append(EvidenceScore(
                    chunk_id=f"pair_{i}",
                    span_id=f"span_{i}",
                    label=label,
                    score=score,
                    raw_score=p_entail,
                ))

        return results

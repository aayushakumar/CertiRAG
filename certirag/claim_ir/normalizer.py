"""
Claim Normalizer
=================

Post-processes claims from the Claim Compiler to ensure:
1. Atomicity — each claim contains exactly one fact
2. Normalization — consistent formatting of numbers, dates, entities
3. Deduplication — merge semantically identical claims
4. Hedge detection — flag or strip hedging language

This is the "cleanup" step between the Claim Compiler and the Verifier.
The normalizer applies deterministic rules (no LLM calls) to ensure
claim quality before verification.

Data Flow:
    ClaimIR (raw) → Normalizer → ClaimIR (clean) → Verifier
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from certirag.schemas.claim_ir import Claim, ClaimIR, ClaimType

logger = logging.getLogger("certirag.claim_ir.normalizer")


# ── Hedge Detection ────────────────────────────────────────────────

# Words/phrases that indicate hedging (uncertainty)
HEDGE_PATTERNS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bpossibly\b",
    r"\bperhaps\b", r"\bprobably\b", r"\blikely\b", r"\bunlikely\b",
    r"\bapparently\b", r"\ballegedly\b", r"\breportedly\b",
    r"\bsome suggest\b", r"\bit is believed\b", r"\bit seems\b",
    r"\bit appears\b", r"\bin some cases\b", r"\bgenerally\b",
    r"\btypically\b", r"\busually\b", r"\boften\b",
]

HEDGE_REGEX = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


# ── Conjunction Patterns (for splitting) ───────────────────────────

# Patterns that indicate a claim should be split into multiple claims.
# We're careful NOT to split entity lists ("cats and dogs").
CONJUNCTION_SPLIT_PATTERNS = [
    # Independent clauses joined by conjunctions
    r",\s+and\s+(?=[A-Z])",    # ", and The..."
    r",\s+but\s+(?=[A-Z])",    # ", but The..."
    r";\s+",                    # semicolons always separate independent facts
    r"\.\s+Also,?\s+",         # ". Also, ..."
    r"\.\s+Additionally,?\s+", # ". Additionally, ..."
    r"\.\s+Furthermore,?\s+",  # ". Furthermore, ..."
    r"\.\s+Moreover,?\s+",     # ". Moreover, ..."
    r",\s+which also\s+",      # ", which also ..."
    r"\s+because\s+",          # "X because Y" → split into X and Y
]


class ClaimNormalizer:
    """
    Normalizes and cleans claims for verification.

    Operations (applied in order):
        1. Strip empty/whitespace-only claims
        2. Detect and flag hedging language
        3. Optionally remove hedges from claim text
        4. Split compound claims into atomic sub-claims
        5. Normalize numeric formats
        6. Deduplicate semantically identical claims
        7. Enforce max claim token length

    Usage:
        normalizer = ClaimNormalizer(split_conjunctions=True)
        clean_ir = normalizer.normalize(raw_claim_ir)

    Args:
        split_conjunctions: Auto-split compound claims.
        remove_hedges: Strip hedging language from claims.
        max_claim_tokens: Max tokens per claim (longer claims are flagged).
        dedup_threshold: Cosine similarity threshold for deduplication.
    """

    def __init__(
        self,
        split_conjunctions: bool = True,
        remove_hedges: bool = False,
        max_claim_tokens: int = 60,
        dedup_threshold: float = 0.92,
    ):
        self.split_conjunctions = split_conjunctions
        self.remove_hedges = remove_hedges
        self.max_claim_tokens = max_claim_tokens
        self.dedup_threshold = dedup_threshold

    def normalize(self, claim_ir: ClaimIR) -> ClaimIR:
        """
        Apply all normalization steps to a ClaimIR.

        Args:
            claim_ir: Raw ClaimIR from the compiler.

        Returns:
            Normalized ClaimIR with clean, atomic claims.
        """
        claims = claim_ir.claims.copy()

        # Step 1: Remove empty claims
        claims = [c for c in claims if c.text.strip()]

        # Step 2: Detect hedges
        claims = [self._detect_hedge(c) for c in claims]

        # Step 3: Optionally remove hedges
        if self.remove_hedges:
            claims = [self._remove_hedge(c) for c in claims]

        # Step 4: Split compound claims
        if self.split_conjunctions:
            split_claims = []
            for claim in claims:
                split_claims.extend(self._split_claim(claim))
            claims = split_claims

        # Step 5: Normalize text
        claims = [self._normalize_text(c) for c in claims]

        # Step 6: Deduplicate
        claims = self._deduplicate(claims)

        # Step 7: Re-number claim IDs
        for i, claim in enumerate(claims):
            claim.id = f"c{i + 1}"

        # Step 8: Flag long claims
        for claim in claims:
            word_count = len(claim.text.split())
            if word_count > self.max_claim_tokens:
                logger.warning(
                    f"Claim {claim.id} has {word_count} words "
                    f"(max: {self.max_claim_tokens}): '{claim.text[:60]}...'"
                )

        logger.info(
            f"Normalized {claim_ir.num_claims} claims → {len(claims)} claims"
        )

        return ClaimIR(
            question=claim_ir.question,
            claims=claims,
            model_id=claim_ir.model_id,
            raw_answer=claim_ir.raw_answer,
        )

    def _detect_hedge(self, claim: Claim) -> Claim:
        """Detect and flag hedging language in a claim."""
        if HEDGE_REGEX.search(claim.text):
            claim.hedge = True
        return claim

    def _remove_hedge(self, claim: Claim) -> Claim:
        """Remove hedging words from claim text."""
        if not claim.hedge:
            return claim

        text = claim.text
        for pattern in HEDGE_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Clean up artifacts
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^[,.\s]+", "", text)
        claim.text = text
        return claim

    def _split_claim(self, claim: Claim) -> list[Claim]:
        """
        Split a compound claim into atomic sub-claims.

        Only splits on patterns that indicate independent facts.
        Does NOT split entity lists (e.g., "cats, dogs, and birds").

        Returns:
            List of claims (1 if no split needed, >1 if split).
        """
        text = claim.text

        # Try each split pattern
        for pattern in CONJUNCTION_SPLIT_PATTERNS:
            parts = re.split(pattern, text)
            if len(parts) > 1:
                # Verify each part is a meaningful claim (>3 words)
                valid_parts = [p.strip() for p in parts if len(p.strip().split()) > 3]
                if len(valid_parts) > 1:
                    sub_claims = []
                    for i, part in enumerate(valid_parts):
                        # Ensure proper sentence ending
                        if not part.endswith((".", "!", "?")):
                            part = part + "."

                        sub_claim = Claim(
                            id=f"{claim.id}_{i + 1}",
                            text=part.strip(),
                            type=claim.type,
                            fields=claim.fields.copy(),
                            evidence_candidates=claim.evidence_candidates.copy(),
                            hedge=claim.hedge,
                            unsupported_by_design=claim.unsupported_by_design,
                        )
                        sub_claims.append(sub_claim)

                    logger.debug(
                        f"Split claim '{text[:50]}...' into "
                        f"{len(sub_claims)} sub-claims"
                    )
                    return sub_claims

        return [claim]

    def _normalize_text(self, claim: Claim) -> Claim:
        """Normalize whitespace and fix common text issues."""
        text = claim.text

        # Collapse whitespace
        text = " ".join(text.split())

        # Ensure ends with period
        if text and not text.endswith((".", "!", "?")):
            text = text + "."

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        claim.text = text
        return claim

    def _deduplicate(self, claims: list[Claim]) -> list[Claim]:
        """
        Remove duplicate claims using text similarity.

        Uses a simple token-overlap heuristic for deduplication
        (no embedding model needed). For more precise dedup,
        use embedding-based similarity in FULL mode.
        """
        if len(claims) <= 1:
            return claims

        unique_claims: list[Claim] = []
        seen_texts: list[set[str]] = []

        for claim in claims:
            tokens = set(claim.text.lower().split())

            is_duplicate = False
            for existing_tokens in seen_texts:
                # Jaccard similarity
                if not tokens or not existing_tokens:
                    continue
                intersection = tokens & existing_tokens
                union = tokens | existing_tokens
                similarity = len(intersection) / len(union) if union else 0

                if similarity >= self.dedup_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_claims.append(claim)
                seen_texts.append(tokens)

        if len(unique_claims) < len(claims):
            logger.info(
                f"Deduplicated: {len(claims)} → {len(unique_claims)} claims"
            )

        return unique_claims

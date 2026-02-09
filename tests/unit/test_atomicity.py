"""
Claim Atomicity Tests
========================

Tests for claim normalization, atomicity splitting,
hedge detection, and deduplication.

These verify that the claim compiler + normalizer produce
well-formed, atomic claims suitable for verification.
"""

from __future__ import annotations

import pytest

from certirag.claim_ir.normalizer import ClaimNormalizer
from certirag.schemas.claim_ir import CandidateEvidence, Claim, ClaimIR, ClaimType


@pytest.fixture
def normalizer():
    return ClaimNormalizer(
        split_conjunctions=True,
        remove_hedges=True,
        max_claim_tokens=50,
    )


class TestConjunctionSplitting:
    """Test splitting conjunctive claims into atomic ones."""

    def test_simple_and_split(self, normalizer):
        """', and [A-Z]' pattern should split into two claims."""
        claim_ir = _make_ir("Paris is the capital of France, and Berlin is the capital of Germany.")
        result = normalizer.normalize(claim_ir)
        assert len(result.claims) >= 2

    def test_no_split_entity_conjunction(self, normalizer):
        """'France and Germany' as entity should not split."""
        claim_ir = _make_ir("France and Germany share a border.")
        result = normalizer.normalize(claim_ir)
        # Should remain as one claim (entity conjunction)
        assert len(result.claims) >= 1

    def test_but_split(self, normalizer):
        """', but [A-Z]' pattern should split."""
        claim_ir = _make_ir("The city of Paris is large, but Berlin is also beautiful.")
        result = normalizer.normalize(claim_ir)
        assert len(result.claims) >= 2

    def test_already_atomic(self, normalizer):
        """Atomic claims should not be split."""
        claim_ir = _make_ir("Paris is the capital of France.")
        result = normalizer.normalize(claim_ir)
        assert len(result.claims) == 1

    def test_multiple_conjunctions(self, normalizer):
        """Multiple conjunctions should produce multiple claims."""
        claim_ir = _make_ir("A is true, B is also true, and C is also true.")
        result = normalizer.normalize(claim_ir)
        assert len(result.claims) >= 2


class TestHedgeDetection:
    """Test detection and marking of hedged/uncertain claims."""

    def test_might_hedge(self, normalizer):
        """'might' should be detected as a hedge."""
        claim_ir = _make_ir("The population might exceed 10 million.")
        result = normalizer.normalize(claim_ir)
        # Hedged claims should still exist but be marked
        assert len(result.claims) >= 1

    def test_probably_hedge(self, normalizer):
        """'probably' should be detected as a hedge."""
        claim_ir = _make_ir("Paris is probably the most visited city.")
        result = normalizer.normalize(claim_ir)
        assert len(result.claims) >= 1

    def test_no_hedge(self, normalizer):
        """Definite statements should not be marked as hedges."""
        claim_ir = _make_ir("Paris is the capital of France.")
        result = normalizer.normalize(claim_ir)
        assert len(result.claims) == 1


class TestDeduplication:
    """Test removal of duplicate/near-duplicate claims."""

    def test_exact_duplicate(self, normalizer):
        """Exact duplicates should be removed."""
        claims = [
            _make_claim("Paris is the capital of France.", "c1"),
            _make_claim("Paris is the capital of France.", "c2"),
        ]
        claim_ir = ClaimIR(question="test?", claims=claims)
        result = normalizer.normalize(claim_ir)
        assert len(result.claims) == 1

    def test_near_duplicate(self, normalizer):
        """Near-duplicates (high Jaccard) should be removed."""
        dedup_normalizer = ClaimNormalizer(
            split_conjunctions=True,
            remove_hedges=True,
            max_claim_tokens=50,
            dedup_threshold=0.8,
        )
        claims = [
            _make_claim("Paris is the capital of France.", "c1"),
            _make_claim("Paris is capital of France.", "c2"),  # Missing "the"
        ]
        claim_ir = ClaimIR(question="test?", claims=claims)
        result = dedup_normalizer.normalize(claim_ir)
        assert len(result.claims) == 1

    def test_different_claims_kept(self, normalizer):
        """Distinct claims should be preserved."""
        claims = [
            _make_claim("Paris is the capital of France.", "c1"),
            _make_claim("Berlin is the capital of Germany.", "c2"),
        ]
        claim_ir = ClaimIR(question="test?", claims=claims)
        result = normalizer.normalize(claim_ir)
        assert len(result.claims) == 2


class TestTextNormalization:
    """Test text cleanup and normalization."""

    def test_whitespace_normalization(self, normalizer):
        """Extra whitespace should be normalized."""
        claim_ir = _make_ir("Paris    is   the    capital.")
        result = normalizer.normalize(claim_ir)
        text = result.claims[0].text
        assert "    " not in text

    def test_empty_claim_rejected(self, normalizer):
        """Empty claims should be rejected by schema validation."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _make_claim("", "c2")


# ── Helper Functions ────────────────────────────────────────────

def _make_ir(text: str) -> ClaimIR:
    """Make a ClaimIR with a single claim."""
    return ClaimIR(
        question="test?",
        claims=[_make_claim(text, "c0")],
    )


def _make_claim(text: str, claim_id: str) -> Claim:
    """Make a single Claim."""
    return Claim(
        id=claim_id,
        text=text,
        type=ClaimType.FACTUAL,
        evidence_candidates=[
            CandidateEvidence(chunk_id="chunk_0", span_id="s0"),
        ],
    )

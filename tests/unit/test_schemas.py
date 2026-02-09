"""
Schema Tests (T-U1 through T-U4)
===================================

Tests all four Pydantic data contracts for:
    - Valid construction
    - Required field enforcement
    - Type validation
    - Cross-field validators
    - Serialization round-trip
    - Invariant preservation

These are the most critical tests — they verify that the
type-level guarantees hold throughout the pipeline.
"""

from __future__ import annotations

import json
import uuid

import pytest
from pydantic import ValidationError

from certirag.schemas.certificate import (
    AuditCertificate,
    PolicySnapshot,
    RendererDecision,
    RenderState,
)
from certirag.schemas.claim_ir import (
    CandidateEvidence,
    Claim,
    ClaimIR,
    ClaimType,
)
from certirag.schemas.evidence import (
    EvidenceChunk,
    EvidenceSpan,
    RetrievalScores,
    SourceInfo,
)
from certirag.schemas.verification import (
    EvidenceScore,
    VerificationLabel,
    VerificationResult,
)

# Import factories from conftest
from tests.conftest import make_chunk, make_claim, make_verification


# ────────────────────────────────────────────────────────────────
# T-U1: EvidenceChunk Schema Tests
# ────────────────────────────────────────────────────────────────

class TestEvidenceChunk:
    """Tests for the EvidenceChunk schema (§6.1)."""

    def test_valid_construction(self):
        """A properly formed chunk should validate without errors."""
        chunk = make_chunk(text="Hello world.", doc_id="d1")
        assert chunk.chunk_id
        assert chunk.doc_id == "d1"
        assert chunk.text == "Hello world."

    def test_span_exceeds_text_rejected(self):
        """Spans exceeding text boundary should fail validation."""
        with pytest.raises(ValidationError):
            EvidenceChunk(
                chunk_id="c", doc_id="d", text="short",
                source=SourceInfo(title="Test"),
                spans=[EvidenceSpan(span_id="s0", start=0, end=100, sentence="x" * 100)],
            )

    def test_negative_offset_rejected(self):
        """Negative span start should fail validation."""
        with pytest.raises(ValidationError):
            EvidenceSpan(span_id="s0", start=-1, end=5, sentence="hello")

    def test_span_integrity(self):
        """Spans must have start < end and sentence length matching."""
        chunk = make_chunk(text="Test sentence here.")
        for span in chunk.spans:
            assert span.start < span.end
            assert span.end - span.start == len(span.sentence)

    def test_serialization_roundtrip(self):
        """Chunk should survive JSON round-trip."""
        original = make_chunk(text="Round trip test.", doc_id="d99")
        data = original.model_dump()
        restored = EvidenceChunk(**data)
        assert restored.text == original.text
        assert restored.chunk_id == original.chunk_id

    def test_json_serialization(self):
        """Chunk should serialize to valid JSON."""
        chunk = make_chunk(text="JSON test.")
        json_str = chunk.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["text"] == "JSON test."


# ────────────────────────────────────────────────────────────────
# T-U2: ClaimIR Schema Tests
# ────────────────────────────────────────────────────────────────

class TestClaimIR:
    """Tests for the ClaimIR schema (§6.2)."""

    def test_valid_construction(self, sample_claim_ir):
        """ClaimIR with valid claims should pass."""
        assert len(sample_claim_ir.claims) == 2
        assert sample_claim_ir.question == "What is the capital of France?"

    def test_claim_types(self):
        """All claim types should be constructable."""
        for ct in ClaimType:
            claim = make_claim(text=f"A {ct.value} claim.", claim_type=ct)
            assert claim.type == ct

    def test_empty_claims_allowed(self):
        """ClaimIR with no claims is valid (question may not decompose)."""
        ir = ClaimIR(question="Test?", claims=[])
        assert len(ir.claims) == 0

    def test_get_claim_lookup(self, sample_claim_ir):
        """get_claim() should find claims by ID."""
        claim = sample_claim_ir.get_claim("c1")
        assert claim is not None
        assert claim.text == "Paris is the capital of France."

    def test_get_claim_missing(self, sample_claim_ir):
        """get_claim() returns None for unknown IDs."""
        assert sample_claim_ir.get_claim("nonexistent") is None

    def test_evidence_pairs(self, sample_claim_ir):
        """get_all_evidence_pairs() yields (claim, evidence) tuples."""
        pairs = sample_claim_ir.get_all_evidence_pairs()
        assert len(pairs) >= 2

    def test_serialization_roundtrip(self, sample_claim_ir):
        """ClaimIR should survive serialization."""
        data = sample_claim_ir.model_dump()
        restored = ClaimIR(**data)
        assert len(restored.claims) == len(sample_claim_ir.claims)

    def test_candidate_evidence_requires_span_id(self):
        """CandidateEvidence requires a span_id."""
        with pytest.raises(ValidationError):
            CandidateEvidence(chunk_id="c1")


# ────────────────────────────────────────────────────────────────
# T-U3: VerificationResult Schema Tests
# ────────────────────────────────────────────────────────────────

class TestVerificationResult:
    """Tests for the VerificationResult schema (§6.3)."""

    def test_valid_construction(self):
        """Valid verification result should pass."""
        vr = make_verification(claim_id="c1", entail=0.9, contradict=0.05)
        assert vr.label == VerificationLabel.ENTAILED
        assert vr.max_entail_score == pytest.approx(0.9)

    def test_three_labels(self):
        """All three labels should be constructable."""
        for label in VerificationLabel:
            vr = make_verification(claim_id="c", label=label)
            assert vr.label == label

    def test_contradiction_detection(self):
        """has_contradiction property should detect contradictions."""
        vr = make_verification(
            label=VerificationLabel.CONTRADICTED,
            entail=0.1, contradict=0.8,
        )
        assert vr.has_contradiction is True

    def test_evidence_score_bounds(self):
        """Scores should be in [0, 1]."""
        with pytest.raises(ValidationError):
            EvidenceScore(
                chunk_id="c1",
                span_id="s0",
                label=VerificationLabel.ENTAILED,
                score=1.5,  # Out of bounds
            )

    def test_multiple_evidence(self):
        """Multiple evidence scores should aggregate correctly."""
        ev1 = EvidenceScore(
            chunk_id="e1", span_id="s0",
            label=VerificationLabel.ENTAILED, score=0.9,
        )
        ev2 = EvidenceScore(
            chunk_id="e2", span_id="s0",
            label=VerificationLabel.ENTAILED, score=0.6,
        )
        vr = VerificationResult(
            claim_id="c1",
            label=VerificationLabel.ENTAILED,
            score=0.9,
            all_scores=[ev1, ev2],
            best_evidence=[ev1, ev2],
        )
        assert vr.max_entail_score == pytest.approx(0.9)
        assert vr.evidence_span_count == 2


# ────────────────────────────────────────────────────────────────
# T-U4: Certificate Schema Tests
# ────────────────────────────────────────────────────────────────

class TestCertificate:
    """Tests for the AuditCertificate schema (§6.4)."""

    def test_render_states(self):
        """All render states should be constructable."""
        for state in RenderState:
            decision = RendererDecision(
                claim_id="c1",
                render_state=state,
                policy=PolicySnapshot(
                    tau_entail=0.5, tau_contradict=0.7,
                    min_evidence_spans=1,
                ),
                reason=f"Test {state.value}",
            )
            assert decision.render_state == state

    def test_certificate_seal(self):
        """Sealed certificate should verify its own integrity."""
        cert = AuditCertificate(
            query_id="test_run",
            question="Test?",
            config_hash="abc123",
            stats={"total_claims": 5, "verified_claims": 3, "blocked_claims": 2},
            policy=PolicySnapshot(
                tau_entail=0.5, tau_contradict=0.7,
                min_evidence_spans=1,
            ),
        )
        cert.seal()
        assert cert.integrity_hash is not None
        assert cert.verify_integrity() is True

    def test_tampered_certificate_fails(self):
        """Modifying a sealed certificate should break integrity."""
        cert = AuditCertificate(
            query_id="test_run",
            question="Test?",
            config_hash="abc123",
            stats={"total_claims": 5, "verified_claims": 3, "blocked_claims": 2},
            policy=PolicySnapshot(
                tau_entail=0.5, tau_contradict=0.7,
                min_evidence_spans=1,
            ),
        )
        cert.seal()
        # Tamper
        cert.stats["verified_claims"] = 10
        assert cert.verify_integrity() is False

    def test_policy_snapshot(self):
        """PolicySnapshot should capture threshold values."""
        policy = PolicySnapshot(
            tau_entail=0.5,
            tau_contradict=0.7,
            min_evidence_spans=1,
        )
        assert policy.tau_entail == 0.5
        assert policy.min_evidence_spans == 1

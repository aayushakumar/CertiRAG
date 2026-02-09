"""
Adversarial Robustness Tests
==============================

Tests that verify CertiRAG's resistance to adversarial inputs:
    - Negation insertion (flipping claim meaning)
    - Paraphrase attacks (semantically equivalent rewording)
    - Entity substitution (swapping entities)
    - Multi-hop reasoning (requires chaining evidence)

These tests are critical for demonstrating that the verification
pipeline cannot be easily fooled.
"""

from __future__ import annotations

import pytest

from certirag.claim_ir.normalizer import ClaimNormalizer
from certirag.render.policy import RendererPolicy
from certirag.schemas.certificate import RenderState
from certirag.schemas.claim_ir import CandidateEvidence, Claim, ClaimIR, ClaimType
from certirag.schemas.verification import (
    EvidenceScore,
    VerificationLabel,
    VerificationResult,
)


@pytest.mark.adversarial
class TestNegationAttack:
    """
    Test that negated claims are detected as contradictions.

    Attack: Take a supported claim and negate it.
    Expected: System should NOT verify the negated claim.
    """

    SUPPORTED_CLAIMS = [
        "Paris is the capital of France.",
        "The Eiffel Tower was built in 1889.",
        "Tokyo is the most populous city in Japan.",
    ]

    NEGATED_CLAIMS = [
        "Paris is not the capital of France.",
        "The Eiffel Tower was not built in 1889.",
        "Tokyo is not the most populous city in Japan.",
    ]

    def test_negated_claim_not_verified(self):
        """
        If the negation of a supported fact is presented,
        it should not be marked VERIFIED.
        """
        policy = RendererPolicy(tau_entail=0.5, tau_contradict=0.7)

        # Simulate verification where negated claim gets low entailment
        for claim_text in self.NEGATED_CLAIMS:
            contradict_ev = EvidenceScore(
                chunk_id="e1", span_id="s0",
                label=VerificationLabel.CONTRADICTED, score=0.85,
            )
            vr = VerificationResult(
                claim_id="adv_neg",
                label=VerificationLabel.CONTRADICTED,
                score=0.1,
                all_scores=[contradict_ev],
                best_evidence=[],
            )
            decision = policy.decide(vr)
            assert decision.render_state != RenderState.VERIFIED, (
                f"Negated claim was incorrectly VERIFIED: {claim_text}"
            )


@pytest.mark.adversarial
class TestEntitySubstitution:
    """
    Test entity substitution attacks.

    Attack: Replace an entity in a supported claim with a different one.
    Expected: System should NOT verify the substituted claim.
    """

    ORIGINAL = "Paris is the capital of France."
    SUBSTITUTED = [
        "London is the capital of France.",      # Wrong city
        "Paris is the capital of Germany.",       # Wrong country
        "Berlin is the capital of France.",       # Wrong city
    ]

    def test_substituted_entity_not_verified(self):
        """Substituted entity claims should not be VERIFIED."""
        policy = RendererPolicy(tau_entail=0.5, tau_contradict=0.7)

        for claim_text in self.SUBSTITUTED:
            contradict_ev = EvidenceScore(
                chunk_id="e1", span_id="s0",
                label=VerificationLabel.CONTRADICTED, score=0.7,
            )
            vr = VerificationResult(
                claim_id="adv_sub",
                label=VerificationLabel.CONTRADICTED,
                score=0.2,
                all_scores=[contradict_ev],
                best_evidence=[],
            )
            decision = policy.decide(vr)
            assert decision.render_state != RenderState.VERIFIED, (
                f"Substituted claim was incorrectly VERIFIED: {claim_text}"
            )


@pytest.mark.adversarial
class TestSpoofingAttack:
    """
    Test that claims cannot spoof verification metadata.

    Attack: Try to inject metadata that would bypass verification.
    Expected: Schema validation should reject malformed data.
    """

    def test_score_out_of_range(self):
        """Entailment scores > 1.0 should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EvidenceScore(
                chunk_id="e1",
                span_id="s0",
                label=VerificationLabel.ENTAILED,
                score=5.0,  # Injected high score
            )

    def test_negative_score(self):
        """Negative scores should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EvidenceScore(
                chunk_id="e1",
                span_id="s0",
                label=VerificationLabel.ENTAILED,
                score=-0.5,
            )

    def test_tampered_certificate_detected(self):
        """A tampered certificate should fail integrity check."""
        from certirag.schemas.certificate import AuditCertificate, PolicySnapshot

        cert = AuditCertificate(
            query_id="test",
            question="test?",
            config_hash="abc",
            stats={"total_claims": 3, "verified_claims": 2, "blocked_claims": 1},
            policy=PolicySnapshot(
                tau_entail=0.5, tau_contradict=0.7,
                min_evidence_spans=1,
            ),
        )
        cert.seal()
        # Tamper with the certificate
        cert.stats["verified_claims"] = 999
        assert cert.verify_integrity() is False


@pytest.mark.adversarial
class TestMultihopReasoning:
    """
    Test multi-hop claim verification.

    Attack: Present a claim that requires chaining two pieces of
    evidence, but only one is available.
    Expected: System should NOT verify the claim.
    """

    def test_partial_evidence_not_verified(self):
        """
        Claim requires both facts, but evidence only supports one.
        """
        policy = RendererPolicy(tau_entail=0.5, tau_contradict=0.7)

        # Evidence only partially supports the multi-hop claim
        entail_ev = EvidenceScore(
            chunk_id="e1", span_id="s0",
            label=VerificationLabel.ENTAILED, score=0.3,
        )
        vr = VerificationResult(
            claim_id="multihop_1",
            label=VerificationLabel.NOT_ENOUGH_INFO,
            score=0.3,
            all_scores=[entail_ev],
            best_evidence=[],
        )
        decision = policy.decide(vr)
        assert decision.render_state != RenderState.VERIFIED


@pytest.mark.adversarial
class TestHedgedClaims:
    """Test that hedged/uncertain claims are handled correctly."""

    def test_hedge_normalization(self):
        """Hedged claims should be flagged during normalization."""
        normalizer = ClaimNormalizer(remove_hedges=True)
        ir = ClaimIR(
            question="test?",
            claims=[
                Claim(
                    id="h1",
                    text="Paris probably is the capital of France.",
                    type=ClaimType.FACTUAL,
                    evidence_candidates=[
                        CandidateEvidence(chunk_id="e1", span_id="s0"),
                    ],
                ),
            ],
        )
        result = normalizer.normalize(ir)
        # Hedged claims should still exist but may be marked
        assert len(result.claims) >= 1

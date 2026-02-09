"""
Renderer Policy Tests
========================

Tests for the fail-closed renderer policy engine.
This is the most safety-critical component — the policy
determines what claims are shown to users.

Tests verify Theorem 1:
    VERIFIED ⟺ (entail ≥ τ_e) ∧ (¬contradict ≥ τ_c) ∧ (evidence ≥ 1)

Coverage:
    - Threshold boundary cases
    - Display mode filtering (strict/mixed/debug)
    - Missing evidence handling
    - No false-positive VERIFIED claims
"""

from __future__ import annotations

import pytest

from certirag.render.policy import RendererPolicy
from certirag.schemas.certificate import PolicySnapshot, RendererDecision, RenderState
from certirag.schemas.claim_ir import ClaimIR
from certirag.schemas.verification import (
    EvidenceScore,
    VerificationLabel,
    VerificationResult,
)
from tests.conftest import make_claim, make_verification


_TEST_POLICY = PolicySnapshot(tau_entail=0.5, tau_contradict=0.7, min_evidence_spans=1)


@pytest.fixture
def policy():
    """Standard policy with known thresholds."""
    return RendererPolicy(
        tau_entail=0.5,
        tau_contradict=0.7,
        min_evidence_spans=1,
    )


class TestPolicyDecisions:
    """Test the core decision logic (Theorem 1)."""

    def test_entailed_above_threshold(self, policy):
        """High entailment + low contradiction → VERIFIED."""
        vr = make_verification(entail=0.8, contradict=0.05)
        decision = policy.decide(vr)
        assert decision.render_state == RenderState.VERIFIED

    def test_entailed_below_threshold(self, policy):
        """Low entailment → UNVERIFIED."""
        vr = make_verification(
            label=VerificationLabel.NOT_ENOUGH_INFO,
            entail=0.3, contradict=0.1,
        )
        decision = policy.decide(vr)
        assert decision.render_state == RenderState.UNVERIFIED

    def test_contradicted(self, policy):
        """High contradiction → BLOCKED."""
        vr = make_verification(
            label=VerificationLabel.CONTRADICTED,
            entail=0.1, contradict=0.85,
        )
        decision = policy.decide(vr)
        assert decision.render_state == RenderState.BLOCKED

    def test_boundary_exact_threshold(self, policy):
        """At exact threshold boundary → VERIFIED (inclusive)."""
        vr = make_verification(entail=0.5, contradict=0.0)
        decision = policy.decide(vr)
        assert decision.render_state == RenderState.VERIFIED

    def test_boundary_just_below(self, policy):
        """Just below entailment threshold → not VERIFIED."""
        vr = make_verification(
            label=VerificationLabel.NOT_ENOUGH_INFO,
            entail=0.499, contradict=0.0,
        )
        decision = policy.decide(vr)
        assert decision.render_state != RenderState.VERIFIED

    def test_no_evidence(self, policy):
        """No evidence scores → UNVERIFIED (fail-closed)."""
        vr = VerificationResult(
            claim_id="c1",
            label=VerificationLabel.NOT_ENOUGH_INFO,
            score=0.0,
        )
        decision = policy.decide(vr)
        assert decision.render_state == RenderState.UNVERIFIED

    def test_entailed_with_high_contradiction(self, policy):
        """High entailment AND high contradiction → BLOCKED."""
        vr = make_verification(
            label=VerificationLabel.CONTRADICTED,
            entail=0.6, contradict=0.8,
        )
        decision = policy.decide(vr)
        # Contradiction should override entailment
        assert decision.render_state == RenderState.BLOCKED


class TestDisplayModes:
    """Test display filtering in different modes."""

    def test_strict_mode_hides_unverified(self, policy, sample_claim_ir):
        """Strict mode should only show VERIFIED claims."""
        decisions = [
            RendererDecision(claim_id="c1", render_state=RenderState.VERIFIED, policy=_TEST_POLICY, reason="ok"),
            RendererDecision(claim_id="c2", render_state=RenderState.UNVERIFIED, policy=_TEST_POLICY, reason="low"),
        ]
        result = policy.filter_for_display(sample_claim_ir, decisions, mode="strict")
        visible = result["visible_claims"]
        hidden = result["hidden_claims"]
        assert len(visible) == 1
        assert len(hidden) == 1
        assert visible[0]["decision"]["render_state"] == "VERIFIED"

    def test_mixed_mode_shows_unverified_tagged(self, policy, sample_claim_ir):
        """Mixed mode should show unverified with warnings."""
        decisions = [
            RendererDecision(claim_id="c1", render_state=RenderState.VERIFIED, policy=_TEST_POLICY, reason="ok"),
            RendererDecision(claim_id="c2", render_state=RenderState.UNVERIFIED, policy=_TEST_POLICY, reason="low"),
        ]
        result = policy.filter_for_display(sample_claim_ir, decisions, mode="mixed")
        visible = result["visible_claims"]
        assert len(visible) == 2

    def test_debug_mode_shows_all(self, policy, sample_claim_ir):
        """Debug mode should show everything including blocked."""
        decisions = [
            RendererDecision(claim_id="c1", render_state=RenderState.VERIFIED, policy=_TEST_POLICY, reason="ok"),
            RendererDecision(claim_id="c2", render_state=RenderState.BLOCKED, policy=_TEST_POLICY, reason="contra"),
        ]
        result = policy.filter_for_display(sample_claim_ir, decisions, mode="debug")
        visible = result["visible_claims"]
        assert len(visible) == 2

    def test_strict_never_shows_blocked(self, policy, sample_claim_ir):
        """BLOCKED claims should never appear in strict mode."""
        decisions = [
            RendererDecision(claim_id="c1", render_state=RenderState.BLOCKED, policy=_TEST_POLICY, reason="contra"),
            RendererDecision(claim_id="c2", render_state=RenderState.BLOCKED, policy=_TEST_POLICY, reason="contra"),
        ]
        result = policy.filter_for_display(sample_claim_ir, decisions, mode="strict")
        assert len(result["visible_claims"]) == 0


class TestBatchRendering:
    """Test rendering multiple claims at once."""

    def test_batch_render(self, policy):
        """Render should produce one decision per verification result."""
        vrs = [
            make_verification("c1", entail=0.9, contradict=0.02),
            make_verification("c2", entail=0.2, contradict=0.1),
            make_verification("c3", entail=0.1, contradict=0.85),
        ]
        decisions = policy.render(vrs)
        assert len(decisions) == 3
        states = [d.render_state for d in decisions]
        assert RenderState.VERIFIED in states
        assert RenderState.BLOCKED in states


class TestSafetyInvariant:
    """
    Critical: No claim should be marked VERIFIED unless it passes
    all three conditions of Theorem 1.
    """

    @pytest.mark.parametrize("entail,contradict,expected_not_verified", [
        (0.0, 0.0, True),     # No entailment
        (0.3, 0.0, True),     # Below threshold
        (0.5, 0.8, True),     # High contradiction
        (1.0, 0.8, True),     # Very high contradiction overrides
    ])
    def test_no_false_verified(self, policy, entail, contradict, expected_not_verified):
        """Claims that shouldn't be verified must not be VERIFIED."""
        label = VerificationLabel.ENTAILED if entail >= 0.5 else VerificationLabel.NOT_ENOUGH_INFO
        if contradict >= 0.7:
            label = VerificationLabel.CONTRADICTED
        vr = make_verification(label=label, entail=entail, contradict=contradict)
        decision = policy.decide(vr)
        if expected_not_verified:
            assert decision.render_state != RenderState.VERIFIED

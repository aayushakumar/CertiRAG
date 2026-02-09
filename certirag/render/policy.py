"""
Renderer Policy Engine
========================

The heart of CertiRAG's fail-closed guarantee. This is a DETERMINISTIC
policy engine that maps VerificationResults to rendering decisions.

CRITICAL INVARIANTS (the "Theorem"):
    1. A claim is VERIFIED iff:
       - max_entail_score >= τ_entail
       - AND max_contradict_score < τ_contradict
       - AND evidence_span_count >= min_evidence_spans
    2. A claim is BLOCKED iff:
       - max_contradict_score >= τ_contradict
    3. All other claims are UNVERIFIED
    4. The LLM's output CANNOT influence the rendering decision
       (badges are assigned ONLY by this policy engine)

Design Philosophy (inspired by Proof-Carrying Numbers):
    Verification is enforced at the PRESENTATION LAYER.
    The renderer is the trust boundary — it reads only from
    VerificationResult objects, never from raw LLM output.

This module contains NO machine learning, NO LLM calls, NO
randomness. It is pure, deterministic, auditable logic.

Data Flow:
    VerificationResult + Policy → RendererDecision
"""

from __future__ import annotations

import logging
from typing import Optional

from certirag.config import CertiRAGConfig
from certirag.schemas.certificate import PolicySnapshot, RenderState, RendererDecision
from certirag.schemas.claim_ir import ClaimIR
from certirag.schemas.verification import VerificationLabel, VerificationResult

logger = logging.getLogger("certirag.render.policy")


class RendererPolicy:
    """
    Deterministic rendering policy engine.

    For each claim, produces a RendererDecision based on the
    verification results and configured thresholds.

    This is the ONLY component that assigns Verified/Unverified/Blocked
    badges. The LLM output cannot influence these decisions.

    Usage:
        policy = RendererPolicy(
            tau_entail=0.85,
            tau_contradict=0.70,
        )
        decisions = policy.render(verification_results)

    Modes:
        - STRICT: Only VERIFIED claims visible in main panel
        - MIXED:  VERIFIED + UNVERIFIED (with warning) visible
        - DEBUG:  Everything visible with full traces

    Args:
        tau_entail: Threshold for entailment (Verified requires score >= tau).
        tau_contradict: Threshold for contradiction (Blocked requires score >= tau).
        min_evidence_spans: Minimum evidence spans for Verified status.
        policy_version: Version string for audit trail.
    """

    def __init__(
        self,
        tau_entail: float = 0.85,
        tau_contradict: float = 0.70,
        min_evidence_spans: int = 1,
        policy_version: str = "v1.0",
    ):
        self.tau_entail = tau_entail
        self.tau_contradict = tau_contradict
        self.min_evidence_spans = min_evidence_spans
        self.policy_version = policy_version

        # Validate thresholds
        assert 0.0 <= tau_entail <= 1.0, f"tau_entail must be in [0,1], got {tau_entail}"
        assert 0.0 <= tau_contradict <= 1.0, f"tau_contradict must be in [0,1], got {tau_contradict}"
        assert min_evidence_spans >= 1, "Must require at least 1 evidence span"

    @classmethod
    def from_config(cls, config: CertiRAGConfig) -> "RendererPolicy":
        """Create a RendererPolicy from CertiRAG config."""
        return cls(
            tau_entail=config.verification.tau_entail,
            tau_contradict=config.verification.tau_contradict,
            min_evidence_spans=config.render.min_evidence_spans,
        )

    @property
    def policy_snapshot(self) -> PolicySnapshot:
        """Get an immutable snapshot of the current policy for certificates."""
        return PolicySnapshot(
            tau_entail=self.tau_entail,
            tau_contradict=self.tau_contradict,
            min_evidence_spans=self.min_evidence_spans,
            policy_version=self.policy_version,
        )

    def decide(self, result: VerificationResult) -> RendererDecision:
        """
        Make a rendering decision for a single claim.

        Decision Logic (DETERMINISTIC):
            1. If max_contradict_score >= tau_contradict → BLOCKED
            2. If max_entail_score >= tau_entail
               AND evidence_span_count >= min_evidence_spans
               AND NOT blocked by contradiction → VERIFIED
            3. Otherwise → UNVERIFIED

        This is the formal property stated as Theorem 1 in the paper:
            VERIFIED ⟺ (entail ≥ τ_e) ∧ (¬contradict ≥ τ_c) ∧ (evidence ≥ 1)

        Args:
            result: VerificationResult for one claim.

        Returns:
            RendererDecision with state, policy reference, and reason.
        """
        max_entail = result.max_entail_score
        max_contradict = result.max_contradict_score
        evidence_count = result.evidence_span_count

        # ── Decision Logic ─────────────────────────────────────────
        #
        # Priority 1: Contradiction blocking (fail-closed safety)
        if max_contradict >= self.tau_contradict:
            return RendererDecision(
                claim_id=result.claim_id,
                render_state=RenderState.BLOCKED,
                policy=self.policy_snapshot,
                reason=(
                    f"Contradiction detected: max_contradict_score={max_contradict:.3f} "
                    f">= tau_contradict={self.tau_contradict}"
                ),
                entail_score=max_entail,
                contradict_score=max_contradict,
                evidence_count=evidence_count,
            )

        # Priority 2: Entailment verification
        if (
            max_entail >= self.tau_entail
            and evidence_count >= self.min_evidence_spans
        ):
            return RendererDecision(
                claim_id=result.claim_id,
                render_state=RenderState.VERIFIED,
                policy=self.policy_snapshot,
                reason=(
                    f"Entailed: max_entail_score={max_entail:.3f} >= tau_entail={self.tau_entail} "
                    f"AND evidence_count={evidence_count} >= {self.min_evidence_spans} "
                    f"AND no contradiction >= tau_contradict={self.tau_contradict}"
                ),
                entail_score=max_entail,
                contradict_score=max_contradict,
                evidence_count=evidence_count,
            )

        # Priority 3: Insufficient evidence or below threshold
        reasons = []
        if max_entail < self.tau_entail:
            reasons.append(
                f"max_entail_score={max_entail:.3f} < tau_entail={self.tau_entail}"
            )
        if evidence_count < self.min_evidence_spans:
            reasons.append(
                f"evidence_count={evidence_count} < min_evidence_spans={self.min_evidence_spans}"
            )

        return RendererDecision(
            claim_id=result.claim_id,
            render_state=RenderState.UNVERIFIED,
            policy=self.policy_snapshot,
            reason="Unverified: " + "; ".join(reasons),
            entail_score=max_entail,
            contradict_score=max_contradict,
            evidence_count=evidence_count,
        )

    def render(
        self, results: list[VerificationResult]
    ) -> list[RendererDecision]:
        """
        Make rendering decisions for all claims.

        Args:
            results: List of VerificationResults (one per claim).

        Returns:
            List of RendererDecisions (one per claim).
        """
        decisions = [self.decide(r) for r in results]

        # Log summary statistics
        verified = sum(1 for d in decisions if d.render_state == RenderState.VERIFIED)
        unverified = sum(1 for d in decisions if d.render_state == RenderState.UNVERIFIED)
        blocked = sum(1 for d in decisions if d.render_state == RenderState.BLOCKED)

        logger.info(
            f"Rendering decisions: {verified} verified, "
            f"{unverified} unverified, {blocked} blocked "
            f"(τ_e={self.tau_entail}, τ_c={self.tau_contradict})"
        )

        return decisions

    def filter_for_display(
        self,
        claim_ir: ClaimIR,
        decisions: list[RendererDecision],
        mode: str = "strict",
    ) -> dict:
        """
        Filter claims for display based on rendering mode.

        Args:
            claim_ir: Original ClaimIR with all claims.
            decisions: Rendering decisions for each claim.
            mode: "strict", "mixed", or "debug".

        Returns:
            Dict with 'visible_claims', 'hidden_claims', 'mode'.
        """
        decision_map = {d.claim_id: d for d in decisions}
        visible = []
        hidden = []

        for claim in claim_ir.claims:
            decision = decision_map.get(claim.id)
            if not decision:
                hidden.append({"claim": claim.model_dump(), "reason": "no decision"})
                continue

            entry = {
                "claim": claim.model_dump(),
                "decision": decision.model_dump(),
            }

            if mode == "strict":
                # Only show VERIFIED claims
                if decision.render_state == RenderState.VERIFIED:
                    visible.append(entry)
                else:
                    hidden.append(entry)

            elif mode == "mixed":
                # Show VERIFIED + UNVERIFIED (with warnings), hide BLOCKED
                if decision.render_state in (RenderState.VERIFIED, RenderState.UNVERIFIED):
                    visible.append(entry)
                else:
                    hidden.append(entry)

            elif mode == "debug":
                # Show everything
                visible.append(entry)

        return {
            "mode": mode,
            "visible_claims": visible,
            "hidden_claims": hidden,
            "stats": {
                "total": len(claim_ir.claims),
                "visible": len(visible),
                "hidden": len(hidden),
            },
        }

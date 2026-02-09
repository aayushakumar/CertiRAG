"""
CertiRAG Test Configuration
==============================

Shared fixtures, factories, and helpers for the entire test suite.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

# ── Ensure test mode ────────────────────────────────────────────
os.environ.setdefault("CERTIRAG_MODE", "lite")
os.environ.setdefault("CERTIRAG_OPENAI_API_KEY", "sk-test-key-for-testing")

from certirag.config import CertiRAGConfig, get_config
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


# ── Markers ─────────────────────────────────────────────────────

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: fast unit tests")
    config.addinivalue_line("markers", "integration: multi-component tests")
    config.addinivalue_line("markers", "adversarial: adversarial robustness tests")
    config.addinivalue_line("markers", "slow: tests that take >5s")
    config.addinivalue_line("markers", "gpu: tests requiring GPU")


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def config() -> CertiRAGConfig:
    """Default test config (lite mode)."""
    return get_config()


@pytest.fixture
def sample_documents() -> list[dict[str, str]]:
    """Small corpus for testing."""
    return [
        {
            "doc_id": "doc_paris",
            "title": "Paris",
            "text": (
                "Paris is the capital of France. It is located on the Seine river. "
                "The city has a population of approximately 2.1 million people. "
                "Paris is known for the Eiffel Tower, built in 1889. "
                "The Louvre Museum in Paris houses the Mona Lisa."
            ),
        },
        {
            "doc_id": "doc_berlin",
            "title": "Berlin",
            "text": (
                "Berlin is the capital of Germany. It is the largest city in Germany "
                "by population. The Berlin Wall divided the city from 1961 to 1989. "
                "Berlin is known for the Brandenburg Gate and Museum Island."
            ),
        },
        {
            "doc_id": "doc_tokyo",
            "title": "Tokyo",
            "text": (
                "Tokyo is the capital of Japan. It is the most populous metropolitan "
                "area in the world. Tokyo hosted the Summer Olympics in 1964 and 2021. "
                "The city is known for Shibuya Crossing and the Imperial Palace."
            ),
        },
    ]


@pytest.fixture
def sample_chunk() -> EvidenceChunk:
    """A single evidence chunk for testing."""
    return make_chunk(
        text="Paris is the capital of France. The Eiffel Tower was built in 1889.",
        doc_id="doc_paris",
    )


@pytest.fixture
def sample_chunks() -> list[EvidenceChunk]:
    """Multiple evidence chunks for testing."""
    return [
        make_chunk(
            text="Paris is the capital of France.",
            doc_id="doc_paris",
            chunk_id="chunk_1",
        ),
        make_chunk(
            text="Berlin is the capital of Germany.",
            doc_id="doc_berlin",
            chunk_id="chunk_2",
        ),
        make_chunk(
            text="Tokyo is the capital of Japan.",
            doc_id="doc_tokyo",
            chunk_id="chunk_3",
        ),
    ]


@pytest.fixture
def sample_claim_ir(sample_chunks) -> ClaimIR:
    """A complete ClaimIR for testing."""
    return ClaimIR(
        question="What is the capital of France?",
        claims=[
            Claim(
                id="c1",
                text="Paris is the capital of France.",
                type=ClaimType.FACTUAL,
                evidence_candidates=[
                    CandidateEvidence(chunk_id="chunk_1", span_id="s0"),
                ],
            ),
            Claim(
                id="c2",
                text="The Eiffel Tower is in Paris.",
                type=ClaimType.FACTUAL,
                evidence_candidates=[
                    CandidateEvidence(chunk_id="chunk_1", span_id="s0"),
                ],
            ),
        ],
    )


@pytest.fixture
def sample_verification_results() -> list[VerificationResult]:
    """Verification results for testing renderer."""
    entail_ev = EvidenceScore(
        chunk_id="chunk_1", span_id="s0",
        label=VerificationLabel.ENTAILED, score=0.92,
    )
    contradict_ev = EvidenceScore(
        chunk_id="chunk_1", span_id="s0",
        label=VerificationLabel.CONTRADICTED, score=0.78,
    )
    return [
        VerificationResult(
            claim_id="c1",
            label=VerificationLabel.ENTAILED,
            score=0.92,
            all_scores=[entail_ev],
            best_evidence=[entail_ev],
        ),
        VerificationResult(
            claim_id="c2",
            label=VerificationLabel.CONTRADICTED,
            score=0.15,
            all_scores=[contradict_ev],
            best_evidence=[],
        ),
    ]


# ── Factories ───────────────────────────────────────────────────

def make_chunk(
    text: str = "Default chunk text.",
    doc_id: str = "doc_0",
    chunk_id: str | None = None,
) -> EvidenceChunk:
    """Factory for creating test evidence chunks."""
    if chunk_id is None:
        chunk_id = f"chunk_{uuid.uuid4().hex[:8]}"
    return EvidenceChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        source=SourceInfo(title=f"Doc {doc_id}"),
        retrieval=RetrievalScores(bm25=0.5, dense=0.5),
        spans=[
            EvidenceSpan(
                span_id="s0",
                start=0,
                end=len(text),
                sentence=text,
            ),
        ],
    )


def make_claim(
    text: str = "Default claim.",
    claim_id: str | None = None,
    claim_type: ClaimType = ClaimType.FACTUAL,
    chunk_ids: list[str] | None = None,
) -> Claim:
    """Factory for creating test claims."""
    if claim_id is None:
        claim_id = f"claim_{uuid.uuid4().hex[:8]}"
    evidence = [
        CandidateEvidence(chunk_id=cid, span_id="s0")
        for cid in (chunk_ids or ["chunk_0"])
    ]
    return Claim(
        id=claim_id,
        text=text,
        type=claim_type,
        evidence_candidates=evidence,
    )


def make_verification(
    claim_id: str = "c1",
    label: VerificationLabel = VerificationLabel.ENTAILED,
    entail: float = 0.9,
    contradict: float = 0.05,
) -> VerificationResult:
    """Factory for creating test verification results."""
    all_scores = []
    best_evidence = []
    if entail > 0:
        entail_ev = EvidenceScore(
            chunk_id="chunk_0", span_id="s0",
            label=VerificationLabel.ENTAILED, score=entail,
        )
        all_scores.append(entail_ev)
        best_evidence.append(entail_ev)
    if contradict > 0:
        contradict_ev = EvidenceScore(
            chunk_id="chunk_0", span_id="s1",
            label=VerificationLabel.CONTRADICTED, score=contradict,
        )
        all_scores.append(contradict_ev)
    return VerificationResult(
        claim_id=claim_id,
        label=label,
        score=entail,
        all_scores=all_scores,
        best_evidence=best_evidence,
    )

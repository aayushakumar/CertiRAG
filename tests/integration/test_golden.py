"""
Golden Path Integration Tests
================================

End-to-end tests using small golden examples where the
expected output is known. These verify that the full pipeline
produces correct results on tractable inputs.

Each test follows the pattern:
    1. Create known documents
    2. Ask a question with a known answer
    3. Verify the pipeline's output matches expectations
"""

from __future__ import annotations

import pytest

from certirag.ingest.chunker import DocumentChunker
from certirag.ingest.indexer import BM25Index, ChunkStore
from certirag.claim_ir.normalizer import ClaimNormalizer
from certirag.render.policy import RendererPolicy
from certirag.schemas.certificate import RenderState
from certirag.schemas.verification import (
    EvidenceScore,
    VerificationLabel,
    VerificationResult,
)


@pytest.mark.integration
class TestGoldenRetrieval:
    """Test that retrieval finds relevant chunks."""

    def test_bm25_retrieves_relevant(self, sample_documents):
        """BM25 should rank Paris doc highest for 'capital of France'."""
        chunker = DocumentChunker(chunk_size=200, overlap=0, use_spacy=False)
        chunks = chunker.chunk_documents(sample_documents)

        index = BM25Index()
        index.build(chunks)

        results = index.query("capital of France", top_k=3)
        # Paris document should be in top results
        top_chunk_ids = [cid for cid, _ in results]
        paris_chunks = [c.chunk_id for c in chunks if c.doc_id == "doc_paris"]
        assert any(pc in top_chunk_ids for pc in paris_chunks), (
            f"Paris chunks not found in top-3. Got: {top_chunk_ids}"
        )

    def test_bm25_retrieves_correct_topic(self, sample_documents):
        """BM25 should rank Tokyo doc highest for 'Olympics in Japan'."""
        chunker = DocumentChunker(chunk_size=200, overlap=0, use_spacy=False)
        chunks = chunker.chunk_documents(sample_documents)

        index = BM25Index()
        index.build(chunks)

        results = index.query("Olympics in Japan", top_k=3)
        top_chunk_ids = [cid for cid, _ in results]
        tokyo_chunks = [c.chunk_id for c in chunks if c.doc_id == "doc_tokyo"]
        assert any(tc in top_chunk_ids for tc in tokyo_chunks)


@pytest.mark.integration
class TestGoldenRenderer:
    """Test that the renderer correctly filters claims."""

    def test_verified_claim_shown_in_strict(self):
        """A well-verified claim should appear in strict mode output."""
        policy = RendererPolicy(tau_entail=0.5, tau_contradict=0.7)

        entail_ev = EvidenceScore(
            chunk_id="e1", span_id="s0",
            label=VerificationLabel.ENTAILED, score=0.95,
        )
        vr = VerificationResult(
            claim_id="c1",
            label=VerificationLabel.ENTAILED,
            score=0.95,
            all_scores=[entail_ev],
            best_evidence=[entail_ev],
        )
        decision = policy.decide(vr)
        assert decision.render_state == RenderState.VERIFIED

    def test_pipeline_rejects_unsupported_claim(self):
        """Unsupported claims should not pass through strict mode."""
        policy = RendererPolicy(tau_entail=0.5, tau_contradict=0.7)

        entail_ev = EvidenceScore(
            chunk_id="e1", span_id="s0",
            label=VerificationLabel.ENTAILED, score=0.15,
        )
        vr = VerificationResult(
            claim_id="c1",
            label=VerificationLabel.NOT_ENOUGH_INFO,
            score=0.15,
            all_scores=[entail_ev],
            best_evidence=[],
        )
        decision = policy.decide(vr)
        assert decision.render_state != RenderState.VERIFIED


@pytest.mark.integration
class TestGoldenNormalization:
    """Test claim normalization on known inputs."""

    def test_compound_claim_splits(self):
        """A clear conjunction should split into atomic claims."""
        from certirag.schemas.claim_ir import (
            CandidateEvidence, Claim, ClaimIR, ClaimType,
        )

        normalizer = ClaimNormalizer(split_conjunctions=True)
        claim_ir = ClaimIR(
            question="test?",
            claims=[
                Claim(
                    id="c1",
                    text="Paris is the capital of France, and Berlin is the capital of Germany.",
                    type=ClaimType.FACTUAL,
                    evidence_candidates=[
                        CandidateEvidence(chunk_id="e1", span_id="s0"),
                    ],
                ),
            ],
        )
        result = normalizer.normalize(claim_ir)
        assert len(result.claims) >= 2

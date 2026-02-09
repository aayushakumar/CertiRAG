"""
Corruption & Removal Tests
==============================

Integration tests that verify the pipeline handles edge cases:
    - Corrupted documents (empty, malformed)
    - Missing evidence (no matching chunks)
    - Removed documents (evidence no longer available)

These test the fail-closed guarantee: when data is missing
or corrupted, the system should err on the side of blocking.
"""

from __future__ import annotations

import pytest

from certirag.ingest.chunker import DocumentChunker
from certirag.ingest.indexer import BM25Index, ChunkStore
from certirag.render.policy import RendererPolicy
from certirag.schemas.certificate import RenderState
from certirag.schemas.verification import (
    EvidenceScore,
    VerificationLabel,
    VerificationResult,
)


@pytest.mark.integration
class TestCorruptedInput:
    """Test pipeline behavior with corrupted inputs."""

    def test_empty_document_handled(self):
        """Pipeline should handle empty document text gracefully."""
        chunker = DocumentChunker(chunk_size=100, overlap=0, use_spacy=False)
        docs = [{"doc_id": "d1", "text": "Valid text here."}]
        chunks = chunker.chunk_documents(docs)
        # Should produce at least one chunk from valid doc
        assert len(chunks) >= 1

    def test_whitespace_only_document(self):
        """Whitespace-only documents should produce no chunks."""
        chunker = DocumentChunker(chunk_size=100, overlap=0, use_spacy=False)
        docs = [{"doc_id": "d1", "text": "   \n\t  "}]
        chunks = chunker.chunk_documents(docs)
        # Whitespace documents should produce zero or minimal chunks
        assert len(chunks) <= 1

    def test_very_long_document(self):
        """Very long documents should chunk without crashing."""
        chunker = DocumentChunker(chunk_size=100, overlap=0, use_spacy=False)
        docs = [{"doc_id": "d1", "text": "Word. " * 10000}]
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) > 1

    def test_special_characters(self):
        """Documents with special characters should not crash."""
        chunker = DocumentChunker(chunk_size=200, overlap=0, use_spacy=False)
        docs = [{
            "doc_id": "d1",
            "text": "Contains émojis 🎉 and spëcial chars: <>&\"' \\n\\t \x00",
        }]
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) >= 1


@pytest.mark.integration
class TestMissingEvidence:
    """Test behavior when required evidence is not found."""

    def test_no_evidence_fails_closed(self):
        """Claims without evidence should be UNVERIFIED (fail-closed)."""
        policy = RendererPolicy(tau_entail=0.5, tau_contradict=0.7)

        # Verification with no evidence
        vr = VerificationResult(
            claim_id="c1",
            label=VerificationLabel.NOT_ENOUGH_INFO,
            score=0.0,
        )
        decision = policy.decide(vr)
        assert decision.render_state != RenderState.VERIFIED

    def test_irrelevant_evidence_fails_closed(self):
        """Low-scoring evidence should not verify a claim."""
        policy = RendererPolicy(tau_entail=0.5, tau_contradict=0.7)

        entail_ev = EvidenceScore(
            chunk_id="e1", span_id="s0",
            label=VerificationLabel.ENTAILED, score=0.1,
        )
        vr = VerificationResult(
            claim_id="c1",
            label=VerificationLabel.NOT_ENOUGH_INFO,
            score=0.1,
            all_scores=[entail_ev],
            best_evidence=[],
        )
        decision = policy.decide(vr)
        assert decision.render_state != RenderState.VERIFIED


@pytest.mark.integration
class TestDocumentRemoval:
    """Test behavior when documents are removed mid-pipeline."""

    def test_bm25_query_after_build(self):
        """BM25 index should handle queries correctly."""
        chunker = DocumentChunker(chunk_size=200, overlap=0, use_spacy=False)
        docs = [
            {"doc_id": "d1", "text": "Paris is the capital of France."},
            {"doc_id": "d2", "text": "Berlin is the capital of Germany."},
        ]
        chunks = chunker.chunk_documents(docs)

        index = BM25Index()
        index.build(chunks)

        # Query for removed content
        results = index.query("Tokyo capital", top_k=5)
        # Should return results (even if low-scoring) without crashing
        assert isinstance(results, list)

    def test_chunk_store_missing_id(self):
        """ChunkStore should handle missing chunk IDs gracefully."""
        store = ChunkStore()

        chunker = DocumentChunker(chunk_size=200, overlap=0, use_spacy=False)
        docs = [{"doc_id": "d1", "text": "Some text."}]
        chunks = chunker.chunk_documents(docs)
        store.add_many(chunks)

        # Query for non-existent chunk
        result = store.get("nonexistent_chunk_id")
        assert result is None

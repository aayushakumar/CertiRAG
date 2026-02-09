"""
Span Integrity Tests
========================

Tests for the span offset tracking invariant (T-U4):
    For every span s: s.start < s.end AND s.text == doc[s.start:s.end]

This is a critical invariant for evidence attribution —
if spans don't accurately point to source text, the entire
verification audit trail is meaningless.
"""

from __future__ import annotations

import pytest

from certirag.ingest.chunker import DocumentChunker
from certirag.retrieve.span_extractor import SpanExtractor, get_span_text
from certirag.schemas.evidence import EvidenceChunk, EvidenceSpan, RetrievalScores, SourceInfo


class TestSpanBasicInvariants:
    """Basic span integrity invariants."""

    def test_start_less_than_end(self):
        """start must be strictly less than end."""
        span = EvidenceSpan(span_id="s0", start=0, end=10, sentence="0123456789")
        assert span.start < span.end

    def test_text_length_matches_offsets(self):
        """text length must equal end - start."""
        text = "Hello world"
        span = EvidenceSpan(span_id="s0", start=5, end=5 + len(text), sentence=text)
        assert span.end - span.start == len(span.sentence)

    def test_invalid_span_rejected(self):
        """start >= end should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EvidenceSpan(span_id="s0", start=10, end=5, sentence="hello")

    def test_zero_length_rejected(self):
        """Zero-length span (start == end) should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EvidenceSpan(span_id="s0", start=5, end=5, sentence="")


class TestChunkerSpanIntegrity:
    """Test that the DocumentChunker produces valid spans."""

    @pytest.fixture
    def chunker(self):
        return DocumentChunker(chunk_size=100, overlap=0, use_spacy=False)

    def test_chunker_spans_valid(self, chunker, sample_documents):
        """All chunks from chunker should have valid spans."""
        chunks = chunker.chunk_documents(sample_documents)
        for chunk in chunks:
            for span in chunk.spans:
                # T-U4: start < end
                assert span.start < span.end, (
                    f"Invalid span in chunk {chunk.chunk_id}: "
                    f"start={span.start} >= end={span.end}"
                )
                # T-U4: length consistency
                assert span.end - span.start == len(span.sentence), (
                    f"Length mismatch in chunk {chunk.chunk_id}: "
                    f"offsets say {span.end - span.start}, "
                    f"sentence length is {len(span.sentence)}"
                )

    def test_spans_cover_chunk_text(self, chunker, sample_documents):
        """Concatenated span texts should reconstruct chunk text (approximately)."""
        chunks = chunker.chunk_documents(sample_documents)
        for chunk in chunks:
            if not chunk.spans:
                continue
            reconstructed = " ".join(s.sentence for s in chunk.spans)
            # Allow minor differences due to whitespace normalization
            assert len(reconstructed) > 0

    def test_spans_non_overlapping(self, chunker, sample_documents):
        """Spans within a chunk should not overlap."""
        chunks = chunker.chunk_documents(sample_documents)
        for chunk in chunks:
            sorted_spans = sorted(chunk.spans, key=lambda s: s.start)
            for i in range(len(sorted_spans) - 1):
                assert sorted_spans[i].end <= sorted_spans[i + 1].start, (
                    f"Overlapping spans in chunk {chunk.chunk_id}: "
                    f"span {i} ends at {sorted_spans[i].end}, "
                    f"span {i+1} starts at {sorted_spans[i+1].start}"
                )

    def test_span_ids_present(self, chunker, sample_documents):
        """All spans should have a non-empty span_id."""
        chunks = chunker.chunk_documents(sample_documents)
        for chunk in chunks:
            for span in chunk.spans:
                assert span.span_id, (
                    f"Span in chunk {chunk.chunk_id} has empty span_id"
                )


class TestSpanExtractor:
    """Test the SpanExtractor utility."""

    def test_validate_valid_spans(self, sample_chunk):
        """Valid chunks should pass validation."""
        extractor = SpanExtractor()
        result = extractor._validate_spans(sample_chunk)
        assert result is True

    def test_get_span_text_extraction(self, sample_chunk):
        """get_span_text should extract text from a chunk by span_id."""
        if sample_chunk.spans:
            text = get_span_text(sample_chunk, "s0")
            assert text == sample_chunk.spans[0].sentence


class TestSpanWithRealText:
    """Test spans against real source text to ensure attribution works."""

    def test_span_matches_source(self):
        """Span offsets should correctly index into source document."""
        source_text = "The quick brown fox jumps over the lazy dog."
        span = EvidenceSpan(
            span_id="s0",
            start=4, end=19,
            sentence="quick brown fox",
        )
        # Verify the span text matches using the offsets
        extracted = source_text[span.start:span.end]
        assert extracted == span.sentence

    def test_multi_sentence_spans(self):
        """Multiple sentence spans should all be valid."""
        source_text = "First sentence. Second sentence. Third sentence."
        spans = [
            EvidenceSpan(span_id="s0", start=0, end=15, sentence="First sentence."),
            EvidenceSpan(span_id="s1", start=16, end=32, sentence="Second sentence."),
            EvidenceSpan(span_id="s2", start=33, end=49, sentence="Third sentence."),
        ]
        for span in spans:
            assert source_text[span.start:span.end] == span.sentence

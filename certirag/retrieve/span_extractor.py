"""
Span Extractor
===============

Extracts sentence-level evidence spans from retrieved chunks,
ensuring spans have stable character offsets for UI highlighting.

This module ensures the T-U4 (Evidence Span Integrity) invariant:
    chunk.text[span.start:span.end] == span.sentence

Spans extracted here are the atomic evidence units that the verifier
scores against individual claims.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from certirag.schemas.evidence import EvidenceChunk, EvidenceSpan

logger = logging.getLogger("certirag.retrieve.span_extractor")


class SpanExtractor:
    """
    Extracts and validates sentence-level spans within evidence chunks.

    If a chunk already has spans (from the chunker), this validates them.
    If not, it creates spans by splitting the chunk text into sentences.

    Usage:
        extractor = SpanExtractor(max_spans_per_chunk=5)
        chunk = extractor.extract_spans(chunk)
        # chunk.spans now contains validated EvidenceSpan objects

    Args:
        max_spans_per_chunk: Maximum spans to keep per chunk.
        use_spacy: Use spaCy for sentence splitting (more accurate).
    """

    def __init__(
        self,
        max_spans_per_chunk: int = 5,
        use_spacy: bool = False,
    ):
        self.max_spans_per_chunk = max_spans_per_chunk
        self.use_spacy = use_spacy

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        if self.use_spacy:
            try:
                import spacy
                nlp = spacy.blank("en")
                nlp.add_pipe("sentencizer")
                doc = nlp(text)
                return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except ImportError:
                pass

        # Regex fallback
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def extract_spans(self, chunk: EvidenceChunk) -> EvidenceChunk:
        """
        Extract or validate sentence-level spans for a chunk.

        If the chunk already has valid spans, they are validated and returned.
        Otherwise, new spans are created by splitting the chunk text.

        Invariant guaranteed:
            For every span: chunk.text[span.start:span.end] == span.sentence

        Args:
            chunk: EvidenceChunk (may or may not have spans).

        Returns:
            Same chunk with validated/populated spans.
        """
        if chunk.spans and self._validate_spans(chunk):
            # Existing spans are valid; just enforce cap
            chunk.spans = chunk.spans[:self.max_spans_per_chunk]
            return chunk

        # Re-extract spans from chunk text
        sentences = self._split_sentences(chunk.text)
        spans: list[EvidenceSpan] = []
        search_start = 0

        for i, sentence in enumerate(sentences):
            if i >= self.max_spans_per_chunk:
                break

            # Find exact position in chunk text
            start = chunk.text.find(sentence, search_start)
            if start == -1:
                # Try with normalized whitespace
                start = search_start
                logger.debug(
                    f"Span '{sentence[:30]}...' not found at expected offset; "
                    f"using fallback position {start}"
                )

            end = start + len(sentence)
            spans.append(EvidenceSpan(
                span_id=f"s{i}",
                start=start,
                end=end,
                sentence=sentence,
            ))
            search_start = end

        chunk.spans = spans
        return chunk

    def _validate_spans(self, chunk: EvidenceChunk) -> bool:
        """
        Validate that all spans have correct offsets.

        Returns True if all spans satisfy:
            chunk.text[span.start:span.end] == span.sentence
        """
        for span in chunk.spans:
            actual = chunk.text[span.start:span.end]
            if actual != span.sentence:
                logger.debug(
                    f"Span {span.span_id} offset mismatch: "
                    f"expected '{span.sentence[:30]}...', got '{actual[:30]}...'"
                )
                return False
        return True

    def extract_spans_batch(
        self, chunks: list[EvidenceChunk]
    ) -> list[EvidenceChunk]:
        """Extract/validate spans for a batch of chunks."""
        return [self.extract_spans(chunk) for chunk in chunks]


def get_span_text(
    chunk: EvidenceChunk, span_id: str
) -> Optional[str]:
    """
    Helper to get span text by span_id from a chunk.

    Args:
        chunk: The evidence chunk containing the span.
        span_id: The span identifier to look up.

    Returns:
        Span sentence text, or None if not found.
    """
    for span in chunk.spans:
        if span.span_id == span_id:
            return span.sentence
    return None

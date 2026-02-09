"""
Document Chunker
=================

Sentence-aware document chunking with character offset tracking.

Architecture:
    Raw Text → Sentence Splitter → Chunk Assembler → EvidenceChunk objects

Key Properties:
    1. Chunks never split mid-sentence (sentence-aligned boundaries)
    2. Each sentence becomes an EvidenceSpan with exact character offsets
    3. Offsets are stable: chunk.text[span.start:span.end] == span.sentence
    4. Configurable chunk size (in tokens) with sentence-aware overlap

Two Modes:
    - FULL mode: Uses spaCy's sentencizer for accurate sentence splitting
    - LITE mode: Uses regex-based sentence splitting (no spaCy dependency)
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from certirag.schemas.evidence import EvidenceChunk, EvidenceSpan, SourceInfo

logger = logging.getLogger("certirag.ingest.chunker")


# ── Sentence Splitters ─────────────────────────────────────────────

def _split_sentences_regex(text: str) -> list[str]:
    """
    Regex-based sentence splitter (LITE mode fallback).

    Splits on period/question/exclamation followed by whitespace and
    an uppercase letter. Not perfect, but works for most English text
    without requiring spaCy.

    Args:
        text: Input text to split into sentences.

    Returns:
        List of sentence strings.
    """
    # Split on sentence-ending punctuation followed by space + uppercase
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    # Filter empty strings and strip whitespace
    return [s.strip() for s in sentences if s.strip()]


def _split_sentences_spacy(text: str) -> list[str]:
    """
    spaCy-based sentence splitter (FULL mode, more accurate).

    Uses spaCy's trained sentencizer for accurate sentence boundary
    detection including abbreviations, titles, etc.

    Args:
        text: Input text to split into sentences.

    Returns:
        List of sentence strings.
    """
    try:
        import spacy
        # Use a lightweight pipeline with just the sentencizer
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
            nlp.add_pipe("sentencizer")
        except OSError:
            # Model not downloaded; fall back to blank + sentencizer
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")

        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except ImportError:
        logger.warning("spaCy not available, falling back to regex splitter")
        return _split_sentences_regex(text)


# ── Core Chunker ───────────────────────────────────────────────────

class DocumentChunker:
    """
    Sentence-aware document chunker with offset tracking.

    Splits documents into overlapping chunks where:
    - Each chunk contains complete sentences (never split mid-sentence)
    - Each sentence becomes a tracked EvidenceSpan
    - Character offsets enable exact highlighting in the UI

    Usage:
        chunker = DocumentChunker(chunk_size=256, overlap=32)
        chunks = chunker.chunk_document(
            text="The quick brown fox...",
            doc_id="doc1",
            source=SourceInfo(title="My Document")
        )

    Args:
        chunk_size: Target chunk size in approximate word count.
        overlap: Number of overlap sentences between consecutive chunks.
        use_spacy: Use spaCy sentencizer (True) or regex fallback (False).
    """

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 2,
        use_spacy: bool = False,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.use_spacy = use_spacy

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using configured method."""
        if self.use_spacy:
            return _split_sentences_spacy(text)
        return _split_sentences_regex(text)

    def _estimate_word_count(self, text: str) -> int:
        """Rough word count estimation."""
        return len(text.split())

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        source: Optional[SourceInfo] = None,
    ) -> list[EvidenceChunk]:
        """
        Split a document into sentence-aligned chunks with span tracking.

        Algorithm:
            1. Split text into sentences
            2. Greedily accumulate sentences into a chunk until chunk_size is reached
            3. Create EvidenceSpan for each sentence with character offsets
            4. Advance with overlap (re-include last N sentences in next chunk)

        Args:
            text: Full document text.
            doc_id: Unique document identifier.
            source: Document provenance metadata.

        Returns:
            List of EvidenceChunk objects with populated spans.

        Example:
            >>> chunker = DocumentChunker(chunk_size=50)
            >>> chunks = chunker.chunk_document("First sentence. Second.", "doc1")
            >>> chunks[0].spans[0].sentence
            'First sentence.'
        """
        if not text.strip():
            return []

        source = source or SourceInfo(title=doc_id)
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks: list[EvidenceChunk] = []
        chunk_idx = 0
        sent_start = 0

        while sent_start < len(sentences):
            # Accumulate sentences until we hit the chunk size target
            chunk_sentences: list[str] = []
            word_count = 0

            sent_end = sent_start
            while sent_end < len(sentences):
                sent_words = self._estimate_word_count(sentences[sent_end])
                if word_count + sent_words > self.chunk_size and chunk_sentences:
                    break  # Don't exceed chunk_size (but always include ≥1 sentence)
                chunk_sentences.append(sentences[sent_end])
                word_count += sent_words
                sent_end += 1

            # Build chunk text and spans with accurate offsets
            chunk_text = " ".join(chunk_sentences)
            spans: list[EvidenceSpan] = []
            current_offset = 0

            for i, sentence in enumerate(chunk_sentences):
                # Find the exact position of this sentence in the chunk text
                start = chunk_text.find(sentence, current_offset)
                if start == -1:
                    start = current_offset  # fallback

                end = start + len(sentence)
                spans.append(EvidenceSpan(
                    span_id=f"s{i}",
                    start=start,
                    end=end,
                    sentence=sentence,
                ))
                current_offset = end

            chunk = EvidenceChunk(
                chunk_id=f"{doc_id}#c{chunk_idx}",
                doc_id=doc_id,
                text=chunk_text,
                source=source,
                spans=spans,
            )
            chunks.append(chunk)
            chunk_idx += 1

            # Advance with overlap
            if sent_end >= len(sentences):
                break
            sent_start = max(sent_start + 1, sent_end - self.overlap)

        logger.info(
            f"Chunked document '{doc_id}': {len(sentences)} sentences → "
            f"{len(chunks)} chunks"
        )
        return chunks

    def chunk_documents(
        self,
        documents: list[dict[str, str]],
    ) -> list[EvidenceChunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of dicts with keys: 'text', 'doc_id',
                       and optionally 'title', 'url', 'section'.

        Returns:
            Flat list of all EvidenceChunk objects across all documents.
        """
        all_chunks: list[EvidenceChunk] = []
        for doc in documents:
            source = SourceInfo(
                title=doc.get("title", doc["doc_id"]),
                url_or_path=doc.get("url", ""),
                section=doc.get("section", ""),
            )
            chunks = self.chunk_document(
                text=doc["text"],
                doc_id=doc["doc_id"],
                source=source,
            )
            all_chunks.extend(chunks)

        logger.info(f"Chunked {len(documents)} documents → {len(all_chunks)} total chunks")
        return all_chunks

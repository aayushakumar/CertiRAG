"""
Evidence Schema (§6.1)
=======================

Defines the structure for retrieved evidence chunks, including:
- Document provenance (doc_id, source metadata)
- Sentence-level spans with character offsets
- Retrieval scores (BM25, dense, reranker)

Design Decisions:
    - Spans use character offsets (not token offsets) for UI highlighting
    - Each span is a complete sentence to ensure coherent evidence display
    - Retrieval scores are preserved for debugging and ablation analysis

Data Flow:
    Ingestion → EvidenceChunk (with spans) → Retriever → Verifier → Certificate
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SourceInfo(BaseModel):
    """
    Provenance metadata for a source document.

    Tracks where a chunk came from, enabling audit trails
    back to the original document.
    """
    title: str = Field(description="Document title")
    url_or_path: str = Field(default="", description="URL or file path to source")
    section: str = Field(default="", description="Section/heading within document")


class RetrievalScores(BaseModel):
    """
    Retrieval scores for a chunk, preserved for debugging.

    These scores are NOT used by the verifier — they're stored
    for ablation analysis and debugging retrieval quality.
    """
    bm25: float = Field(default=0.0, description="BM25 score")
    dense: float = Field(default=0.0, description="Dense (cosine) similarity score")
    rerank: float = Field(default=0.0, description="Cross-encoder reranker score")
    rrf: float = Field(default=0.0, description="Reciprocal Rank Fusion combined score")


class EvidenceSpan(BaseModel):
    """
    A sentence-level span within an evidence chunk.

    Spans are the atomic unit of evidence. The verifier checks
    claims against individual spans (not whole chunks), enabling
    fine-grained evidence attribution.

    Invariant:
        chunk.text[start:end] == sentence
        (enforced by model_validator)
    """
    span_id: str = Field(description="Unique span identifier (format: 's{N}')")
    start: int = Field(ge=0, description="Start character offset within chunk text")
    end: int = Field(gt=0, description="End character offset within chunk text (exclusive)")
    sentence: str = Field(description="The verbatim sentence text")

    @model_validator(mode="after")
    def validate_offsets(self) -> "EvidenceSpan":
        """Ensure start < end and sentence length matches offset range."""
        if self.start >= self.end:
            raise ValueError(f"Span start ({self.start}) must be < end ({self.end})")
        expected_len = self.end - self.start
        actual_len = len(self.sentence)
        if abs(expected_len - actual_len) > 2:  # allow minor whitespace variance
            raise ValueError(
                f"Span length mismatch: offsets suggest {expected_len} chars, "
                f"but sentence has {actual_len} chars"
            )
        return self


class EvidenceChunk(BaseModel):
    """
    A retrieved evidence chunk with full provenance.

    This is the primary data structure flowing through the retrieval
    pipeline. Each chunk contains:
    - The chunk text (typically 1-5 sentences)
    - Source provenance (document, section, URL)
    - Sentence-level spans with stable offsets
    - Retrieval scores for debugging

    Schema (§6.1):
        {
          "chunk_id": "doc42#c17",
          "doc_id": "doc42",
          "text": "...",
          "source": {"title": "...", "url_or_path": "...", "section": "..."},
          "spans": [{"span_id": "s1", "start": 120, "end": 212, "sentence": "..."}],
          "retrieval": {"bm25": 12.3, "dense": 0.71, "rerank": 0.62}
        }
    """
    chunk_id: str = Field(description="Unique chunk ID (format: '{doc_id}#c{N}')")
    doc_id: str = Field(description="Parent document ID")
    text: str = Field(description="Full chunk text")
    source: SourceInfo = Field(default_factory=SourceInfo, description="Document provenance")
    spans: list[EvidenceSpan] = Field(
        default_factory=list,
        description="Sentence-level spans within this chunk"
    )
    retrieval: RetrievalScores = Field(
        default_factory=RetrievalScores,
        description="Retrieval scores (for debugging, not used by verifier)"
    )

    @model_validator(mode="after")
    def validate_spans_within_text(self) -> "EvidenceChunk":
        """Ensure all spans fall within the chunk text boundaries."""
        for span in self.spans:
            if span.end > len(self.text):
                raise ValueError(
                    f"Span {span.span_id} end ({span.end}) exceeds "
                    f"chunk text length ({len(self.text)})"
                )
        return self

    @property
    def num_spans(self) -> int:
        """Number of sentence-level spans in this chunk."""
        return len(self.spans)

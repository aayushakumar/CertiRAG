"""
Document Indexer
=================

Builds and manages search indices for hybrid retrieval:
- BM25 index (keyword matching) via rank_bm25
- Dense vector index (semantic matching) via FAISS or numpy fallback

Two Modes:
    - FULL mode: FAISS index (IndexFlatIP for exact search)
    - LITE mode: Numpy-based brute-force cosine similarity (no FAISS needed)

Both modes store chunk metadata in a sidecar Python dict for lookup.

Data Flow:
    EvidenceChunks + Embeddings → Indexer → Persisted indices → Retriever
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from certirag.schemas.evidence import EvidenceChunk

logger = logging.getLogger("certirag.ingest.indexer")


class BM25Index:
    """
    BM25 keyword index over chunked documents.

    Uses rank_bm25's BM25Okapi implementation for efficient
    keyword-based retrieval. Tokens are lowercased and split
    on whitespace (simple but effective for English text).

    Usage:
        idx = BM25Index()
        idx.build(chunks)
        scores = idx.query("what is transformer?", top_k=10)
        # scores: list of (chunk_id, score) tuples
    """

    def __init__(self):
        self._bm25 = None
        self._chunk_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()

    def build(self, chunks: list[EvidenceChunk]) -> None:
        """
        Build the BM25 index over a list of evidence chunks.

        Args:
            chunks: List of EvidenceChunk objects to index.
        """
        from rank_bm25 import BM25Okapi

        self._chunk_ids = [c.chunk_id for c in chunks]
        self._tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"Built BM25 index over {len(chunks)} chunks")

    def query(self, query_text: str, top_k: int = 50) -> list[tuple[str, float]]:
        """
        Query the BM25 index.

        Args:
            query_text: Query string.
            top_k: Number of results to return.

        Returns:
            List of (chunk_id, bm25_score) tuples, sorted by score descending.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokenized_query = self._tokenize(query_text)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices (return all with non-trivial scores)
        top_indices = np.argsort(scores)[::-1][:top_k]
        # BM25 scores can be negative for small corpora (IDF issue)
        # Use relative threshold: exclude only truly zero-relevance docs
        threshold = -1e9 if len(self._chunk_ids) < 10 else 0.0
        results = [
            (self._chunk_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > threshold
        ]
        return results

    def save(self, path: Path) -> None:
        """Save BM25 index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "chunk_ids": self._chunk_ids,
                "tokenized_corpus": self._tokenized_corpus,
            }, f)
        logger.info(f"Saved BM25 index to {path}")

    def load(self, path: Path) -> None:
        """Load BM25 index from disk."""
        from rank_bm25 import BM25Okapi

        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunk_ids = data["chunk_ids"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"Loaded BM25 index from {path} ({len(self._chunk_ids)} chunks)")


class DenseIndex:
    """
    Dense vector index for semantic retrieval.

    Two backends:
    - FAISS (FULL mode): Fast, GPU-capable, scalable
    - numpy (LITE mode): Simple brute-force, no extra dependencies

    Both use inner product (= cosine similarity for L2-normalized vectors).

    Usage:
        idx = DenseIndex(mode="lite")
        idx.build(chunk_ids, embeddings)
        results = idx.query(query_vector, top_k=10)
    """

    def __init__(self, mode: str = "lite"):
        self.mode = mode
        self._chunk_ids: list[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._faiss_index = None

    def build(self, chunk_ids: list[str], embeddings: np.ndarray) -> None:
        """
        Build the dense index.

        Args:
            chunk_ids: List of chunk IDs corresponding to embedding rows.
            embeddings: Numpy array of shape (n_chunks, dimension), L2-normalized.
        """
        self._chunk_ids = chunk_ids
        self._embeddings = embeddings.astype(np.float32)

        if self.mode == "full":
            self._build_faiss(embeddings)
        else:
            logger.info(
                f"Built numpy dense index: {len(chunk_ids)} chunks, "
                f"dim={embeddings.shape[1]}"
            )

    def _build_faiss(self, embeddings: np.ndarray) -> None:
        """Build FAISS index (FULL mode)."""
        try:
            import faiss

            dimension = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dimension)  # inner product
            self._faiss_index.add(embeddings.astype(np.float32))
            logger.info(
                f"Built FAISS IndexFlatIP: {embeddings.shape[0]} vectors, "
                f"dim={dimension}"
            )
        except ImportError:
            logger.warning("FAISS not available, falling back to numpy")
            self.mode = "lite"

    def query(
        self, query_embedding: np.ndarray, top_k: int = 50
    ) -> list[tuple[str, float]]:
        """
        Query the dense index.

        Args:
            query_embedding: Query vector of shape (1, dimension), L2-normalized.
            top_k: Number of results to return.

        Returns:
            List of (chunk_id, similarity_score) tuples, sorted by score descending.
        """
        if self._embeddings is None and self._faiss_index is None:
            raise RuntimeError("Dense index not built. Call build() first.")

        query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if self.mode == "full" and self._faiss_index is not None:
            return self._query_faiss(query_embedding, top_k)
        else:
            return self._query_numpy(query_embedding, top_k)

    def _query_faiss(
        self, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[str, float]]:
        """Query using FAISS (FULL mode)."""
        import faiss  # noqa: F811

        scores, indices = self._faiss_index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS uses -1 for padding
                results.append((self._chunk_ids[idx], float(score)))
        return results

    def _query_numpy(
        self, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[str, float]]:
        """Query using numpy brute-force (LITE mode)."""
        # Cosine similarity via dot product (vectors are L2-normalized)
        similarities = np.dot(self._embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            (self._chunk_ids[i], float(similarities[i]))
            for i in top_indices
        ]

    def save(self, path: Path) -> None:
        """Save dense index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.mode == "full" and self._faiss_index is not None:
            import faiss
            faiss.write_index(self._faiss_index, str(path.with_suffix(".faiss")))

        # Always save metadata + embeddings
        with open(path.with_suffix(".meta"), "wb") as f:
            pickle.dump({
                "chunk_ids": self._chunk_ids,
                "embeddings": self._embeddings,
                "mode": self.mode,
            }, f)
        logger.info(f"Saved dense index to {path}")

    def load(self, path: Path) -> None:
        """Load dense index from disk."""
        with open(path.with_suffix(".meta"), "rb") as f:
            data = pickle.load(f)
        self._chunk_ids = data["chunk_ids"]
        self._embeddings = data["embeddings"]
        self.mode = data.get("mode", self.mode)

        if self.mode == "full":
            try:
                import faiss
                faiss_path = path.with_suffix(".faiss")
                if faiss_path.exists():
                    self._faiss_index = faiss.read_index(str(faiss_path))
                else:
                    self._build_faiss(self._embeddings)
            except ImportError:
                logger.warning("FAISS not available, using numpy fallback")
                self.mode = "lite"

        logger.info(f"Loaded dense index from {path} ({len(self._chunk_ids)} chunks)")


class ChunkStore:
    """
    Metadata store for EvidenceChunk objects.

    Provides fast lookup by chunk_id and serialization to disk.
    This is the "sidecar" that accompanies the search indices.
    """

    def __init__(self):
        self._chunks: dict[str, EvidenceChunk] = {}

    def add(self, chunk: EvidenceChunk) -> None:
        """Add a chunk to the store."""
        self._chunks[chunk.chunk_id] = chunk

    def add_many(self, chunks: list[EvidenceChunk]) -> None:
        """Add multiple chunks to the store."""
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def get(self, chunk_id: str) -> Optional[EvidenceChunk]:
        """Look up a chunk by ID. Returns None if not found."""
        return self._chunks.get(chunk_id)

    def get_many(self, chunk_ids: list[str]) -> list[EvidenceChunk]:
        """Look up multiple chunks by ID, preserving order."""
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def get_span(self, chunk_id: str, span_id: str) -> Optional[str]:
        """
        Get the text of a specific span within a chunk.

        Args:
            chunk_id: The chunk identifier.
            span_id: The span identifier within the chunk.

        Returns:
            Span sentence text, or None if not found.
        """
        chunk = self.get(chunk_id)
        if chunk is None:
            return None
        for span in chunk.spans:
            if span.span_id == span_id:
                return span.sentence
        return None

    @property
    def size(self) -> int:
        """Number of chunks in the store."""
        return len(self._chunks)

    def all_chunk_ids(self) -> list[str]:
        """Return all chunk IDs."""
        return list(self._chunks.keys())

    def save(self, path: Path) -> None:
        """Save chunk store to disk as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {cid: chunk.model_dump() for cid, chunk in self._chunks.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved chunk store to {path} ({self.size} chunks)")

    def load(self, path: Path) -> None:
        """Load chunk store from disk."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._chunks = {
            cid: EvidenceChunk.model_validate(chunk_data)
            for cid, chunk_data in data.items()
        }
        logger.info(f"Loaded chunk store from {path} ({self.size} chunks)")

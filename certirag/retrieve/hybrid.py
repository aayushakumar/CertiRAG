"""
Hybrid Retriever
=================

Combines BM25 (keyword) and dense (semantic) retrieval using
Reciprocal Rank Fusion (RRF), then applies cross-encoder reranking.

Architecture:
    Query → [BM25 top-k, Dense top-k] → RRF Fusion → Reranker → top-k chunks

Key Design Decisions:
    - RRF is used over linear interpolation because it's rank-based
      (no score normalization needed between BM25 and dense)
    - Reranking uses a cross-encoder for high accuracy on the fused set
    - Evidence spans are extracted at the sentence level for fine-grained
      verification (the verifier checks spans, not whole chunks)

Two Modes:
    - FULL mode: Cross-encoder reranker (bge-reranker-base)
    - LITE mode: No reranking (RRF scores used directly)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from certirag.config import CertiRAGConfig
from certirag.ingest.embedder import DocumentEmbedder
from certirag.ingest.indexer import BM25Index, ChunkStore, DenseIndex
from certirag.schemas.evidence import EvidenceChunk, RetrievalScores

logger = logging.getLogger("certirag.retrieve.hybrid")


class HybridRetriever:
    """
    Hybrid BM25 + dense retrieval with optional cross-encoder reranking.

    Pipeline:
        1. BM25 retrieval → top-k_bm25 candidates
        2. Dense retrieval → top-k_dense candidates
        3. Reciprocal Rank Fusion (RRF) → combined ranking
        4. Cross-encoder reranking (FULL mode) → final top-k

    Usage:
        retriever = HybridRetriever(config, bm25_index, dense_index,
                                     embedder, chunk_store)
        results = retriever.retrieve("What is transformer architecture?")

    Args:
        config: CertiRAG configuration.
        bm25_index: Built BM25 index.
        dense_index: Built dense vector index.
        embedder: Document embedder (for query embedding).
        chunk_store: Chunk metadata store.
    """

    def __init__(
        self,
        config: CertiRAGConfig,
        bm25_index: BM25Index,
        dense_index: DenseIndex,
        embedder: DocumentEmbedder,
        chunk_store: ChunkStore,
    ):
        self.config = config
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.embedder = embedder
        self.chunk_store = chunk_store
        self._reranker = None

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[tuple[str, float]],
        dense_results: list[tuple[str, float]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """
        Combine BM25 and dense rankings using Reciprocal Rank Fusion.

        RRF score = Σ 1/(k + rank_i) for each ranking system i.

        This is preferred over linear interpolation because:
        - It's rank-based (no score normalization needed)
        - It's robust to score distribution differences between systems
        - It has a single hyperparameter (k) that's easy to tune

        Args:
            bm25_results: BM25 results as (chunk_id, score) tuples.
            dense_results: Dense results as (chunk_id, score) tuples.
            k: RRF constant (default 60, standard in literature).

        Returns:
            Fused ranking as (chunk_id, rrf_score) tuples, sorted descending.
        """
        rrf_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        dense_scores: dict[str, float] = {}

        # BM25 contribution
        for rank, (chunk_id, score) in enumerate(bm25_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            bm25_scores[chunk_id] = score

        # Dense contribution
        for rank, (chunk_id, score) in enumerate(dense_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            dense_scores[chunk_id] = score

        # Sort by RRF score descending
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Update chunk retrieval scores for debugging
        for chunk_id, rrf_score in sorted_results:
            chunk = self.chunk_store.get(chunk_id)
            if chunk:
                chunk.retrieval = RetrievalScores(
                    bm25=bm25_scores.get(chunk_id, 0.0),
                    dense=dense_scores.get(chunk_id, 0.0),
                    rrf=rrf_score,
                )

        return sorted_results

    def _rerank(
        self, query: str, chunk_ids: list[str], top_k: int
    ) -> list[tuple[str, float]]:
        """
        Rerank candidates using a cross-encoder (FULL mode only).

        The cross-encoder scores each (query, chunk_text) pair directly,
        producing more accurate relevance scores than bi-encoder similarity.

        Args:
            query: Query text.
            chunk_ids: Candidate chunk IDs to rerank.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked (chunk_id, reranker_score) tuples.
        """
        if not self.config.is_full:
            # LITE mode: no reranking, return as-is
            return [(cid, 0.0) for cid in chunk_ids[:top_k]]

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            if self._reranker is None:
                model_name = self.config.retrieval.reranker_model
                logger.info(f"Loading reranker: {model_name}")
                self._reranker = {
                    "tokenizer": AutoTokenizer.from_pretrained(model_name),
                    "model": AutoModelForSequenceClassification.from_pretrained(model_name),
                }
                if torch.cuda.is_available():
                    self._reranker["model"] = self._reranker["model"].cuda()
                self._reranker["model"].eval()

            # Score each (query, chunk) pair
            chunks = self.chunk_store.get_many(chunk_ids)
            pairs = [(query, chunk.text) for chunk in chunks]

            tokenizer = self._reranker["tokenizer"]
            model = self._reranker["model"]

            scores = []
            batch_size = 16
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                inputs = tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                    if batch_scores.ndim == 0:
                        batch_scores = np.array([float(batch_scores)])
                    scores.extend(batch_scores.tolist())

            # Sort by reranker score
            scored = list(zip(chunk_ids[:len(scores)], scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            # Update retrieval scores
            for chunk_id, rerank_score in scored:
                chunk = self.chunk_store.get(chunk_id)
                if chunk:
                    chunk.retrieval.rerank = rerank_score

            return scored[:top_k]

        except ImportError:
            logger.warning("transformers not available, skipping reranking")
            return [(cid, 0.0) for cid in chunk_ids[:top_k]]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[EvidenceChunk]:
        """
        Retrieve evidence chunks for a query using hybrid retrieval.

        Full Pipeline:
            1. BM25 retrieval → top-50 by keyword match
            2. Dense retrieval → top-50 by semantic similarity
            3. RRF fusion → combined ranking
            4. Reranking (FULL mode) → refined top-k
            5. Return EvidenceChunk objects with populated retrieval scores

        Args:
            query: User's question text.
            top_k: Override for max chunks to return (default: config value).

        Returns:
            List of EvidenceChunk objects, sorted by relevance.
            Each chunk has populated retrieval scores and sentence spans.
        """
        top_k = top_k or self.config.retrieval.max_chunks
        rc = self.config.retrieval

        logger.info(f"Retrieving for query: '{query[:80]}...'")

        # Step 1: BM25 retrieval
        bm25_results = self.bm25_index.query(query, top_k=rc.top_k_bm25)
        logger.debug(f"BM25 returned {len(bm25_results)} candidates")

        # Step 2: Dense retrieval
        query_embedding = self.embedder.embed_query(query)
        dense_results = self.dense_index.query(query_embedding, top_k=rc.top_k_dense)
        logger.debug(f"Dense returned {len(dense_results)} candidates")

        # Step 3: RRF fusion
        fused_results = self._reciprocal_rank_fusion(
            bm25_results, dense_results, k=rc.rrf_k
        )
        logger.debug(f"RRF fused {len(fused_results)} unique candidates")

        # Step 4: Reranking (FULL mode uses cross-encoder)
        candidate_ids = [cid for cid, _ in fused_results[:rc.top_k_rerank * 3]]
        if self.config.is_full and len(candidate_ids) > 0:
            reranked = self._rerank(query, candidate_ids, top_k=rc.top_k_rerank)
            final_ids = [cid for cid, _ in reranked[:top_k]]
        else:
            final_ids = [cid for cid, _ in fused_results[:top_k]]

        # Step 5: Return EvidenceChunks
        chunks = self.chunk_store.get_many(final_ids)

        # Enforce hard cap on spans per chunk
        for chunk in chunks:
            if len(chunk.spans) > rc.max_spans_per_chunk:
                chunk.spans = chunk.spans[:rc.max_spans_per_chunk]

        logger.info(f"Retrieved {len(chunks)} chunks for query")
        return chunks


class SimpleRetriever:
    """
    Simple BM25-only retriever for LITE mode / testing.

    This is used when dense embeddings are not available (e.g., quick
    testing in Codespaces without any embedding model).

    Usage:
        retriever = SimpleRetriever(bm25_index, chunk_store)
        results = retriever.retrieve("What is X?", top_k=5)
    """

    def __init__(self, bm25_index: BM25Index, chunk_store: ChunkStore):
        self.bm25_index = bm25_index
        self.chunk_store = chunk_store

    def retrieve(self, query: str, top_k: int = 10) -> list[EvidenceChunk]:
        """Retrieve chunks using BM25 only."""
        results = self.bm25_index.query(query, top_k=top_k)
        chunk_ids = [cid for cid, _ in results]
        chunks = self.chunk_store.get_many(chunk_ids)

        # Update BM25 scores on chunks
        score_map = dict(results)
        for chunk in chunks:
            chunk.retrieval = RetrievalScores(bm25=score_map.get(chunk.chunk_id, 0.0))

        return chunks

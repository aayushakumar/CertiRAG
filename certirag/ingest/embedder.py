"""
Document Embedder
==================

Dense embedding generation for document chunks.

Two Modes:
    - FULL mode: Uses sentence-transformers (e5-base-v2) for local GPU embedding
    - LITE mode: Uses OpenAI embeddings API (text-embedding-3-small)

All embeddings are L2-normalized such that cosine similarity = dot product.

Data Flow:
    EvidenceChunk.text → Embedder → numpy array → FAISS index
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("certirag.ingest.embedder")


class DocumentEmbedder:
    """
    Generates dense vector embeddings for document chunks.

    Supports both local (sentence-transformers) and API-based (OpenAI)
    embedding models. Embeddings are L2-normalized for cosine similarity.

    Usage:
        # LITE mode (API-based)
        embedder = DocumentEmbedder(mode="lite", api_key="sk-...")
        vectors = embedder.embed(["Hello world", "Foo bar"])

        # FULL mode (local GPU)
        embedder = DocumentEmbedder(mode="full", model_name="intfloat/e5-base-v2")
        vectors = embedder.embed(["Hello world", "Foo bar"])

    Args:
        mode: "lite" (API) or "full" (local model).
        model_name: HuggingFace model ID (full mode) or OpenAI model name.
        api_key: OpenAI API key (lite mode only).
        batch_size: Batch size for encoding.
        device: PyTorch device ("cuda", "cpu", "auto").
    """

    def __init__(
        self,
        mode: str = "lite",
        model_name: str = "intfloat/e5-base-v2",
        api_key: Optional[str] = None,
        batch_size: int = 64,
        device: str = "auto",
    ):
        self.mode = mode
        self.model_name = model_name
        self.api_key = api_key
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        if self._dimension is not None:
            return self._dimension

        if self.mode == "lite":
            # OpenAI text-embedding-3-small default
            self._dimension = 1536
        else:
            # Load model to get dimension
            self._load_model()
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._dimension

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model (FULL mode only)."""
        if self._model is not None:
            return

        if self.mode == "full":
            try:
                from sentence_transformers import SentenceTransformer

                device = self.device
                if device == "auto":
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                logger.info(f"Loading embedding model: {self.model_name} on {device}")
                self._model = SentenceTransformer(self.model_name, device=device)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Embedding model loaded. Dimension: {self._dimension}")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers required for FULL mode. "
                    "Install with: pip install sentence-transformers"
                )

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), dimension), L2-normalized.

        Raises:
            RuntimeError: If the embedding backend is not available.
        """
        if not texts:
            return np.array([])

        if self.mode == "lite":
            return self._embed_openai(texts)
        else:
            return self._embed_local(texts)

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API (LITE mode)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai package required for LITE mode. "
                "Install with: pip install openai"
            )

        if not self.api_key:
            raise ValueError("OpenAI API key required for LITE mode embedding")

        client = OpenAI(api_key=self.api_key)
        all_embeddings = []

        # Batch API calls
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.debug(f"Embedding batch {i // self.batch_size + 1}")

            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        embeddings = np.array(all_embeddings, dtype=np.float32)
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # avoid division by zero
        embeddings = embeddings / norms

        self._dimension = embeddings.shape[1]
        return embeddings

    def _embed_local(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers (FULL mode)."""
        self._load_model()

        # e5 models expect "passage: " prefix for documents
        if "e5" in self.model_name.lower():
            texts = [f"passage: {t}" for t in texts]

        logger.info(f"Embedding {len(texts)} texts with batch_size={self.batch_size}")
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # L2 normalization
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query text.

        For e5 models, adds the "query: " prefix (different from document prefix).

        Args:
            query: Query text string.

        Returns:
            numpy array of shape (1, dimension), L2-normalized.
        """
        if self.mode == "full" and "e5" in self.model_name.lower():
            text = f"query: {query}"
        else:
            text = query

        if self.mode == "lite":
            return self._embed_openai([text])
        else:
            self._load_model()
            emb = self._model.encode([text], normalize_embeddings=True)
            return np.array(emb, dtype=np.float32)

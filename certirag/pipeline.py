"""
CertiRAG End-to-End Pipeline
==============================

Orchestrates the full CertiRAG pipeline:
    Query → Retrieve → Compile Claims → Normalize → Verify → MSE → Render → Certificate

This is the single entry point for running CertiRAG on a query.
It manages component initialization, execution ordering, timing,
and error handling.

Two Modes:
    - LITE: CPU/API mode for development (Codespaces)
    - FULL: GPU mode for benchmarking (Colab Pro / local GPU)

Usage:
    from certirag.pipeline import CertiRAGPipeline

    pipeline = CertiRAGPipeline.from_config(config)
    pipeline.ingest(documents)               # one-time setup
    result = pipeline.run("What is X?")       # per-query
    print(result["visible_claims"])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from certirag.config import CertiRAGConfig, get_config
from certirag.schemas.certificate import AuditCertificate
from certirag.schemas.claim_ir import ClaimIR
from certirag.schemas.evidence import EvidenceChunk

logger = logging.getLogger("certirag.pipeline")


@dataclass
class PipelineResult:
    """
    Complete output of a CertiRAG pipeline run.

    Contains everything needed for display, debugging, and auditing.
    """
    question: str
    claim_ir: ClaimIR
    evidence_chunks: list[EvidenceChunk]
    verification_results: list
    renderer_decisions: list
    certificate: Optional[AuditCertificate]
    display: dict[str, Any]  # filtered claims for UI
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def verified_claims(self) -> list[dict]:
        """Claims that passed verification."""
        return self.display.get("visible_claims", [])

    @property
    def hidden_claims(self) -> list[dict]:
        """Claims that were hidden/blocked."""
        return self.display.get("hidden_claims", [])

    @property
    def stats(self) -> dict:
        """Summary statistics."""
        return self.display.get("stats", {})


class CertiRAGPipeline:
    """
    End-to-end CertiRAG pipeline orchestrator.

    Manages the complete flow from query to verified answer:
        1. Retrieve evidence chunks (hybrid BM25 + dense)
        2. Compile claims (LLM → ClaimIR)
        3. Normalize claims (split, deduplicate, clean)
        4. Verify claims (MiniCheck / NLI / LLM-judge)
        5. Select MSE (Minimal Sufficient Evidence)
        6. Render decisions (policy engine)
        7. Build certificate (audit trail)

    Usage:
        pipeline = CertiRAGPipeline(config)
        pipeline.ingest([{"doc_id": "d1", "text": "...", "title": "..."}])
        result = pipeline.run("What is the capital of France?")

    Args:
        config: CertiRAG configuration.
    """

    def __init__(self, config: Optional[CertiRAGConfig] = None):
        self.config = config or get_config()
        self.config.ensure_dirs()
        self._is_ingested = False

        # Components (lazy-initialized)
        self._chunker = None
        self._embedder = None
        self._bm25_index = None
        self._dense_index = None
        self._chunk_store = None
        self._retriever = None
        self._compiler = None
        self._normalizer = None
        self._verifier = None
        self._mse_selector = None
        self._renderer = None
        self._cert_builder = None

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> "CertiRAGPipeline":
        """Create pipeline from config file or environment."""
        config = get_config(config_path)
        return cls(config)

    def _init_components(self) -> None:
        """Initialize all pipeline components based on config."""
        from certirag.ingest.chunker import DocumentChunker
        from certirag.ingest.indexer import BM25Index, ChunkStore, DenseIndex
        from certirag.claim_ir.compiler import ClaimCompiler
        from certirag.claim_ir.normalizer import ClaimNormalizer
        from certirag.verify.mse import MSESelector
        from certirag.render.policy import RendererPolicy
        from certirag.render.certificate import CertificateBuilder

        # Chunker
        self._chunker = DocumentChunker(
            chunk_size=self.config.retrieval.chunk_size,
            overlap=2,
            use_spacy=self.config.is_full,
        )

        # Indexing
        self._bm25_index = BM25Index()
        self._dense_index = DenseIndex(mode=self.config.mode.value)
        self._chunk_store = ChunkStore()

        # Embedder (only for FULL mode or if API key available)
        if self.config.is_full or self.config.openai_api_key:
            from certirag.ingest.embedder import DocumentEmbedder
            self._embedder = DocumentEmbedder(
                mode=self.config.mode.value,
                model_name=self.config.retrieval.embedding_model,
                api_key=self.config.openai_api_key,
            )

        # Claim compiler — prefer Gemini (free), fall back to OpenAI
        if self.config.gemini_api_key:
            from certirag.claim_ir.gemini_compiler import GeminiClaimCompiler
            self._compiler = GeminiClaimCompiler(self.config)
        else:
            from certirag.claim_ir.compiler import ClaimCompiler
            self._compiler = ClaimCompiler(self.config)

        # Normalizer
        self._normalizer = ClaimNormalizer(
            split_conjunctions=self.config.claim.split_conjunctions,
            remove_hedges=self.config.claim.remove_hedges,
            max_claim_tokens=self.config.claim.max_claim_tokens,
        )

        # Verifier
        self._init_verifier()

        # MSE selector
        self._mse_selector = MSESelector(
            tau_entail=self.config.verification.tau_entail,
        )

        # Renderer
        self._renderer = RendererPolicy.from_config(self.config)

        # Certificate builder
        self._cert_builder = CertificateBuilder(self.config)

        logger.info(f"Pipeline initialized in {self.config.mode.value} mode")

    def _init_verifier(self) -> None:
        """Initialize the appropriate verifier based on config.

        Priority order:
            1. FULL mode → MiniCheck GPU model
            2. Gemini API key → Gemini LLM-as-judge (free tier)
            3. OpenAI API key → GPT-4o LLM-as-judge
            4. torch available → Local HF NLI model (CPU, no API key)
            5. Fallback → MiniCheck lite (word-overlap heuristic)
        """
        if self.config.is_full:
            from certirag.verify.minicheck import MiniCheckVerifier
            self._verifier = MiniCheckVerifier(
                model_name=self.config.verification.verifier_model,
                mode="full",
                batch_size=self.config.verification.verifier_batch_size,
                max_pairs=self.config.verification.max_verifier_pairs,
            )
        elif self.config.gemini_api_key:
            from certirag.verify.gemini_verifier import GeminiVerifier
            self._verifier = GeminiVerifier(
                api_key=self.config.gemini_api_key,
                model=self.config.gemini_model,
                max_pairs=self.config.verification.max_verifier_pairs,
            )
        elif self.config.openai_api_key:
            from certirag.verify.llm_judge import LLMJudgeVerifier
            self._verifier = LLMJudgeVerifier(
                api_key=self.config.openai_api_key,
                model=self.config.openai_model,
                max_pairs=self.config.verification.max_verifier_pairs,
            )
        else:
            # Try local HF NLI model (no API key needed)
            try:
                import torch  # noqa: F401
                from certirag.verify.hf_nli_verifier import HFNLIVerifier
                self._verifier = HFNLIVerifier(
                    max_pairs=self.config.verification.max_verifier_pairs,
                )
                logger.info("Using local HF NLI verifier (CPU, no API key)")
            except ImportError:
                from certirag.verify.minicheck import MiniCheckVerifier
                self._verifier = MiniCheckVerifier(
                    mode="lite",
                    max_pairs=self.config.verification.max_verifier_pairs,
                )

    def ingest(self, documents: list[dict[str, str]]) -> None:
        """
        Ingest documents into the retrieval index.

        Args:
            documents: List of dicts with keys:
                - doc_id (required): Unique document identifier
                - text (required): Document text content
                - title (optional): Document title
                - url (optional): Source URL
                - section (optional): Section heading
        """
        if not self._chunker:
            self._init_components()

        logger.info(f"Ingesting {len(documents)} documents...")
        t0 = time.time()

        # Chunk documents
        all_chunks = self._chunker.chunk_documents(documents)
        self._chunk_store.add_many(all_chunks)

        # Build BM25 index
        self._bm25_index.build(all_chunks)

        # Build dense index (if embedder available)
        if self._embedder:
            texts = [c.text for c in all_chunks]
            embeddings = self._embedder.embed(texts)
            chunk_ids = [c.chunk_id for c in all_chunks]
            self._dense_index.build(chunk_ids, embeddings)

        self._is_ingested = True
        elapsed = time.time() - t0
        logger.info(
            f"Ingestion complete: {len(documents)} docs → "
            f"{len(all_chunks)} chunks in {elapsed:.1f}s"
        )

    def run(
        self,
        question: str,
        mode: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run the full CertiRAG pipeline on a question.

        Steps:
            1. Retrieve evidence
            2. Compile claims
            3. Normalize claims
            4. Verify claims
            5. Select MSE
            6. Render decisions
            7. Build certificate

        Args:
            question: User's question.
            mode: Override display mode ("strict", "mixed", "debug").

        Returns:
            PipelineResult with all outputs.

        Raises:
            RuntimeError: If documents haven't been ingested yet.
        """
        if not self._is_ingested:
            raise RuntimeError("No documents ingested. Call pipeline.ingest() first.")

        mode = mode or self.config.render.default_mode
        timings: dict[str, float] = {}
        total_start = time.time()

        # ── Step 1: Retrieve ───────────────────────────────────────
        t0 = time.time()
        if self._embedder:
            from certirag.retrieve.hybrid import HybridRetriever
            retriever = HybridRetriever(
                config=self.config,
                bm25_index=self._bm25_index,
                dense_index=self._dense_index,
                embedder=self._embedder,
                chunk_store=self._chunk_store,
            )
            evidence_chunks = retriever.retrieve(question)
        else:
            from certirag.retrieve.hybrid import SimpleRetriever
            retriever = SimpleRetriever(self._bm25_index, self._chunk_store)
            evidence_chunks = retriever.retrieve(question)
        timings["retrieve_ms"] = (time.time() - t0) * 1000

        # ── Step 2: Compile Claims ─────────────────────────────────
        t0 = time.time()
        claim_ir = self._compiler.compile(question, evidence_chunks)
        timings["compile_ms"] = (time.time() - t0) * 1000

        # ── Step 3: Normalize Claims ───────────────────────────────
        t0 = time.time()
        claim_ir = self._normalizer.normalize(claim_ir)
        timings["normalize_ms"] = (time.time() - t0) * 1000

        # ── Step 4: Verify Claims ──────────────────────────────────
        t0 = time.time()
        verification_results = self._verifier.verify_claims(
            claim_ir, evidence_chunks
        )
        timings["verify_ms"] = (time.time() - t0) * 1000

        # ── Step 5: MSE Selection ──────────────────────────────────
        t0 = time.time()
        if self.config.verification.use_mse:
            verification_results = self._mse_selector.select_batch(verification_results)
        timings["mse_ms"] = (time.time() - t0) * 1000

        # ── Step 6: Render ─────────────────────────────────────────
        t0 = time.time()
        renderer_decisions = self._renderer.render(verification_results)
        display = self._renderer.filter_for_display(claim_ir, renderer_decisions, mode)
        timings["render_ms"] = (time.time() - t0) * 1000

        # ── Step 7: Certificate ────────────────────────────────────
        t0 = time.time()
        total_ms = (time.time() - total_start) * 1000
        certificate = None
        if self.config.render.include_certificate:
            certificate = self._cert_builder.build(
                question=question,
                claim_ir=claim_ir,
                evidence_chunks=evidence_chunks,
                verification_results=verification_results,
                renderer_decisions=renderer_decisions,
                policy=self._renderer.policy_snapshot,
                latency_ms=total_ms,
            )
        timings["certificate_ms"] = (time.time() - t0) * 1000
        timings["total_ms"] = (time.time() - total_start) * 1000

        result = PipelineResult(
            question=question,
            claim_ir=claim_ir,
            evidence_chunks=evidence_chunks,
            verification_results=verification_results,
            renderer_decisions=renderer_decisions,
            certificate=certificate,
            display=display,
            timings=timings,
        )

        logger.info(
            f"Pipeline complete: {display['stats']} | "
            f"Total: {timings['total_ms']:.0f}ms"
        )
        return result

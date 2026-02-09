"""
CertiRAG Configuration System
==============================

Central configuration using Pydantic Settings. Supports:
- Environment variables (CERTIRAG_ prefix)
- .env file loading
- YAML config file overrides
- Two execution modes: "lite" (CPU/API) and "full" (GPU/local models)

The config produces a deterministic hash for reproducibility tracking.
Every certificate and output artifact is stamped with this hash.

Usage:
    from certirag.config import get_config
    cfg = get_config()                    # loads from env / .env
    cfg = get_config("configs/strict.yaml")  # loads with YAML overrides
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Execution Mode ─────────────────────────────────────────────────
class ExecutionMode(str, Enum):
    """
    Controls which backends are used throughout the pipeline.

    - LITE: CPU-only, uses OpenAI API for LLM calls, mock/simple verifier.
            Suitable for Codespaces, CI, and development.
    - FULL: GPU-accelerated, uses local HuggingFace models for everything.
            Suitable for Colab Pro, local GPU workstations, benchmarking.
    """
    LITE = "lite"
    FULL = "full"


# ── Sub-configs ────────────────────────────────────────────────────
class RetrievalConfig(BaseModel):
    """Configuration for the hybrid retrieval pipeline."""
    max_chunks: int = Field(default=10, description="Max chunks returned per query")
    max_spans_per_chunk: int = Field(default=5, description="Max sentence spans per chunk")
    top_k_bm25: int = Field(default=50, description="BM25 candidates before fusion")
    top_k_dense: int = Field(default=50, description="Dense candidates before fusion")
    top_k_rerank: int = Field(default=10, description="Candidates after reranking")
    rrf_k: int = Field(default=60, description="RRF constant (controls rank fusion smoothing)")
    chunk_size: int = Field(default=256, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=32, description="Overlap between consecutive chunks in tokens")
    # Model choices (used in FULL mode)
    embedding_model: str = Field(
        default="intfloat/e5-base-v2",
        description="Dense embedding model (HuggingFace ID)"
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Cross-encoder reranker model (HuggingFace ID)"
    )


class ClaimConfig(BaseModel):
    """Configuration for the claim compiler and normalizer."""
    max_claims: int = Field(default=12, description="Max claims per query response")
    max_claim_tokens: int = Field(default=60, description="Max tokens per atomic claim")
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM for claim compilation (OpenAI model name or HF ID)"
    )
    temperature: float = Field(default=0.0, description="LLM temperature for claim compilation")
    split_conjunctions: bool = Field(default=True, description="Auto-split compound claims")
    remove_hedges: bool = Field(default=False, description="Strip hedging language before verification")


class VerificationConfig(BaseModel):
    """Configuration for the verification pipeline."""
    tau_entail: float = Field(
        default=0.85,
        ge=0.0, le=1.0,
        description="Entailment threshold: claim is Verified if score >= tau_entail"
    )
    tau_contradict: float = Field(
        default=0.70,
        ge=0.0, le=1.0,
        description="Contradiction threshold: claim is Blocked if score >= tau_contradict"
    )
    max_verifier_pairs: int = Field(
        default=240,
        description="Hard cap on total (claim, span) pairs per query"
    )
    max_evidence_per_claim: int = Field(
        default=20,
        description="Max evidence spans considered per claim"
    )
    use_mse: bool = Field(
        default=True,
        description="Use Minimal Sufficient Evidence selection"
    )
    verifier_model: str = Field(
        default="lytang/MiniCheck-RoBERTa-large",
        description="Primary verifier model (HuggingFace ID)"
    )
    verifier_batch_size: int = Field(default=32, description="Batch size for verifier inference")
    calibration_method: str = Field(
        default="isotonic",
        description="Calibration method: 'temperature', 'isotonic', or 'none'"
    )


class RenderConfig(BaseModel):
    """Configuration for the renderer / policy engine."""
    default_mode: str = Field(
        default="strict",
        description="Default rendering mode: 'strict', 'mixed', or 'debug'"
    )
    min_evidence_spans: int = Field(
        default=1,
        description="Minimum evidence spans required for Verified status"
    )
    include_certificate: bool = Field(default=True, description="Attach audit certificate to output")


class EvalConfig(BaseModel):
    """Configuration for the evaluation harness."""
    seed: int = Field(default=42, description="Global random seed for reproducibility")
    human_eval_sample_size: int = Field(default=200, description="Samples for human evaluation")
    num_workers: int = Field(default=4, description="Parallel workers for batch evaluation")


# ── Main Config ────────────────────────────────────────────────────
class CertiRAGConfig(BaseSettings):
    """
    Root configuration for the CertiRAG system.

    Loads from environment variables (CERTIRAG_ prefix) and .env file.
    Can be extended with YAML overrides via `get_config(yaml_path)`.

    Example:
        export CERTIRAG_MODE=full
        export CERTIRAG_TAU_ENTAIL=0.9
    """
    model_config = SettingsConfigDict(
        env_prefix="CERTIRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Top-level settings ─────────────────────────────────────────
    mode: ExecutionMode = Field(
        default=ExecutionMode.LITE,
        description="Execution mode: 'lite' (CPU/API) or 'full' (GPU/local)"
    )
    data_dir: Path = Field(default=Path("./data"), description="Data directory")
    output_dir: Path = Field(default=Path("./outputs"), description="Output directory")
    model_dir: Path = Field(default=Path("./models"), description="Model cache directory")
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: 'json' or 'text'")

    # ── OpenAI API (for lite mode) ─────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model for generation")

    # ── Google Gemini API (free tier) ──────────────────────────────
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model for LLM calls")

    # ── Sub-configs ────────────────────────────────────────────────
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    claim: ClaimConfig = Field(default_factory=ClaimConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @property
    def is_lite(self) -> bool:
        """Check if running in lite (CPU/API) mode."""
        return self.mode == ExecutionMode.LITE

    @property
    def is_full(self) -> bool:
        """Check if running in full (GPU) mode."""
        return self.mode == ExecutionMode.FULL

    def config_hash(self) -> str:
        """
        Produce a deterministic SHA-256 hash of the configuration.

        This hash is stamped on every certificate and output artifact
        to ensure reproducibility. Two runs with the same config hash
        should produce identical results (given the same data + seeds).
        """
        # Serialize config to a canonical JSON string (sorted keys)
        config_dict = self.model_dump(mode="json", exclude={"openai_api_key"})
        canonical = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def ensure_dirs(self) -> None:
        """Create data, output, and model directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


# ── Config Loading ─────────────────────────────────────────────────
def get_config(yaml_path: Optional[str] = None) -> CertiRAGConfig:
    """
    Load CertiRAG configuration.

    Priority (highest to lowest):
        1. Environment variables (CERTIRAG_ prefix)
        2. .env file
        3. YAML config file (if provided)
        4. Default values

    Args:
        yaml_path: Optional path to a YAML config file for overrides.

    Returns:
        Fully resolved CertiRAGConfig instance.

    Example:
        cfg = get_config()                          # defaults + env
        cfg = get_config("configs/strict.yaml")     # with YAML overrides
    """
    if yaml_path:
        import yaml
        with open(yaml_path) as f:
            overrides = yaml.safe_load(f) or {}
        return CertiRAGConfig(**overrides)
    return CertiRAGConfig()

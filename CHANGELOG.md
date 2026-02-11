# Changelog

All notable changes to CertiRAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-11

### Added

- **Core Pipeline**: Full 7-stage verification pipeline
  - Document ingestion with sentence-aware chunking
  - BM25 + dense hybrid retrieval with RRF fusion
  - LLM-based claim extraction and normalization
  - Cross-encoder NLI verification with calibration
  - Deterministic renderer policy engine (Theorem 1)
  - Tamper-evident audit certificates with SHA-256 integrity

- **Schemas**: Pydantic v2 data models for all pipeline stages
  - `ClaimIR` — claim extraction intermediate representation
  - `EvidenceChunk` / `EvidenceSpan` — evidence with retrieval scores
  - `VerificationResult` / `EvidenceScore` — NLI verification output
  - `RendererDecision` / `AuditCertificate` — rendering + audit

- **Dual-Mode Architecture**
  - LITE mode: CPU-only with API-based LLMs (Codespaces-friendly)
  - FULL mode: GPU-accelerated with local models (Colab Pro)

- **Verification Engine**
  - MiniCheck integration for lightweight NLI
  - Cross-encoder NLI with DeBERTa-v3
  - LLM-as-judge verification
  - Maximum Spanning Evidence (MSE) selection
  - Score calibration with temperature scaling

- **Renderer Policy Engine**
  - Fail-closed rendering guarantee
  - Configurable thresholds (τ_entail, τ_contradict)
  - Three display modes: strict, mixed, debug
  - Policy snapshots for audit trail

- **CLI**: `certirag` command with subcommands
  - `ingest` — document ingestion
  - `run` — pipeline execution
  - `eval` — benchmark evaluation
  - `validate` — schema validation
  - `export-schemas` — JSON schema export

- **Web UI**: Streamlit-based dashboard
  - Glassmorphic dark theme with neon accents
  - Interactive claim verification
  - Evidence explorer with retrieval scores
  - Live audit certificate generation
  - Pipeline step visualization

- **Evaluation Framework**
  - AggreFact benchmark support
  - ALCE factual precision metrics
  - RAGTruth dataset integration
  - Automated metric computation

- **Testing**: 134 tests across four suites
  - Unit tests (75) — schemas, renderer, metrics, spans
  - Integration tests (26) — pipeline flows, corruption, e2e verification
  - Adversarial tests (12) — robustness, boundary cases
  - Benchmark tests (21) — NLI dataset evaluation (slow)

- **Documentation**
  - Comprehensive README with installation guides
  - Architecture diagrams (Mermaid)
  - API reference with examples
  - Contributing guidelines

[Unreleased]: https://github.com/aayushakumar/CertiRAG/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aayushakumar/CertiRAG/releases/tag/v0.1.0

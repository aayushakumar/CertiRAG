<p align="center">
  <img src="https://img.shields.io/badge/CertiRAG-v0.1.0-blue?style=for-the-badge" alt="CertiRAG Badge"/>
  <img src="https://img.shields.io/badge/python-3.10%2B-green?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/license-MIT-orange?style=for-the-badge" alt="MIT License"/>
  <img src="https://img.shields.io/badge/tests-104%20passed-brightgreen?style=for-the-badge&logo=pytest" alt="Tests"/>
  <img src="https://img.shields.io/badge/pydantic-v2-purple?style=for-the-badge" alt="Pydantic v2"/>
</p>

<h1 align="center">🛡️ CertiRAG</h1>
<h3 align="center"><em>Certified Retrieval-Augmented Generation with Fail-Closed Verification</em></h3>

<p align="center">
CertiRAG decomposes LLM-generated answers into atomic claims, verifies each claim against retrieved evidence using NLI models, and produces a tamper-evident audit certificate. Unverified or contradicted claims are <strong>blocked before reaching the user</strong> — a fail-closed design that provably prevents unfaithful content from being displayed.
</p>

---

## Table of Contents

- [Why CertiRAG?](#why-certirag)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Verification Guarantee (Theorem 1)](#verification-guarantee-theorem-1)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [LITE Mode (CPU / Codespaces)](#lite-mode-cpu--codespaces)
  - [FULL Mode (GPU / Colab Pro)](#full-mode-gpu--colab-pro)
  - [Development Install](#development-install)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [YAML Configuration Files](#yaml-configuration-files)
  - [Configuration Reference](#configuration-reference)
- [Quick Start Guide](#quick-start-guide)
  - [Step 1: Ingest Documents](#step-1-ingest-documents)
  - [Step 2: Run Verification](#step-2-run-verification)
  - [Step 3: Inspect the Certificate](#step-3-inspect-the-certificate)
- [Using CertiRAG as a Library](#using-certirag-as-a-library)
  - [Basic Usage](#basic-usage)
  - [Custom Configuration](#custom-configuration)
  - [Working with Schemas](#working-with-schemas)
  - [Using Individual Components](#using-individual-components)
- [CLI Reference](#cli-reference)
- [Data Schemas](#data-schemas)
  - [EvidenceChunk & EvidenceSpan](#evidencechunk--evidencespan)
  - [ClaimIR & Claim](#claimir--claim)
  - [VerificationResult & EvidenceScore](#verificationresult--evidencescore)
  - [AuditCertificate & RendererDecision](#auditcertificate--rendererdecision)
- [Pipeline Deep Dive](#pipeline-deep-dive)
  - [Stage 1: Hybrid Retrieval](#stage-1-hybrid-retrieval)
  - [Stage 2: Claim Compilation](#stage-2-claim-compilation)
  - [Stage 3: Claim Normalization](#stage-3-claim-normalization)
  - [Stage 4: Verification](#stage-4-verification)
  - [Stage 5: Minimal Sufficient Evidence (MSE)](#stage-5-minimal-sufficient-evidence-mse)
  - [Stage 6: Rendering & Policy](#stage-6-rendering--policy)
  - [Stage 7: Certificate Generation](#stage-7-certificate-generation)
- [Evaluation Framework](#evaluation-framework)
- [Web UI](#web-ui)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [Development Workflow](#development-workflow)
- [Notebooks](#notebooks)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Why CertiRAG?

Standard RAG systems retrieve documents and generate answers, but they **cannot guarantee** that the generated text is faithful to the source evidence. This leads to:

| Problem | Impact |
|---------|--------|
| **Hallucinated facts** | Users receive fabricated information presented as truth |
| **Unsupported claims** | Answers contain statements with no evidence backing |
| **Contradictory outputs** | Generated text directly contradicts source documents |
| **No auditability** | No way to trace which evidence supports which claim |
| **No reproducibility** | Results vary without any record of configuration |

**CertiRAG solves all of these** by introducing a formal verification layer between generation and display:

```
Traditional RAG:  Query → Retrieve → Generate → Display (hope for the best)
CertiRAG:         Query → Retrieve → Generate → Decompose → Verify → Certify → Display (guaranteed)
```

Every claim is individually verified. Every decision is recorded. Every certificate is tamper-evident. **If a claim can't be proven, it doesn't get shown.**

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Atomic Claim Decomposition** | LLM decomposes answers into individual, verifiable claims using few-shot prompting |
| 🔍 **Hybrid Retrieval** | BM25 (sparse) + Dense embeddings + Reciprocal Rank Fusion + optional cross-encoder reranking |
| ✅ **Multi-Verifier Architecture** | MiniCheck-RoBERTa-large (primary), NLI-DeBERTa (ablation), LLM-as-judge (ablation) |
| 🚫 **Fail-Closed Policy** | Theorem 1: claims are VERIFIED only when ALL conditions are met — otherwise BLOCKED or UNVERIFIED |
| 📏 **Minimal Sufficient Evidence** | Greedy MSE algorithm selects the smallest evidence set that achieves the verification threshold |
| 🔒 **Tamper-Evident Certificates** | SHA-256 sealed audit certificates with config hashing for full reproducibility |
| ⚖️ **Score Calibration** | Temperature scaling + isotonic regression for well-calibrated confidence scores |
| 📊 **Comprehensive Evaluation** | Built-in harness for ALCE, RAGTruth, and AggreFact benchmarks with ablation support |
| 🖥️ **Dual Execution Modes** | LITE (CPU/API for Codespaces) and FULL (GPU for Colab Pro / local) |
| 🎨 **Interactive Web UI** | Modern web interface for real-time verification visualization |
| 📦 **pip-installable** | `pip install certirag` — use as a library or CLI tool |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          CertiRAG Pipeline                               │
│                                                                          │
│  ┌──────────┐    ┌───────────┐    ┌───────────┐    ┌──────────────────┐  │
│  │ Hybrid   │───▶│  Claim    │───▶│  Claim    │───▶│   Verifier       │  │
│  │ Retriever│    │  Compiler │    │ Normalizer│    │ (MiniCheck/NLI)  │  │
│  └──────────┘    └───────────┘    └───────────┘    └────────┬─────────┘  │
│       │                                                      │           │
│       │  BM25 + Dense                                        │           │
│       │  + RRF Fusion                                        ▼           │
│       │  + Reranking            ┌─────────┐     ┌────────────────────┐   │
│       │                         │   MSE   │◀────│  VerificationResult│   │
│       │                         │Selector │     │  per claim         │   │
│       │                         └────┬────┘     └────────────────────┘   │
│       │                              │                                   │
│       │                              ▼                                   │
│       │                    ┌──────────────────┐                          │
│       │                    │ Renderer Policy  │                          │
│       │                    │  (Theorem 1)     │                          │
│       │                    └────────┬─────────┘                          │
│       │                             │                                    │
│       │                             ▼                                    │
│       │                    ┌──────────────────┐                          │
│       │                    │   Certificate    │                          │
│       │                    │    Builder       │                          │
│       │                    └────────┬─────────┘                          │
│       │                             │                                    │
└───────┼─────────────────────────────┼────────────────────────────────────┘
        │                             ▼
   ┌────┴────┐              ┌──────────────────┐
   │Documents│              │ AuditCertificate │
   │ Corpus  │              │ + Rendered Answer│
   └─────────┘              └──────────────────┘
```

For detailed Mermaid diagrams of every subsystem, see [`docs/architecture.md`](docs/architecture.md).

---

## Verification Guarantee (Theorem 1)

CertiRAG enforces a **formal verification predicate** on every claim:

$$
\text{VERIFIED}(c_i) \iff \underbrace{\hat{p}_{\text{entail}}(c_i) \ge \tau_e}_{\text{sufficient support}} \;\wedge\; \underbrace{\hat{p}_{\text{contradict}}(c_i) < \tau_c}_{\text{no contradiction}} \;\wedge\; \underbrace{|E_{\text{MSE}}(c_i)| \ge 1}_{\text{evidence exists}}
$$

| State | Condition | User Sees |
|-------|-----------|-----------|
| 🟢 **VERIFIED** | All three conditions met | Claim displayed with evidence |
| 🟡 **UNVERIFIED** | Insufficient entailment or no evidence | Claim hidden (strict) or flagged (mixed) |
| 🔴 **BLOCKED** | Evidence actively contradicts the claim | Claim never shown — hard block |

**The guarantee**: No claim reaches the user as "verified" unless it passes all three gates. This is enforced deterministically in `RendererPolicy.decide()` — no stochastic component.

---

## Project Structure

```
CertiRAG/
├── certirag/                        # Core library (pip-installable)
│   ├── __init__.py                  # Package root, version = "0.1.0"
│   ├── config.py                    # Central config (Pydantic Settings + YAML + env)
│   ├── utils.py                     # Seeds, hashing, logging, text processing
│   ├── pipeline.py                  # End-to-end orchestrator (CertiRAGPipeline)
│   ├── cli.py                       # CLI: ingest / run / eval / validate / export-schemas
│   │
│   ├── schemas/                     # Pydantic v2 data contracts
│   │   ├── evidence.py              # EvidenceChunk, EvidenceSpan, SourceInfo
│   │   ├── claim_ir.py              # ClaimIR, Claim, CandidateEvidence, ClaimType
│   │   ├── verification.py          # VerificationResult, EvidenceScore, VerificationLabel
│   │   └── certificate.py           # AuditCertificate, RendererDecision, PolicySnapshot
│   │
│   ├── ingest/                      # Document ingestion pipeline
│   │   ├── chunker.py               # Sentence-aware chunking (regex / spaCy)
│   │   ├── embedder.py              # Dense embeddings (e5-base-v2 / OpenAI)
│   │   └── indexer.py               # BM25Index + DenseIndex + ChunkStore
│   │
│   ├── retrieve/                    # Evidence retrieval
│   │   ├── hybrid.py                # HybridRetriever (BM25+Dense+RRF+rerank)
│   │   └── span_extractor.py        # SpanExtractor + get_span_text()
│   │
│   ├── claim_ir/                    # Claim compilation & normalization
│   │   ├── compiler.py              # LLM-based claim decomposition
│   │   ├── normalizer.py            # Atomicity (split / dehedge / dedup)
│   │   └── validator.py             # JSON Schema validation for all contracts
│   │
│   ├── verify/                      # Verification system
│   │   ├── verifier.py              # BaseVerifier (ABC)
│   │   ├── minicheck.py             # MiniCheck-RoBERTa-large (primary)
│   │   ├── nli_verifier.py          # NLI-DeBERTa-v3 (ablation baseline)
│   │   ├── llm_judge.py             # LLM-as-judge via OpenAI (ablation)
│   │   ├── calibrator.py            # Temperature scaling + isotonic regression
│   │   └── mse.py                   # Minimal Sufficient Evidence selector
│   │
│   └── render/                      # Output rendering
│       ├── policy.py                # RendererPolicy (Theorem 1 enforcement)
│       └── certificate.py           # CertificateBuilder (SHA-256 sealing)
│
├── eval/                            # Evaluation harness
│   ├── datasets/                    # Benchmark data loaders
│   │   ├── alce.py                  # ALCE (Gao et al., 2023)
│   │   ├── ragtruth.py              # RAGTruth (Wu et al., 2024)
│   │   └── aggrefact.py             # AggreFact (Tang et al., 2022)
│   ├── metrics.py                   # claim_f1, faithfulness, AUROC, ECE, yield
│   ├── runner.py                    # EvalRunner orchestrator
│   ├── baselines.py                 # StandardRAG, PosthocChecker, MultiQueryRAG
│   ├── ablations.py                 # 10 ablation configurations
│   └── plots.py                     # Publication-quality matplotlib figures
│
├── tests/                           # 104 tests (pytest)
│   ├── conftest.py                  # Shared fixtures & factories
│   ├── unit/                        # Unit tests
│   │   ├── test_schemas.py          # Schema construction, validation, roundtrips
│   │   ├── test_atomicity.py        # Conjunction splitting, hedge detection, dedup
│   │   ├── test_renderer.py         # Theorem 1 decisions, display modes, safety
│   │   ├── test_spans.py            # Span integrity, offset validation
│   │   └── test_metrics.py          # Evaluation metric correctness
│   ├── integration/                 # Integration tests
│   │   ├── test_golden.py           # End-to-end golden path scenarios
│   │   └── test_corruption.py       # Error handling, edge cases
│   └── adversarial/                 # Adversarial robustness tests
│       └── test_adversarial.py      # Negation, entity substitution, spoofing
│
├── ui/                              # Web interface
│   └── app.py                       # Streamlit demo application
│
├── notebooks/                       # Jupyter analysis notebooks
│   ├── 01_data_exploration.ipynb    # Chunking, spans, indexing analysis
│   ├── 02_retrieval_analysis.ipynb  # BM25 vs Dense vs Hybrid comparison
│   ├── 03_verifier_calibration.ipynb # Calibration, reliability, thresholds
│   └── 04_publication_plots.ipynb   # Paper tables, LaTeX export
│
├── docs/                            # Documentation
│   └── architecture.md              # 8 Mermaid architecture diagrams
│
├── configs/                         # Configuration presets
│   ├── default.yaml                 # Balanced defaults
│   ├── strict.yaml                  # High-precision (τ_e=0.7, τ_c=0.5)
│   └── mixed.yaml                   # Mixed display mode
│
├── pyproject.toml                   # Build config, dependencies, tool settings
├── Makefile                         # Development command shortcuts
├── Dockerfile                       # Multi-stage: CPU (lite) + GPU (full)
├── LICENSE                          # MIT License
├── PLAN.md                          # Implementation roadmap
├── README_SPEC.md                   # Original research specification
└── README.md                        # This file
```

---

## Installation

### Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- **pip** 24.x+ (if using pip 26+, ensure setuptools compatibility)
- **Git** for cloning

### LITE Mode (CPU / Codespaces)

Best for development, testing, and environments without GPUs. Uses OpenAI API for embeddings and claim compilation.

```bash
# Clone
git clone https://github.com/aayushakumar/CertiRAG.git
cd CertiRAG

# Install with LITE dependencies
pip install -e ".[lite,dev]"

# Set up environment
echo 'CERTIRAG_MODE=lite' > .env
echo 'CERTIRAG_OPENAI_API_KEY=sk-your-key-here' >> .env
```

**LITE mode uses:**
- OpenAI API for embeddings and claim decomposition
- NumPy-based cosine similarity (no FAISS)
- Word-overlap heuristic for verification (no GPU model)
- No cross-encoder reranking

### FULL Mode (GPU / Colab Pro)

For production evaluation and paper experiments. Runs all models locally on GPU.

```bash
# Install with FULL dependencies (requires CUDA)
pip install -e ".[full,dev]"
```

**FULL mode uses:**
- `sentence-transformers` for local embeddings (e5-base-v2)
- FAISS GPU index for dense retrieval
- MiniCheck-RoBERTa-large for verification (local inference)
- Cross-encoder reranking (BAAI/bge-reranker-base)

### Development Install

```bash
# Install everything (LITE + FULL + dev tools)
pip install -e ".[lite,full,dev]"

# Or use Make
make install-dev
```

### Google Colab Pro Setup

```python
# In a Colab notebook cell:
!git clone https://github.com/aayushakumar/CertiRAG.git
%cd CertiRAG
!pip install -e ".[full,dev]"

import os
os.environ["CERTIRAG_MODE"] = "full"
os.environ["CERTIRAG_OPENAI_API_KEY"] = "sk-..."  # For claim compilation

from certirag.pipeline import CertiRAGPipeline
pipeline = CertiRAGPipeline.from_config()
```

---

## Configuration

CertiRAG uses a layered configuration system: **defaults → YAML file → environment variables → constructor args**.

### Environment Variables

All environment variables use the `CERTIRAG_` prefix:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CERTIRAG_MODE` | No | `lite` | Execution mode: `lite` or `full` |
| `CERTIRAG_OPENAI_API_KEY` | Yes (LITE) | — | OpenAI API key for claim compilation & embeddings |
| `CERTIRAG_OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model for claim compilation |
| `CERTIRAG_DATA_DIR` | No | `./data` | Directory for document data and indices |
| `CERTIRAG_OUTPUT_DIR` | No | `./outputs` | Directory for certificates and results |
| `CERTIRAG_MODEL_DIR` | No | `./models` | Directory for cached model weights |
| `CERTIRAG_LOG_LEVEL` | No | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `CERTIRAG_LOG_FORMAT` | No | `json` | Log format: `json` or `text` |

Create a `.env` file in the project root:

```bash
# .env
CERTIRAG_MODE=lite
CERTIRAG_OPENAI_API_KEY=sk-your-key-here
CERTIRAG_OPENAI_MODEL=gpt-4o-mini
CERTIRAG_LOG_LEVEL=INFO
```

### YAML Configuration Files

Override any setting via YAML. Three presets are included:

**`configs/default.yaml`** — Balanced for development:
```yaml
mode: lite
chunk_size: 8
top_k: 5
tau_entail: 0.5
tau_contradict: 0.7
max_claims: 20
split_conjunctions: true
remove_hedges: true
dedup_threshold: 0.8
calibration_method: temperature
```

**`configs/strict.yaml`** — High precision:
```yaml
mode: lite
tau_entail: 0.7
tau_contradict: 0.5
default_mode: strict
```

**`configs/mixed.yaml`** — Show unverified claims with warnings:
```yaml
mode: lite
tau_entail: 0.5
tau_contradict: 0.7
default_mode: mixed
```

Use a config file:
```bash
certirag --config configs/strict.yaml run "What is the capital of France?"
```

### Configuration Reference

<details>
<summary><strong>Click to expand full configuration reference</strong></summary>

#### Root Config (`CertiRAGConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `ExecutionMode` | `LITE` | `LITE` or `FULL` |
| `data_dir` | `str` | `./data` | Data storage directory |
| `output_dir` | `str` | `./outputs` | Output directory for certificates |
| `model_dir` | `str` | `./models` | Model cache directory |
| `log_level` | `str` | `INFO` | Logging level |
| `log_format` | `str` | `json` | `json` or `text` |
| `openai_api_key` | `str?` | `None` | OpenAI API key |
| `openai_model` | `str` | `gpt-4o-mini` | OpenAI model name |

#### Retrieval Config (`RetrievalConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_chunks` | `int` | `10` | Maximum chunks to retrieve |
| `max_spans_per_chunk` | `int` | `5` | Maximum spans per chunk |
| `top_k_bm25` | `int` | `50` | BM25 candidate pool size |
| `top_k_dense` | `int` | `50` | Dense retrieval candidate pool size |
| `top_k_rerank` | `int` | `10` | Final reranked result count |
| `rrf_k` | `int` | `60` | RRF smoothing parameter |
| `chunk_size` | `int` | `256` | Target chunk size in tokens |
| `chunk_overlap` | `int` | `32` | Token overlap between adjacent chunks |
| `embedding_model` | `str` | `intfloat/e5-base-v2` | Embedding model name |
| `reranker_model` | `str` | `BAAI/bge-reranker-base` | Cross-encoder reranker model |

#### Claim Config (`ClaimConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_claims` | `int` | `12` | Maximum claims to extract per answer |
| `max_claim_tokens` | `int` | `60` | Maximum tokens per individual claim |
| `llm_model` | `str` | `gpt-4o-mini` | LLM for claim decomposition |
| `temperature` | `float` | `0.0` | LLM sampling temperature (0.0 = deterministic) |
| `split_conjunctions` | `bool` | `True` | Split compound "and"/"but" claims |
| `remove_hedges` | `bool` | `False` | Strip hedging language ("might", "probably") |

#### Verification Config (`VerificationConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tau_entail` | `float` | `0.85` | Entailment threshold (Theorem 1) |
| `tau_contradict` | `float` | `0.70` | Contradiction threshold (Theorem 1) |
| `max_verifier_pairs` | `int` | `240` | Cost cap: max (claim, span) pairs per query |
| `max_evidence_per_claim` | `int` | `20` | Max evidence spans considered per claim |
| `use_mse` | `bool` | `True` | Enable MSE evidence selection |
| `verifier_model` | `str` | `lytang/MiniCheck-RoBERTa-large` | Verifier model |
| `verifier_batch_size` | `int` | `32` | Batch size for NLI inference |
| `calibration_method` | `str` | `isotonic` | `isotonic`, `temperature`, or `none` |

#### Render Config (`RenderConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_mode` | `str` | `strict` | Display filter: `strict`, `mixed`, `debug` |
| `min_evidence_spans` | `int` | `1` | Minimum evidence spans for VERIFIED |
| `include_certificate` | `bool` | `True` | Generate audit certificate per query |

#### Eval Config (`EvalConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seed` | `int` | `42` | Random seed for reproducibility |
| `human_eval_sample_size` | `int` | `200` | Samples for human evaluation |
| `num_workers` | `int` | `4` | Worker threads for parallel evaluation |

</details>

---

## Quick Start Guide

### Step 1: Ingest Documents

Prepare your documents as JSONL (one JSON object per line with `id`, `text`, and optional `source`):

```jsonl
{"id": "doc001", "text": "The Eiffel Tower is a 330-metre tall wrought-iron lattice tower in Paris, France. Named after engineer Gustave Eiffel, whose company built the tower from 1887 to 1889.", "source": "wikipedia"}
{"id": "doc002", "text": "The Great Wall of China spans approximately 21,196 kilometres across the historical northern borders of China.", "source": "encyclopedia"}
```

Ingest via CLI:

```bash
certirag ingest --docs documents.jsonl --index-dir data/index
```

Or via Python:

```python
from certirag.pipeline import CertiRAGPipeline

pipeline = CertiRAGPipeline.from_config()
pipeline.ingest([
    {"id": "doc001", "text": "The Eiffel Tower is 330 metres tall...", "source": "wikipedia"},
    {"id": "doc002", "text": "The Great Wall spans 21,196 km...", "source": "encyclopedia"},
])
```

### Step 2: Run Verification

```bash
certirag run "How tall is the Eiffel Tower?" --display strict
```

Example output:

```
CertiRAG Verification Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Question: How tall is the Eiffel Tower?

Claims:
  ✅ VERIFIED   "The Eiffel Tower is 330 metres tall."
                Evidence: doc001, span s0 (score: 0.94)
  🟡 UNVERIFIED "It is the tallest structure in Europe."
                Reason: Insufficient entailment (0.42 < 0.85)

Stats: 1 verified, 1 unverified, 0 blocked
Certificate: SHA-256 seal = a3f8c2d1...
```

### Step 3: Inspect the Certificate

```bash
# Validate a saved certificate
certirag validate --input outputs/certificate_20260209.json --schema certificate
```

```python
# Programmatic validation
from certirag.render.certificate import CertificateBuilder
from certirag.config import get_config

config = get_config()
builder = CertificateBuilder(config)
result = builder.verify_certificate("outputs/certificate.json")
print(f"Valid: {result['valid']}")       # True if untampered
print(f"Errors: {result['errors']}")     # [] if everything is OK
```

---

## Using CertiRAG as a Library

### Basic Usage

```python
from certirag.pipeline import CertiRAGPipeline

# Initialize (reads config from env / .env / YAML)
pipeline = CertiRAGPipeline.from_config()

# Ingest your documents
pipeline.ingest([
    {"id": "paris", "text": "Paris is the capital of France. It has a population of 2.1 million."},
    {"id": "london", "text": "London is the capital of the United Kingdom and England."},
])

# Run verification on a question
result = pipeline.run("What is the capital of France?")

# Access results
print(f"Total claims: {result.stats['total']}")
print(f"Verified:     {result.stats['verified']}")
print(f"Blocked:      {result.stats['blocked']}")

# Iterate over individual decisions
for decision in result.renderer_decisions:
    print(f"  [{decision.render_state.value}] Claim {decision.claim_id}: "
          f"entail={decision.entail_score:.2f}, "
          f"contradict={decision.contradict_score:.2f}")

# Access the sealed certificate
cert = result.certificate
print(f"Certificate seal: {cert.integrity_hash[:16]}...")
print(f"Tamper check: {'✅ valid' if cert.verify_integrity() else '❌ TAMPERED'}")
```

### Custom Configuration

```python
from certirag.config import CertiRAGConfig, ExecutionMode

# Construct programmatically
config = CertiRAGConfig(
    mode=ExecutionMode.FULL,
    openai_api_key="sk-...",
)
pipeline = CertiRAGPipeline(config=config)

# Or load from YAML
from certirag.config import get_config
config = get_config("configs/strict.yaml")
pipeline = CertiRAGPipeline(config=config)
```

### Working with Schemas

All data flows through strictly-typed Pydantic v2 models:

```python
from certirag.schemas.evidence import EvidenceChunk, EvidenceSpan, SourceInfo
from certirag.schemas.claim_ir import ClaimIR, Claim, CandidateEvidence, ClaimType
from certirag.schemas.verification import VerificationResult, EvidenceScore, VerificationLabel
from certirag.schemas.certificate import (
    AuditCertificate, RendererDecision, PolicySnapshot, RenderState
)

# Create an evidence chunk
chunk = EvidenceChunk(
    chunk_id="c1",
    doc_id="doc001",
    text="Paris is the capital of France.",
    source=SourceInfo(title="Wikipedia"),
    spans=[
        EvidenceSpan(span_id="s0", start=0, end=31,
                     sentence="Paris is the capital of France.")
    ]
)

# JSON round-trip
json_str = chunk.model_dump_json(indent=2)
restored = EvidenceChunk.model_validate_json(json_str)
assert restored.chunk_id == chunk.chunk_id  # ✅ Always works

# Export JSON schemas for external validation tools
from certirag.claim_ir.validator import export_all_schemas
export_all_schemas("schemas/")
```

### Using Individual Components

Each pipeline stage can be used independently:

```python
# --- Chunking ---
from certirag.ingest.chunker import DocumentChunker
chunker = DocumentChunker(chunk_size=256, overlap=2)
chunks = chunker.chunk_document("Your document text...", doc_id="d1")

# --- Indexing ---
from certirag.ingest.indexer import BM25Index, ChunkStore
store = ChunkStore()
store.add_many(chunks)
bm25 = BM25Index()
bm25.build(chunks)
results = bm25.query("search query", top_k=5)  # [(chunk_id, score), ...]

# --- Retrieval ---
from certirag.retrieve.hybrid import SimpleRetriever
retriever = SimpleRetriever(bm25_index=bm25, chunk_store=store)
evidence = retriever.retrieve("What is X?", top_k=5)

# --- Claim Normalization ---
from certirag.claim_ir.normalizer import ClaimNormalizer
normalizer = ClaimNormalizer(split_conjunctions=True, dedup_threshold=0.92)
normalized_ir = normalizer.normalize(claim_ir)

# --- Verification ---
from certirag.verify.minicheck import MiniCheckVerifier
verifier = MiniCheckVerifier(mode="full", device="cuda")
results = verifier.verify_claims(claim_ir, evidence_chunks)

# --- Rendering ---
from certirag.render.policy import RendererPolicy
policy = RendererPolicy(tau_entail=0.85, tau_contradict=0.70)
decisions = policy.render(verification_results)
display = policy.filter_for_display(claim_ir, decisions, mode="strict")

# --- Calibration ---
from certirag.verify.calibrator import VerifierCalibrator
calibrator = VerifierCalibrator(method="isotonic")
calibrator.fit(raw_scores_array, true_labels_array)
calibrated = calibrator.calibrate(raw_scores_array)
print(f"ECE: {calibrator.compute_ece(calibrated, true_labels_array):.4f}")

# --- MSE Selection ---
from certirag.verify.mse import MSESelector
mse = MSESelector(tau_entail=0.85, max_spans=5)
trimmed = mse.select_batch(verification_results)
stats = mse.compute_stats(trimmed)
print(f"Compression: {stats['compression_ratio']:.2f}")
```

---

## CLI Reference

```bash
certirag [--config <path>] [--mode {lite,full}] [-v/--verbose] <command> [args]
```

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `ingest` | Index documents for retrieval | `certirag ingest --docs data.jsonl` |
| `run` | Verify a question against indexed docs | `certirag run "What is X?"` |
| `eval` | Run evaluation benchmarks | `certirag eval --benchmark alce --data-dir data/alce` |
| `validate` | Validate JSON against a schema | `certirag validate --input cert.json --schema certificate` |
| `export-schemas` | Export all JSON schemas to files | `certirag export-schemas --output-dir schemas/` |

<details>
<summary><strong>Click to expand full CLI argument reference</strong></summary>

#### `ingest`

```bash
certirag ingest --docs <path> [--index-dir data/index]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--docs` | Yes | — | JSONL file or directory of .txt/.json files |
| `--index-dir` | No | `data/index` | Output directory for BM25 + dense indices |

#### `run`

```bash
certirag run "Your question" [--display {strict,mixed,debug}] [--output path.json]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `<question>` | Yes | — | Question to verify (positional) |
| `--display` | No | `strict` | `strict` / `mixed` / `debug` |
| `--output` | No | — | Save full result JSON to this path |

#### `eval`

```bash
certirag eval --benchmark {alce,ragtruth,aggrefact} --data-dir <path> [--subset <name>] [--max-examples <n>]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--benchmark` | Yes | — | Which benchmark to run |
| `--data-dir` | Yes | — | Path to benchmark data |
| `--subset` | No | — | Benchmark subset (e.g., `asqa` for ALCE) |
| `--max-examples` | No | all | Limit number of evaluation examples |
| `--output-dir` | No | `eval_results` | Output directory for results |

#### `validate`

```bash
certirag validate --input <path> --schema {evidence,claim_ir,verification,certificate}
```

#### `export-schemas`

```bash
certirag export-schemas [--output-dir schemas/]
```

</details>

---

## Data Schemas

### EvidenceChunk & EvidenceSpan

```python
class EvidenceSpan(BaseModel):
    span_id: str                    # Unique span identifier ("s0", "s1", ...)
    start: int                      # Start character offset (>= 0)
    end: int                        # End character offset (> 0, > start)
    sentence: str                   # The actual sentence text

class EvidenceChunk(BaseModel):
    chunk_id: str                   # Unique chunk identifier
    doc_id: str                     # Parent document ID
    text: str                       # Full chunk text
    source: SourceInfo              # Source metadata (title, url_or_path, section)
    spans: list[EvidenceSpan] = []  # Sentence-level spans
    retrieval: RetrievalScores = RetrievalScores()  # bm25, dense, rerank, rrf
```

**Invariant**: `chunk.text[span.start:span.end]` must equal `span.sentence` (±2 chars). Enforced by Pydantic validator.

### ClaimIR & Claim

```python
class Claim(BaseModel):
    id: str                                            # Claim ID ("c0", "c1", ...)
    text: str                                          # Claim text (non-empty)
    type: ClaimType = ClaimType.FACTUAL                # Classification
    fields: dict[str, Any] = {}                        # Extra metadata
    evidence_candidates: list[CandidateEvidence] = []  # Linked evidence
    hedge: bool = False                                # Contains hedging language
    unsupported_by_design: bool = False                # Intentionally unsupported

class ClaimIR(BaseModel):
    question: str                     # Original user question
    claims: list[Claim] = []          # Decomposed atomic claims
    model_id: Optional[str] = None    # LLM used for decomposition  
    raw_answer: Optional[str] = None  # Original generated answer
```

**Claim types**: `FACTUAL` · `TEMPORAL_FACT` · `NUMERIC_FACT` · `CAUSAL` · `COMPARATIVE` · `DEFINITION` · `PROCEDURAL` · `OTHER`

### VerificationResult & EvidenceScore

```python
class EvidenceScore(BaseModel):
    chunk_id: str                    # Evidence chunk ID
    span_id: str                     # Evidence span ID
    label: VerificationLabel         # ENTAILED / CONTRADICTED / NOT_ENOUGH_INFO
    score: float                     # Confidence score [0.0, 1.0]
    raw_score: float | None = None   # Pre-calibration score

class VerificationResult(BaseModel):
    claim_id: str                          # Claim being verified
    label: VerificationLabel               # Aggregate label
    score: float                           # Aggregate score [0.0, 1.0]
    best_evidence: list[EvidenceScore] = []  # MSE-selected evidence
    all_scores: list[EvidenceScore] = []     # All (claim, span) scores
```

### AuditCertificate & RendererDecision

```python
class RendererDecision(BaseModel):
    claim_id: str                    # Claim ID
    render_state: RenderState        # VERIFIED / UNVERIFIED / BLOCKED
    policy: PolicySnapshot           # Thresholds used for this decision
    reason: str                      # Human-readable explanation  
    entail_score: float = 0.0        # Max entailment score
    contradict_score: float = 0.0    # Max contradiction score
    evidence_count: int = 0          # Number of supporting evidence spans

class AuditCertificate(BaseModel):
    query_id: str                    # Unique query identifier
    question: str                    # Original question
    timestamp: str                   # ISO timestamp (auto-generated)
    config_hash: str                 # SHA-256 of configuration
    policy: PolicySnapshot | None    # Policy thresholds used
    stats: dict                      # {total, verified, blocked, unverified}
    integrity_hash: str              # SHA-256 seal for tamper detection
```

---

## Pipeline Deep Dive

### Stage 1: Hybrid Retrieval

```
Query → BM25 Index (sparse, lexical matching) → Top-50 candidates
      → Dense Index (embedding cosine similarity) → Top-50 candidates  
      → Reciprocal Rank Fusion (k=60) → Merged ranking
      → [FULL mode] Cross-Encoder Reranking → Final Top-k
```

**RRF formula**: $\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}$ where $k = 60$.

### Stage 2: Claim Compilation

Uses LLM (GPT-4o-mini default) with few-shot prompting to decompose generated answers into atomic, independently verifiable claims. Each claim is typed (factual, temporal, numeric, etc.).

### Stage 3: Claim Normalization

1. **Conjunction splitting**: "Paris is the capital and largest city" → 2 claims
2. **Hedge detection**: Flags "might", "probably", "approximately"  
3. **Deduplication**: TF-IDF cosine similarity > 0.92 → merge

### Stage 4: Verification

Scores every (claim, evidence_span) pair using MiniCheck or NLI model → 3-way label (ENTAILED / CONTRADICTED / NOT_ENOUGH_INFO) with confidence score. Cost-capped at 240 pairs per query.

### Stage 5: Minimal Sufficient Evidence (MSE)

Greedy algorithm: sort evidence by entailment score descending, accumulate until threshold is met. Reduces evidence by ~66% while maintaining verification guarantees.

### Stage 6: Rendering & Policy

Deterministic Theorem 1 evaluation:

| Check | Result | 
|-------|--------|
| `max_contradict ≥ τ_c` | → **BLOCKED** |
| `max_entail ≥ τ_e AND evidence ≥ min_spans` | → **VERIFIED** |
| Otherwise | → **UNVERIFIED** |

Display modes control visibility: `strict` (verified only), `mixed` (all labeled), `debug` (full scores).

### Stage 7: Certificate Generation

Assembles all pipeline outputs into a sealed `AuditCertificate`:
- SHA-256 config hash for reproducibility
- SHA-256 integrity hash over the full certificate body
- Verification via `certificate.verify_integrity()` returns `True/False`

---

## Evaluation Framework

### Supported Benchmarks

| Benchmark | Paper | Focus |
|-----------|-------|-------|
| **ALCE** | Gao et al., 2023 | Citation verification (ASQA, ELI5, ExpertQA subsets) |
| **RAGTruth** | Wu et al., 2024 | Hallucination detection in RAG outputs |
| **AggreFact** | Tang et al., 2022 | Factual consistency in summarization |

### Metrics

| Metric | Description |
|--------|-------------|
| `claim_precision` | Of VERIFIED claims, fraction truly entailed |
| `claim_recall` | Of truly entailed claims, fraction VERIFIED |
| `claim_f1` | Harmonic mean of precision and recall |
| `faithfulness_rate` | Overall faithful rendering rate |
| `AUROC` | Entailment scoring discrimination |
| `ECE` | Expected Calibration Error |
| `MSE_compression` | Evidence volume reduction from MSE |
| `verification_yield` | Fraction of claims passing verification |

### Running Evaluations

```bash
certirag eval --benchmark alce --data-dir data/alce --subset asqa
certirag eval --benchmark ragtruth --data-dir data/ragtruth
certirag eval --benchmark aggrefact --data-dir data/aggrefact
# Or all at once:
make eval
```

### Ablation Studies

10 built-in ablation configurations available via `eval.ablations.get_ablation_configs()`:

| Ablation | What changes |
|----------|-------------|
| `no_decomposition` | Skip claim decomposition |
| `no_mse` | Skip MSE evidence selection |
| `bm25_only` | BM25-only retrieval (no dense) |
| `dense_only` | Dense-only retrieval (no BM25) |
| `no_reranker` | Skip cross-encoder reranking |
| `nli_verifier` | NLI-DeBERTa instead of MiniCheck |
| `llm_judge` | LLM-as-judge instead of MiniCheck |
| `no_calibration` | Raw scores (no calibration) |
| `low_threshold` | τ_e=0.5, τ_c=0.3 (permissive) |
| `high_threshold` | τ_e=0.95, τ_c=0.9 (restrictive) |

---

## Web UI

CertiRAG includes a modern web interface for interactive verification demos.

```bash
# Start the UI
make demo
# Or directly:
streamlit run ui/app.py --server.port 8501
```

**Features**:
- Sidebar: mode selection, threshold sliders, display mode toggles
- Real-time claim verification with color-coded results
- Evidence inspector with span scores and progress bars  
- Certificate viewer with integrity validation
- Works in LITE mode without GPU or API keys (uses simplified heuristics)

---

## Docker Deployment

```bash
# Build CPU image
docker build -t certirag:latest .

# Build GPU image
docker build --target gpu -t certirag:gpu .

# Run (CPU)
docker run -p 8501:8501 \
  -e CERTIRAG_OPENAI_API_KEY=sk-... \
  -v $(pwd)/data:/app/data \
  certirag:latest

# Run (GPU)
docker run --gpus all -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  certirag:gpu
```

Health check: `http://localhost:8501/_stcore/health`

---

## Testing

**104 tests** across three categories, all passing:

```bash
make test              # Unit tests only (fast, < 2s)
make test-all          # All tests
make test-coverage     # With HTML coverage report
```

| Suite | Count | Focus |
|-------|-------|-------|
| Unit | 75 | Schemas, atomicity, rendering, spans, metrics |
| Integration | 17 | Golden paths, error handling, edge cases |
| Adversarial | 12 | Negation attacks, spoofing, entity substitution |

---

## Development Workflow

```bash
make install-dev       # Install all dependencies
make format            # Auto-format with ruff
make lint              # Lint check
make typecheck         # mypy static analysis
make test-all          # Full test suite
make test-coverage     # Coverage report

# Full CI check
make lint && make typecheck && make test-all
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Chunking behavior, span visualization, BM25 indexing |
| `02_retrieval_analysis.ipynb` | BM25 vs Dense vs Hybrid (RRF) comparison |
| `03_verifier_calibration.ipynb` | Reliability diagrams, calibration methods, threshold analysis |
| `04_publication_plots.ipynb` | Paper-ready tables, ablation results, LaTeX export |

```bash
pip install jupyter
jupyter notebook notebooks/
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Install dev dependencies: `pip install -e ".[lite,full,dev]"`
4. Make changes with tests: ensure `make lint && make test-all` passes
5. Submit a pull request

**Code style**: Pydantic v2 models, type hints everywhere, Google-style docstrings, tests for all new functionality.

---

## Citation

```bibtex
@article{certirag2026,
  title={CertiRAG: Certified Retrieval-Augmented Generation 
         with Fail-Closed Claim Verification},
  author={Kumar, Aayush},
  year={2026},
  url={https://github.com/aayushakumar/CertiRAG}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with 🛡️ for trustworthy AI</strong>
</p>

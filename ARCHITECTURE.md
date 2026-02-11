# CertiRAG — Architecture & Component Guide

> A comprehensive reference for every module, class, and design decision in the CertiRAG codebase.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Core Library — `certirag/`](#core-library--certirag)
  - [Entry Points & Configuration](#entry-points--configuration)
  - [Schemas](#schemas--certiragschemas)
  - [Claim IR (Intermediate Representation)](#claim-ir--certiragclaim_ir)
  - [Ingest](#ingest--certiragingest)
  - [Retrieve](#retrieve--certiragretrieve)
  - [Verify](#verify--certiragverify)
  - [Render & Certificate](#render--certiragrender)
- [Evaluation Framework — `eval/`](#evaluation-framework--eval)
- [Test Suite — `tests/`](#test-suite--tests)
- [Streamlit UI — `ui/`](#streamlit-ui--ui)
- [Configuration Files](#configuration-files)
- [Data Flow Diagram](#data-flow-diagram)

---

## Project Overview

**CertiRAG** (Certified Retrieval-Augmented Generation) is a fail-closed verification pipeline that:

1. **Decomposes** LLM answers into atomic, typed claims
2. **Verifies** each claim against retrieved evidence using NLI models
3. **Blocks** unverified or contradicted claims before they reach the user
4. **Produces** tamper-evident audit certificates for every query

Two execution modes are supported:

| Mode   | Compute   | Verifiers                         | Use Case              |
|--------|-----------|-----------------------------------|-----------------------|
| `LITE` | CPU / API | Word-overlap, OpenAI, Gemini, Groq | Development, demos   |
| `FULL` | GPU       | DeBERTa NLI, MiniCheck, Ensemble  | Production, research  |

### Core Guarantee — Theorem 1 (Fail-Closed)

A claim is rendered as **VERIFIED** if and only if:
- Its entailment score ≥ τ\_entail (default 0.85)
- It has ≥ min_evidence_spans supporting spans (default 1)
- No contradiction score ≥ τ\_contradict (default 0.70)

All other claims are **UNVERIFIED** or **BLOCKED**. This policy is purely deterministic — no ML model or LLM can influence rendering decisions.

---

## Directory Structure

```
CertiRAG/
├── certirag/                  # Core library (pip-installable)
│   ├── __init__.py            # Package version & docstring
│   ├── __main__.py            # `python -m certirag` entry point
│   ├── cli.py                 # CLI subcommands (ingest, run, eval, validate)
│   ├── config.py              # Pydantic Settings configuration
│   ├── pipeline.py            # End-to-end orchestrator
│   ├── utils.py               # Seeding, hashing, logging helpers
│   ├── schemas/               # Pydantic v2 data contracts
│   │   ├── evidence.py        # EvidenceChunk, EvidenceSpan
│   │   ├── claim_ir.py        # ClaimIR, Claim, ClaimType
│   │   ├── verification.py    # VerificationResult, VerificationLabel
│   │   └── certificate.py     # AuditCertificate, RendererDecision
│   ├── claim_ir/              # Claim compilation & normalization
│   │   ├── compiler.py        # LLM-based claim decomposition
│   │   ├── gemini_compiler.py # Gemini API alternative compiler
│   │   ├── normalizer.py      # Atomicity, dedup, hedge detection
│   │   └── validator.py       # Schema + semantic validation
│   ├── ingest/                # Document ingestion pipeline
│   │   ├── chunker.py         # Sentence-aware chunking with offsets
│   │   ├── embedder.py        # Dense embeddings (OpenAI / local)
│   │   └── indexer.py         # BM25 + FAISS/numpy indices
│   ├── retrieve/              # Evidence retrieval
│   │   ├── hybrid.py          # Hybrid BM25+dense with RRF + reranking
│   │   └── span_extractor.py  # Sentence-level span validation
│   ├── verify/                # Claim verification engines
│   │   ├── verifier.py        # Abstract base class (Template Method)
│   │   ├── hf_nli_verifier.py # Local DeBERTa NLI (CPU, no API)
│   │   ├── nli_verifier.py    # Standard NLI baseline
│   │   ├── ensemble_verifier.py # Multi-model ensemble with calibration
│   │   ├── groq_verifier.py   # LLM-as-judge via Groq (free tier)
│   │   ├── gemini_verifier.py # LLM-as-judge via Gemini (free tier)
│   │   ├── minicheck.py       # MiniCheck factuality model
│   │   ├── llm_judge.py       # GPT-4o judge (accuracy ceiling)
│   │   ├── calibrator.py      # Platt scaling / isotonic calibration
│   │   └── mse.py             # Minimal Sufficient Evidence selection
│   └── render/                # Output rendering & certification
│       ├── policy.py          # Fail-closed policy engine (Theorem 1)
│       └── certificate.py     # Audit certificate builder
├── eval/                      # Evaluation & benchmarking framework
│   ├── benchmark.py           # Pluggable benchmark runner + CLI
│   ├── scoring.py             # Metrics: Acc, F1, AUROC, ECE
│   ├── optimizer.py           # Threshold grid search + optimization
│   ├── run_full_eval.py       # Full eval suite (all models × all datasets)
│   └── datasets/
│       └── loaders.py         # VitaminC, ANLI, FEVER-NLI dataset loaders
├── tests/                     # 134 tests (113 fast + 21 benchmark)
│   ├── conftest.py            # Shared fixtures & factories
│   ├── unit/                  # Schema, atomicity, metrics, renderer, spans
│   ├── integration/           # E2E verification, golden paths, corruption
│   ├── adversarial/           # Negation, spoofing, entity substitution
│   └── benchmark/             # NLI dataset evaluation tests
├── ui/                        # Streamlit interactive dashboard
│   ├── app.py                 # Main Streamlit app
│   ├── pipeline.py            # Demo pipeline (CPU-only, no API)
│   └── styles.py              # Cyberpunk CSS theme
├── eval_results/              # Saved benchmark JSON results
├── docs/                      # Additional documentation
├── pyproject.toml             # Build config, dependencies, pytest setup
├── README.md                  # Project overview & user guide
├── BENCHMARK_RESULTS.md       # Detailed benchmark analysis
├── CHANGELOG.md               # Version history
└── CONTRIBUTING.md            # Contribution guidelines
```

---

## Core Library — `certirag/`

### Entry Points & Configuration

#### `certirag/__init__.py`
Package root. Declares `__version__ = "0.1.0"` and documents the high-level architecture flow.

#### `certirag/__main__.py`
Enables `python -m certirag` by delegating to `cli.main()`.

#### `certirag/config.py`
Central configuration hub using **Pydantic Settings v2**. Environment variables use the `CERTIRAG_` prefix; YAML file overrides are supported.

| Config Class        | Key Fields                                                                 |
|---------------------|---------------------------------------------------------------------------|
| `ExecutionMode`     | Enum: `LITE` (CPU/API) or `FULL` (GPU/local)                             |
| `RetrievalConfig`   | `chunk_size` (512), `chunk_overlap` (64), `top_k` (5), `rrf_k` (60)     |
| `ClaimConfig`       | `max_claims` (20), `temperature` (0.1), `model` (gpt-4o-mini)           |
| `VerificationConfig`| `tau_entail` (0.85), `tau_contradict` (0.70), `batch_size` (32)          |
| `RenderConfig`      | `default_display_mode` (mixed), `min_evidence_spans` (1)                 |
| `CertiRAGConfig`    | Root model combining all above; `config_hash()` → SHA-256 fingerprint    |

**Design:** `get_config(yaml_path)` factory merges defaults → env vars → YAML overrides. `config_hash()` produces a deterministic SHA-256 for reproducibility stamping in certificates.

#### `certirag/cli.py`
Argparse CLI with five subcommands:

| Command           | Description                                           |
|-------------------|-------------------------------------------------------|
| `certirag ingest` | Read documents from JSONL or directory, build indices |
| `certirag run`    | Run a single query through the full pipeline          |
| `certirag eval`   | Run evaluation benchmarks (ALCE/RAGTruth/AggreFact)  |
| `certirag validate` | Schema-validate JSON files                         |
| `certirag export-schemas` | Export Pydantic JSON schemas to files         |

#### `certirag/pipeline.py`
**The main orchestrator.** Manages the complete 7-stage pipeline:

```
Query → Retrieve → Compile Claims → Normalize → Verify → MSE Select → Render → Certificate
```

Key components:

- **`CertiRAGPipeline`** — Lazy-initializes all components; configurable via `CertiRAGConfig`
- **`_init_verifier()`** — Priority chain: FULL mode GPU models → Gemini API → OpenAI API → local HF NLI → MiniCheck-lite
- **`PipelineResult`** — Dataclass holding the complete output: claims, evidence, verification results, renderer decisions, certificate, and timing data
- **`run(question, mode)`** — Executes the full pipeline, returns `PipelineResult`

#### `certirag/utils.py`
Shared utilities:

| Function               | Purpose                                                     |
|------------------------|-------------------------------------------------------------|
| `set_all_seeds(seed)`  | Seeds Python, NumPy, and optionally PyTorch for determinism |
| `generate_run_id()`    | Format: `certirag-{timestamp}-{short_uuid}`                |
| `compute_hash(data)`   | Truncated SHA-256 for config/cert hashing                  |
| `compute_content_hash` | Content-addressable hash for certificate integrity          |
| `setup_logging()`      | Structured logging (JSON or text format)                   |

---

### Schemas — `certirag/schemas/`

The four Pydantic v2 data contracts that flow through the pipeline. All schemas enforce structural invariants via model validators at construction time.

#### `evidence.py` — Evidence Chunks

```
EvidenceChunk
├── chunk_id: str          # Unique chunk identifier
├── doc_id: str            # Source document ID
├── text: str              # Full chunk text
├── source: SourceInfo     # Provenance (title, URL, section)
├── spans: list[EvidenceSpan]  # Sentence-level spans
│   ├── span_id: str
│   ├── start: int         # Character offset (inclusive)
│   ├── end: int           # Character offset (exclusive)
│   └── sentence: str      # Extracted sentence text
└── retrieval: RetrievalScores  # BM25, dense, rerank, RRF scores
```

**Invariant (T-U4):** `chunk.text[span.start:span.end] == span.sentence` — enforced by validators at construction.

#### `claim_ir.py` — Claim Intermediate Representation

```
ClaimIR
├── question: str              # Original user query
├── claims: list[Claim]
│   ├── id: str                # Unique claim identifier
│   ├── text: str              # Atomic claim text
│   ├── type: ClaimType        # factual, temporal_fact, numeric_fact, causal, ...
│   ├── fields: dict           # Type-specific structured fields
│   ├── evidence_candidates: list[CandidateEvidence]  # (chunk_id, span_id) pointers
│   ├── hedge: bool            # Contains hedging language
│   └── unsupported_by_design: bool  # Expected to lack support
├── model_id: str              # LLM model used for compilation
└── raw_answer: str            # Original LLM response
```

**Claim taxonomy** (`ClaimType`): factual, temporal_fact, numeric_fact, causal, comparative, definition, procedural, other.

#### `verification.py` — Verification Results

```
VerificationResult
├── claim_id: str
├── label: VerificationLabel   # ENTAILED | CONTRADICTED | NOT_ENOUGH_INFO
├── score: float [0, 1]       # Confidence score
├── best_evidence: list[EvidenceScore]  # MSE-selected set
├── all_scores: list[EvidenceScore]
│   ├── chunk_id / span_id    # Evidence pointer
│   ├── label: VerificationLabel
│   ├── score: float [0, 1]
│   └── raw_score: float      # Pre-calibration score
└── Properties: max_entail_score, max_contradict_score, has_contradiction
```

#### `certificate.py` — Audit Certificates

```
AuditCertificate
├── query_id: str              # Unique query identifier
├── question: str              # User's question
├── timestamp: datetime
├── retrieval_summary: dict    # Retrieved doc/chunk counts
├── claims: list[dict]         # Serialized claims
├── verification_results: list[dict]
├── renderer_decisions: list[RendererDecision]
│   ├── claim_id: str
│   ├── render_state: RenderState  # VERIFIED | UNVERIFIED | BLOCKED
│   ├── policy: PolicySnapshot     # Frozen threshold values
│   └── reason: str
├── config_hash: str           # SHA-256 of config at runtime
├── model_versions: dict       # Verifier model identifiers
├── integrity_hash: str        # SHA-256 of entire certificate content
└── Methods: seal(), verify_integrity()
```

**Tamper detection:** `seal()` computes `integrity_hash` over all content fields. `verify_integrity()` recomputes and compares.

---

### Claim IR — `certirag/claim_ir/`

#### `compiler.py` — Claim Compiler

Decomposes LLM answers into atomic, evidence-linked `ClaimIR` using structured few-shot prompting.

- **Few-shot examples** cover OpenAI founding facts and Apple revenue claims
- **8 compilation rules** enforce atomicity, evidence linkage, type classification
- **Two backends:** OpenAI API (LITE, JSON mode) and local HuggingFace (FULL)
- **Parsing pipeline:** LLM output → JSON extraction → schema validation → claim cap

#### `gemini_compiler.py` — Gemini Claim Compiler

Drop-in alternative using Google Gemini API (free tier). Inherits from `ClaimCompiler`, overrides API call only — reuses all prompts, parsing, and validation logic.

#### `normalizer.py` — Claim Normalizer

Deterministic post-processing pipeline (no LLM calls):

| Step | Transformation | Example |
|------|---------------|---------|
| 1 | Strip empty claims | `""` → removed |
| 2 | Detect hedges | `"X might be Y"` → `hedge=True` |
| 3 | Remove hedge words | `"X probably causes Y"` → `"X causes Y"` |
| 4 | Split conjunctions | `"A is B, and C is D"` → 2 claims |
| 5 | Normalize text | Collapse whitespace, add period, capitalize |
| 6 | Deduplicate | Jaccard token overlap > threshold → remove |
| 7 | Re-number IDs | Sequential `claim_001`, `claim_002`, ... |
| 8 | Flag long claims | Warn if > threshold tokens |

**Notable patterns:**
- Conjunction splitting uses `", and [A-Z]"`, `"; [A-Z]"`, `", but [A-Z]"` — preserves entity conjunctions like "Alice and Bob"
- Deduplication uses Jaccard similarity on lowercased token sets

#### `validator.py` — Schema & Semantic Validator

Validates data against Pydantic JSON schemas plus semantic checks:
- Conjunction atomicity (no hidden compound claims)
- Evidence linkage (referenced span IDs exist)
- Unique claim IDs
- Certificate integrity hash verification
- Exports JSON schemas for external tooling

---

### Ingest — `certirag/ingest/`

#### `chunker.py` — Document Chunker

Sentence-aware chunking that never splits mid-sentence. Each sentence becomes an `EvidenceSpan` with exact character offsets.

**Algorithm:** Greedy sentence accumulation up to `chunk_size` (default 512 chars). Overlap is achieved by re-including trailing sentences from the previous chunk.

**Two sentence splitting modes:**
- `LITE`: Regex-based splitter (`[.!?]` boundaries)
- `FULL`: spaCy sentencizer (higher accuracy)

**Output:** List of `EvidenceChunk` objects, each containing 1+ `EvidenceSpan` with validated `(start, end)` character offsets into the chunk text.

#### `embedder.py` — Document Embedder

Dense embedding generation for the vector index.

| Mode   | Backend               | Model                       | Notes                |
|--------|-----------------------|-----------------------------|----------------------|
| `LITE` | OpenAI API            | `text-embedding-3-small`    | Batched API calls    |
| `FULL` | sentence-transformers | Configurable (e.g., e5)     | Local GPU inference  |

All embeddings are L2-normalized so cosine similarity = dot product.

#### `indexer.py` — Search Indices

Three index components:

| Component    | Implementation              | Purpose                         |
|-------------|-----------------------------|---------------------------------|
| `BM25Index`  | `rank_bm25.BM25Okapi`      | Keyword/term-frequency search   |
| `DenseIndex` | FAISS `IndexFlatIP` or NumPy | Exact inner-product vector search |
| `ChunkStore` | Dict-based metadata store   | Chunk/span lookup by ID         |

All indices support `save(path)` / `load(path)` for persistence.

---

### Retrieve — `certirag/retrieve/`

#### `hybrid.py` — Hybrid Retriever

Combines keyword and semantic retrieval with rank fusion:

```
Query → BM25 top-50 → ┐
                       ├─ RRF fusion → Top-K → (optional) Cross-encoder rerank → Results
Query → Dense top-50 → ┘
```

**Reciprocal Rank Fusion (RRF):** `score(d) = Σ 1/(k + rank(d))` with `k=60`. Preferred over linear interpolation because it's rank-invariant.

**Cross-encoder reranking** (FULL mode only): Uses `bge-reranker-base` to re-score the top candidates for higher precision.

**`SimpleRetriever`:** BM25-only fallback for testing and LITE mode.

#### `span_extractor.py` — Span Extractor

Validates and repairs sentence-level spans within chunks. Enforces the critical invariant:

```
chunk.text[span.start:span.end] == span.sentence
```

If spans are invalid, re-extracts them from the chunk text using sentence splitting.

---

### Verify — `certirag/verify/`

The verification subsystem uses the **Strategy pattern** — all verifiers implement `BaseVerifier` and are interchangeable.

#### `verifier.py` — Abstract Base Class

The **Template Method pattern** base class:

```python
class BaseVerifier(ABC):
    @abstractmethod
    def verify_batch(self, pairs) -> list[EvidenceScore]:
        """Subclasses implement NLI inference here."""

    def verify_claims(self, claim_ir, evidence_chunks, max_pairs=None):
        """Template method: collect pairs → cap → batch verify → aggregate."""
```

`_aggregate_scores()` determines the overall label per claim: max entailment score vs. max contradiction score, with label assignment.

#### `hf_nli_verifier.py` — HuggingFace NLI Verifier (Recommended)

**The default verifier for local CPU inference.** No API keys needed, ~50ms per pair.

- **Default model:** `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (2-class: entail/not-entail)
- **Auto-detection:** Dynamically maps model's `id2label` config to find entailment/contradiction/neutral indices, supporting both 2-class and 3-class models
- **NLI format:** premise = evidence, hypothesis = claim
- **Batched inference** with configurable batch size

#### `nli_verifier.py` — Standard NLI Baseline

Ablation baseline using `DeBERTa-v3-large-MNLI`. Maps standard 3-class NLI outputs (entailment/neutral/contradiction) to verification labels.

#### `ensemble_verifier.py` — Ensemble NLI Verifier

Multi-model ensemble with weighted probability averaging and optional calibration.

**Three presets:**

| Preset     | Models                                           | Weights | Use Case              |
|------------|--------------------------------------------------|---------|----------------------|
| `accurate` | DeBERTa-v3-base-mnli-fever-anli (3-class)        | 1.0     | Best 3-class accuracy |
| `sota`     | 3-class DeBERTa (0.6) + 2-class zeroshot (0.4)  | 0.6/0.4 | Highest AUROC         |
| `large`    | DeBERTa-v3-large-mnli-fever-anli-ling-wanli      | 1.0     | Maximum accuracy      |

**Ensemble mechanism:**
1. Each model produces `(N, 3)` probability array `[P(entail), P(neutral), P(contradict)]`
2. For 2-class models, the "not-entailment" probability is split equally between neutral and contradiction
3. Arrays are weighted and averaged across models
4. Optional isotonic regression calibration via `fit_calibrator()`

#### `groq_verifier.py` — Groq LLM-as-Judge

Fact verification via Groq's **free-tier** inference API using large language models.

| Model Alias     | Groq Model ID                   | Parameters |
|----------------|----------------------------------|------------|
| `llama-3.3-70b` | `llama-3.3-70b-versatile`       | 70B        |
| `mixtral-8x7b`  | `mixtral-8x7b-32768`            | 46.7B MoE  |
| `gemma2-9b`     | `gemma2-9b-it`                  | 9B         |

Features:
- **OpenAI-compatible API** — uses the openai SDK with Groq base URL
- **Rate limiting** — Token-bucket at 30 RPM (configurable)
- **Batch prompting** — Multiple pairs in a single prompt for efficiency
- **JSON parsing** — Structured output with markdown stripping and fallback parsing

#### `gemini_verifier.py` — Gemini LLM-as-Judge

Similar to Groq but uses Google Gemini API (free tier, 15 RPM). One API call per claim-evidence pair with structured JSON output.

#### `minicheck.py` — MiniCheck Verifier

Based on [MiniCheck (Tang et al., 2024)](https://arxiv.org/abs/2404.10774), a ~400M parameter model specifically trained for document-grounded fact verification.

| Mode   | Implementation          | Notes                            |
|--------|------------------------|----------------------------------|
| `FULL` | HuggingFace GPU model   | Full MiniCheck inference         |
| `LITE` | Word-overlap heuristic  | Development placeholder only     |

#### `llm_judge.py` — GPT-4o Judge

LLM-as-judge using GPT-4o via OpenAI API. Serves as the **accuracy ceiling** for ablation experiments. Most expensive option (1 API call per pair).

#### `calibrator.py` — Confidence Calibration

Calibrates raw verifier confidence scores to reflect true probabilities.

| Method              | Implementation                    | Use Case                         |
|--------------------|-----------------------------------|----------------------------------|
| Platt scaling       | Temperature `T` via NLL minimization | Parametric, few parameters      |
| Isotonic regression | `sklearn.isotonic.IsotonicRegression` | Non-parametric, more flexible   |

Also provides `compute_ece()` for Expected Calibration Error measurement.

#### `mse.py` — Minimal Sufficient Evidence

Greedy algorithm to find the **smallest set** of entailing evidence spans that achieves the τ\_entail threshold.

**Algorithm:**
1. Sort entailing spans by score (descending)
2. Accumulate spans until cumulative evidence meets threshold or `max_spans` reached
3. Return the minimal sufficient set

Reports `compute_stats()`: average spans per claim, max spans, % single-span claims.

---

### Render — `certirag/render/`

#### `policy.py` — Fail-Closed Policy Engine

**The heart of CertiRAG's safety guarantee.** A purely deterministic policy engine — no ML, no LLM, no randomness.

```python
def decide(result: VerificationResult) -> RendererDecision:
    if result.max_contradict_score >= tau_contradict:
        return BLOCKED          # Contradiction detected
    if (result.max_entail_score >= tau_entail
        and len(result.best_evidence) >= min_spans
        and not result.has_contradiction):
        return VERIFIED         # Sufficient evidence
    return UNVERIFIED           # Insufficient evidence
```

**Display modes** for `filter_for_display()`:

| Mode    | Shows              | Use Case                |
|---------|-------------------|------------------------|
| `strict` | VERIFIED only     | Production (safest)    |
| `mixed`  | VERIFIED + UNVERIFIED | Default display     |
| `debug`  | All including BLOCKED | Development/debugging |

**Critical invariant:** LLM output cannot influence rendering decisions. Only this engine assigns badges.

#### `certificate.py` — Certificate Builder

Constructs `AuditCertificate` objects — machine-readable proofs of correct policy application.

- **`build()`** — Assembles the full certificate with retrieval summary, stats, model versions
- **`seal()`** — Computes integrity hash via `compute_content_hash()`
- **`export_json()`** — Serializes to JSON file
- **`verify_certificate()`** — Loads and verifies certificate integrity (tamper detection)

---

## Evaluation Framework — `eval/`

### `benchmark.py` — Benchmark Runner

Pluggable benchmark framework that runs any `BaseVerifier`-compatible model against standardized NLI datasets.

**CLI usage:**
```bash
python eval/benchmark.py --model hf_nli --dataset vitaminc --max-examples 200
```

**Available model choices:**
`hf_nli`, `minicheck_lite`, `gemini`, `groq`, `groq_mixtral`, `groq_gemma`, `ensemble_accurate`, `ensemble_sota`, `ensemble_large`

**Workflow:** Load dataset → instantiate verifier → evaluate examples → compute scores → save JSON results.

Uses `VerifierProtocol` (runtime-checkable Protocol) for duck-typed verifier compatibility.

### `scoring.py` — Metrics Framework

Comprehensive metrics computation:

| Metric Category   | Metrics                                                |
|-------------------|--------------------------------------------------------|
| Overall           | Accuracy, macro-F1                                     |
| Per-class         | Precision, Recall, F1, Support (for each label)       |
| Binary            | Binary accuracy (entailed vs. not-entailed)            |
| Probabilistic     | AUROC (using entailment scores), ECE                   |
| Auxiliary          | Confusion matrix, average latency, total examples      |

Key data structures:
- **`BenchmarkPrediction`** — Per-example: gold label, predicted label, scores, latency
- **`BenchmarkScores`** — Full scoring report with `summary_table()` and `to_dict()`
- **`compare_models()`** — Multi-model comparison leaderboard

### `optimizer.py` — Threshold Optimizer

Grid search over entailment and contradiction thresholds to maximize model performance.

**Algorithm:**
1. **Stratified split:** Partition predictions into calibration (40%) and test (60%) sets
2. **Grid search:** Sweep `entailment_threshold` × `contradict_threshold` over the calibration set
3. **Optimize:** Select thresholds maximizing macro-F1
4. **Evaluate:** Apply optimal thresholds to held-out test set

### `run_full_eval.py` — Full Evaluation Suite

End-to-end evaluation script: all models × all datasets with threshold optimization and calibration.

**Model registry:** `deberta_2class`, `deberta_3class`, `ensemble_sota`, `deberta_large`, Groq variants.

**For each (model, dataset) combination, runs:**
1. Baseline evaluation (default thresholds)
2. Optimized evaluation (grid-searched thresholds)
3. Calibrated evaluation (isotonic regression)

Produces a complete leaderboard saved to `eval_results/full_eval_report.json`.

### `datasets/loaders.py` — Dataset Loaders

Downloads and normalizes public NLI benchmarks:

| Dataset     | Source       | Labels                         | Size    |
|-------------|-------------|--------------------------------|---------|
| VitaminC    | HuggingFace | SUPPORTS, REFUTES, NEI         | ~400K   |
| ANLI R1/R2/R3 | HuggingFace | Entailment (0), Neutral (1), Contradiction (2) | ~170K |
| FEVER-NLI   | HuggingFace | SUPPORTS, REFUTES, NOT ENOUGH INFO | ~185K |

All datasets are normalized to `BenchmarkExample` with fields: `claim`, `evidence`, `gold_label`, `dataset`, `example_id`.

**`DATASET_REGISTRY`** — Dict mapping short names to loader functions for programmatic access.

**Notable fix:** `_hf_load_dataset()` works around the local `eval/datasets/` directory shadowing the HuggingFace `datasets` package by manipulating `sys.modules`.

---

## Test Suite — `tests/`

134 total tests: **113 fast** (run in ~5s) + **21 benchmark/slow** (excluded by default).

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures, factories, markers
├── unit/                    # Fast, isolated tests
│   ├── test_schemas.py      # 23 tests — Pydantic data contracts
│   ├── test_atomicity.py    # 13 tests — Claim normalization
│   ├── test_metrics.py      # 20 tests — Eval metrics
│   ├── test_renderer.py     # 16 tests — Policy engine (Theorem 1)
│   └── test_spans.py        # 12 tests — Span offset tracking
├── integration/             # Cross-component tests
│   ├── test_corruption.py   # 8 tests — Edge cases & error handling
│   ├── test_e2e_verification.py # 5 tests — Real NLI model (20 cases)
│   └── test_golden.py       # 5 tests — Known-correct golden paths
├── adversarial/             # Security & robustness
│   └── test_adversarial.py  # 7 tests — Attacks & spoofing
└── benchmark/               # NLI evaluation (slow, excluded by default)
    └── test_benchmark.py    # 21 tests — Dataset loading, model benchmarks
```

### `conftest.py` — Shared Test Infrastructure

- **Markers:** `unit`, `integration`, `adversarial`, `benchmark`, `smoke`, `slow`, `gpu`
- **Environment:** Forces `CERTIRAG_MODE=lite` for all tests
- **Fixtures:** `config`, `sample_documents` (Paris/Berlin/Tokyo), `sample_chunk`, `sample_chunks`, `sample_claim_ir`, `sample_verification_results`
- **Factories:** `make_chunk(text, doc_id)`, `make_claim(text, id)`, `make_verification(claim_id, entail, contradict)`

### Unit Tests

| Test File | Tests | What It Validates |
|-----------|-------|-------------------|
| `test_schemas.py` | 23 | All four Pydantic contracts: valid construction, required fields, type validation, cross-field validators, serialization round-trips, invariant preservation |
| `test_atomicity.py` | 13 | Conjunction splitting (", and [A-Z]", ", but"), entity preservation ("Alice and Bob"), hedge detection, Jaccard deduplication, text normalization |
| `test_metrics.py` | 20 | `eval.metrics` module: claim accuracy, faithfulness rate, ECE, AUROC, verification yield, MSE efficiency, latency stats — perfect/zero/partial/empty cases |
| `test_renderer.py` | 16 | Theorem 1 boundary cases, display mode filtering (strict/mixed/debug), batch rendering, parameterized safety invariant (4 cases that must NOT be VERIFIED) |
| `test_spans.py` | 12 | Span offset invariant (T-U4), `start < end`, length matches, chunker produces valid non-overlapping spans, `SpanExtractor` validation |

### Integration Tests

| Test File | Tests | What It Validates |
|-----------|-------|-------------------|
| `test_corruption.py` | 8 | Empty/whitespace docs, very long docs, special characters, missing evidence → fail-closed, missing chunk IDs |
| `test_e2e_verification.py` | 5 | Real DeBERTa-v3-xsmall model on 20 hand-curated cases (7 entailed, 5 contradicted, 8 NEI). Asserts binary accuracy ≥ 60%, entailment F1 ≥ 0.5 |
| `test_golden.py` | 5 | BM25 retrieves Paris for "capital of France", renderer correctly filters verified/unsupported, normalizer splits compound claims |

### Adversarial Tests

| Test | Attack Type | Expectation |
|------|-------------|-------------|
| Negation insertion | Negate a supported claim | Must NOT be VERIFIED |
| Entity substitution | Swap entities in supported claim | Must NOT be VERIFIED |
| Score spoofing | EvidenceScore > 1.0 or < 0 | Pydantic rejects |
| Certificate tampering | Modify sealed certificate | `verify_integrity()` fails |
| Multi-hop gap | Partial evidence for multi-hop claim | Not VERIFIED |
| Hedged claims | Claims with "might", "possibly" | Flagged during normalization |

### Benchmark Tests (Slow — excluded by default)

Run via `python -m pytest tests/benchmark/ -m slow` or `python -m pytest tests/benchmark/ -m smoke`.

| Test Class | Tests | Purpose |
|-----------|-------|---------|
| `TestDatasetLoading` | 4 | Verify VitaminC, ANLI R1, FEVER-NLI download and parse correctly |
| `TestBenchmarkSmoke` | 3 | Quick 50-example smoke tests per dataset |
| `TestVitaminCBenchmark` | 2 | Full 200-example benchmark with quality gates |
| `TestANLIBenchmark` | 3 | ANLI R1/R2/R3 benchmarks |
| `TestFEVERBenchmark` | 1 | FEVER-NLI benchmark |
| `TestModelComparison` | 1 | Multi-model leaderboard |
| `TestScoringFramework` | 5 | Scoring module validation (runs fast) |
| `TestFullBenchmarkSuite` | 1 | End-to-end full suite |

**pytest configuration** (in `pyproject.toml`):
```toml
[tool.pytest.ini_options]
addopts = "-v --tb=short --strict-markers -m 'not slow and not smoke'"
```

---

## Streamlit UI — `ui/`

### `app.py` — Main Dashboard

Interactive glassmorphic dashboard for claim-level verification.

**Features:**
- Sidebar controls: execution mode, display mode, threshold sliders, top-k, advanced options
- Hero banner with animated gradient
- Pipeline step visualization (Ingest → Retrieve → Claims → Normalize → Verify → Render → Certify)
- Metrics dashboard (verified/blocked/unverified counts)
- Claim cards with color-coded badges and linked evidence spans
- Live audit certificate display

### `pipeline.py` — Demo Pipeline Runner

CPU-only pipeline for the demo UI (no GPU or API keys required):
- BM25 retrieval
- Heuristic claim extraction (sentence splitting, not LLM)
- Jaccard word-overlap verification (not NLI model)
- Full certificate generation

### `styles.py` — Cyberpunk CSS Theme

800+ lines of CSS implementing a futuristic dark theme:
- Glassmorphism effects (frosted glass cards)
- Neon accent colors (cyan, purple, green, amber, red)
- Animated gradient backgrounds and grid overlays
- JetBrains Mono + Inter font stack

**Launch:** `streamlit run ui/app.py`

---

## Configuration Files

### `pyproject.toml`

| Section | Key Settings |
|---------|-------------|
| Build | setuptools + wheel |
| Project | `certirag` v0.1.0, Python ≥3.10, MIT license |
| Core deps | pydantic ≥2.5, pydantic-settings, jsonschema, pyyaml, rich, click, tqdm |
| `[lite]` extras | spacy, rank-bm25, numpy, scikit-learn, openai, tiktoken, httpx, streamlit, plotly |
| `[full]` extras | torch, transformers, sentence-transformers, faiss-cpu, accelerate, datasets |
| Console script | `certirag = certirag.cli:main` |
| pytest | Strict markers, verbose, excludes slow/smoke by default |

### Other Config Files

| File | Purpose |
|------|---------|
| `README.md` | User-facing guide: installation, CLI, architecture, benchmarks |
| `BENCHMARK_RESULTS.md` | Detailed evaluation results and model comparison |
| `CHANGELOG.md` | Version history and release notes |
| `CONTRIBUTING.md` | Contribution guidelines and development setup |
| `.github/` | GitHub Actions workflows and templates |

---

## Data Flow Diagram

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────┐
│   Retrieve   │────▶│ Evidence     │
│  (hybrid.py) │     │ Chunks       │
└─────────────┘     └──────┬───────┘
                           │
    ┌──────────────────────┘
    │
    ▼
┌─────────────┐     ┌──────────────┐
│  Compile     │────▶│ ClaimIR      │
│ (compiler.py)│     │ (atomic      │
└─────────────┘     │  claims)     │
                    └──────┬───────┘
                           │
    ┌──────────────────────┘
    │
    ▼
┌─────────────┐     ┌──────────────┐
│  Normalize   │────▶│ Cleaned      │
│(normalizer.py│     │ ClaimIR      │
└─────────────┘     └──────┬───────┘
                           │
    ┌──────────────────────┤
    │                      │
    ▼                      ▼
┌─────────────┐     ┌──────────────┐
│   Verify     │────▶│ Verification │
│ (verifier.py)│     │ Results      │
└─────────────┘     └──────┬───────┘
                           │
    ┌──────────────────────┘
    │
    ▼
┌─────────────┐     ┌──────────────┐
│  MSE Select  │────▶│ Minimal      │
│   (mse.py)   │     │ Evidence Set │
└─────────────┘     └──────┬───────┘
                           │
    ┌──────────────────────┘
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Render     │────▶│ Renderer     │────▶│   Audit      │
│ (policy.py)  │     │ Decisions    │     │ Certificate  │
└─────────────┘     │ ✅ VERIFIED  │     │ (sealed &    │
                    │ ⚠️ UNVERIFIED│     │  tamper-proof)│
                    │ ❌ BLOCKED   │     └──────────────┘
                    └──────────────┘
```

---

## Key Design Patterns

| Pattern | Where Used | Why |
|---------|-----------|-----|
| **Strategy** | `verify/` — all verifiers are interchangeable | Swap NLI models without changing pipeline |
| **Template Method** | `BaseVerifier.verify_claims()` | Common orchestration; subclasses implement only `verify_batch()` |
| **Builder** | `CertificateBuilder` | Complex certificate construction with sealing |
| **Factory** | `_init_verifier()`, `create_verifier()` | Runtime verifier selection based on mode/config |
| **Pydantic Validators** | All schemas | Structural invariants enforced at construction |
| **Content-Addressable Hashing** | Certificates, config | Tamper detection and reproducibility |

---

*This document was auto-generated from the CertiRAG codebase. For usage instructions, see [README.md](README.md). For benchmark results, see [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).*

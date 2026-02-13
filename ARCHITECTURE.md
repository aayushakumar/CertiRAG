# CertiRAG — Complete Architecture & Codebase Walkthrough

> **Fail-Closed Grounding for RAG with Verified Rendering**
>
> CertiRAG decomposes LLM answers into atomic claims, verifies each claim against retrieved evidence, and renders only verified claims with a trust badge — enforced at the presentation layer, not by the LLM.

---

## 1. High-Level Architecture

```mermaid
flowchart LR
    Q["🔍 User Query"] --> R["📚 Retrieve"]
    R --> C["✂️ Compile Claims"]
    C --> N["🧹 Normalize"]
    N --> V["✅ Verify"]
    V --> M["📎 MSE Select"]
    M --> P["🏛️ Policy Engine"]
    P --> CERT["📜 Certificate"]

    style Q fill:#4a90d9,color:#fff
    style R fill:#6c5ce7,color:#fff
    style C fill:#e17055,color:#fff
    style N fill:#fdcb6e,color:#333
    style V fill:#00b894,color:#fff
    style M fill:#0984e3,color:#fff
    style P fill:#d63031,color:#fff
    style CERT fill:#2d3436,color:#fff
```

The pipeline is orchestrated by [`CertiRAGPipeline`](certirag/pipeline.py) with 7 sequential stages. Every stage feeds structured Pydantic objects to the next — no raw LLM output ever reaches the renderer.

---

## 2. Execution Modes

| Aspect | **LITE** (CPU / API) | **FULL** (GPU) |
|---|---|---|
| Embeddings | OpenAI API | sentence-transformers (local) |
| Dense Index | NumPy brute-force | FAISS |
| Sentence Splitting | Regex | spaCy |
| Claim Compiler | OpenAI / Gemini API | Local HuggingFace model |
| Verifier | Word-overlap heuristic | MiniCheck / Cascade / Gemini |
| Reranker | ❌ Disabled | Cross-encoder reranking |

Controlled by `execution_mode` in [`config.py`](certirag/config.py).

---

## 3. Data Model Flow

```mermaid
flowchart TD
    subgraph Schemas["📦 Pydantic Schemas"]
        EC["EvidenceChunk\n─ chunk_id, text\n─ source, spans\n─ retrieval scores"]
        CIR["ClaimIR\n─ question, claims[]\n─ model_id, raw_answer"]
        VR["VerificationResult\n─ claim_id, label\n─ score, best_evidence\n─ all_scores"]
        RD["RendererDecision\n─ claim_id, render_state\n─ policy snapshot\n─ entail/contradict scores"]
        AC["AuditCertificate\n─ query_id, question\n─ claims, verif results\n─ decisions, integrity_hash"]
    end

    EC --> CIR
    CIR --> VR
    VR --> RD
    RD --> AC

    style EC fill:#a29bfe,color:#fff
    style CIR fill:#fd79a8,color:#fff
    style VR fill:#55efc4,color:#333
    style RD fill:#ffeaa7,color:#333
    style AC fill:#636e72,color:#fff
```

Schema files:

- [`evidence.py`](certirag/schemas/evidence.py) — `SourceInfo`, `RetrievalScores`, `EvidenceSpan`, `EvidenceChunk`
- [`claim_ir.py`](certirag/schemas/claim_ir.py) — `ClaimType`, `Claim`, `ClaimIR`
- [`verification.py`](certirag/schemas/verification.py) — `VerificationLabel`, `EvidenceScore`, `VerificationResult`
- [`certificate.py`](certirag/schemas/certificate.py) — `RenderState`, `PolicySnapshot`, `RendererDecision`, `AuditCertificate`

---

## 4. Module Deep-Dives

### 4.1 Ingest Pipeline

```mermaid
flowchart LR
    DOC["📄 Raw Document"] --> CHUNK["DocumentChunker\nSentence-aligned splits"]
    CHUNK --> EMBED["DocumentEmbedder\nDense vectors"]
    EMBED --> IDX["Indexer\nBM25 + Dense"]

    CHUNK -->|EvidenceChunk| STORE["ChunkStore\nMetadata DB"]

    style DOC fill:#dfe6e9,color:#333
    style CHUNK fill:#74b9ff,color:#333
    style EMBED fill:#a29bfe,color:#fff
    style IDX fill:#6c5ce7,color:#fff
    style STORE fill:#636e72,color:#fff
```

| File | Class | Key Behavior |
|---|---|---|
| [`chunker.py`](certirag/ingest/chunker.py) | `DocumentChunker` | Sentence-aligned chunks with **character offsets**. Invariant: `chunk.text[span.start:span.end] == span.sentence` |
| [`embedder.py`](certirag/ingest/embedder.py) | `DocumentEmbedder` | L2-normalized embeddings (OpenAI API or sentence-transformers) |
| [`indexer.py`](certirag/ingest/indexer.py) | `BM25Index`, `DenseIndex`, `ChunkStore` | BM25 (rank_bm25), FAISS/NumPy dense, metadata store |

---

### 4.2 Retrieval

```mermaid
flowchart TD
    Q["Query"] --> BM25["BM25 Search\ntop-k keyword"]
    Q --> DENSE["Dense Search\ntop-k semantic"]
    BM25 --> RRF["Reciprocal Rank Fusion\nscore = Sum 1 / k+rank"]
    DENSE --> RRF
    RRF --> RERANK["Cross-Encoder Reranker\n(FULL mode only)"]
    RERANK --> SPANS["SpanExtractor\nSentence-level spans"]
    SPANS --> OUT["EvidenceChunks with\nRetrievalScores populated"]

    style Q fill:#4a90d9,color:#fff
    style BM25 fill:#e17055,color:#fff
    style DENSE fill:#6c5ce7,color:#fff
    style RRF fill:#00b894,color:#fff
    style RERANK fill:#fdcb6e,color:#333
    style SPANS fill:#0984e3,color:#fff
    style OUT fill:#2d3436,color:#fff
```

| File | Class | Key Behavior |
|---|---|---|
| [`hybrid.py`](certirag/retrieve/hybrid.py) | `HybridRetriever` | BM25 + dense → RRF fusion → cross-encoder rerank (FULL) |
| [`hybrid.py`](certirag/retrieve/hybrid.py) | `SimpleRetriever` | BM25-only fallback for LITE mode |
| [`span_extractor.py`](certirag/retrieve/span_extractor.py) | `SpanExtractor` | Validates sentence spans with T-U4 offset invariant |

---

### 4.3 Claim Compilation & Normalization

```mermaid
flowchart LR
    ANS["LLM Answer +\nEvidence"] --> COMP["ClaimCompiler\nFew-shot prompt → JSON"]
    COMP --> NORM["ClaimNormalizer"]
    NORM --> VAL["Validator\nSchema + heuristics"]

    subgraph Normalization Steps
        H["Hedge detection\nand removal"]
        S["Compound claim\nsplitting"]
        T["Text normalization\nlowercase, whitespace"]
        D["Deduplication\nword overlap ge 0.8"]
    end

    NORM --- H
    NORM --- S
    NORM --- T
    NORM --- D

    style ANS fill:#dfe6e9,color:#333
    style COMP fill:#e17055,color:#fff
    style NORM fill:#fdcb6e,color:#333
    style VAL fill:#00b894,color:#fff
```

| File | Class | Key Behavior |
|---|---|---|
| [`compiler.py`](certirag/claim_ir/compiler.py) | `ClaimCompiler` | Decomposes LLM answers into atomic `ClaimIR` via structured prompts |
| [`gemini_compiler.py`](certirag/claim_ir/gemini_compiler.py) | `GeminiClaimCompiler` | Alternative using Gemini API (free tier) |
| [`normalizer.py`](certirag/claim_ir/normalizer.py) | `ClaimNormalizer` | Hedge removal, splitting, dedup, text normalization |
| [`validator.py`](certirag/claim_ir/validator.py) | Functions | JSON schema validation + atomicity heuristics |

---

### 4.4 Verification

```mermaid
flowchart TD
    CLAIMS["Claims +\nEvidence Spans"] --> BASE["BaseVerifier\nverify_claims()"]
    BASE --> PAIRS["Collect all\n(claim, evidence) pairs"]
    PAIRS --> CAP["Cost cap\nmax_pairs le 240"]
    CAP --> BATCH["verify_batch()\n(abstract method)"]
    BATCH --> AGG["Aggregate scores\nper claim"]
    AGG --> VR["VerificationResult"]

    subgraph "Verifier Backends"
        MC["MiniCheckVerifier\nRoBERTa-large\n(FULL mode)"]
        MC_LITE["MiniCheck LITE\nword-overlap heuristic"]
        GV["GeminiVerifier\nLLM-as-Judge\n(free tier, 15 RPM)"]
        CV["CascadeVerifier\n2-class gate → 3-class resolver"]
    end

    BATCH --- MC
    BATCH --- MC_LITE
    BATCH --- GV
    BATCH --- CV

    style CLAIMS fill:#dfe6e9,color:#333
    style BASE fill:#636e72,color:#fff
    style BATCH fill:#00b894,color:#fff
    style VR fill:#55efc4,color:#333
    style MC fill:#6c5ce7,color:#fff
    style MC_LITE fill:#a29bfe,color:#fff
    style GV fill:#e17055,color:#fff
    style CV fill:#0984e3,color:#fff
```

| File | Class | Key Behavior |
|---|---|---|
| [`verifier.py`](certirag/verify/verifier.py) | `BaseVerifier` (ABC) | Pair collection → cost cap → batch scoring → aggregation |
| [`minicheck.py`](certirag/verify/minicheck.py) | `MiniCheckVerifier` | RoBERTa-large NLI model (~400M params); LITE heuristic fallback |
| [`gemini_verifier.py`](certirag/verify/gemini_verifier.py) | `GeminiVerifier` | LLM-as-Judge via Gemini 2.0 Flash; rate-limited |
| [`cascade_verifier.py`](certirag/verify/cascade_verifier.py) | `CascadeVerifier` | **Novel**: 2-class DeBERTa gate (fast entailment) → 3-class DeBERTa resolver (contradiction detection) |
| [`calibrator.py`](certirag/verify/calibrator.py) | `VerifierCalibrator` | Temperature scaling or isotonic regression; ECE computation |
| [`mse.py`](certirag/verify/mse.py) | `MSESelector` | Minimal Sufficient Evidence: greedy span selection until τ\_entail met |

#### Cascade Verifier Architecture (Novel Contribution)

```mermaid
flowchart TD
    ALL["All claim-evidence pairs"] --> GATE["Stage 1: 2-class DeBERTa\nzero-shot entailment gate"]
    GATE -->|"P entail ge 0.92"| PASS["ENTAILED\nhigh confidence"]
    GATE -->|"P entail lt 0.92"| RESOLVE["Stage 2: 3-class DeBERTa\ncontradiction resolver"]
    RESOLVE --> E2["ENTAILED"]
    RESOLVE --> C2["CONTRADICTED"]
    RESOLVE --> N2["NOT_ENOUGH_INFO"]

    style ALL fill:#dfe6e9,color:#333
    style GATE fill:#fdcb6e,color:#333
    style PASS fill:#00b894,color:#fff
    style RESOLVE fill:#6c5ce7,color:#fff
    style E2 fill:#00b894,color:#fff
    style C2 fill:#d63031,color:#fff
    style N2 fill:#636e72,color:#fff
```

> **Why cascade?** The 2-class model has 96.7% binary accuracy but cannot detect contradictions (F1=0). The 3-class model detects contradictions (F1=0.748) but is conservative on entailment. The cascade **routes** rather than **blends**, avoiding dilution of the 3-class signal.

---

### 4.5 Rendering & Policy Engine

```mermaid
flowchart TD
    VR["VerificationResult"] --> POLICY["RendererPolicy\nDETERMINISTIC"]

    POLICY --> CHECK1{"max contradict\nge tau_c?"}
    CHECK1 -->|Yes| BLOCKED["BLOCKED"]
    CHECK1 -->|No| CHECK2{"max entail ge tau_e\nAND evidence ge 1?"}
    CHECK2 -->|Yes| VERIFIED["VERIFIED"]
    CHECK2 -->|No| UNVERIFIED["UNVERIFIED"]

    VERIFIED --> CERT["CertificateBuilder\n→ AuditCertificate"]
    UNVERIFIED --> CERT
    BLOCKED --> CERT

    style VR fill:#55efc4,color:#333
    style POLICY fill:#d63031,color:#fff
    style BLOCKED fill:#e74c3c,color:#fff
    style VERIFIED fill:#27ae60,color:#fff
    style UNVERIFIED fill:#f39c12,color:#fff
    style CERT fill:#2d3436,color:#fff
```

**Theorem 1 (Fail-Closed):**

```
VERIFIED ⟺ (entail ≥ τ_e) ∧ (¬contradict ≥ τ_c) ∧ (evidence ≥ 1)
```

> **Important:** The renderer is the **trust boundary**. It reads **only** from `VerificationResult` objects, never from raw LLM output. The LLM cannot influence badge assignment.

| File | Class | Key Behavior |
|---|---|---|
| [`policy.py`](certirag/render/policy.py) | `RendererPolicy` | Deterministic decision logic; display modes: strict / mixed / debug |
| [`certificate.py`](certirag/render/certificate.py) | `CertificateBuilder` | Builds sealed `AuditCertificate` with integrity hash; supports verification |

---

## 5. End-to-End Pipeline Flow

```mermaid
sequenceDiagram
    participant U as User
    participant P as Pipeline
    participant R as Retriever
    participant CC as ClaimCompiler
    participant N as Normalizer
    participant V as Verifier
    participant MSE as MSESelector
    participant RP as RendererPolicy
    participant CB as CertificateBuilder

    U ->> P: query(question, documents)
    P ->> R: retrieve(question)
    R -->> P: EvidenceChunks

    P ->> CC: compile(question, answer, chunks)
    CC -->> P: ClaimIR

    P ->> N: normalize(ClaimIR)
    N -->> P: ClaimIR cleaned

    P ->> V: verify_claims(ClaimIR, chunks)
    V -->> P: VerificationResults

    P ->> MSE: select_batch(results)
    MSE -->> P: VerificationResults trimmed

    P ->> RP: render(results)
    RP -->> P: RendererDecisions

    P ->> CB: build(question, claims, results, decisions)
    CB -->> P: AuditCertificate sealed

    P -->> U: PipelineResult
```

---

## 6. Evaluation Framework

```mermaid
flowchart TD
    subgraph "Benchmark Datasets"
        VC["VitaminC"]
        ANLI["ANLI R1/R2/R3"]
        FV["FEVER-NLI"]
        SN["SNLI"]
    end

    subgraph "Eval Pipeline"
        BR["BenchmarkRunner\nPluggable verifier"]
        SP["score_predictions\nAll metrics"]
        CMP["compare_models\nCross-model tables"]
        BOOT["bootstrap_confidence_intervals\n95pct CIs"]
        MCN["mcnemars_test\nStatistical significance"]
    end

    VC --> BR
    ANLI --> BR
    FV --> BR
    SN --> BR
    BR --> SP
    SP --> CMP
    SP --> BOOT
    SP --> MCN

    style BR fill:#6c5ce7,color:#fff
    style SP fill:#00b894,color:#fff
```

| File | Purpose |
|---|---|
| [`benchmark.py`](eval/benchmark.py) | `BenchmarkRunner` — pluggable verifier evaluation against NLI datasets |
| [`scoring.py`](eval/scoring.py) | `score_predictions()` — accuracy, P/R/F1, AUROC, ECE, confusion matrix, bootstrap CIs, McNemar's test |
| [`metrics.py`](eval/metrics.py) | Shared metrics: faithfulness rate, AUROC, ECE, verification yield, MSE efficiency, latency stats |
| [`runner.py`](eval/runner.py) | `EvalRunner` — end-to-end pipeline evaluation on ALCE, RAGTruth, AggreFact |

---

## 7. Configuration System

```mermaid
flowchart TD
    ENV[".env file"] --> CS["CertiRAGConfig\n(Pydantic Settings)"]
    YAML["YAML overrides"] --> CS
    ENVVAR["Environment vars"] --> CS

    CS --> RC["RetrievalConfig\ntop_k, models, RRF k"]
    CS --> CC["ClaimConfig\nLLM model, max_claims"]
    CS --> VC["VerificationConfig\ntau_entail, tau_contradict, verifier"]
    CS --> RnC["RenderConfig\nmin_evidence, display_mode"]
    CS --> EC["EvalConfig\nbenchmarks, max_examples"]

    CS -->|"config_hash()"| HASH["SHA-256 digest\nfor reproducibility"]

    style CS fill:#e17055,color:#fff
    style HASH fill:#2d3436,color:#fff
```

Defined in [`config.py`](certirag/config.py). Defaults are tuned for LITE mode (Codespaces-friendly).

---

## 8. UI (Streamlit Dashboard)

[`app.py`](ui/app.py) provides a glassmorphic dark-themed interface with:

| Section | Function | Purpose |
|---|---|---|
| Sidebar | `render_sidebar()` | Mode selection, threshold sliders, API key input |
| Hero | `render_hero()` | Branding banner |
| Pipeline Steps | `render_pipeline_steps()` | Visual progress indicator |
| Metrics | `render_metrics()` | Verified / Unverified / Blocked counts |
| Claim Cards | `render_claims()` | Color-coded claim cards with scores |
| Evidence | `render_evidence()` | Evidence explorer with highlighted spans |
| Verification | `render_verification_detail()` | Per-claim NLI score breakdown |
| Certificate | `render_certificate()` | JSON audit certificate viewer |

---

## 9. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Fail-closed rendering** | LLM output never controls trust badges — safety invariant |
| **Sentence-level spans with offsets** | Enables precise UI highlighting; `chunk.text[start:end] == sentence` |
| **RRF fusion** | Robust combination of BM25 + dense without score normalization |
| **Cascade verifier** | Routes (not blends) — avoids 2-class model diluting contradiction signal |
| **MSE selection** | Minimizes certificate size while preserving verification quality |
| **Sealed certificates** | SHA-256 integrity hash enables tamper detection |
| **LITE/FULL modes** | Development on CPU, production on GPU — same pipeline, different backends |
| **Pydantic schemas** | Type safety + JSON schema export + automatic validation throughout |
| **Config hashing** | Exact SHA-256 of all params → bit-for-bit reproducibility tracking |

---

## 10. Repository Map

```
CertiRAG/
├── certirag/                  # Core package
│   ├── __init__.py            # Architecture overview
│   ├── pipeline.py            # 7-step orchestrator
│   ├── config.py              # Pydantic Settings configuration
│   ├── cli.py                 # CLI: ingest, run, eval, validate
│   ├── utils.py               # Seeds, hashing, logging, I/O
│   ├── schemas/               # Pydantic data models
│   │   ├── evidence.py        # EvidenceChunk, EvidenceSpan
│   │   ├── claim_ir.py        # ClaimIR, Claim, ClaimType
│   │   ├── verification.py    # VerificationResult, EvidenceScore
│   │   └── certificate.py     # AuditCertificate, RendererDecision
│   ├── ingest/                # Document processing
│   │   ├── chunker.py         # Sentence-aligned chunking
│   │   ├── embedder.py        # Dense embeddings
│   │   └── indexer.py         # BM25 + FAISS/NumPy indices
│   ├── retrieve/              # Evidence retrieval
│   │   ├── hybrid.py          # RRF fusion + cross-encoder reranking
│   │   └── span_extractor.py  # Sentence span extraction/validation
│   ├── claim_ir/              # Claim decomposition
│   │   ├── compiler.py        # LLM → atomic claims
│   │   ├── gemini_compiler.py # Gemini API variant
│   │   ├── normalizer.py      # Hedging, splitting, dedup
│   │   └── validator.py       # Schema + heuristic validation
│   ├── verify/                # Fact verification
│   │   ├── verifier.py        # BaseVerifier ABC
│   │   ├── minicheck.py       # MiniCheck (RoBERTa-large)
│   │   ├── gemini_verifier.py # LLM-as-Judge (Gemini)
│   │   ├── cascade_verifier.py# 2-stage cascade (novel)
│   │   ├── calibrator.py      # Score calibration (Platt/isotonic)
│   │   └── mse.py             # Minimal Sufficient Evidence
│   └── render/                # Trust boundary
│       ├── policy.py          # Deterministic policy engine
│       └── certificate.py     # Audit certificate builder
├── eval/                      # Evaluation framework
│   ├── benchmark.py           # BenchmarkRunner
│   ├── scoring.py             # Metrics + bootstrap CIs + McNemar
│   ├── metrics.py             # Shared metric functions
│   └── runner.py              # End-to-end eval orchestrator
├── ui/
│   └── app.py                 # Streamlit dashboard
├── configs/                   # YAML config presets
├── tests/                     # Test suite
└── notebooks/                 # Analysis notebooks
```

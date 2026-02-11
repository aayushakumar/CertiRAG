# CertiRAG — Implementation Plan for a Top-Tier Publication

> **Target venues:** ACL / EMNLP / NeurIPS / ICML (main conference) or TACL / JMLR (journal)
> **Estimated timeline:** 8–10 focused weeks (solo/small team)
> **Hardware assumption:** 1× A100 or Colab Pro (L4/A100); no multi-node training required

---

## Phase 0 — Project Scaffolding & Infrastructure (Days 1–3)

### 0.1 Repository Structure

```
CertiRAG/
├── certirag/
│   ├── __init__.py
│   ├── config.py                 # Pydantic settings, threshold defaults, model paths
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── evidence.py           # EvidenceChunk, Span dataclasses
│   │   ├── claim_ir.py           # ClaimIR, Claim, CandidateEvidence
│   │   ├── verification.py       # VerificationResult
│   │   └── certificate.py        # RendererDecision / AuditCertificate
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── chunker.py            # Sentence-aware chunking with offset tracking
│   │   ├── embedder.py           # Dense embedding (e5-base-v2 / bge-base)
│   │   └── indexer.py            # FAISS + BM25 (rank_bm25) index builder
│   ├── retrieve/
│   │   ├── __init__.py
│   │   ├── hybrid.py             # Hybrid BM25 + dense retrieval
│   │   ├── reranker.py           # Cross-encoder reranker (bge-reranker-base)
│   │   └── span_extractor.py     # Sentence-level span extraction with stable offsets
│   ├── claim_ir/
│   │   ├── __init__.py
│   │   ├── compiler.py           # LLM-based claim compiler (prompt → ClaimIR JSON)
│   │   ├── normalizer.py         # Claim splitter, deduplication, hedge removal
│   │   └── validator.py          # JSON-schema validation of Claim IR
│   ├── verify/
│   │   ├── __init__.py
│   │   ├── verifier.py           # Abstract verifier interface
│   │   ├── minicheck.py          # MiniCheck adapter (primary)
│   │   ├── nli_verifier.py       # NLI baseline (DeBERTa-v3-large-mnli)
│   │   ├── llm_judge.py          # LLM-as-judge baseline (GPT-4o / Llama-3)
│   │   ├── calibrator.py         # Temperature scaling / isotonic regression
│   │   └── mse.py                # Minimal Sufficient Evidence selection
│   ├── render/
│   │   ├── __init__.py
│   │   ├── policy.py             # Deterministic renderer policy engine
│   │   └── certificate.py        # Certificate serializer + integrity hash
│   ├── pipeline.py               # End-to-end orchestrator (query → certificate)
│   └── utils.py                  # Logging, hashing, reproducibility helpers
├── eval/
│   ├── __init__.py
│   ├── alce/
│   │   ├── loader.py             # ALCE dataset loader (ASQA, QAMPARI, ELI5)
│   │   ├── metrics.py            # Citation precision/recall, correctness, fluency
│   │   └── run_alce.py           # ALCE benchmark runner
│   ├── ragtruth/
│   │   ├── loader.py             # RAGTruth corpus loader
│   │   ├── metrics.py            # UER, CER, verifier catch rate
│   │   └── run_ragtruth.py       # RAGTruth benchmark runner
│   ├── aggrefact/
│   │   ├── loader.py             # LLM-AggreFact loader (verifier intrinsic eval)
│   │   └── run_aggrefact.py      # Verifier quality benchmark
│   ├── baselines/
│   │   ├── standard_rag.py       # Baseline 1: RAG + citations, no enforcement
│   │   ├── multi_query_rag.py    # Baseline 2: Query expansion RAG
│   │   └── posthoc_checker.py    # Baseline 3: Post-hoc fact-check, no fail-closed
│   ├── ablations.py              # Threshold sweeps, component ablations
│   ├── calibration_eval.py       # Reliability diagrams, ECE computation
│   └── plots.py                  # Publication-quality matplotlib/seaborn figures
├── ui/
│   ├── app.py                    # Streamlit/Gradio demo app
│   ├── components/               # Claim chips, evidence drawer, mode toggles
│   └── static/                   # CSS overrides, icons
├── tests/
│   ├── unit/
│   │   ├── test_schemas.py       # T-U1: Schema validation
│   │   ├── test_atomicity.py     # T-U2: Claim atomicity checks
│   │   ├── test_renderer.py      # T-U3: Renderer invariant tests
│   │   └── test_spans.py         # T-U4: Evidence span integrity
│   ├── integration/
│   │   ├── test_golden.py        # T-I1: End-to-end golden tests
│   │   ├── test_corruption.py    # T-I2: Retrieval corruption
│   │   └── test_removal.py       # T-I3: Evidence removal downgrades
│   └── adversarial/
│       ├── test_spoofing.py      # T-R1: Citation spoofing
│       ├── test_negation.py      # T-R2: Negation flips
│       ├── test_paraphrase.py    # T-R3: Paraphrase stress
│       └── test_multihop.py      # T-R4: Multi-hop synthesis
├── scripts/
│   ├── download_data.sh          # Download ALCE, RAGTruth, AggreFact
│   ├── ingest_corpus.py          # CLI: ingest documents
│   ├── run_pipeline.py           # CLI: run full pipeline
│   └── run_eval.py               # CLI: run benchmarks
├── configs/
│   ├── default.yaml              # Default experiment config
│   ├── strict.yaml               # Strict mode thresholds
│   ├── mixed.yaml                # Mixed mode thresholds
│   └── ablations/                # Per-ablation config overrides
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_retrieval_analysis.ipynb
│   ├── 03_verifier_calibration.ipynb
│   └── 04_money_plots.ipynb
├── data/                         # .gitignored; local data cache
├── outputs/                      # .gitignored; run artifacts
├── pyproject.toml
├── Makefile                      # make ingest, make eval, make test, make demo
├── Dockerfile
├── .env.example
├── .gitignore
├── LICENSE
├── PLAN.md
└── README.md
```

### 0.2 Toolchain Decisions

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Config | Pydantic + Hydra/YAML | Typed, composable, config hashes for reproducibility |
| Schemas | Pydantic v2 + JSON Schema export | Runtime validation + paper artifact |
| Embeddings | `intfloat/e5-base-v2` or `BAAI/bge-base-en-v1.5` | Strong open embedding, <500 MB |
| BM25 | `rank_bm25` | Pure Python, no server needed |
| Vector index | FAISS (flat or IVF) | Standard, fast, GPU-optional |
| Reranker | `BAAI/bge-reranker-base` | Strong, open, fits on consumer GPU |
| LLM (claim compiler) | `gpt-4o-mini` via API **or** `meta-llama/Llama-3.1-8B-Instruct` | Cheap; local option for reproducibility |
| Verifier (primary) | MiniCheck (`lytang/MiniCheck-RoBERTa-large`) | Sub-400M, designed for grounding, SOTA cost/quality |
| Verifier (ablation) | DeBERTa-v3-large-MNLI, GPT-4o judge | For ablation table |
| Calibration | scikit-learn isotonic + temperature scaling | Standard, well-understood |
| UI | Streamlit | Fast to build, paper demo video |
| Testing | pytest + hypothesis (property tests) | Industry standard |
| Plots | matplotlib + seaborn, LaTeX fonts | Publication-ready |

### 0.3 Day-1 Setup Tasks

- [ ] `pyproject.toml` with all deps, pinned versions
- [ ] `Makefile` with targets: `setup`, `ingest`, `test`, `eval`, `demo`, `plots`
- [ ] `.gitignore` (data/, outputs/, *.pyc, __pycache__, .env)
- [ ] Pydantic schemas for all 4 data contracts (§6 of README)
- [ ] JSON Schema auto-export for each schema
- [ ] Logging config (structured JSON logs with run_id)
- [ ] Seed management utility (set all seeds: Python, NumPy, torch, transformers)
- [ ] Config hash utility (SHA-256 of frozen config → stamped on every certificate)

---

## Phase 1 — Ingestion & Retrieval (Week 1)

### 1.1 Ingestion Pipeline

**Goal:** Turn raw documents into indexed, span-tracked chunks.

| Task | Details | Acceptance Criteria |
|------|---------|---------------------|
| Sentence-aware chunker | Use `spaCy` sentence splitter. Target chunk size ~256 tokens with sentence-boundary alignment. Track `(doc_id, chunk_id, start_char, end_char)` for each chunk. | Chunks never split mid-sentence; offsets round-trip correctly. |
| Span extraction | Each chunk produces sentence-level spans with `(span_id, start, end, sentence_text)`. | `chunk.text[span.start:span.end] == span.sentence` for all spans. |
| Dense embedding | Batch embed all chunks with `e5-base-v2`. Normalize to unit vectors. | Cosine similarity search works correctly on a 100-doc mini corpus. |
| FAISS indexing | Build `IndexFlatIP` (exact) for dev; `IndexIVFFlat` for larger corpora. Store chunk metadata in a sidecar SQLite or pickle. | k-NN retrieval returns correct chunks on 10 hand-crafted queries. |
| BM25 indexing | Tokenize chunks, build `rank_bm25.BM25Okapi` index. | BM25 retrieval returns expected keyword-matched chunks. |
| Provenance store | Serialize chunk metadata (doc_id, source title, URL, section) alongside index. | Metadata survives save/load cycle. |

**Unit tests:** `test_spans.py` — T-U4 (span integrity); offset round-trip property test.

### 1.2 Hybrid Retrieval + Reranking

| Task | Details | Acceptance Criteria |
|------|---------|---------------------|
| Hybrid scorer | Reciprocal rank fusion (RRF) of BM25 + dense rankings. Configurable α weight. | On synthetic queries, hybrid outperforms either alone (manual check on 20 queries). |
| Reranker | Cross-encoder `bge-reranker-base`. Re-score top-50 hybrid candidates → return top-k. | Reranked top-5 consistently contains ground-truth chunk on golden set. |
| Span-level extraction | After reranking, extract sentence-level spans from selected chunks. Return `EvidenceChunk` objects. | Spans are valid, offsets correct. |
| Hard caps | `max_chunks=10`, `max_spans_per_chunk=5` (configurable). | Never returns more than caps regardless of corpus. |

**Integration test:** `test_golden.py` — 20 queries against mini corpus; check recall@10.

### 1.3 Standard RAG Baseline

Build a simple `query → retrieve → generate answer with citations` baseline using the same retriever + an LLM. This is **Baseline 1** for the paper.

---

## Phase 2 — Claim IR Compiler & Normalizer (Week 2)

### 2.1 Claim Compiler

| Task | Details | Acceptance Criteria |
|------|---------|---------------------|
| Prompt design | Few-shot prompt instructing LLM to decompose answer into atomic claims with evidence pointers. Output must be valid ClaimIR JSON. Include 3 exemplars covering: simple fact, temporal fact, multi-evidence claim. | On 20 test questions, ≥90% outputs parse as valid ClaimIR. |
| Schema-constrained generation | Use JSON mode (OpenAI) or grammar-constrained generation (vLLM/llama.cpp GBNF). | 100% of outputs are valid JSON matching schema. |
| Evidence linking | Each claim must reference ≥1 `(chunk_id, span_id)` from the retrieved set. If LLM can't ground a claim, it must flag `"unsupported_by_design": true`. | No orphan claims (claims with zero evidence candidates unless explicitly flagged). |
| Claim typing | Tag each claim with type: `factual`, `temporal_fact`, `numeric_fact`, `causal`, `comparative`, `definition`, `procedural`. | Type distribution is plausible on 50-query sample. |

### 2.2 Claim Normalizer

| Task | Details | Acceptance Criteria |
|------|---------|---------------------|
| Conjunction splitter | Rule-based: split on "and", "which also", "because" when they join independent facts. Use dependency parse to distinguish conjunctive clauses from entity lists ("cats and dogs" should NOT split). | Precision ≥95% on 100 hand-labeled cases. |
| Hedge detector | Detect hedging language ("may", "possibly", "some suggest"). Mark `hedge: true` on claim or strip hedge for verification. | Correctly identifies hedges in 50 test claims. |
| Numeric normalizer | Normalize "2.5 billion" → `2500000000`, "Q3 2023" → `"2023-Q3"` etc. into `fields` dict. | 20 numeric normalization test cases pass. |
| Deduplication | Cosine similarity of claim embeddings; merge claims with sim > 0.92. | Duplicate claims from multi-query RAG are merged. |

**Unit tests:** `test_atomicity.py` — T-U2 (no "and/because/which" in claims); max-token-length enforcement.

**Schema test:** `test_schemas.py` — T-U1 (validate against exported JSON Schema).

---

## Phase 3 — Verifier Integration & Calibration (Week 3)

### 3.1 MiniCheck Verifier (Primary)

| Task | Details | Acceptance Criteria |
|------|---------|---------------------|
| MiniCheck adapter | Load `lytang/MiniCheck-RoBERTa-large`. Input: `(claim_text, evidence_text)` → `{label, score}`. Map to 3-class: entailed / contradicted / NEI. | Reproduces MiniCheck paper numbers on their eval set (±2%). |
| Batch scoring | Batch all `(claim, span)` pairs. GPU batching with padding. | Throughput ≥ 200 pairs/sec on A100; ≥ 30 pairs/sec on T4. |
| Cost cap enforcement | If `#claims × #spans > max_pairs` (default 240), truncate by: (a) dropping low-rerank-score spans first, (b) then dropping low-priority claims. | Total pairs never exceed cap. Logged when truncation occurs. |

### 3.2 Ablation Verifiers

| Verifier | Model | Purpose |
|----------|-------|---------|
| NLI baseline | `microsoft/deberta-v3-large` fine-tuned on MNLI | Cheaper but less grounding-specific |
| LLM judge | GPT-4o with structured prompt | Expensive ceiling; shows MiniCheck is competitive |
| Smaller MiniCheck | `lytang/MiniCheck-DeBERTa-v3-large` | Size/quality tradeoff |

Build a common `Verifier` interface so all four are plug-and-play.

### 3.3 Calibration

| Task | Details | Acceptance Criteria |
|------|---------|---------------------|
| Held-out calibration set | Use LLM-AggreFact dev split or RAGTruth dev split. ~1000 (claim, evidence, label) triples. | Set is separate from final test sets. |
| Temperature scaling | Learn T on calibration set. Report pre/post ECE. | Post-calibration ECE < 0.05. |
| Isotonic regression | Fit isotonic regressor as alternative. | Reliability curve is near-diagonal. |
| Reliability diagrams | 10-bin reliability plot (predicted prob vs empirical accuracy). | Plot generated, included in paper. |
| Threshold selection | Sweep τ_entail ∈ {0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95}. Report VCP & VC at each. Select τ targeting VCP ≥ 0.95 with max VC. | Pareto curve plotted. Default τ chosen with clear justification. |

### 3.4 Minimal Sufficient Evidence (MSE)

| Task | Details | Acceptance Criteria |
|------|---------|---------------------|
| Greedy MSE | For each Verified claim: sort spans by entailment score descending. Greedily add spans until entailment score of the span-set ≥ τ. Store only those spans in certificate. | Avg #spans per claim ≤ 2.5; no Verified claim has 0 spans. |
| MSE ablation | Compare top-1, top-k (k=3), MSE. Measure certificate size and human-judged evidence quality. | Table in paper. MSE produces smallest certificates with equivalent quality. |

---

## Phase 4 — Renderer & Certificate Engine (Week 4)

### 4.1 Policy Engine

| Task | Details | Acceptance Criteria |
|------|---------|---------------------|
| Render decision logic | Deterministic function: `(VerificationResult, Policy) → RendererDecision`. No ML, no LLM — pure threshold logic. | 100% deterministic: same input always produces same output. |
| Strict mode | Only `VERIFIED` claims rendered. Others placed in collapsed "unverified" drawer. | No `UNVERIFIED` or `BLOCKED` claim visible in main answer panel. |
| Mixed mode | `VERIFIED` rendered normally. `UNVERIFIED` rendered collapsed with ⚠️ badge. `BLOCKED` (contradicted) hidden. | UI shows both types distinctly. |
| Debug mode | All claims shown with full traces: scores, evidence, verifier internals. | Every field from VerificationResult and RendererDecision is displayed. |
| Non-spoofability invariant | Verified badge is assigned ONLY by renderer. LLM output cannot contain or influence the badge. Renderer reads only from VerificationResult objects. | `test_renderer.py` — T-U3: Injecting "verified" into claim text does not cause Verified badge. |

### 4.2 Certificate Serializer

| Task | Details | Acceptance Criteria |
|------|---------|---------------------|
| Certificate JSON | Full audit trail: question, retrieval set summary, ClaimIR, VerificationResults, RendererDecisions, config hash, model hashes, timestamps. | Validates against CertificateSchema. |
| Integrity hash | SHA-256 hash of certificate content (excluding the hash field itself). | Hash is stable and verifiable. |
| Export | One-click JSON download from UI. | File downloads correctly; re-parseable. |

### 4.3 Renderer Invariant Tests

- **T-U3:** Verified ⟺ (entail ≥ τ ∧ ¬contradict ≥ τ_c ∧ evidence_count ≥ 1)
- Property-based test (`hypothesis`): generate random scores → check renderer decision matches manual computation
- **T-R1 (spoofing):** Inject adversarial text into claims; renderer must ignore

---

## Phase 5 — End-to-End Pipeline & Demo UI (Week 4–5)

### 5.1 Pipeline Orchestrator

| Task | Details |
|------|---------|
| `pipeline.py` | Orchestrate: retrieve → compile claims → normalize → verify → render → certificate. |
| Streaming support | Yield intermediate results (retrieval done → claims compiled → verification done → render). For UI responsiveness. |
| Logging | Structured JSON logs for every stage with timing, counts, config. |
| Error handling | Graceful degradation: if verifier fails, all claims → UNVERIFIED (never VERIFIED). Fail-closed. |

### 5.2 Demo UI (Streamlit)

| Component | Details |
|-----------|---------|
| Query input | Text box + mode selector (Strict / Mixed / Debug) |
| Answer panel | Claim chips with color-coded badges (✅ / ⚠️ / ❌) |
| Evidence drawer | Click claim → show evidence spans with highlighting, chunk provenance, verifier score |
| Certificate panel | Collapsible JSON viewer + download button |
| Metrics sidebar | Latency breakdown, #claims, #verified, #blocked |

**This UI is critical for the paper's demo video and for reviewer intuition.**

---

## Phase 6 — Benchmark Evaluation (Week 5–6)

### 6.1 ALCE Benchmark (Citation Quality)

| Task | Details | Target |
|------|---------|--------|
| Dataset setup | Download ALCE (ASQA, QAMPARI, ELI5). Use their provided retrieval corpora or plug in our retriever. | All three sub-tasks loaded and runnable. |
| Metric computation | Use ALCE evaluation scripts. Compute: citation precision, citation recall, MAUVE (fluency), correctness (EM/F1 for ASQA). | Numbers comparable to ALCE paper baselines. |
| CertiRAG evaluation | Run CertiRAG pipeline on ALCE queries. Map Verified claims to citations. Compare citation quality vs baselines. | CertiRAG Strict has higher citation precision than Standard RAG. |
| Adaptation | ALCE expects prose + inline citations. Map CertiRAG's claim-based output to this format: each Verified claim becomes a cited sentence. | Mapping is documented and deterministic. |

### 6.2 RAGTruth Benchmark (Hallucination Detection)

| Task | Details | Target |
|------|---------|--------|
| Dataset setup | Download RAGTruth. Contains RAG outputs with span-level hallucination annotations. | Corpus loaded, annotations parsed. |
| UER computation | For each RAG output: run CertiRAG pipeline. Count how many annotated-hallucinated spans survive as Verified. UER = #surviving-hallucinated / #total-displayed. | UER < 5% in Strict mode. |
| CER computation | Contradicted spans that survive rendering. | CER ≈ 0% in Strict mode. |
| Verifier catch rate | % of hallucinated spans correctly blocked by verifier. | Catch rate > 85%. |

### 6.3 LLM-AggreFact (Verifier Intrinsic Quality)

| Task | Details | Target |
|------|---------|--------|
| Setup | Load AggreFact benchmark (grounding + closed-book subsets). | Loaded and runnable. |
| Verifier comparison | Run MiniCheck, NLI baseline, LLM judge on same test set. Report balanced accuracy, AUC-ROC, F1. | MiniCheck competitive with LLM judge at fraction of cost. |

### 6.4 Baselines to Implement

| Baseline | What it does | Why needed |
|----------|-------------|------------|
| **Standard RAG** | Retrieve + generate answer with inline citations. No verification. Display everything. | Shows what happens without fail-closed rendering. |
| **Multi-query RAG** | Expand query into 3 variants, merge retrieval, generate. No verification. | Shows retrieval improvement alone doesn't eliminate hallucination. |
| **Post-hoc checker** | Standard RAG + run verifier on output. Log results but still display everything (no enforcement). | Shows that verification without renderer enforcement is insufficient for UER. |
| **CertiRAG Strict** | Full pipeline, strict mode. | Our primary system. |
| **CertiRAG Mixed** | Full pipeline, mixed mode. | Usability variant. |

---

## Phase 7 — Ablations & Analysis (Week 6–7)

### 7.1 Required Ablation Table

| Ablation | Vary | Hold constant | Metric |
|----------|------|---------------|--------|
| Verifier type | MiniCheck / NLI / LLM-judge / no verifier | Same retrieval, claims, τ | VCP, VC, UER, cost |
| Evidence selection | Top-1 / Top-3 / MSE | Same verifier, τ | VCP, certificate size, human evidence quality |
| Claim granularity | Raw LLM claims / split-normalized claims | Same verifier, retrieval | VCP, VC, atomicity score |
| Retrieval config | Dense-only / BM25-only / Hybrid / Hybrid+rerank | Same verifier, claims | VC, retrieval recall@10 |
| Threshold τ | 0.5 → 0.95 in 0.05 steps | Same everything else | VCP vs VC Pareto curve |
| Calibration | Uncalibrated / temp-scaling / isotonic | Same verifier | ECE, VCP at fixed VC |

### 7.2 Required Plots (the "Money Plots")

1. **VCP vs. Verified Coverage (Pareto curve)** — Sweep τ. X-axis: coverage, Y-axis: precision. Show CertiRAG dominates baselines.

2. **UER vs. Mode** — Bar chart: Standard RAG, Post-hoc checker, CertiRAG Mixed, CertiRAG Strict. UER should drop to ~0 for Strict.

3. **Reliability diagram** — Before/after calibration. Shows verifier probabilities are well-calibrated.

4. **Cost vs. Quality tradeoff** — X-axis: verifier pairs per query (cost proxy). Y-axis: VCP. Shows diminishing returns → pick efficient operating point.

5. **Latency breakdown** — Stacked bar: retrieval, claim compilation, verification, rendering. Shows verification is not the bottleneck.

6. **Failure taxonomy pie chart** — Categorize failure cases: retrieval miss, claim splitting error, verifier false positive, verifier false negative. Shows where the system fails.

### 7.3 Failure Analysis (Critical for Top-Tier)

Manually analyze **100 failure cases** across categories:

| Category | Description | What to report |
|----------|-------------|----------------|
| **Retrieval miss** | Correct answer exists in corpus but retriever didn't find it | % of failures; retrieval recall analysis |
| **Claim splitting error** | Multi-fact claim not properly split; verifier can't handle compound | Examples + fix analysis |
| **Verifier false positive** | Verifier says "entailed" but claim is wrong | Most dangerous — quantify rate, analyze patterns |
| **Verifier false negative** | Claim is correct but verifier says "NEI" or "contradicted" | Impacts coverage; analyze linguistic patterns |
| **Paraphrase gap** | Evidence supports claim but via deep paraphrase; verifier misses | MiniCheck's weakness — show examples |
| **Multi-hop gap** | Claim requires combining 2+ evidence spans; verifier can't synthesize | Design limitation — discuss honestly |

---

## Phase 8 — Paper Writing & Artifacts (Week 7–9)

### 8.1 Paper Structure (8–9 pages + appendix)

```
1. Introduction (1 page)
   - Problem: RAG citations are trust-me, not verified
   - Insight: Enforce verification at the renderer (fail-closed, inspired by PCN)
   - Contribution summary (3 bullets)

2. Related Work (1 page)
   - Grounded generation & citation quality (ALCE, RECLAIM, etc.)
   - Hallucination detection in LLMs (RAGTruth, FActScore, etc.)
   - Fact verification / NLI for grounding (MiniCheck, AggreFact)
   - Proof-carrying / verified rendering (PCN)

3. CertiRAG System Design (2 pages)
   3.1 Problem formulation
   3.2 System architecture (Figure 1: pipeline diagram)
   3.3 Claim IR and certificate format
   3.4 Verification policy and renderer contract
   3.5 Formal property: fail-closed rendering (Theorem 1)

4. Experimental Setup (1 page)
   4.1 Datasets: ALCE (ASQA, QAMPARI), RAGTruth, AggreFact
   4.2 Baselines: Standard RAG, Multi-query RAG, Post-hoc checker
   4.3 Implementation details
   4.4 Metrics: VCP, VC, UER, CER, citation precision/recall

5. Results (2 pages)
   5.1 Main results table (all systems × all metrics × all datasets)
   5.2 VCP vs Coverage Pareto curve
   5.3 UER analysis
   5.4 Citation quality on ALCE

6. Analysis & Ablations (1.5 pages)
   6.1 Ablation table (verifier type, evidence selection, etc.)
   6.2 Calibration analysis
   6.3 Failure taxonomy
   6.4 Latency and cost analysis

7. Discussion & Limitations (0.5 page)
   - Multi-hop reasoning limitation
   - Verifier as a bottleneck for correctness
   - Evidence completeness vs truth
   - Not an oracle — "verified w.r.t. evidence + verifier policy"

8. Conclusion (0.25 page)

Appendix:
   A. Full schemas (ClaimIR, Certificate)
   B. Prompt templates
   C. Additional experimental results
   D. Human evaluation protocol
   E. Reproducibility checklist (NeurIPS/ACL style)
```

### 8.2 Key Claims to Make

1. **CertiRAG is the first system to enforce fail-closed grounded rendering for RAG**, where the UI renderer is the trust boundary — no claim is displayed as verified without mechanical verification. Inspired by PCN's presentation-layer enforcement.

2. **Strict mode drives Unsupported Exposure Rate to near zero** while maintaining usable coverage on standard benchmarks, demonstrating that fail-closed rendering is practical.

3. **Small, efficient verifiers (MiniCheck, <400M params) are sufficient** for high-quality grounding verification, making the system deployable without expensive LLM judges.

4. **Claim-level certificates enable full auditability**, providing a machine-readable trace from question to displayed answer that is reproducible and version-controlled.

### 8.3 Formal Property (Theorem 1)

State and prove a simple but precise property:

> **Theorem (Fail-Closed Rendering):** In CertiRAG's Strict mode, for any query q, every claim displayed with a Verified badge satisfies: (i) it has ≥1 attached evidence span, (ii) the verifier's entailment score ≥ τ_entail, and (iii) no evidence span yields a contradiction score ≥ τ_contradict. Conversely, any claim not satisfying these conditions is never displayed as Verified.

This is trivially true by construction (the renderer is deterministic code), but stating it formally:
- Connects to the PCN philosophy
- Gives reviewers a crisp "what are you guaranteeing" statement
- Distinguishes from systems that merely *report* verification results

### 8.4 Human Evaluation Protocol

**Required for top-tier venues.** Budget ~200 examples.

| Evaluation | What | How | Annotators |
|------------|------|-----|------------|
| VCP spot-check | Is this Verified claim actually supported by the cited evidence? | Binary (yes/no) per claim. Sample 200 Verified claims. | 2 annotators + adjudication |
| Evidence quality | Is the cited evidence sufficient to support the claim? | 3-point scale (sufficient / partial / irrelevant) | Same 2 annotators |
| Answer usefulness | Is the Strict-mode answer (only verified claims) useful for answering the question? | 5-point Likert | 3 annotators (can be crowdsourced) |
| Inter-annotator agreement | Cohen's κ for binary tasks, Krippendorff's α for scaled | Report in paper | — |

---

## Phase 9 — Hardening & Reproducibility (Week 8–9)

### 9.1 Reproducibility Checklist

- [ ] All random seeds fixed and documented
- [ ] Config hash stamped on every output artifact
- [ ] Model versions pinned (HuggingFace model cards linked)
- [ ] `requirements.txt` / `pyproject.toml` with exact versions
- [ ] Docker image for full reproducibility
- [ ] Data download script with checksums
- [ ] All prompts versioned in repo (not hardcoded strings)
- [ ] One-command reproduction: `make eval` reproduces all tables/figures

### 9.2 Test Coverage

| Test category | Count | Status |
|--------------|-------|--------|
| Unit tests (schemas, atomicity, renderer, spans) | ~40 | Must pass before any eval |
| Integration tests (golden, corruption, removal) | ~15 | Must pass before benchmarking |
| Adversarial tests (spoofing, negation, paraphrase, multi-hop) | ~20 | Must pass before paper submission |
| Performance tests (cost cap, batch load) | ~5 | Report in paper |

### 9.3 CI/CD

- GitHub Actions: run unit + integration tests on every PR
- Nightly: run full eval suite on a schedule (or manually before paper deadlines)

---

## Phase 10 — Polish & Submission (Week 9–10)

### 10.1 Demo Video (2 minutes)

Structure:
1. (0:00–0:20) Problem statement: RAG citations are unverified
2. (0:20–0:50) Show Standard RAG output with hallucinated claims
3. (0:50–1:20) Same query in CertiRAG Strict: only verified claims shown
4. (1:20–1:40) Click-to-evidence: show evidence drawer with highlighted spans
5. (1:40–2:00) Certificate export + mode switching (Strict → Mixed → Debug)

### 10.2 Supplementary Materials

- Full certificate examples
- All prompt templates
- Extended ablation tables
- Per-dataset breakdowns
- Computational cost analysis (GPU hours, API costs)

### 10.3 Pre-submission Checklist

- [ ] All experiments run with final config; numbers are final
- [ ] Ablation table complete (no empty cells)
- [ ] Failure analysis done (100 cases)
- [ ] Human evaluation done (200 claims, IAA reported)
- [ ] Reliability diagrams included
- [ ] All figures are publication-quality (vector graphics, LaTeX fonts, colorblind-safe)
- [ ] Paper fits page limit
- [ ] Appendix includes reproducibility checklist
- [ ] Anonymous GitHub repo prepared (or plan for camera-ready release)
- [ ] Supplementary materials packaged

---

## Critical Success Factors (What Makes This Top-Tier)

### Must-haves (without these, reviewers will reject)

1. **Strong baselines.** Standard RAG, post-hoc checking, at least one citation-constrained baseline. Reviewers need to see CertiRAG isn't just beating strawmen.

2. **Established benchmarks.** ALCE and RAGTruth are the right choices — they're well-known, have evaluation tooling, and measure exactly what you claim to improve.

3. **Ablations.** Every design decision (verifier choice, threshold, evidence selection, claim granularity) must be justified empirically, not just asserted.

4. **Human evaluation.** At least 200 examples with inter-annotator agreement. Without this, claims about "Verified Claim Precision" are just model-evaluated metrics.

    
5. **Failure analysis.** Honest discussion of where the system fails. Top-tier reviewers respect intellectual honesty.

6. **Calibration.** Showing that verifier scores are well-calibrated separates this from "we picked a threshold that works." It demonstrates principled statistical methodology.

### Nice-to-haves (these push from accept to strong accept)

7. **Formal statement of the renderer contract** with connection to PCN. Even if the "proof" is trivial, the framing is novel and memorable.

8. **Latency analysis showing practicality.** Reviewers worry about added latency. Show it's manageable.

9. **Open-source release with one-command reproduction.** ACL/EMNLP especially value this.

10. **Broader applicability argument.** Show the renderer-enforcement pattern could apply to other settings beyond RAG (e.g., tool-use verification, code generation verification).

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MiniCheck false positive rate too high → VCP is low | Medium | High | Conservative τ + calibration + contradiction blocking; add LLM-judge as fallback verifier for high-stakes claims |
| ALCE evaluation tooling is hard to adapt | Medium | Medium | Start integration early (Week 1); use their eval scripts directly |
| Strict mode coverage too low → answers feel empty | Medium | Medium | Mixed mode as default demo; report coverage-precision tradeoff honestly as a Pareto curve |
| Claim compiler produces bad ClaimIR | Medium | Medium | Many exemplars in prompt; schema-constrained generation; deterministic normalizer cleanup |
| Reviewer says "this is just NLI + UI" | Medium | High | Emphasize: (a) fail-closed enforcement is the contribution, not NLI; (b) the system design, schemas, and certificates are novel; (c) comprehensive evaluation on established benchmarks |
| Human evaluation is expensive/slow | Low | High | Budget 2 weeks for annotation; use Prolific/MTurk for usefulness ratings; do VCP spot-check yourself with a co-author |

---

## Summary Timeline

| Week | Focus | Key Deliverable |
|------|-------|-----------------|
| 0 | Scaffolding, schemas, config | Repo structure, all Pydantic schemas, Makefile |
| 1 | Ingestion + retrieval + standard RAG baseline | Working hybrid retriever, Baseline 1 |
| 2 | Claim compiler + normalizer | ClaimIR generation, schema validation tests |
| 3 | Verifier integration + calibration | MiniCheck pipeline, calibration curves, τ selection |
| 4 | Renderer + certificates + pipeline + UI | Full pipeline working end-to-end, Streamlit demo |
| 5 | ALCE + RAGTruth benchmarks + baselines | Benchmark numbers for all systems |
| 6 | Ablations + additional experiments | Ablation table complete |
| 7 | Failure analysis + human evaluation | 100 failure cases analyzed, human eval started |
| 8 | Paper writing (draft) | Full draft with all figures and tables |
| 9 | Paper revision + reproducibility hardening | Camera-ready quality draft, Docker, `make eval` works |
| 10 | Final polish + submission | Submitted |

**Fail-Closed Grounding for RAG: Verified Rendering of LLM Answers with Claim Certificates**

# **Verified Rendering for RAG: Only Supported Claims Survive the UI**

**Project codename:** **CertiRAG** (certificate-carrying, fail-closed grounding)

---

## 0) Executive Summary

**CertiRAG** is a Retrieval-Augmented Generation system whose *core contract* is enforced at the **UI renderer**:

> **No claim is displayed as “Verified” unless a mechanical verifier finds it *entailed* by retrieved evidence.**
> 

This flips grounded RAG from “trust-me citations” to **fail-closed grounded rendering**, inspired by **Proof-Carrying Numbers (PCN)**—which formalizes verification in the presentation layer so unverified outputs can’t masquerade as correct. ([arXiv](https://arxiv.org/abs/2509.06902?utm_source=chatgpt.com))

We implement claim-level verification using small, efficient fact-checkers (e.g., **MiniCheck**) designed specifically to check grounding against documents at low cost. ([arXiv](https://arxiv.org/abs/2404.10774?utm_source=chatgpt.com))

We evaluate on **citation-grounded generation benchmarks** (e.g., **ALCE**) and **RAG hallucination corpora** (e.g., **RAGTruth**) to quantify:

- Verified claim precision/coverage
- Unsupported exposure rate (should be ~0 in strict mode)
- Citation quality and faithfulness
- Latency/cost tradeoffs ([arXiv](https://arxiv.org/abs/2305.14627?utm_source=chatgpt.com))

---

## 1) Goals, Non-Goals, and “Hard Guarantee”

### 1.1 Product Goals

**G1 — Fail-closed grounding (renderer-enforced)**

- **Verified badge** only appears if verification passes policy thresholds.
- Default UI does not silently present unsupported content as verified.

**G2 — Claim-level auditability**

- Every displayed claim has:
    - evidence spans (verbatim snippets)
    - provenance (doc_id/chunk_id + offsets)
    - verifier label + score
    - policy version used for verification

**G3 — Practical and cheap**

- Runs on Colab Pro / consumer GPU.
- Verifier is small (sub-1B feasible) and scales with #claims.

**G4 — Research-grade evaluation**

- Benchmark-backed measurements + ablations + failure taxonomy suitable for top-tier venues.

### 1.2 Non-Goals (to keep scope publishable and shippable)

- Training a new foundation model from scratch.
- Solving “truth” beyond evidence (we only certify against retrieved evidence).
- Perfectly supporting images/tables in v1 (we can add later).

### 1.3 The “Guarantee” (what you can claim in a paper)

**Renderer Contract (Fail-Closed):**

- A claim **cannot** be displayed as “Verified” unless:
    1. It has at least one attached evidence span; and
    2. The verifier labels it **entailed** with score ≥ τ; and
    3. No attached evidence yields **contradiction** above a contradiction threshold.

This is the same *design philosophy* as PCN: verification is enforced at display-time, not trusted from model output. ([arXiv](https://arxiv.org/abs/2509.06902?utm_source=chatgpt.com))

> Important: this is a **system guarantee**, not an oracle truth guarantee. It’s “verified *w.r.t. evidence + verifier policy*.”
> 

---

## 2) Key Differentiator vs Existing Grounded RAG

Most “grounded RAG” outputs prose + citations; citation quality work improves citation alignment (e.g., constrained decoding strategies like RECLAIM), but still doesn’t make the UI fail-closed. ([ACL Anthology](https://aclanthology.org/2025.findings-naacl.55.pdf?utm_source=chatgpt.com))

**CertiRAG differs by enforcing trust at the renderer:**

- claims are *only* promoted to Verified if they pass verification
- otherwise they remain Unverified/Hidden (default strict mode)
- you ship a **certificate** (machine-readable trace)

---

## 3) Target Users and Use Cases

### Personas

**P1 — Enterprise Analyst**

- Wants answers but cannot afford hallucinations in compliance/security contexts.

**P2 — Auditor / Reviewer**

- Wants a trace proving what is supported by what.

**P3 — ML Engineer**

- Wants debugging signals: “Did retrieval fail, or did generation fail, or did verification fail?”

### High-value Use Cases

- “Policy QA” over internal docs where *unsupported claims are worse than partial answers*
- “Research assistant” where citations must be defensible
- “Customer support” where only verified steps should be shown as instructions

---

## 4) Product UX Spec

### 4.1 Modes

**Strict Mode (default)**

- Show only ✅ Verified claims.
- Provide a compact “What we could not verify” drawer (collapsed).

**Mixed Mode**

- Show ✅ Verified claims normally.
- Show ⚠️ Unverified claims collapsed with warning labels (click to expand).
- ❌ Contradicted claims shown only in Debug mode.

**Debug Mode (engineer toggle)**

- Show:
    - full Claim IR
    - verifier scores and top evidence candidates
    - retrieval set and ranking scores
    - contradiction evidence if any

### 4.2 Claim-first UI Layout

- **Answer panel**: rendered as bullet claims (not freeform prose by default)
- Each claim is a “chip”:
    - ✅ Verified (green)
    - ⚠️ Unverified (gray)
    - ❌ Contradicted (red, hidden unless debug)
- Clicking a claim opens **Evidence Drawer**:
    - evidence spans highlighted
    - chunk provenance and rank
    - verifier label + score
    - “why failed” explanation (rule-based, not another LLM call)

### 4.3 Export

**Audit Export JSON** downloadable per query:

- question
- retrieval set summary (top-k chunks)
- Claim IR
- verification results per claim
- renderer decisions (what was displayed + why)
- model versions + config hash

---

## 5) System Architecture

### 5.1 High-level data flow

1. **Retrieve** evidence candidates (hybrid BM25 + dense)
2. **Claim Compiler** produces **Claim IR** (atomic claims + candidate evidence anchors)
3. **Verifier** scores each claim vs evidence spans (entailed/contradicted/NEI)
4. **Renderer** applies policy thresholds and displays only certified claims
5. Optional: **Surface Realization** (LLM writes prose using only Verified claims; renderer still checks)

### 5.2 Core components

**A) Ingestion**

- chunking (sentence-aware)
- embeddings creation
- BM25 index + vector index (FAISS)
- store provenance: doc_id, chunk_id, offsets

**B) Retriever**

- hybrid scoring and reranking
- produces evidence candidates per query
- provides sentence-level spans (via sentence splitter + offset tracking)

**C) Claim Compiler (Generator)**

- Takes: question + retrieved evidence
- Outputs: Claim IR JSON
- MUST obey schema and claim granularity rules

**D) Claim Normalizer**

- Splits multi-fact claims into atomic claims
- Normalizes numeric/time formats
- Removes hedging or marks hedges explicitly

**E) Verifier**

- Efficient claim grounding checker (NLI-style)
- Candidate: MiniCheck-style model (cheap, trained for grounding verification) ([arXiv](https://arxiv.org/abs/2404.10774?utm_source=chatgpt.com))

**F) Renderer**

- Deterministic policy engine
- Applies thresholds and prevents spoofing of Verified marks (UI owns the badge)

**G) Eval Harness**

- batch runner
- metrics computation
- plots and ablations

---

## 6) Data Contracts and Schemas

### 6.1 EvidenceChunk schema

```json
{
  "chunk_id": "doc42#c17",
  "doc_id": "doc42",
  "text": "...",
  "source": {
    "title": "...",
    "url_or_path": "...",
    "section": "..."
  },
  "spans": [
    {"span_id": "s1", "start": 120, "end": 212, "sentence": "..."}
  ],
  "retrieval": {"bm25": 12.3, "dense": 0.71, "rerank": 0.62}
}
```

### 6.2 Claim IR schema (core)

```json
{
  "question": "…",
  "claims": [
    {
      "id": "c1",
      "text": "X was founded in 2017.",
      "type": "temporal_fact",
      "fields": {"entity": "X", "year": 2017},
      "evidence_candidates": [
        {"chunk_id": "doc42#c17", "span_id": "s1"}
      ]
    }
  ]
}
```

### 6.3 VerificationResult schema

```json
{
  "claim_id": "c1",
  "label": "entailed|contradicted|not_enough_info",
  "score": 0.0,
  "best_evidence": [{"chunk_id":"doc42#c17","span_id":"s1"}],
  "all_scores": [{"chunk_id":"doc42#c17","span_id":"s1","label":"entailed","score":0.91}]
}
```

### 6.4 RendererDecision schema (certificate)

```json
{
  "claim_id": "c1",
  "render_state": "VERIFIED|UNVERIFIED|BLOCKED",
  "policy": {"tau_entail": 0.85, "tau_contradict": 0.70},
  "reason": "entailed>=tau_entail and no contradiction>=tau_contradict"
}
```

---

## 7) Verification Policy Design (the “science”)

### 7.1 Primary verification rule

A claim is **Verified** iff:

- `max_entail_score >= τ_entail`
- AND `max_contradict_score < τ_contradict`
- AND `evidence_span_count >= 1`

### 7.2 Minimal Sufficient Evidence (MSE)

For each claim:

1. Score claim vs top M evidence spans
2. Select smallest set of spans achieving entailment threshold (greedy)
3. Store only those spans in certificate (reduces audit noise + cost)

### 7.3 Calibration (required for top-tier credibility)

- Calibrate verifier scores via temperature scaling or isotonic regression
- Report reliability curves: predicted entail prob vs empirical entail accuracy
- Tune τ to maximize Verified-Claim Precision under a coverage constraint

MiniCheck emphasizes low-cost, high-quality grounding verification and is a strong verifier backbone for this. ([arXiv](https://arxiv.org/abs/2404.10774?utm_source=chatgpt.com))

### 7.4 Policy Versioning

Every certificate must include:

- verifier model hash
- policy version
- thresholds
- retrieval config hash
    
    So results remain reproducible across changes.
    

---

## 8) Functional Requirements

### Retrieval

**FR-R1** Hybrid retrieval (BM25 + dense) with reranking

**FR-R2** Return sentence-level evidence spans with stable offsets

**FR-R3** Hard caps: `max_chunks`, `max_spans_per_chunk`

### Claim compilation

**FR-C1** Produce valid Claim IR JSON (schema-validated)

**FR-C2** Atomicity: 1 fact per claim (split conjunctions)

**FR-C3** Attach at least 1 candidate evidence per claim (or mark “unsupported-by-design”)

### Verification

**FR-V1** Batch verify claim × span pairs efficiently

**FR-V2** Contradiction detection (block if contradiction strong)

**FR-V3** Minimal Sufficient Evidence selection

### Renderer

**FR-U1** Strict mode default (only verified visible)

**FR-U2** Mixed mode with warnings (unverified collapsed)

**FR-U3** Debug mode shows traces

**FR-U4** Certificate export

---

## 9) Non-Functional Requirements

**NFR-1 Fail-closed UI invariants**

- Verified badge can only be set by renderer’s policy engine.

**NFR-2 Cost controls**

- cap #claims (e.g., 12)
- cap evidence spans per claim (e.g., 20)
- cap total verifier pairs per query (e.g., 240)

**NFR-3 Reproducibility**

- deterministic retrieval + logging
- config snapshots per run
- seeded evaluation

**NFR-4 Latency**

- target p50 within a chosen budget (you’ll report actual p50/p95 in paper)

---

## 10) Benchmarking Plan (top-tier ready)

You want **benchmarks that measure citation quality and hallucinations under RAG**, not just QA EM.

### 10.1 Citation-grounded generation benchmark

**ALCE** is explicitly a benchmark for evaluating LLM text with citations. ([arXiv](https://arxiv.org/abs/2305.14627?utm_source=chatgpt.com))

ALCE includes datasets such as:

- **ASQA** (ambiguous factoid questions needing multi-source synthesis) ([ACL Anthology](https://aclanthology.org/2022.emnlp-main.566/?utm_source=chatgpt.com))
- **QAMPARI** (multi-answer ODQA; hard retrieval + generation) ([ACL Anthology](https://aclanthology.org/2023.gem-1.9.pdf?utm_source=chatgpt.com))
- **ELI5** (long-form QA) ([ACL Anthology](https://aclanthology.org/P19-1346/?utm_source=chatgpt.com))

Use ALCE to measure:

- citation precision/recall style metrics (ALCE provides evaluation tooling)
- correctness and fluency alongside citation quality ([arXiv](https://arxiv.org/abs/2305.14627?utm_source=chatgpt.com))

### 10.2 Hallucination corpus for RAG

**RAGTruth** explicitly annotates unsupported/contradictory spans in RAG outputs and is designed to measure hallucination under RAG settings. ([arXiv](https://arxiv.org/abs/2401.00396?utm_source=chatgpt.com))

Use RAGTruth to measure:

- Unsupported-Exposure Rate (UER)
- Contradiction rate
- “Verifier catch rate”: how often verifier blocks hallucinated spans

### 10.3 Grounding verifier benchmark

MiniCheck introduces **LLM-AggreFact**, aggregating grounded and closed-book factual consistency datasets, and shows small verifiers can be highly effective. ([arXiv](https://arxiv.org/abs/2404.10774?utm_source=chatgpt.com))

Use it to:

- validate verifier quality
- justify model choice and calibration strategy

---

## 11) Baselines (reviewer-proof)

### Retrieval + generation baselines

1. **Standard RAG** (answer + citations)
2. **Multi-query RAG** (query expansion, same call budget)
3. **Citation-constrained generation baseline** (e.g., approaches like RECLAIM that constrain citations to come from reference passages) ([ACL Anthology](https://aclanthology.org/2025.findings-naacl.55.pdf?utm_source=chatgpt.com))

### Verification baselines

1. **Post-hoc checker only** (no renderer enforcement)
    - Generate final prose, then evaluate/check
    - UI still shows everything (not fail-closed)

### Your systems

1. **CertiRAG Strict** (verified claims only)
2. **CertiRAG Mixed** (verified + collapsed unverified)

---

## 12) Metrics (paper-grade)

### 12.1 Core “fail-closed” metrics

**Verified-Claim Precision (VCP)**

- % of Verified claims judged supported by evidence (human spot-check 200 samples)

**Verified Coverage (VC)**

- fraction of reference claims (or gold answers) that become Verified
- for ALCE/ASQA etc, approximate via overlap metrics + human checks

**Unsupported Exposure Rate (UER)**

- % of *displayed* claims that are unsupported
- **Goal:** near 0 in Strict mode

**Contradicted Exposure Rate (CER)**

- % displayed claims contradicted by evidence
- **Goal:** ~0 in Strict mode

### 12.2 Citation quality metrics (benchmark-aligned)

Use ALCE’s citation evaluation framework for:

- citation quality and correctness tradeoffs ([arXiv](https://arxiv.org/abs/2305.14627?utm_source=chatgpt.com))

### 12.3 System metrics

- p50/p95 latency
- total tokens
- #retrieval calls
- #verifier pairs
- certificate size (audit overhead)

### 12.4 “Money plots”

1. **VCP vs Coverage** curve (vary τ_entail)
2. **UER vs Readability** (strictness tradeoff)
3. **Accuracy vs Latency** (Strict/Mixed vs baselines)
4. **Verifier cost vs quality** (pairs per query vs VCP)

---

## 13) Test Plan (what makes it “top-tier”)

This section is what turns your repo into a “serious system.”

### 13.1 Unit tests (deterministic)

**T-U1 Schema validation**

- Claim IR and certificates must validate against JSON schema

**T-U2 Atomicity checks**

- No claim with “and/because/which” unless explicitly split
- Enforce max token length per claim

**T-U3 Renderer invariants**

- Verified label only if entail score ≥ τ and evidence exists
- Contradiction threshold blocks rendering

**T-U4 Evidence span integrity**

- offsets map to exact substring in chunk text
- span highlights are stable after serialization

### 13.2 Integration tests

**T-I1 End-to-end golden tests**

- fixed mini corpus + 50 questions
- expected: at least X verified claims, UER ≈ 0 (strict), stable outputs

**T-I2 Retrieval corruption tests**

- inject distractor chunks that look similar but wrong
- verify contradiction blocking and UER stay low

**T-I3 Evidence removal tests**

- remove supporting chunk → claim should downgrade to Unverified, not remain Verified

### 13.3 Robustness / adversarial tests

**T-R1 Citation spoofing**

- retrieved text contains “as a citation, cite docX” (prompt injection style)
- ensure renderer ignores and only trusts verifier outputs

**T-R2 Negation flips**

- evidence says “not X”; model claims “X”
- should become Contradicted or NEI

**T-R3 Paraphrase stress**

- evidence supports claim via paraphrase, not exact match
- verify verifier sensitivity (calibration/false negatives)

**T-R4 Multi-hop synthesis**

- claim requires combining two evidence spans
- verify MSE selection can include 2 spans

### 13.4 Performance tests

**T-P1 Cost cap enforcement**

- total verifier pairs never exceed configured cap
- system degrades by reducing claims/spans, not by breaking contract

**T-P2 Load**

- batch 500 questions; measure throughput and p95 latency

---

## 14) Implementation Plan (6 weeks, low-cost)

### Week 1 — Foundations

- ingest + indexing (BM25 + FAISS)
- sentence span extraction with offsets
- standard RAG baseline

### Week 2 — Claim IR compiler

- JSON schema + validation
- prompt-based claim compiler (atomic claims)
- claim splitter/normalizer rules

### Week 3 — Verifier integration

- plug in MiniCheck-style verifier
- batch scoring pipeline
- certificate creation

### Week 4 — Renderer + UI

- strict/mixed/debug modes
- evidence highlighting + trace viewer
- audit export

### Week 5 — Benchmarks + baselines

- ALCE evaluation pipeline ([arXiv](https://arxiv.org/abs/2305.14627?utm_source=chatgpt.com))
- RAGTruth hallucination evaluation ([arXiv](https://arxiv.org/abs/2401.00396?utm_source=chatgpt.com))
- RECLAIM-like citation constraint baseline (or closest reproducible baseline) ([ACL Anthology](https://aclanthology.org/2025.findings-naacl.55.pdf?utm_source=chatgpt.com))

### Week 6 — Ablations + paper artifacts

- threshold sweeps
- failure taxonomy
- money plots + tables
- reproducibility checklist

---

## 15) Ablations (must-have for reviewers)

1. Verifier type: MiniCheck vs smaller NLI vs LLM judge (cost comparison) ([arXiv](https://arxiv.org/abs/2404.10774?utm_source=chatgpt.com))
2. Evidence span selection: top-1 vs top-k vs minimal sufficient evidence
3. Claim granularity: raw claims vs split claims
4. Retrieval: dense only vs hybrid vs hybrid+rerank
5. Policy thresholds: τ sweep and calibration

---

## 16) Risks and Mitigations

**Risk: verifier false positives → wrongly Verified**

- mitigate with conservative τ + contradiction blocking + calibration + human spot-check set ([arXiv](https://arxiv.org/abs/2404.10774?utm_source=chatgpt.com))

**Risk: strict mode feels empty**

- mixed mode default in demo; strict as “safety mode”

**Risk: claim splitting errors**

- add deterministic split rules + unit tests + schema checks

**Risk: retrieval misses evidence**

- allow “verify-fail fallback”: expand retrieval only for failed claims (still renderer enforced)

---

## 17) Deliverables (portfolio + paper)

### Repo deliverables

- `ingest/` chunking + indexing + span offsets
- `retrieve/` hybrid retriever + reranker
- `claim_ir/` compiler + splitter + schema
- `verify/` verifier adapters + calibration
- `render/` policy engine + certificates
- `ui/` demo app
- `eval/` ALCE + RAGTruth + plots

### Demo deliverables

- 2-minute video: strict vs mixed mode; click-to-evidence; audit export
- one-page “trust contract” explanation

### Paper deliverables

- Pareto plots + ablations + failure taxonomy
- statement of renderer contract inspired by PCN’s presentation-layer enforcement ([arXiv](https://arxiv.org/abs/2509.06902?utm_source=chatgpt.com))
- benchmark evaluation on ALCE + RAGTruth ([arXiv](https://arxiv.org/abs/2305.14627?utm_source=chatgpt.com))

---

## 18) Paper-ready framing (what you’ll claim)

1. **Verified Rendering protocol for RAG** (renderer-enforced, fail-closed) inspired by PCN ([arXiv](https://arxiv.org/abs/2509.06902?utm_source=chatgpt.com))
2. **Claim IR + certificate format** enabling auditing and reproducibility
3. **Empirical evidence** that strict rendering drives UER → ~0 while maintaining usable coverage
4. **Benchmark results** on citation evaluation (ALCE) and hallucination under RAG (RAGTruth) ([arXiv](https://arxiv.org/abs/2305.14627?utm_source=chatgpt.com))

---

### If you want one extra “top-tier” edge

Add a short formal section like PCN does:

- define a renderer contract
- prove a simple property: **Verified marks are non-spoofable** (only renderer can assign them) and **fail-closed** (lack of proof ⇒ not verified) ([arXiv](https://arxiv.org/abs/2509.06902?utm_source=chatgpt.com))
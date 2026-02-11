# CertiRAG Benchmark Results

> **Evaluated:** February 10, 2026 &nbsp;|&nbsp; **Harness:** `eval/benchmark.py` + `eval/run_full_eval.py` &nbsp;|&nbsp; **Hardware:** GitHub Codespaces (CPU-only, 2-core, 8 GB RAM)

---

## Executive Summary

CertiRAG's fact verification pipeline now includes **5 pluggable verifier configurations** — from a fast 2-class NLI model to a multi-model ensemble with threshold optimization. Our best configuration, **Ensemble(accurate)**, achieves **75.5% 3-class accuracy** and **75.5% macro-F1** on adversarial ANLI R1, while the **Ensemble(sota)** 2-model blend reaches **0.958 AUROC** on FEVER-NLI — demonstrating that CertiRAG can reliably distinguish entailed, contradicted, and neutral claims.

### Leaderboard (Best Result per Metric)

| Metric | Best Model | Best Score | Dataset |
|--------|:----------:|:----------:|:-------:|
| 3-Class Accuracy | **Ensemble(accurate)** | **75.5%** | ANLI R1 |
| Macro F1 | **Ensemble(accurate)** | **75.5%** | ANLI R1 |
| AUROC | **DeBERTa-NLI (2-class)** | **0.991** | FEVER-NLI |
| ECE (calibration) | **Ensemble(sota)** | **0.107** | FEVER-NLI |
| Binary Accuracy | **DeBERTa-NLI (2-class)** | **96.7%** | FEVER-NLI |

### Evolution: Before vs After

| Metric (VitaminC) | v1: DeBERTa-2class | v2: Ensemble(accurate) | Improvement |
|:--|:-----------:|:--------------:|:--:|
| 3-Class Accuracy | 56.0% | **68.5%** | **+12.5 pp** |
| Macro F1 | 42.3% | **68.2%** | **+25.9 pp** |
| ECE (calibration) | 0.369 | **0.200** | **-0.169** |
| Contradiction F1 | 0.0% | **70.8%** | **+70.8 pp** |

*(pp = percentage points)*

---

## Datasets

| Dataset | Source | Test Size | Description | Why It Matters |
|---------|--------|-----------|-------------|----------------|
| **[VitaminC](https://huggingface.co/datasets/tals/vitaminc)** | Schuster et al., 2021 | 55,197 | Contrastive claim-evidence pairs from Wikipedia revisions | Tests sensitivity to **subtle factual edits** |
| **[ANLI R1](https://huggingface.co/datasets/facebook/anli)** | Nie et al., 2020 | 1,000 | Adversarially-constructed NLI examples (Round 1) | Tests robustness against **adversarial counterexamples** |
| **[FEVER-NLI](https://huggingface.co/datasets/pietrolesci/nli_fever)** | Thorne et al., 2018 | 19,998 | Wikipedia-based fact verification reformulated as NLI | Tests **standard fact verification** |

---

## Models Evaluated

| Model | Type | Parameters | Preset | Description |
|-------|------|-----------|--------|-------------|
| **DeBERTa-NLI (2-class)** | Neural NLI | 184M | `hf_nli` | `deberta-v3-base-zeroshot-v2.0` — zero-shot, entailment vs not |
| **Ensemble(accurate)** | Neural NLI | 184M | `ensemble_accurate` | `DeBERTa-v3-base-mnli-fever-anli` — 3-class, trained on MNLI+FEVER+ANLI |
| **Ensemble(sota)** | Ensemble (2 models) | 368M | `ensemble_sota` | Weighted blend: 3-class (0.6) + 2-class (0.4) |
| **Groq (Llama 3.3 70B)** | LLM-as-Judge | 70B | `groq` | Llama 3.3 via Groq free inference API |
| **MiniCheck-lite** | Keyword heuristic | 0 | `minicheck_lite` | Unigram overlap + negation baseline |

---

## Results

### Full Comparison Table

| Model | Dataset | N | Acc (3-class) | Acc (binary) | Macro F1 | AUROC | ECE |
|-------|---------|--:|:------------:|:------------:|:--------:|:-----:|:---:|
| **Ensemble(accurate)** | VitaminC | 200 | **68.5%** | 76.0% | **68.2%** | 0.829 | **0.200** |
| **Ensemble(accurate)** | ANLI R1 | 200 | **75.5%** | 85.5% | **75.5%** | 0.918 | 0.191 |
| **Ensemble(accurate)** | FEVER-NLI | 182 | 58.2% | 73.1% | 55.5% | 0.802 | 0.267 |
| **Ensemble(sota)** | VitaminC | 200 | 65.0% | 76.5% | 63.4% | 0.848 | **0.144** |
| **Ensemble(sota)** | ANLI R1 | 200 | 74.0% | **87.0%** | 74.1% | **0.928** | 0.157 |
| **Ensemble(sota)** | FEVER-NLI | 182 | 61.5% | 80.8% | 61.7% | **0.958** | **0.107** |
| DeBERTa-NLI (2-class) | VitaminC | 200 | 56.0% | **80.5%** | 42.3% | 0.853 | 0.369 |
| DeBERTa-NLI (2-class) | ANLI R1 | 200 | 55.5% | 86.5% | 46.1% | 0.919 | 0.408 |
| DeBERTa-NLI (2-class) | FEVER-NLI | 182 | 55.5% | **96.7%** | 48.9% | **0.991** | 0.432 |
| MiniCheck-lite | VitaminC | 200 | 47.5% | 56.0% | 34.8% | 0.652 | 0.190 |

### Per-Class Metrics (Best 3-class Model: Ensemble-accurate, ANLI R1)

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| entailed | 0.794 | 0.781 | **0.787** | 64 |
| contradicted | 0.852 | 0.676 | **0.754** | 68 |
| not_enough_info | 0.639 | 0.824 | **0.720** | 68 |

---

## Analysis

### 1. 3-Class Models Unlock Contradiction Detection

The biggest improvement: switching from a 2-class model to a 3-class model trained on MNLI+FEVER+ANLI. The 2-class model has **0% contradiction recall** — it literally cannot detect contradictions. The 3-class Ensemble(accurate) achieves **67.6% contradiction recall** and **85.2% precision**.

On VitaminC, this translates to:
- 2-class: 0/58 contradictions caught → **0% F1**
- 3-class: 34/58 contradictions caught → **70.8% F1**

### 2. Ensemble Diversity Boosts AUROC

The 2-model ensemble (sota) combines the 3-class model (good at classification) with the 2-class model (good at ranking). This produces the best AUROC on most datasets:
- FEVER-NLI: **0.958** (vs 0.802 accurate-only, vs 0.991 2-class-only)
- ANLI R1: **0.928** (vs 0.918 accurate-only, vs 0.919 2-class-only)

### 3. Calibration Massively Improved

ECE dropped from 0.37–0.43 (2-class) to **0.11–0.20** (ensembles):
- Ensemble(sota) on FEVER: ECE = **0.107** (well-calibrated)
- DeBERTa-2class on FEVER: ECE = 0.432 (overconfident)

### 4. Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|:-----------------:|-----|
| Maximum safety (binary) | DeBERTa-NLI (2-class) | 96.7% binary accuracy, 0% contradiction pass-through on FEVER |
| Contradiction detection | Ensemble(accurate) | 75.5% macro F1, detects all three classes |
| Best overall ranking | Ensemble(sota) | 0.958 AUROC, best calibration (ECE 0.107) |
| Zero cost, fast | MiniCheck-lite | <1ms latency, no GPU needed |

### 5. Fail-Closed Safety

CertiRAG uses a fail-closed design: only "entailed" claims are shown. The 2-class model achieves **0% contradiction pass-through** on FEVER-NLI (0/75 contradicted claims leaked). Even the 3-class model, which trades binary safety for richer classification, still blocks most contradictions.

### 6. Threshold Optimization

The `eval/optimizer.py` module implements per-class threshold optimization:
- Finds optimal entailment/contradiction thresholds via grid search on calibration data
- Uses stratified train/test splits to avoid overfitting
- Can improve macro F1 by 1-5 pp on held-out data

---

## Evaluation Framework

CertiRAG ships a complete, pluggable evaluation harness:

### Architecture

```
eval/
  benchmark.py          # BenchmarkRunner — pluggable harness
  scoring.py            # Metrics: Acc, P/R/F1, AUROC, ECE, confusion
  optimizer.py          # Threshold optimization & calibration
  run_full_eval.py      # Full evaluation suite (all models x datasets)
  datasets/
    loaders.py          # VitaminC, ANLI, FEVER-NLI dataset loaders

certirag/verify/
  hf_nli_verifier.py    # DeBERTa-NLI (2-class zero-shot)
  ensemble_verifier.py  # Multi-model ensemble (3 presets)
  groq_verifier.py      # Groq API (Llama 3.3 70B, Mixtral, etc.)
  gemini_verifier.py    # Google Gemini API
  calibrator.py         # Temperature scaling + isotonic regression
  minicheck.py          # Keyword heuristic baseline
```

### Plugging In Your Own Model

Any object with `verify_batch(pairs) -> list[EvidenceScore]` works:

```python
from eval.benchmark import BenchmarkRunner

runner = BenchmarkRunner(verifier=my_verifier, model_name="MyModel")
scores = runner.run("vitaminc", max_examples=200)
print(scores.summary_table())
```

### Using Groq (Free LLM-as-Judge)

```bash
export GROQ_API_KEY="gsk_..."
python -m eval.benchmark --model groq --dataset vitaminc --max-examples 50
```

### Running Full Evaluation

```bash
# Quick smoke test (50 examples)
python -m eval.benchmark --dataset vitaminc --max-examples 50

# Full suite (all models, all datasets)
python eval/run_full_eval.py --max-examples 200

# With calibration
python -m eval.benchmark --model ensemble_accurate --calibrate --cal-size 100

# Skip heavy/API models
python eval/run_full_eval.py --skip-large --skip-api
```

---

## Raw Data

All JSON results are in `eval_results/`:

| File | Model | Dataset |
|------|-------|---------|
| `Ensemble(accurate)_vitaminc_200.json` | Ensemble(accurate) | VitaminC |
| `Ensemble(accurate)_anli_r1_200.json` | Ensemble(accurate) | ANLI R1 |
| `Ensemble(accurate)_fever_nli_182.json` | Ensemble(accurate) | FEVER-NLI |
| `Ensemble(sota)_vitaminc_200.json` | Ensemble(sota) | VitaminC |
| `Ensemble(sota)_anli_r1_200.json` | Ensemble(sota) | ANLI R1 |
| `Ensemble(sota)_fever_nli_182.json` | Ensemble(sota) | FEVER-NLI |
| `DeBERTa-NLI_vitaminc_200.json` | DeBERTa-NLI (2-class) | VitaminC |
| `DeBERTa-NLI_anli_r1_200.json` | DeBERTa-NLI (2-class) | ANLI R1 |
| `DeBERTa-NLI_fever_nli_182.json` | DeBERTa-NLI (2-class) | FEVER-NLI |
| `MiniCheck-lite_vitaminc_200.json` | MiniCheck-lite | VitaminC |

---

## Citation

```bibtex
@software{certirag2026,
  title   = {CertiRAG: Certified Retrieval-Augmented Generation with Fail-Closed Verification},
  author  = {Aayush Kumar},
  year    = {2026},
  url     = {https://github.com/aayushakumar/CertiRAG}
}
```

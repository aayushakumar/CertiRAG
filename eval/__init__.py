"""
CertiRAG Evaluation Harness
============================

Benchmark and ablation evaluation code for reproducing
all tables and figures from the paper.

Modules:
    datasets/   - Benchmark dataset loaders (ALCE, RAGTruth, AggreFact)
    metrics.py  - Shared metrics (faithfulness, AUROC, ECE)
    runner.py   - Evaluation orchestrator
    baselines.py - Baseline systems for comparison
    ablations.py - Ablation study configurations
    plots.py    - Publication-quality figures
"""

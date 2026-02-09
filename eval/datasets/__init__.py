"""
Benchmark Dataset Loaders
==========================

CertiRAG evaluates on three benchmarks:
    - ALCE (Gao et al., 2023): Citation-based QA faithfulness
    - RAGTruth (Wu et al., 2024): RAG hallucination detection  
    - AggreFact (Tang et al., 2022): Summarization factuality (transfer)

Each loader provides a uniform interface:
    examples = load_<dataset>(split="test")
    # Each example: {"question": str, "documents": [...], "gold_claims": [...]}
"""

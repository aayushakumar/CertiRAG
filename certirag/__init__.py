"""
CertiRAG — Fail-Closed Grounding for RAG with Verified Rendering
=================================================================

CertiRAG is a Retrieval-Augmented Generation system that enforces
verification at the UI renderer: no claim is displayed as "Verified"
unless a mechanical verifier finds it entailed by retrieved evidence.

Architecture Overview:
    Query → Retrieve → Compile Claims → Verify → Render (fail-closed)

Modules:
    - ingest:    Document chunking, embedding, and indexing
    - retrieve:  Hybrid BM25 + dense retrieval with reranking
    - claim_ir:  Claim compilation, normalization, and validation
    - verify:    Verifier adapters (MiniCheck, NLI, LLM-judge) + calibration
    - render:    Deterministic policy engine + certificate generation
    - pipeline:  End-to-end orchestrator
"""

__version__ = "0.1.0"
__all__ = ["__version__"]

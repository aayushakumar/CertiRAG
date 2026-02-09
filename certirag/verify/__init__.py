"""
CertiRAG Verification System
==============================

Verifier adapters, calibration, and Minimal Sufficient Evidence selection.

Components:
    - verifier.py:        Abstract verifier interface
    - minicheck.py:       MiniCheck adapter (primary, GPU)
    - nli_verifier.py:    NLI baseline (DeBERTa-MNLI)
    - llm_judge.py:       LLM-as-judge (OpenAI API)
    - gemini_verifier.py: Gemini-as-judge (free tier)
    - calibrator.py:      Score calibration (temperature scaling, isotonic)
    - mse.py:             Minimal Sufficient Evidence selection
"""

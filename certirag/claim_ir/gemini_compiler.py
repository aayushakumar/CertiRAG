"""
Gemini Claim Compiler
=====================

Uses Google Gemini API (free tier) as the LLM backend for
decomposing answers into atomic, evidence-linked claims.

Drop-in alternative to the OpenAI-based compiler, following
the same ClaimCompiler interface and prompt templates.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from certirag.config import CertiRAGConfig
from certirag.schemas.claim_ir import CandidateEvidence, Claim, ClaimIR, ClaimType
from certirag.schemas.evidence import EvidenceChunk
from certirag.claim_ir.compiler import ClaimCompiler, SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

logger = logging.getLogger("certirag.claim_ir.gemini_compiler")


class GeminiClaimCompiler(ClaimCompiler):
    """
    Claim compiler using Google Gemini API (free tier).

    Overrides the OpenAI _call_openai method to use Gemini instead,
    reusing the same prompt templates and parsing logic.

    Usage:
        compiler = GeminiClaimCompiler(config)
        claim_ir = compiler.compile(question, evidence_chunks)

    Args:
        config: CertiRAG configuration (uses gemini_api_key).
    """

    def __init__(self, config: CertiRAGConfig):
        super().__init__(config)
        self._gemini_client = None

    def _get_gemini_client(self):
        """Lazy-initialize Gemini client."""
        if self._gemini_client is None:
            try:
                from google import genai
                api_key = getattr(self.config, "gemini_api_key", None)
                if not api_key:
                    raise ValueError(
                        "Gemini API key required. Set CERTIRAG_GEMINI_API_KEY "
                        "or pass gemini_api_key in config."
                    )
                self._gemini_client = genai.Client(api_key=api_key)
            except ImportError:
                raise RuntimeError(
                    "google-genai package required. "
                    "Install with: pip install google-genai"
                )
        return self._gemini_client

    def _call_openai(self, messages: list[dict[str, str]]) -> str:
        """Override: call Gemini instead of OpenAI."""
        return self._call_gemini(messages)

    def _call_gemini(self, messages: list[dict[str, str]]) -> str:
        """Call Gemini API for claim compilation."""
        client = self._get_gemini_client()

        # Combine system + user messages into a single prompt
        system_msg = ""
        user_msg = ""
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                user_msg = msg["content"]

        full_prompt = f"{system_msg}\n\n{user_msg}"

        response = client.models.generate_content(
            model=getattr(self.config, "gemini_model", "gemini-2.0-flash"),
            contents=full_prompt,
            config={
                "temperature": self.config.claim.temperature,
                "max_output_tokens": 2000,
                "response_mime_type": "application/json",
            },
        )

        return response.text

"""
Claim Compiler
===============

Decomposes an LLM's answer into atomic, independently verifiable claims
linked to candidate evidence spans.

Architecture:
    Question + Evidence Chunks → LLM (structured output) → ClaimIR

Two Modes:
    - FULL mode: Uses local HuggingFace model (Llama-3-8B-Instruct)
    - LITE mode: Uses OpenAI API (gpt-4o-mini) with JSON mode

The compiler uses a carefully designed prompt with few-shot exemplars
to produce valid ClaimIR JSON. Schema-constrained generation ensures
100% valid output.

Key Requirements (FR-C1, FR-C2, FR-C3):
    - Output must be valid ClaimIR JSON
    - Each claim must be atomic (1 fact per claim)
    - Each claim must reference ≥1 evidence span
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from certirag.config import CertiRAGConfig
from certirag.schemas.claim_ir import CandidateEvidence, Claim, ClaimIR, ClaimType
from certirag.schemas.evidence import EvidenceChunk

logger = logging.getLogger("certirag.claim_ir.compiler")


# ── Prompt Templates ───────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise claim decomposition engine for a fact-verification system.

Your task: Given a question and retrieved evidence passages, generate an answer decomposed into ATOMIC claims.

RULES:
1. Each claim must contain exactly ONE verifiable fact
2. Each claim must be self-contained (understandable without other claims)
3. Each claim must reference at least one evidence passage by chunk_id and span_id
4. If a fact cannot be grounded in any evidence, set "unsupported_by_design": true
5. Classify each claim with a type from: factual, temporal_fact, numeric_fact, causal, comparative, definition, procedural, other
6. Extract structured fields where applicable (entity, year, quantity, etc.)
7. Do NOT include hedging language (may, possibly, might) in verified claims
8. Maximum {max_claims} claims per response

OUTPUT FORMAT: Return valid JSON matching this schema:
{{
  "question": "the original question",
  "claims": [
    {{
      "id": "c1",
      "text": "The claim text (one atomic fact)",
      "type": "factual|temporal_fact|numeric_fact|causal|comparative|definition|procedural|other",
      "fields": {{"entity": "...", "year": 2017}},
      "evidence_candidates": [
        {{"chunk_id": "doc1#c0", "span_id": "s0"}}
      ],
      "hedge": false,
      "unsupported_by_design": false
    }}
  ]
}}"""

FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Question: When was OpenAI founded and by whom?
Evidence:
  chunk_id: doc1#c0, spans:
    s0: "OpenAI was founded in December 2015."
    s1: "The founders include Sam Altman, Greg Brockman, and Ilya Sutskever."

Output:
{
  "question": "When was OpenAI founded and by whom?",
  "claims": [
    {
      "id": "c1",
      "text": "OpenAI was founded in December 2015.",
      "type": "temporal_fact",
      "fields": {"entity": "OpenAI", "year": 2015, "month": "December"},
      "evidence_candidates": [{"chunk_id": "doc1#c0", "span_id": "s0"}],
      "hedge": false,
      "unsupported_by_design": false
    },
    {
      "id": "c2",
      "text": "Sam Altman is one of the founders of OpenAI.",
      "type": "factual",
      "fields": {"entity": "Sam Altman", "organization": "OpenAI"},
      "evidence_candidates": [{"chunk_id": "doc1#c0", "span_id": "s1"}],
      "hedge": false,
      "unsupported_by_design": false
    },
    {
      "id": "c3",
      "text": "Greg Brockman is one of the founders of OpenAI.",
      "type": "factual",
      "fields": {"entity": "Greg Brockman", "organization": "OpenAI"},
      "evidence_candidates": [{"chunk_id": "doc1#c0", "span_id": "s1"}],
      "hedge": false,
      "unsupported_by_design": false
    },
    {
      "id": "c4",
      "text": "Ilya Sutskever is one of the founders of OpenAI.",
      "type": "factual",
      "fields": {"entity": "Ilya Sutskever", "organization": "OpenAI"},
      "evidence_candidates": [{"chunk_id": "doc1#c0", "span_id": "s1"}],
      "hedge": false,
      "unsupported_by_design": false
    }
  ]
}

EXAMPLE 2:
Question: How much revenue did Apple generate in Q3 2023?
Evidence:
  chunk_id: doc5#c2, spans:
    s0: "Apple reported quarterly revenue of $81.8 billion for Q3 2023."
    s1: "This represents a 1% year-over-year decline."

Output:
{
  "question": "How much revenue did Apple generate in Q3 2023?",
  "claims": [
    {
      "id": "c1",
      "text": "Apple reported quarterly revenue of $81.8 billion for Q3 2023.",
      "type": "numeric_fact",
      "fields": {"entity": "Apple", "metric": "revenue", "value": 81.8, "unit": "billion USD", "period": "Q3 2023"},
      "evidence_candidates": [{"chunk_id": "doc5#c2", "span_id": "s0"}],
      "hedge": false,
      "unsupported_by_design": false
    },
    {
      "id": "c2",
      "text": "Apple's Q3 2023 revenue represents a 1% year-over-year decline.",
      "type": "comparative",
      "fields": {"entity": "Apple", "change": -1, "unit": "percent", "period": "Q3 2023"},
      "evidence_candidates": [{"chunk_id": "doc5#c2", "span_id": "s1"}],
      "hedge": false,
      "unsupported_by_design": false
    }
  ]
}
"""


class ClaimCompiler:
    """
    Compiles LLM answers into structured Claim IR.

    Takes a question and retrieved evidence chunks, produces a ClaimIR
    object with atomic, evidence-linked claims.

    Two modes:
        - LITE: Uses OpenAI API (gpt-4o-mini) with JSON mode
        - FULL: Uses local HuggingFace model with structured generation

    Usage:
        compiler = ClaimCompiler(config)
        claim_ir = compiler.compile(
            question="When was X founded?",
            evidence_chunks=[chunk1, chunk2],
        )

    Args:
        config: CertiRAG configuration.
    """

    def __init__(self, config: CertiRAGConfig):
        self.config = config
        self._client = None

    def _format_evidence(self, chunks: list[EvidenceChunk]) -> str:
        """
        Format evidence chunks into a structured text block for the prompt.

        Each chunk is rendered with its chunk_id and sentence-level spans,
        making it easy for the LLM to reference specific evidence.
        """
        evidence_parts = []
        for chunk in chunks:
            part = f"  chunk_id: {chunk.chunk_id}, spans:"
            for span in chunk.spans:
                part += f"\n    {span.span_id}: \"{span.sentence}\""
            evidence_parts.append(part)
        return "\n".join(evidence_parts)

    def _build_prompt(
        self, question: str, evidence_chunks: list[EvidenceChunk]
    ) -> list[dict[str, str]]:
        """Build the chat messages for claim compilation."""
        system = SYSTEM_PROMPT.format(max_claims=self.config.claim.max_claims)
        evidence_text = self._format_evidence(evidence_chunks)

        user_message = f"""Question: {question}

Evidence:
{evidence_text}

Decompose the answer into atomic claims. Return valid JSON only."""

        return [
            {"role": "system", "content": system + "\n\n" + FEW_SHOT_EXAMPLES},
            {"role": "user", "content": user_message},
        ]

    def compile(
        self,
        question: str,
        evidence_chunks: list[EvidenceChunk],
    ) -> ClaimIR:
        """
        Compile a question + evidence into structured Claim IR.

        Pipeline:
            1. Format evidence chunks into prompt
            2. Call LLM (API or local) with JSON mode
            3. Parse and validate output as ClaimIR
            4. Enforce claim count cap

        Args:
            question: The user's question.
            evidence_chunks: Retrieved evidence chunks with spans.

        Returns:
            Validated ClaimIR object.

        Raises:
            ValueError: If LLM output cannot be parsed as valid ClaimIR.
        """
        if not evidence_chunks:
            logger.warning("No evidence chunks provided; generating empty ClaimIR")
            return ClaimIR(question=question, claims=[])

        messages = self._build_prompt(question, evidence_chunks)

        if self.config.is_lite:
            raw_output = self._call_openai(messages)
        else:
            raw_output = self._call_local(messages)

        return self._parse_output(raw_output, question)

    def _call_openai(self, messages: list[dict[str, str]]) -> str:
        """Call OpenAI API with JSON mode."""
        try:
            from openai import OpenAI

            if self._client is None:
                self._client = OpenAI(api_key=self.config.openai_api_key)

            response = self._client.chat.completions.create(
                model=self.config.claim.llm_model,
                messages=messages,
                temperature=self.config.claim.temperature,
                response_format={"type": "json_object"},
                max_tokens=2000,
            )
            return response.choices[0].message.content

        except ImportError:
            raise RuntimeError("openai package required. Install with: pip install openai")
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def _call_local(self, messages: list[dict[str, str]]) -> str:
        """Call local HuggingFace model for claim compilation."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_name = self.config.claim.llm_model
            logger.info(f"Loading local LLM: {model_name}")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2000,
                    temperature=max(self.config.claim.temperature, 0.01),
                    do_sample=self.config.claim.temperature > 0,
                )

            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            return generated

        except ImportError:
            raise RuntimeError(
                "transformers + torch required for FULL mode. "
                "Install with: pip install 'certirag[full]'"
            )

    def _parse_output(self, raw_output: str, question: str) -> ClaimIR:
        """
        Parse LLM output into a validated ClaimIR object.

        Handles common LLM output issues:
        - JSON wrapped in markdown code blocks
        - Missing fields (filled with defaults)
        - Extra/invalid claims (truncated to max_claims)
        """
        # Strip markdown code blocks if present
        output = raw_output.strip()
        if output.startswith("```"):
            lines = output.split("\n")
            output = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM output as JSON: {e}")
            logger.debug(f"Raw output: {raw_output[:500]}")
            # Return empty ClaimIR rather than crashing (fail-safe)
            return ClaimIR(question=question, claims=[], raw_answer=raw_output)

        # Build ClaimIR from parsed data
        claims = []
        raw_claims = data.get("claims", [])

        for i, raw_claim in enumerate(raw_claims[:self.config.claim.max_claims]):
            try:
                claim = Claim(
                    id=raw_claim.get("id", f"c{i + 1}"),
                    text=raw_claim.get("text", ""),
                    type=ClaimType(raw_claim.get("type", "factual")),
                    fields=raw_claim.get("fields", {}),
                    evidence_candidates=[
                        CandidateEvidence(**ec)
                        for ec in raw_claim.get("evidence_candidates", [])
                    ],
                    hedge=raw_claim.get("hedge", False),
                    unsupported_by_design=raw_claim.get("unsupported_by_design", False),
                )
                claims.append(claim)
            except Exception as e:
                logger.warning(f"Failed to parse claim {i}: {e}")
                continue

        claim_ir = ClaimIR(
            question=question,
            claims=claims,
            model_id=self.config.claim.llm_model,
            raw_answer=raw_output,
        )

        logger.info(
            f"Compiled {claim_ir.num_claims} claims "
            f"({claim_ir.num_grounded_claims} grounded)"
        )
        return claim_ir

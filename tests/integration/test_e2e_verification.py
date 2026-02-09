"""
End-to-End Tests with Real NLI Verification & Metrics
======================================================

Tests the full CertiRAG pipeline with a real NLI model (DeBERTa-v3-xsmall)
running on CPU. Measures precision, recall, F1, and accuracy for
fact-verification quality.

No API keys needed — uses local HuggingFace model.

Test Dataset:
    - Factual claims with supporting evidence → should be ENTAILED
    - Contradicted claims → should be CONTRADICTED
    - Unrelated evidence → should be NOT_ENOUGH_INFO

Metrics tracked:
    - Verification accuracy (3-class)
    - Entailment precision / recall / F1
    - Contradiction detection rate
    - Pipeline end-to-end latency
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pytest

from certirag.schemas.evidence import EvidenceChunk, EvidenceSpan, RetrievalScores, SourceInfo
from certirag.schemas.claim_ir import CandidateEvidence, Claim, ClaimIR, ClaimType
from certirag.schemas.verification import VerificationLabel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Test Dataset ─────────────────────────────────────────────────

@dataclass
class VerificationTestCase:
    """A single verification test case with expected label."""
    claim: str
    evidence: str
    expected_label: VerificationLabel
    description: str = ""


# Ground truth dataset for measuring verification quality
VERIFICATION_DATASET: list[VerificationTestCase] = [
    # ── ENTAILED cases (claim is supported by evidence) ──────────
    VerificationTestCase(
        claim="Paris is the capital of France.",
        evidence="Paris is the capital and most populous city of France.",
        expected_label=VerificationLabel.ENTAILED,
        description="Direct factual match",
    ),
    VerificationTestCase(
        claim="The Eiffel Tower was built in 1889.",
        evidence="The Eiffel Tower is a wrought-iron lattice tower constructed in 1889 for the World's Fair.",
        expected_label=VerificationLabel.ENTAILED,
        description="Temporal fact with supporting detail",
    ),
    VerificationTestCase(
        claim="Water boils at 100 degrees Celsius at sea level.",
        evidence="At standard atmospheric pressure (sea level), the boiling point of water is 100°C or 212°F.",
        expected_label=VerificationLabel.ENTAILED,
        description="Scientific fact with unit conversion",
    ),
    VerificationTestCase(
        claim="Albert Einstein developed the theory of general relativity.",
        evidence="Einstein published his general theory of relativity in 1915, fundamentally changing our understanding of gravity.",
        expected_label=VerificationLabel.ENTAILED,
        description="Attribution with temporal context",
    ),
    VerificationTestCase(
        claim="The Sun is a star.",
        evidence="The Sun is the star at the center of the Solar System, a nearly perfect sphere of hot plasma.",
        expected_label=VerificationLabel.ENTAILED,
        description="Simple definition match",
    ),
    VerificationTestCase(
        claim="Amazon was founded by Jeff Bezos.",
        evidence="Jeff Bezos founded Amazon in 1994 in his garage in Bellevue, Washington.",
        expected_label=VerificationLabel.ENTAILED,
        description="Founder attribution",
    ),
    VerificationTestCase(
        claim="DNA has a double helix structure.",
        evidence="The structure of DNA is a double helix, as discovered by Watson and Crick in 1953.",
        expected_label=VerificationLabel.ENTAILED,
        description="Scientific structure fact",
    ),
    VerificationTestCase(
        claim="The Great Wall of China is visible from space.",
        evidence="The Great Wall of China is a series of fortifications stretching over 13,000 miles.",
        expected_label=VerificationLabel.NOT_ENOUGH_INFO,
        description="Evidence doesn't mention visibility from space",
    ),

    # ── CONTRADICTED cases (evidence contradicts the claim) ──────
    VerificationTestCase(
        claim="Berlin is the capital of France.",
        evidence="Paris is the capital of France. Berlin is the capital of Germany.",
        expected_label=VerificationLabel.CONTRADICTED,
        description="Wrong capital attribution",
    ),
    VerificationTestCase(
        claim="The Eiffel Tower was built in 1920.",
        evidence="The Eiffel Tower was constructed in 1889 for the 1889 World's Fair held in Paris.",
        expected_label=VerificationLabel.CONTRADICTED,
        description="Wrong date",
    ),
    VerificationTestCase(
        claim="Water boils at 50 degrees Celsius.",
        evidence="Water boils at 100 degrees Celsius (212°F) at standard atmospheric pressure.",
        expected_label=VerificationLabel.CONTRADICTED,
        description="Wrong temperature",
    ),
    VerificationTestCase(
        claim="Tokyo is the capital of China.",
        evidence="Tokyo is the capital city of Japan, while Beijing is the capital of China.",
        expected_label=VerificationLabel.CONTRADICTED,
        description="Wrong country for capital",
    ),
    VerificationTestCase(
        claim="Amazon was founded by Bill Gates.",
        evidence="Amazon was founded by Jeff Bezos in 1994. Bill Gates founded Microsoft.",
        expected_label=VerificationLabel.CONTRADICTED,
        description="Wrong founder",
    ),

    # ── NOT_ENOUGH_INFO cases (evidence is unrelated) ────────────
    VerificationTestCase(
        claim="The population of Tokyo is 14 million.",
        evidence="Paris is the capital of France with a population of about 2.1 million.",
        expected_label=VerificationLabel.NOT_ENOUGH_INFO,
        description="Completely unrelated evidence",
    ),
    VerificationTestCase(
        claim="Python was created by Guido van Rossum.",
        evidence="JavaScript is a high-level programming language used primarily for web development.",
        expected_label=VerificationLabel.NOT_ENOUGH_INFO,
        description="Different programming language",
    ),
    VerificationTestCase(
        claim="Mount Everest is 8,849 meters tall.",
        evidence="The Mariana Trench is the deepest oceanic trench on Earth, reaching depths of 10,994 meters.",
        expected_label=VerificationLabel.NOT_ENOUGH_INFO,
        description="Unrelated geographic feature",
    ),
    VerificationTestCase(
        claim="The speed of light is approximately 300,000 km/s.",
        evidence="Sound travels at approximately 343 meters per second in air at room temperature.",
        expected_label=VerificationLabel.NOT_ENOUGH_INFO,
        description="Different physical constant",
    ),
    VerificationTestCase(
        claim="Shakespeare wrote Romeo and Juliet.",
        evidence="Charles Dickens wrote Oliver Twist and A Tale of Two Cities in the 19th century.",
        expected_label=VerificationLabel.NOT_ENOUGH_INFO,
        description="Different author, different works",
    ),
]


@dataclass
class MetricsResult:
    """Holds evaluation metrics for verification quality."""
    total: int = 0
    correct: int = 0
    # Per-label counts (binary: entailed vs not-entailed)
    true_entail: int = 0
    pred_entail: int = 0
    tp_entail: int = 0
    true_not_entail: int = 0
    pred_not_entail: int = 0
    tp_not_entail: int = 0
    # 3-class counts (for detailed reporting)
    true_contradict: int = 0
    pred_contradict: int = 0
    true_nei: int = 0
    pred_nei: int = 0
    # Binary accuracy (entailed vs not)
    binary_correct: int = 0
    # Timing
    total_time_ms: float = 0.0
    per_pair_ms: float = 0.0
    # Detailed results
    results: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """3-class accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def binary_accuracy(self) -> float:
        """Binary accuracy: entailed vs not-entailed."""
        return self.binary_correct / self.total if self.total > 0 else 0.0

    @property
    def entail_precision(self) -> float:
        return self.tp_entail / self.pred_entail if self.pred_entail > 0 else 0.0

    @property
    def entail_recall(self) -> float:
        return self.tp_entail / self.true_entail if self.true_entail > 0 else 0.0

    @property
    def entail_f1(self) -> float:
        p, r = self.entail_precision, self.entail_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def not_entail_precision(self) -> float:
        return self.tp_not_entail / self.pred_not_entail if self.pred_not_entail > 0 else 0.0

    @property
    def not_entail_recall(self) -> float:
        return self.tp_not_entail / self.true_not_entail if self.true_not_entail > 0 else 0.0

    @property
    def not_entail_f1(self) -> float:
        p, r = self.not_entail_precision, self.not_entail_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "VERIFICATION METRICS REPORT",
            "=" * 60,
            f"Total test cases:        {self.total}",
            f"3-class accuracy:        {self.accuracy:.1%}",
            f"Binary accuracy:         {self.binary_accuracy:.1%}",
            f"  (entailed vs not-entailed — the metric that matters)",
            "",
            "── Entailment Detection (PRIMARY METRIC) ──",
            f"  Precision: {self.entail_precision:.1%}  "
            f"Recall: {self.entail_recall:.1%}  "
            f"F1: {self.entail_f1:.1%}",
            f"  (TP={self.tp_entail}, Pred={self.pred_entail}, True={self.true_entail})",
            "",
            "── Not-Entailed Detection ──",
            f"  Precision: {self.not_entail_precision:.1%}  "
            f"Recall: {self.not_entail_recall:.1%}  "
            f"F1: {self.not_entail_f1:.1%}",
            f"  (TP={self.tp_not_entail}, Pred={self.pred_not_entail}, True={self.true_not_entail})",
            "",
            "── Timing ──",
            f"  Total: {self.total_time_ms:.0f}ms  Per-pair: {self.per_pair_ms:.0f}ms",
            "=" * 60,
        ]
        return "\n".join(lines)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def nli_verifier():
    """Load the HF NLI verifier once per test module (expensive)."""
    try:
        from certirag.verify.hf_nli_verifier import HFNLIVerifier
        verifier = HFNLIVerifier(device="cpu")
        return verifier
    except ImportError:
        pytest.skip("torch/transformers not available")


@pytest.fixture(scope="module")
def verification_dataset():
    """Return the ground truth verification dataset."""
    return VERIFICATION_DATASET


# ── Core Verification Tests ──────────────────────────────────────

class TestHFNLIVerifier:
    """Test the HuggingFace NLI verifier with real model inference."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_model_loads(self, nli_verifier):
        """Verify the NLI model loads successfully."""
        assert nli_verifier is not None
        assert "deberta" in nli_verifier._model_name.lower()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_single_entailment(self, nli_verifier):
        """Test a clear entailment case."""
        pairs = [("Paris is the capital of France.", "Paris is the capital and most populous city of France.")]
        scores = nli_verifier.verify_batch(pairs)
        assert len(scores) == 1
        assert scores[0].label == VerificationLabel.ENTAILED
        assert scores[0].score > 0.8

    @pytest.mark.integration
    @pytest.mark.slow
    def test_single_contradiction(self, nli_verifier):
        """Test a clear contradiction case.
        
        Note: 2-class models map contradictions to NOT_ENOUGH_INFO,
        which is correct for CertiRAG's fail-closed design.
        The key invariant: contradictions must NOT be ENTAILED.
        """
        pairs = [("Berlin is the capital of France.", "Paris is the capital of France.")]
        scores = nli_verifier.verify_batch(pairs)
        assert len(scores) == 1
        assert scores[0].label != VerificationLabel.ENTAILED

    @pytest.mark.integration
    @pytest.mark.slow
    def test_single_nei(self, nli_verifier):
        """Test a clear not-enough-info case (unrelated evidence)."""
        pairs = [("Python was created by Guido van Rossum.", "JavaScript is used for web development.")]
        scores = nli_verifier.verify_batch(pairs)
        assert len(scores) == 1
        # For 2-class model: NEI maps to NOT_ENOUGH_INFO
        # The key invariant: it should NOT be ENTAILED
        assert scores[0].label != VerificationLabel.ENTAILED

    @pytest.mark.integration
    @pytest.mark.slow
    def test_batch_verification(self, nli_verifier):
        """Test batch verification with multiple pairs."""
        pairs = [
            ("Paris is the capital of France.", "Paris is the capital of France."),
            ("Berlin is the capital of France.", "Paris is the capital of France."),
            ("The sky is blue.", "Water freezes at 0 degrees."),
        ]
        scores = nli_verifier.verify_batch(pairs)
        assert len(scores) == 3
        assert scores[0].label == VerificationLabel.ENTAILED
        # #2 and #3 should NOT be entailed (contradiction or NEI)
        assert scores[1].label != VerificationLabel.ENTAILED
        assert scores[2].label != VerificationLabel.ENTAILED


class TestVerificationMetrics:
    """Run full verification benchmark and compute metrics."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_verification_accuracy(self, nli_verifier, verification_dataset):
        """
        Run all test cases and compute accuracy metrics.

        Primary quality gate: ≥90% BINARY accuracy (entailed vs not-entailed).
        This is the metric that matters for CertiRAG's fail-closed design.
        """
        metrics = MetricsResult()
        pairs = [(tc.claim, tc.evidence) for tc in verification_dataset]

        t0 = time.time()
        scores = nli_verifier.verify_batch(pairs)
        elapsed_ms = (time.time() - t0) * 1000

        metrics.total = len(verification_dataset)
        metrics.total_time_ms = elapsed_ms
        metrics.per_pair_ms = elapsed_ms / len(pairs) if pairs else 0

        for tc, score in zip(verification_dataset, scores):
            expected = tc.expected_label
            predicted = score.label

            # Binary classification: entailed vs not-entailed
            expected_entailed = expected == VerificationLabel.ENTAILED
            predicted_entailed = predicted == VerificationLabel.ENTAILED

            # Count true/predicted for binary metrics
            if expected_entailed:
                metrics.true_entail += 1
            else:
                metrics.true_not_entail += 1

            if predicted_entailed:
                metrics.pred_entail += 1
            else:
                metrics.pred_not_entail += 1

            # Binary correct
            if expected_entailed == predicted_entailed:
                metrics.binary_correct += 1
                if expected_entailed:
                    metrics.tp_entail += 1
                else:
                    metrics.tp_not_entail += 1

            # 3-class correct
            if predicted == expected:
                metrics.correct += 1

            # Track 3-class distribution
            if expected == VerificationLabel.CONTRADICTED:
                metrics.true_contradict += 1
            elif expected == VerificationLabel.NOT_ENOUGH_INFO:
                metrics.true_nei += 1
            if predicted == VerificationLabel.CONTRADICTED:
                metrics.pred_contradict += 1
            elif predicted == VerificationLabel.NOT_ENOUGH_INFO:
                metrics.pred_nei += 1

            metrics.results.append({
                "claim": tc.claim[:50],
                "expected": expected.value,
                "predicted": predicted.value,
                "score": score.score,
                "binary_correct": expected_entailed == predicted_entailed,
                "exact_correct": predicted == expected,
                "description": tc.description,
            })

        # Print detailed report
        print("\n" + metrics.summary())
        print("\nDetailed Results:")
        for r in metrics.results:
            status = "✓" if r["binary_correct"] else "✗"
            exact = "=" if r["exact_correct"] else "≈" if r["binary_correct"] else "✗"
            print(
                f"  {status}{exact} [{r['expected']:>15s} → {r['predicted']:>15s}] "
                f"(score={r['score']:.3f}) {r['description']}"
            )

        # ── Quality gates (binary) ──
        assert metrics.binary_accuracy >= 0.90, (
            f"Binary accuracy {metrics.binary_accuracy:.1%} below 90% threshold"
        )
        assert metrics.entail_f1 >= 0.85, (
            f"Entailment F1 {metrics.entail_f1:.1%} below 85% threshold"
        )
        assert metrics.not_entail_f1 >= 0.85, (
            f"Not-entailment F1 {metrics.not_entail_f1:.1%} below 85% threshold"
        )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_verification_latency(self, nli_verifier):
        """Verify that per-pair latency is acceptable (<2000ms on CPU)."""
        pairs = [
            ("Paris is the capital of France.", "Paris is the capital of France."),
            ("Berlin is the capital of Germany.", "Berlin is the capital of Germany."),
            ("Tokyo is the capital of Japan.", "Tokyo is the capital of Japan."),
            ("London is the capital of England.", "London is the capital of England."),
        ]
        t0 = time.time()
        nli_verifier.verify_batch(pairs)
        elapsed_ms = (time.time() - t0) * 1000
        per_pair_ms = elapsed_ms / len(pairs)
        print(f"\nLatency: {elapsed_ms:.0f}ms total, {per_pair_ms:.0f}ms per pair")
        # First batch includes model loading overhead; allow generous threshold
        assert per_pair_ms < 2000, f"Per-pair latency {per_pair_ms:.0f}ms exceeds 2000ms"


class TestPipelineIntegration:
    """Test the full pipeline end-to-end with real verification."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_with_hf_verifier(self):
        """
        Run full pipeline: ingest → retrieve → compile (mock) → verify → render.
        Uses HF NLI verifier for real verification.
        """
        from certirag.config import CertiRAGConfig, ExecutionMode
        from certirag.verify.hf_nli_verifier import HFNLIVerifier
        from certirag.verify.mse import MSESelector
        from certirag.render.policy import RendererPolicy
        from certirag.claim_ir.normalizer import ClaimNormalizer

        # Build a ClaimIR manually (simulating claim compilation)
        evidence_chunks = [
            EvidenceChunk(
                chunk_id="doc1#c0",
                doc_id="doc1",
                text="Paris is the capital of France. The Eiffel Tower was built in 1889.",
                source=SourceInfo(title="Paris Facts"),
                retrieval=RetrievalScores(bm25=0.8),
                spans=[
                    EvidenceSpan(span_id="s0", start=0, end=30,
                                 sentence="Paris is the capital of France."),
                    EvidenceSpan(span_id="s1", start=31, end=66,
                                 sentence="The Eiffel Tower was built in 1889."),
                ],
            ),
            EvidenceChunk(
                chunk_id="doc2#c0",
                doc_id="doc2",
                text="Berlin is the capital of Germany. The Berlin Wall fell in 1989.",
                source=SourceInfo(title="Berlin Facts"),
                retrieval=RetrievalScores(bm25=0.6),
                spans=[
                    EvidenceSpan(span_id="s0", start=0, end=32,
                                 sentence="Berlin is the capital of Germany."),
                    EvidenceSpan(span_id="s1", start=33, end=62,
                                 sentence="The Berlin Wall fell in 1989."),
                ],
            ),
        ]

        claim_ir = ClaimIR(
            question="What is the capital of France?",
            claims=[
                Claim(
                    id="c1",
                    text="Paris is the capital of France.",
                    type=ClaimType.FACTUAL,
                    evidence_candidates=[
                        CandidateEvidence(chunk_id="doc1#c0", span_id="s0"),
                    ],
                ),
                Claim(
                    id="c2",
                    text="The Eiffel Tower was built in 1889.",
                    type=ClaimType.TEMPORAL_FACT,
                    evidence_candidates=[
                        CandidateEvidence(chunk_id="doc1#c0", span_id="s1"),
                    ],
                ),
                Claim(
                    id="c3",
                    text="Berlin is the capital of France.",
                    type=ClaimType.FACTUAL,
                    evidence_candidates=[
                        CandidateEvidence(chunk_id="doc2#c0", span_id="s0"),
                    ],
                ),
            ],
        )

        # Step 1: Verify with real NLI model
        verifier = HFNLIVerifier(device="cpu")
        verification_results = verifier.verify_claims(claim_ir, evidence_chunks)

        assert len(verification_results) == 3

        # c1: "Paris is the capital of France" vs "Paris is the capital of France."
        # → should be ENTAILED
        c1_result = next(r for r in verification_results if r.claim_id == "c1")
        assert c1_result.label == VerificationLabel.ENTAILED
        assert c1_result.score > 0.8
        print(f"\n  c1 (Paris/France): {c1_result.label.value} ({c1_result.score:.3f})")

        # c2: "Eiffel Tower built in 1889" vs "The Eiffel Tower was built in 1889."
        # → should be ENTAILED
        c2_result = next(r for r in verification_results if r.claim_id == "c2")
        assert c2_result.label == VerificationLabel.ENTAILED
        assert c2_result.score > 0.8
        print(f"  c2 (Eiffel/1889):  {c2_result.label.value} ({c2_result.score:.3f})")

        # c3: "Berlin is the capital of France" vs "Berlin is the capital of Germany."
        # → 2-class model: NOT_ENOUGH_INFO; 3-class: CONTRADICTED
        # Key invariant: must NOT be ENTAILED (fail-closed)
        c3_result = next(r for r in verification_results if r.claim_id == "c3")
        assert c3_result.label != VerificationLabel.ENTAILED
        print(f"  c3 (Berlin/France): {c3_result.label.value} ({c3_result.score:.3f})")

        # Step 2: MSE selection
        config = CertiRAGConfig(mode=ExecutionMode.LITE)
        mse_selector = MSESelector(tau_entail=config.verification.tau_entail)
        mse_results = mse_selector.select_batch(verification_results)

        # Step 3: Render decisions
        renderer = RendererPolicy.from_config(config)
        decisions = renderer.render(mse_results)

        # Verify rendering
        print(f"\n  Renderer decisions: {len(decisions)}")
        for d in decisions:
            print(f"    {d.claim_id}: {d.render_state.value}")

        # c1 should be VERIFIED, c3 should be BLOCKED or UNVERIFIED
        c1_decision = next(d for d in decisions if d.claim_id == "c1")
        c3_decision = next(d for d in decisions if d.claim_id == "c3")
        assert c1_decision.render_state.value in ("VERIFIED",)
        assert c3_decision.render_state.value in ("BLOCKED", "UNVERIFIED")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline_with_ingest(self):
        """
        Test the complete pipeline including document ingestion.
        Uses BM25-only retrieval (no embedding model needed).
        """
        from certirag.config import CertiRAGConfig, ExecutionMode
        from certirag.ingest.chunker import DocumentChunker
        from certirag.ingest.indexer import BM25Index, ChunkStore
        from certirag.retrieve.hybrid import SimpleRetriever

        # Build retrieval components directly (no pipeline overhead)
        chunker = DocumentChunker(chunk_size=50, overlap=1, use_spacy=False)
        bm25_index = BM25Index()
        chunk_store = ChunkStore()

        documents = [
            {
                "doc_id": "doc_paris",
                "title": "Paris",
                "text": (
                    "Paris is the capital of France. It is located on the Seine river. "
                    "The city has a population of approximately 2.1 million people. "
                    "Paris is known for the Eiffel Tower, built in 1889. "
                    "The Louvre Museum in Paris houses the Mona Lisa."
                ),
            },
            {
                "doc_id": "doc_berlin",
                "title": "Berlin",
                "text": (
                    "Berlin is the capital of Germany. It is the largest city in Germany "
                    "by population. The Berlin Wall divided the city from 1961 to 1989. "
                    "Berlin is known for the Brandenburg Gate and Museum Island."
                ),
            },
        ]

        # Chunk and index
        all_chunks = chunker.chunk_documents(documents)
        chunk_store.add_many(all_chunks)
        bm25_index.build(all_chunks)

        print(f"\n  Ingested: {len(documents)} docs → {len(all_chunks)} chunks")
        for c in all_chunks:
            print(f"    {c.chunk_id}: {c.text[:60]}... ({len(c.spans)} spans)")

        # Retrieve
        retriever = SimpleRetriever(bm25_index, chunk_store)
        chunks = retriever.retrieve("What is the capital of France?", top_k=3)

        assert len(chunks) > 0, "BM25 retrieval returned no chunks"
        print(f"  Retrieved {len(chunks)} chunks")

        # Find the chunk most likely to contain the answer
        # Use verify_batch directly (bypasses span-lookup complexity)
        from certirag.verify.hf_nli_verifier import HFNLIVerifier
        verifier = HFNLIVerifier(device="cpu")

        claim = "Paris is the capital of France."
        # Find the Paris chunk (BM25 may not return it first)
        paris_chunk = next(
            (c for c in chunks if "Paris" in c.text), chunks[0]
        )
        best_evidence = paris_chunk.text
        print(f"  Best chunk text: {best_evidence[:80]}...")

        pairs = [(claim, best_evidence)]
        results = verifier.verify_batch(pairs)
        assert len(results) == 1
        print(f"  Verification: {results[0].label.value} ({results[0].score:.3f})")

        # The claim "Paris is the capital of France" should be entailed
        assert results[0].label == VerificationLabel.ENTAILED
        assert results[0].score > 0.7


class TestGeminiVerifier:
    """Test the Gemini verifier (requires API key)."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_gemini_verifier_basic(self):
        """Test Gemini verifier if API key is available and quota allows."""
        import os
        api_key = os.environ.get("CERTIRAG_GEMINI_API_KEY")
        if not api_key:
            pytest.skip("CERTIRAG_GEMINI_API_KEY not set")

        from certirag.verify.gemini_verifier import GeminiVerifier
        verifier = GeminiVerifier(api_key=api_key)

        pairs = [
            ("Paris is the capital of France.", "Paris is the capital and most populous city of France."),
        ]
        scores = verifier.verify_batch(pairs)
        assert len(scores) == 1
        # If quota is exhausted, Gemini falls back to NOT_ENOUGH_INFO with score 0.5
        if scores[0].label == VerificationLabel.NOT_ENOUGH_INFO and scores[0].score <= 0.5:
            pytest.skip("Gemini API quota exhausted — skipping")
        assert scores[0].label == VerificationLabel.ENTAILED
        print(f"\n  Gemini result: {scores[0].label.value} ({scores[0].score:.3f})")

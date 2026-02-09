"""
CertiRAG — Futuristic Web UI
================================

A glassmorphic, dark-themed Streamlit dashboard for interactive
claim-level verification with live audit certificates.

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import streamlit as st

# ── Ensure project root is on sys.path ──────────────────────────
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ui.pipeline import run_pipeline  # noqa: E402
from ui.styles import get_css  # noqa: E402

# ── Page Config (must be first Streamlit call) ──────────────────
st.set_page_config(
    page_title="CertiRAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS
st.markdown(get_css(), unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_sidebar() -> dict:
    """Render the sidebar controls and return settings."""
    with st.sidebar:
        st.markdown("### 🛡️ CertiRAG")
        st.markdown(
            '<p style="color: #64748b; font-family: JetBrains Mono, monospace; '
            'font-size: 0.7rem; letter-spacing: 1.5px; text-transform: uppercase; '
            'margin-top: -12px;">VERIFICATION ENGINE</p>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Mode selector
        mode = st.selectbox(
            "Execution Mode",
            ["lite", "full"],
            help="LITE = CPU/API (Codespaces), FULL = GPU (Colab Pro)",
        )

        # Display mode
        display_mode = st.selectbox(
            "Display Mode",
            ["mixed", "strict", "debug"],
            help=(
                "STRICT = only verified claims. "
                "MIXED = verified + unverified. "
                "DEBUG = all claims with traces."
            ),
        )

        st.markdown("---")

        # Threshold controls
        st.markdown(
            '<p style="color: #00e5ff; font-family: JetBrains Mono, monospace; '
            'font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase;">'
            '⚙ Thresholds</p>',
            unsafe_allow_html=True,
        )

        tau_entail = st.slider(
            "τ_entail",
            0.0, 1.0, 0.40, 0.05,
            help="Minimum entailment score for VERIFIED status.",
        )
        tau_contradict = st.slider(
            "τ_contradict",
            0.0, 1.0, 0.70, 0.05,
            help="Maximum contradiction score before BLOCKED status.",
        )

        st.markdown("---")

        # Advanced options
        with st.expander("⚡ Advanced"):
            top_k = st.number_input("Top-K Evidence", 1, 20, 5)
            max_claims = st.number_input("Max Claims", 1, 50, 20)

        st.markdown("---")

        # System status
        st.markdown(
            '<div style="background: rgba(15, 23, 42, 0.6); '
            'border: 1px solid rgba(100, 116, 139, 0.2); border-radius: 10px; '
            'padding: 12px; font-family: JetBrains Mono, monospace; font-size: 0.7rem;">'
            '<div style="color: #64748b; text-transform: uppercase; letter-spacing: 1px; '
            'margin-bottom: 8px;">System</div>'
            '<div style="color: #94a3b8;">Mode: <span style="color: #00e5ff;">'
            f'{mode.upper()}</span></div>'
            '<div style="color: #94a3b8;">Engine: <span style="color: #22c55e;">'
            'ONLINE</span></div>'
            '<div style="color: #94a3b8;">Policy: <span style="color: #a855f7;">'
            'v1.0</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    return {
        "mode": mode,
        "display_mode": display_mode,
        "tau_entail": tau_entail,
        "tau_contradict": tau_contradict,
        "top_k": top_k,
        "max_claims": max_claims,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HERO BANNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_hero():
    """Render the hero banner at the top."""
    st.markdown(
        '<div class="hero-banner">'
        '<div class="hero-subtitle">Certified Retrieval-Augmented Generation</div>'
        '<h1 style="margin: 0; font-size: 2.4rem;">CertiRAG</h1>'
        '<div class="hero-tagline">'
        'Claim-level verification with fail-closed rendering guarantees. '
        'Every claim is grounded in evidence, scored by NLI, and sealed '
        'in a tamper-evident audit certificate.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PIPELINE STEP VISUALIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_pipeline_steps(steps: list[tuple[str, int]] | None = None):
    """Render the pipeline step indicators."""
    step_names = [
        ("📥", "Ingest"),
        ("🔍", "Retrieve"),
        ("📝", "Claims"),
        ("⚖️", "Normalize"),
        ("✅", "Verify"),
        ("🎨", "Render"),
        ("📜", "Certify"),
    ]

    html_parts = ['<div class="pipeline-steps">']
    for i, (icon, name) in enumerate(step_names):
        css = "done" if steps else ""
        count = ""
        if steps and i < len(steps):
            count = f' <span style="opacity: 0.6;">({steps[i][1]})</span>'
        html_parts.append(
            f'<div class="pipeline-step {css}">{icon} {name}{count}</div>'
        )
        if i < len(step_names) - 1:
            html_parts.append('<span class="pipeline-arrow">→</span>')
    html_parts.append("</div>")

    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  METRICS DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_metrics(result: dict):
    """Render the metrics dashboard row."""
    stats = result["display"]["stats"]
    decisions = result["decisions"]

    verified = sum(1 for d in decisions if d.render_state.value == "VERIFIED")
    blocked = sum(1 for d in decisions if d.render_state.value == "BLOCKED")
    unverified = stats["total"] - verified - blocked

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Claims", stats["total"])
    with c2:
        st.metric("✅ Verified", verified)
    with c3:
        st.metric("⚠️ Unverified", unverified)
    with c4:
        st.metric("❌ Blocked", blocked)
    with c5:
        st.metric("⚡ Latency", f"{result['elapsed']*1000:.0f}ms")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLAIM CARDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _render_score_bar(label: str, value: float, bar_class: str) -> str:
    """Generate HTML for a score bar."""
    pct = max(0, min(100, value * 100))
    return (
        f'<div class="score-bar-container">'
        f'<div class="score-bar-label">'
        f'<span>{label}</span><span>{value:.3f}</span>'
        f'</div>'
        f'<div class="score-bar">'
        f'<div class="score-bar-fill {bar_class}" style="width: {pct}%;"></div>'
        f'</div>'
        f'</div>'
    )


def render_claims(result: dict):
    """Render claim verification cards."""
    visible = result["display"]["visible_claims"]
    hidden = result["display"]["hidden_claims"]

    if not visible and not hidden:
        st.info("No claims extracted. Try a more detailed question.")
        return

    # Visible claims
    for entry in visible:
        claim = entry["claim"]
        decision = entry["decision"]
        state = decision["render_state"]
        css_class = state.lower()

        icon_map = {"VERIFIED": "✅", "UNVERIFIED": "⚠️", "BLOCKED": "❌"}
        icon = icon_map.get(state, "❓")

        # Scores from decision
        entail = decision.get("entail_score", 0.0)
        contradict = decision.get("contradict_score", 0.0)
        ev_count = decision.get("evidence_count", 0)

        # Build score bars HTML
        score_html = (
            _render_score_bar("Entailment", entail, "entail")
            + _render_score_bar("Contradiction", contradict, "contradict")
        )

        st.markdown(
            f'<div class="claim-card {css_class}">'
            f'<div class="badge {css_class}">{icon} {state}</div>'
            f'<div class="claim-text">{claim["text"]}</div>'
            f'{score_html}'
            f'<div class="claim-meta">'
            f'<span>📊 Evidence: {ev_count}</span>'
            f'<span>🏷️ Type: {claim.get("type", "factual")}</span>'
            f'<span>🔑 ID: {claim["id"]}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Hidden claims
    if hidden:
        with st.expander(f"🚫 Hidden Claims ({len(hidden)})"):
            for entry in hidden:
                claim = entry.get("claim", {})
                decision = entry.get("decision", {})
                reason = decision.get("reason", "No decision available")

                st.markdown(
                    f'<div class="claim-card blocked" style="opacity: 0.7;">'
                    f'<div class="badge blocked">❌ HIDDEN</div>'
                    f'<div class="claim-text" style="text-decoration: line-through; '
                    f'opacity: 0.6;">{claim.get("text", "")}</div>'
                    f'<div class="claim-meta"><span>💬 {reason}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EVIDENCE EXPLORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_evidence(result: dict):
    """Render the evidence explorer panel."""
    evidence = result["evidence"]

    if not evidence:
        st.info("No evidence chunks retrieved.")
        return

    for ev in evidence[:10]:
        # Build span tags
        span_html = ""
        if ev.spans:
            spans = "".join(
                f'<span class="span-tag">{s.span_id}: "{s.sentence[:50]}..."</span>'
                if len(s.sentence) > 50
                else f'<span class="span-tag">{s.span_id}: "{s.sentence}"</span>'
                for s in ev.spans[:5]
            )
            span_html = f'<div class="ev-spans">{spans}</div>'

        # Retrieval scores
        bm25 = ev.retrieval.bm25
        dense = ev.retrieval.dense
        rrf = ev.retrieval.rrf

        score_html = (
            _render_score_bar("BM25", min(bm25, 1.0), "entail")
            + _render_score_bar("Dense", min(dense, 1.0), "neutral")
            + _render_score_bar("RRF", min(rrf, 1.0), "entail")
        )

        st.markdown(
            f'<div class="evidence-panel">'
            f'<div class="ev-header">'
            f'<span class="ev-source">📄 {ev.doc_id} / {ev.chunk_id}</span>'
            f'<span class="ev-score">BM25: {bm25:.3f}</span>'
            f'</div>'
            f'<div class="ev-text">{ev.text}</div>'
            f'{score_html}'
            f'{span_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VERIFICATION DETAIL TAB
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_verification_detail(result: dict):
    """Render detailed verification scores per claim."""
    claim_map = {c.id: c for c in result["claim_ir"].claims}

    for vr in result["verification_results"]:
        claim = claim_map.get(vr.claim_id)
        claim_text = claim.text if claim else vr.claim_id

        label_icon = {
            "entailed": "✅",
            "contradicted": "❌",
            "not_enough_info": "⚠️",
        }
        icon = label_icon.get(vr.label.value, "❓")

        with st.expander(f"{icon} {vr.label.value.upper()} — {claim_text[:80]}"):
            st.markdown(
                f'<div class="info-box">'
                f'Overall: <code>{vr.label.value}</code> · '
                f'Score: <code>{vr.score:.3f}</code> · '
                f'Evidence spans: <code>{vr.evidence_span_count}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Per-evidence scores
            st.markdown("#### Evidence Scores")
            for es in vr.all_scores:
                label_css = {
                    "entailed": "entail",
                    "contradicted": "contradict",
                    "not_enough_info": "neutral",
                }.get(es.label.value, "neutral")

                st.markdown(
                    f'<div style="background: rgba(15, 23, 42, 0.5); '
                    f'border: 1px solid rgba(100, 116, 139, 0.15); '
                    f'border-radius: 8px; padding: 10px; margin: 6px 0; '
                    f'font-family: JetBrains Mono, monospace; font-size: 0.78rem;">'
                    f'<span style="color: #94a3b8;">Chunk:</span> '
                    f'<span style="color: #3b82f6;">{es.chunk_id}</span> · '
                    f'<span style="color: #94a3b8;">Span:</span> '
                    f'<span style="color: #a855f7;">{es.span_id}</span> · '
                    f'<span style="color: #94a3b8;">Label:</span> '
                    f'<span style="color: #00e5ff;">{es.label.value}</span>'
                    f'{_render_score_bar("Score", es.score, label_css)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AUDIT CERTIFICATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_certificate(result: dict):
    """Render the audit certificate."""
    cert = result["certificate"]

    integrity_ok = cert.verify_integrity()
    integrity_icon = "✅" if integrity_ok else "❌"
    integrity_color = "#22c55e" if integrity_ok else "#ef4444"

    st.markdown(
        f'<div class="audit-cert">'
        f'<div class="cert-header">'
        f'<span style="font-size: 1.5rem;">📜</span>'
        f'<span class="cert-title">Audit Certificate</span>'
        f'<span style="margin-left: auto; color: {integrity_color}; font-size: 0.75rem;">'
        f'{integrity_icon} Integrity {"VERIFIED" if integrity_ok else "FAILED"}</span>'
        f'</div>'
        f'<div class="cert-field">'
        f'<span class="cert-key">Query ID</span>'
        f'<span class="cert-value">{cert.query_id}</span>'
        f'</div>'
        f'<div class="cert-field">'
        f'<span class="cert-key">Timestamp</span>'
        f'<span class="cert-value">{cert.timestamp}</span>'
        f'</div>'
        f'<div class="cert-field">'
        f'<span class="cert-key">Question</span>'
        f'<span class="cert-value">{cert.question[:60]}...</span>'
        f'</div>'
        f'<div class="cert-field">'
        f'<span class="cert-key">Claims</span>'
        f'<span class="cert-value">{len(cert.claims)}</span>'
        f'</div>'
        f'<div class="cert-field">'
        f'<span class="cert-key">Policy</span>'
        f'<span class="cert-value">τ_e={cert.policy.tau_entail} · '
        f'τ_c={cert.policy.tau_contradict}</span>'
        f'</div>'
        f'<div class="cert-field">'
        f'<span class="cert-key">Latency</span>'
        f'<span class="cert-value">{cert.stats.get("latency_ms", 0)}ms</span>'
        f'</div>'
        f'<div class="cert-field">'
        f'<span class="cert-key">Integrity Hash</span>'
        f'<span class="cert-hash">{cert.integrity_hash}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Raw JSON download
    with st.expander("📋 Raw Certificate JSON"):
        cert_json = cert.model_dump_json(indent=2)
        st.code(cert_json, language="json")
        st.download_button(
            "⬇ Download Certificate",
            data=cert_json,
            file_name=f"certirag_cert_{cert.query_id}.json",
            mime="application/json",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  INPUT SECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_input():
    """Render the input section and return (question, documents)."""
    col1, col2 = st.columns([3, 2])

    with col1:
        question = st.text_area(
            "💬 Question",
            value=(
                "What is the capital of France and what is it known for?"
            ),
            height=120,
            placeholder="Enter a question to verify against documents...",
        )

    with col2:
        documents = st.text_area(
            "📄 Documents",
            value=(
                "Paris is the capital and most populous city of France. "
                "It is situated on the Seine River. Paris is known for "
                "the Eiffel Tower, which was completed in 1889. The "
                "Louvre Museum in Paris is the world's most-visited museum."
            ),
            height=120,
            placeholder="Paste reference documents (one paragraph per line)...",
        )

    return question, documents


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN APPLICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    """Main application entry point."""
    settings = render_sidebar()
    render_hero()

    # Pipeline step visualization (inactive state)
    render_pipeline_steps()

    # Input
    question, documents = render_input()

    # Run button
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        run_clicked = st.button(
            "🔍  VERIFY CLAIMS",
            type="primary",
            use_container_width=True,
        )

    if run_clicked:
        if not question.strip() or not documents.strip():
            st.warning("Please enter both a question and documents.")
            return

        # Animated progress
        progress_placeholder = st.empty()

        steps = [
            "Ingesting documents...",
            "Building BM25 index...",
            "Extracting claims...",
            "Normalizing claims...",
            "Running verification...",
            "Applying render policy...",
            "Sealing audit certificate...",
        ]

        with st.spinner(""):
            progress_bar = progress_placeholder.progress(0, text="Initializing...")
            for i, step_text in enumerate(steps):
                progress_bar.progress(
                    (i + 1) / len(steps),
                    text=f"⚡ {step_text}",
                )
                time.sleep(0.15)  # Brief visual delay for animation effect

            try:
                result = run_pipeline(question, documents, settings)
            except Exception as e:
                progress_placeholder.empty()
                st.error(f"Pipeline error: {e}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
                return

        progress_placeholder.empty()

        if result is None:
            st.error("Pipeline returned no results.")
            return

        # Store result in session state for tab persistence
        st.session_state["result"] = result

    # ── Render results if available ─────────────────────────────
    result = st.session_state.get("result")
    if result:
        st.markdown("---")

        # Updated pipeline steps
        render_pipeline_steps(result.get("pipeline_steps"))

        # Metrics
        render_metrics(result)

        st.markdown("---")

        # Tabbed results
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Claims",
            "📚 Evidence",
            "🔬 Verification",
            "📜 Certificate",
        ])

        with tab1:
            render_claims(result)

        with tab2:
            render_evidence(result)

        with tab3:
            render_verification_detail(result)

        with tab4:
            render_certificate(result)


# ── Entry Point ─────────────────────────────────────────────────

if __name__ == "__main__":
    main()
else:
    # When run via `streamlit run ui/app.py`
    main()

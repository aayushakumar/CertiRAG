"""
CertiRAG Futuristic UI — CSS Theme
====================================

Glassmorphism · Neon Accents · Animated Backgrounds
Dark cyberpunk aesthetic with sci-fi dashboard feel.
"""

# ── Color Tokens ────────────────────────────────────────────────
COLORS = {
    "bg_deep":      "#050510",
    "bg_surface":   "rgba(10, 15, 30, 0.85)",
    "bg_glass":     "rgba(15, 23, 42, 0.60)",
    "bg_glass_alt": "rgba(25, 33, 52, 0.55)",
    "bg_input":     "rgba(15, 20, 40, 0.90)",
    "accent_cyan":  "#00e5ff",
    "accent_purple":"#a855f7",
    "accent_blue":  "#3b82f6",
    "accent_green": "#22c55e",
    "accent_amber": "#f59e0b",
    "accent_red":   "#ef4444",
    "text_primary": "#e2e8f0",
    "text_secondary":"#94a3b8",
    "text_muted":   "#64748b",
    "border":       "rgba(100, 116, 139, 0.20)",
    "border_glow":  "rgba(0, 229, 255, 0.25)",
}


def get_css() -> str:
    """Return the full CSS stylesheet for the futuristic UI."""
    return f"""
<style>
/* ━━━━━━━━━━ ROOT / RESET ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {{
    --bg-deep:       {COLORS["bg_deep"]};
    --bg-surface:    {COLORS["bg_surface"]};
    --bg-glass:      {COLORS["bg_glass"]};
    --accent-cyan:   {COLORS["accent_cyan"]};
    --accent-purple: {COLORS["accent_purple"]};
    --accent-blue:   {COLORS["accent_blue"]};
    --accent-green:  {COLORS["accent_green"]};
    --accent-amber:  {COLORS["accent_amber"]};
    --accent-red:    {COLORS["accent_red"]};
    --text-primary:  {COLORS["text_primary"]};
    --text-secondary:{COLORS["text_secondary"]};
    --text-muted:    {COLORS["text_muted"]};
    --border:        {COLORS["border"]};
    --glow-cyan:     0 0 15px rgba(0, 229, 255, 0.35), 0 0 40px rgba(0, 229, 255, 0.10);
    --glow-purple:   0 0 15px rgba(168, 85, 247, 0.35), 0 0 40px rgba(168, 85, 247, 0.10);
    --glow-green:    0 0 15px rgba(34, 197, 94, 0.35), 0 0 40px rgba(34, 197, 94, 0.10);
    --glow-red:      0 0 15px rgba(239, 68, 68, 0.35), 0 0 40px rgba(239, 68, 68, 0.10);
    --glass-blur:    blur(20px);
    --radius:        12px;
    --radius-lg:     16px;
    --transition:    all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    color-scheme: dark;
}}

/* ━━━━━━━━━━ ANIMATED BACKGROUND ━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

.stApp {{
    background: var(--bg-deep) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--text-primary) !important;
}}

.stApp::before {{
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 800px 600px at 20% 30%, rgba(0, 229, 255, 0.06) 0%, transparent 60%),
        radial-gradient(ellipse 600px 500px at 80% 70%, rgba(168, 85, 247, 0.06) 0%, transparent 60%),
        radial-gradient(ellipse 500px 400px at 50% 50%, rgba(59, 130, 246, 0.03) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: bgShift 20s ease-in-out infinite alternate;
}}

@keyframes bgShift {{
    0%   {{ opacity: 0.8; transform: scale(1); }}
    50%  {{ opacity: 1;   transform: scale(1.05); }}
    100% {{ opacity: 0.8; transform: scale(1); }}
}}

/* Grid overlay */
.stApp::after {{
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0, 229, 255, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 229, 255, 0.02) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}}

/* ━━━━━━━━━━ SIDEBAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(10, 15, 35, 0.95), rgba(5, 10, 25, 0.98)) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: var(--glass-blur) !important;
}}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {{
    color: var(--accent-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    text-shadow: 0 0 20px rgba(0, 229, 255, 0.3);
}}

section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
section[data-testid="stSidebar"] .stSlider label {{
    color: var(--text-secondary) !important;
}}

section[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}}

/* ━━━━━━━━━━ HEADINGS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

.stApp h1 {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 900 !important;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple), var(--accent-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-fill-color: transparent;
    filter: drop-shadow(0 0 30px rgba(0, 229, 255, 0.2));
}}

.stApp h2 {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    border-bottom: 2px solid rgba(0, 229, 255, 0.2);
    padding-bottom: 8px;
}}

.stApp h3 {{
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    color: var(--accent-cyan) !important;
    font-size: 0.95rem !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}

/* ━━━━━━━━━━ BUTTONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

.stButton > button {{
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.15), rgba(168, 85, 247, 0.15)) !important;
    border: 1px solid rgba(0, 229, 255, 0.3) !important;
    color: var(--accent-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 1px;
    border-radius: var(--radius) !important;
    transition: var(--transition) !important;
    text-transform: uppercase;
    padding: 0.6rem 1.5rem !important;
}}

.stButton > button:hover {{
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.30), rgba(168, 85, 247, 0.30)) !important;
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
    transform: translateY(-1px);
}}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.25), rgba(59, 130, 246, 0.25)) !important;
    border: 1px solid var(--accent-cyan) !important;
    color: #fff !important;
    box-shadow: var(--glow-cyan) !important;
}}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.40), rgba(59, 130, 246, 0.40)) !important;
    box-shadow: 0 0 25px rgba(0, 229, 255, 0.5), 0 0 60px rgba(0, 229, 255, 0.15) !important;
}}

/* ━━━━━━━━━━ INPUTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

.stTextArea textarea,
.stTextInput input {{
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    border-radius: var(--radius) !important;
    transition: var(--transition) !important;
}}

.stTextArea textarea:focus,
.stTextInput input:focus {{
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 10px rgba(0, 229, 255, 0.15) !important;
}}

/* ━━━━━━━━━━ TABS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

.stTabs [data-baseweb="tab-list"] {{
    background: var(--bg-glass) !important;
    border-radius: var(--radius-lg) !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
    backdrop-filter: var(--glass-blur) !important;
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    transition: var(--transition) !important;
    padding: 8px 20px !important;
}}

.stTabs [data-baseweb="tab"]:hover {{
    background: rgba(0, 229, 255, 0.08) !important;
    color: var(--accent-cyan) !important;
}}

.stTabs [aria-selected="true"] {{
    background: rgba(0, 229, 255, 0.12) !important;
    color: var(--accent-cyan) !important;
    box-shadow: inset 0 0 15px rgba(0, 229, 255, 0.08) !important;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: var(--accent-cyan) !important;
    height: 2px !important;
}}

/* ━━━━━━━━━━ METRIC CARDS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

[data-testid="stMetric"] {{
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 16px 20px !important;
    backdrop-filter: var(--glass-blur) !important;
    transition: var(--transition) !important;
}}

[data-testid="stMetric"]:hover {{
    border-color: rgba(0, 229, 255, 0.3) !important;
    box-shadow: var(--glow-cyan) !important;
    transform: translateY(-2px);
}}

[data-testid="stMetricLabel"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    color: var(--text-muted) !important;
}}

[data-testid="stMetricValue"] {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

/* ━━━━━━━━━━ EXPANDERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

[data-testid="stExpander"] {{
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    backdrop-filter: var(--glass-blur) !important;
    overflow: hidden;
    transition: var(--transition) !important;
}}

[data-testid="stExpander"]:hover {{
    border-color: rgba(0, 229, 255, 0.25) !important;
}}

[data-testid="stExpander"] summary {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
}}

/* ━━━━━━━━━━ DIVIDERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

hr {{
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-purple), transparent) !important;
    opacity: 0.3 !important;
    margin: 1.5rem 0 !important;
}}

/* ━━━━━━━━━━ PROGRESS BARS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

.stProgress > div > div {{
    background: rgba(30, 41, 59, 0.6) !important;
    border-radius: 8px !important;
}}

.stProgress > div > div > div {{
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple)) !important;
    border-radius: 8px !important;
}}

/* ━━━━━━━━━━ CUSTOM COMPONENTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

/* Hero banner */
.hero-banner {{
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.08), rgba(168, 85, 247, 0.08));
    border: 1px solid rgba(0, 229, 255, 0.15);
    border-radius: var(--radius-lg);
    padding: 28px 32px;
    margin-bottom: 24px;
    backdrop-filter: var(--glass-blur);
    position: relative;
    overflow: hidden;
}}

.hero-banner::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 0deg, transparent, rgba(0, 229, 255, 0.03), transparent 30%);
    animation: heroRotate 15s linear infinite;
}}

@keyframes heroRotate {{
    100% {{ transform: rotate(360deg); }}
}}

.hero-subtitle {{
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
}}

.hero-tagline {{
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-top: 8px;
    line-height: 1.5;
}}

/* Claim cards */
.claim-card {{
    background: var(--bg-glass);
    border-left: 3px solid var(--text-muted);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 16px 20px;
    margin: 10px 0;
    backdrop-filter: var(--glass-blur);
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}}

.claim-card::after {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, rgba(0, 229, 255, 0.02), transparent);
    pointer-events: none;
}}

.claim-card:hover {{
    transform: translateX(4px);
}}

.claim-card.verified {{
    border-left-color: var(--accent-green);
    box-shadow: -2px 0 15px rgba(34, 197, 94, 0.15);
}}

.claim-card.unverified {{
    border-left-color: var(--accent-amber);
    box-shadow: -2px 0 15px rgba(245, 158, 11, 0.15);
}}

.claim-card.blocked {{
    border-left-color: var(--accent-red);
    box-shadow: -2px 0 15px rgba(239, 68, 68, 0.15);
}}

.claim-card .badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}}

.badge.verified {{
    background: rgba(34, 197, 94, 0.15);
    color: var(--accent-green);
    border: 1px solid rgba(34, 197, 94, 0.3);
}}

.badge.unverified {{
    background: rgba(245, 158, 11, 0.15);
    color: var(--accent-amber);
    border: 1px solid rgba(245, 158, 11, 0.3);
}}

.badge.blocked {{
    background: rgba(239, 68, 68, 0.15);
    color: var(--accent-red);
    border: 1px solid rgba(239, 68, 68, 0.3);
}}

.claim-text {{
    color: var(--text-primary);
    font-size: 0.95rem;
    line-height: 1.6;
}}

.claim-meta {{
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    margin-top: 8px;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}}

.claim-meta span {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
}}

/* Evidence panel */
.evidence-panel {{
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin: 8px 0;
    backdrop-filter: var(--glass-blur);
    transition: var(--transition);
    position: relative;
}}

.evidence-panel:hover {{
    border-color: rgba(59, 130, 246, 0.3);
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.08);
}}

.evidence-panel .ev-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}}

.evidence-panel .ev-source {{
    color: var(--accent-blue);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
}}

.evidence-panel .ev-score {{
    color: var(--accent-cyan);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    background: rgba(0, 229, 255, 0.08);
    padding: 2px 8px;
    border-radius: 10px;
    border: 1px solid rgba(0, 229, 255, 0.2);
}}

.evidence-panel .ev-text {{
    color: var(--text-primary);
    font-size: 0.9rem;
    line-height: 1.7;
}}

.evidence-panel .ev-spans {{
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
}}

.evidence-panel .span-tag {{
    display: inline-block;
    background: rgba(168, 85, 247, 0.1);
    color: var(--accent-purple);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 6px;
    margin: 2px 4px 2px 0;
    border: 1px solid rgba(168, 85, 247, 0.2);
}}

/* Score bars */
.score-bar-container {{
    margin: 6px 0;
}}

.score-bar-label {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
}}

.score-bar {{
    height: 6px;
    background: rgba(30, 41, 59, 0.6);
    border-radius: 6px;
    overflow: hidden;
    position: relative;
}}

.score-bar-fill {{
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}}

.score-bar-fill.entail {{
    background: linear-gradient(90deg, var(--accent-green), var(--accent-cyan));
}}

.score-bar-fill.contradict {{
    background: linear-gradient(90deg, var(--accent-red), var(--accent-amber));
}}

.score-bar-fill.neutral {{
    background: linear-gradient(90deg, var(--text-muted), var(--accent-blue));
}}

/* Audit certificate */
.audit-cert {{
    background: linear-gradient(135deg, rgba(10, 15, 35, 0.90), rgba(15, 20, 40, 0.85));
    border: 1px solid rgba(0, 229, 255, 0.2);
    border-radius: var(--radius-lg);
    padding: 24px;
    backdrop-filter: var(--glass-blur);
    position: relative;
    overflow: hidden;
    font-family: 'JetBrains Mono', monospace;
}}

.audit-cert::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple), var(--accent-blue));
}}

.audit-cert .cert-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}}

.audit-cert .cert-title {{
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 2px;
}}

.audit-cert .cert-field {{
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 0.78rem;
    border-bottom: 1px dashed rgba(100, 116, 139, 0.15);
}}

.audit-cert .cert-key {{
    color: var(--text-muted);
}}

.audit-cert .cert-value {{
    color: var(--text-primary);
}}

.audit-cert .cert-hash {{
    color: var(--accent-purple);
    font-size: 0.72rem;
    word-break: break-all;
}}

/* Pipeline steps */
.pipeline-steps {{
    display: flex;
    gap: 4px;
    align-items: center;
    margin: 20px 0;
    flex-wrap: wrap;
    justify-content: center;
}}

.pipeline-step {{
    display: flex;
    align-items: center;
    gap: 4px;
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.5px;
    text-transform: uppercase;
    backdrop-filter: var(--glass-blur);
    transition: var(--transition);
}}

.pipeline-step.active {{
    border-color: var(--accent-cyan);
    color: var(--accent-cyan);
    box-shadow: var(--glow-cyan);
    animation: stepPulse 2s ease-in-out infinite;
}}

.pipeline-step.done {{
    border-color: var(--accent-green);
    color: var(--accent-green);
}}

.pipeline-arrow {{
    color: var(--text-muted);
    font-size: 0.65rem;
    opacity: 0.5;
}}

@keyframes stepPulse {{
    0%, 100% {{ box-shadow: 0 0 5px rgba(0, 229, 255, 0.2); }}
    50%      {{ box-shadow: 0 0 20px rgba(0, 229, 255, 0.4); }}
}}

/* Status indicator */
.status-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: statusPulse 2s ease-in-out infinite;
}}

.status-dot.green {{ background: var(--accent-green); box-shadow: var(--glow-green); }}
.status-dot.amber {{ background: var(--accent-amber); }}
.status-dot.red   {{ background: var(--accent-red); box-shadow: var(--glow-red); }}

@keyframes statusPulse {{
    0%, 100% {{ opacity: 1; }}
    50%      {{ opacity: 0.5; }}
}}

/* Glass container */
.glass-container {{
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px;
    backdrop-filter: var(--glass-blur);
    margin: 12px 0;
}}

/* Info box */
.info-box {{
    background: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: var(--radius);
    padding: 12px 16px;
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.6;
}}

.info-box code {{
    background: rgba(0, 229, 255, 0.1);
    color: var(--accent-cyan);
    padding: 1px 5px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}}

/* Custom scrollbar */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}

::-webkit-scrollbar-track {{
    background: rgba(15, 23, 42, 0.3);
}}

::-webkit-scrollbar-thumb {{
    background: rgba(0, 229, 255, 0.2);
    border-radius: 3px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: rgba(0, 229, 255, 0.4);
}}

/* Toast / alert override */
.stAlert {{
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    backdrop-filter: var(--glass-blur) !important;
}}

/* Spinner override */
.stSpinner > div {{
    border-top-color: var(--accent-cyan) !important;
}}

/* Hide Streamlit branding */
#MainMenu, footer, header {{
    visibility: hidden;
}}

/* Selectbox */
[data-baseweb="select"] {{
    background: var(--bg-surface) !important;
}}

[data-baseweb="select"] > div {{
    background: var(--bg-surface) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius) !important;
}}

/* Column gap */
[data-testid="stHorizontalBlock"] {{
    gap: 12px;
}}

/* Remove default container padding for cleaner look */
.block-container {{
    padding-top: 2rem !important;
    max-width: 1200px !important;
}}
</style>
"""

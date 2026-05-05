"""
app.py
======
Basel II Credit Risk Dashboard — Production Grade
--------------------------------------------------
Bloomberg-style dark UI | Three pages:
  1. Inference    — Applicant scoring with gauge, credit band, approval logic
  2. Portfolio    — High-density metrics, RWA, stress testing
  3. Monitoring   — ROC-AUC, Gini, KS plot, Basel threshold checks

Run: streamlit run src/app.py
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from scipy.special import expit

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Basel II Credit Risk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR     = Path(__file__).resolve().parent
OUTPUTS_DIR = SRC_DIR.parent / "scorecard_outputs"

# ── Bloomberg-style CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0a0a0e;
    --surface:   #111116;
    --surface2:  #18181f;
    --border:    #252530;
    --border2:   #32323f;
    --text:      #e8e8f0;
    --muted:     #6b6b80;
    --dim:       #404055;
    --accent:    #00d4aa;
    --accent2:   #4a9eff;
    --warn:      #ffb347;
    --danger:    #ff5555;
    --good:      #50fa7b;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg); }

/* ── Header bar ── */
.bb-header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 10px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin: -1rem -1rem 1.5rem -1rem;
}
.bb-logo {
    font-family: var(--mono);
    font-size: 15px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.08em;
}
.bb-title {
    font-family: var(--sans);
    font-size: 13px;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.bb-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ── Metric cards ── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.blue::before  { background: var(--accent2); }
.metric-card.teal::before  { background: var(--accent); }
.metric-card.warn::before  { background: var(--warn); }
.metric-card.danger::before{ background: var(--danger); }
.metric-card.good::before  { background: var(--good); }

.metric-label {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--muted);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-family: var(--mono);
    font-size: 26px;
    font-weight: 600;
    line-height: 1;
    letter-spacing: -0.02em;
}
.metric-sub {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    margin-top: 6px;
}
.metric-delta-pos { color: var(--good); font-size: 11px; }
.metric-delta-neg { color: var(--danger); font-size: 11px; }

/* ── Section titles ── */
.section-title {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    border-left: 2px solid var(--accent);
    padding-left: 10px;
    margin-bottom: 14px;
    margin-top: 24px;
}

/* ── Form inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
    border-radius: 3px !important;
}
.stSelectbox > div > div > div:hover { border-color: var(--accent) !important; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    padding: 8px 20px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    color: var(--muted) !important;
}

/* ── Tables ── */
.stDataFrame { background: var(--surface) !important; }

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Badges ── */
.badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 2px;
    letter-spacing: 0.08em;
}
.badge-approve { background: rgba(80,250,123,0.12); color: var(--good); border: 1px solid var(--good); }
.badge-refer   { background: rgba(255,179,71,0.12);  color: var(--warn); border: 1px solid var(--warn); }
.badge-decline { background: rgba(255,85,85,0.12);   color: var(--danger); border: 1px solid var(--danger); }

/* ── Score band pill ── */
.band-pill {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 2px;
    display: inline-block;
    margin-top: 6px;
}
.band-exc  { background: rgba(80,250,123,0.15); color: #50fa7b; border: 1px solid #50fa7b44; }
.band-good { background: rgba(74,158,255,0.15); color: #4a9eff; border: 1px solid #4a9eff44; }
.band-fav  { background: rgba(0,212,170,0.15);  color: #00d4aa; border: 1px solid #00d4aa44; }
.band-avg  { background: rgba(255,179,71,0.15); color: #ffb347; border: 1px solid #ffb34744; }
.band-bad  { background: rgba(255,85,85,0.15);  color: #ff5555; border: 1px solid #ff555544; }

/* ── Stress toggle ── */
.stToggle label { font-family: var(--mono) !important; font-size: 12px !important; }

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — cached, error-handled
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load all model artifacts. Returns dict with None for missing files."""
    arts = {}

    def _load(key, path):
        try:
            with open(path, "rb") as f:
                arts[key] = pickle.load(f)
        except Exception as e:
            arts[key] = None
            arts[f"_{key}_error"] = str(e)

    _load("model",        OUTPUTS_DIR / "model.pkl")
    _load("features",     OUTPUTS_DIR / "feature_names.pkl")
    _load("woe_mappings", OUTPUTS_DIR / "woe_mappings.pkl")
    _load("metrics",      OUTPUTS_DIR / "model_metrics.pkl")
    _load("lgd_model",    SRC_DIR      / "lgd_model.pkl")

    # Portfolio data — try multiple locations
    for el_path in [
        OUTPUTS_DIR / "el_results.pkl",
        OUTPUTS_DIR / "el_results.csv",
        SRC_DIR     / "el_results.pkl",
    ]:
        if el_path.exists():
            try:
                if el_path.suffix == ".csv":
                    arts["portfolio"] = pd.read_csv(el_path)
                else:
                    with open(el_path, "rb") as f:
                        arts["portfolio"] = pickle.load(f)
            except Exception:
                pass
            break

    if "portfolio" not in arts:
        arts["portfolio"] = None

    return arts


def align_features(model, feature_names: list) -> list:
    """
    Ensure feature_names length matches model.coef_ width.
    Trims or pads with zeros to prevent shape mismatches.
    """
    if model is None or feature_names is None:
        return feature_names or []
    n_coef = model.coef_.shape[1]
    n_feat = len(feature_names)
    if n_coef == n_feat:
        return feature_names
    if n_coef < n_feat:
        return feature_names[:n_coef]
    # pad: shouldn't happen but guard anyway
    return feature_names + [f"_pad_{i}" for i in range(n_coef - n_feat)]


def get_col(df: pd.DataFrame, *candidates: str, default=None):
    """Flexibly detect a column from multiple candidate names."""
    df_lower = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return df[c]
        if c.lower() in df_lower:
            return df[df_lower[c.lower()]]
    if default is not None:
        return pd.Series([default] * len(df), index=df.index)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SCORECARD LOGIC
# ══════════════════════════════════════════════════════════════════════════════

SA_BASE_SCORE = 620
SA_BASE_ODDS  = 8
SA_PDO        = 20
SA_APPROVAL   = 550          # approval cut-off (configurable)
SA_MIN_SCORE  = 300
SA_MAX_SCORE  = 850

SA_BANDS = [
    (767, 850, "Excellent",      "exc"),
    (681, 766, "Good",           "good"),
    (614, 680, "Favourable",     "fav"),
    (583, 613, "Average",        "avg"),
    (527, 582, "Below Average",  "bad"),
    (487, 526, "Unfavourable",   "bad"),
    (300, 486, "Poor",           "bad"),
]


def score_from_pd(pd_val: float) -> int:
    """Convert PD to scorecard score using PDO formula."""
    factor = SA_PDO / np.log(2)
    offset = SA_BASE_SCORE - factor * np.log(SA_BASE_ODDS)
    odds   = max((1 - pd_val) / max(pd_val, 1e-9), 0.001)
    raw    = offset + factor * np.log(odds)
    return int(np.clip(round(raw), SA_MIN_SCORE, SA_MAX_SCORE))


def get_band(score: int) -> tuple:
    for lo, hi, label, css in SA_BANDS:
        if lo <= score <= hi:
            return label, css
    return "Poor", "bad"


def get_decision(score: int, cutoff: int = SA_APPROVAL) -> tuple:
    if score >= cutoff + 15:
        return "APPROVE",  "approve"
    elif score >= cutoff - 10:
        return "REFER",    "refer"
    else:
        return "DECLINE",  "decline"


def compute_pd(model, woe_vec: np.ndarray) -> float:
    """Predict P(Good) then return P(Bad) = 1 − P(Good)."""
    if model is None:
        return 0.15
    logit = model.intercept_[0] + np.dot(model.coef_[0], woe_vec)
    p_good = expit(logit)
    return float(1 - p_good)


# ══════════════════════════════════════════════════════════════════════════════
# BASEL II RWA LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def basel_correlation(pd_val: float) -> float:
    """
    Basel II IRB correlation formula:
    R = 0.12 × (1 − e^{-50×PD}) / (1 − e^{-50}) + 0.24 × (1 − (1 − e^{-50×PD}) / (1 − e^{-50}))
    """
    e50   = np.exp(-50)
    denom = 1 - e50
    r1    = (1 - np.exp(-50 * pd_val)) / denom
    R     = 0.12 * r1 + 0.24 * (1 - r1)
    return R


def regulatory_capital(pd_val: float, lgd: float, maturity: float = 2.5,
                        size_adj: float = 0.0) -> float:
    """
    Basel II IRB formula for regulatory capital K.
    K = LGD × N[(N^{-1}(PD) + sqrt(R)×N^{-1}(0.999)) / sqrt(1-R)] − LGD × PD × Maturity adj
    """
    from scipy.stats import norm
    if pd_val <= 0 or pd_val >= 1:
        return 0.0
    R  = basel_correlation(pd_val)
    sr = np.sqrt(R)
    sr1= np.sqrt(1 - R)
    N  = norm.cdf
    Ni = norm.ppf
    # Maturity adjustment
    b  = (0.11852 - 0.05478 * np.log(max(pd_val, 1e-6))) ** 2
    ma = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
    K  = lgd * (N((Ni(pd_val) + sr * Ni(0.999)) / sr1) - pd_val) * ma
    return max(float(K), 0.0)


def compute_rwa(pd_val: float, lgd: float, ead: float) -> dict:
    K   = regulatory_capital(pd_val, lgd)
    RWA = K * ead * 12.5
    EL  = pd_val * lgd * ead
    return {"K": K, "RWA": RWA, "EL": EL}


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY HELPERS — Bloomberg dark style
# ══════════════════════════════════════════════════════════════════════════════

DARK_LAYOUT = dict(
    paper_bgcolor="#111116",
    plot_bgcolor="#111116",
    font=dict(family="IBM Plex Mono", color="#e8e8f0", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="#18181f", bordercolor="#252530", borderwidth=1,
                font=dict(size=10)),
    xaxis=dict(gridcolor="#252530", zerolinecolor="#252530",
               tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#252530", zerolinecolor="#252530",
               tickfont=dict(size=10)),
)


def gauge_chart(score: int, cutoff: int = SA_APPROVAL) -> go.Figure:
    pct   = (score - SA_MIN_SCORE) / (SA_MAX_SCORE - SA_MIN_SCORE)
    color = "#50fa7b" if score >= cutoff + 15 else \
            "#ffb347" if score >= cutoff - 10 else "#ff5555"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"font": {"family": "IBM Plex Mono", "size": 40,
                         "color": color}, "suffix": ""},
        gauge={
            "axis": {"range": [SA_MIN_SCORE, SA_MAX_SCORE],
                     "tickwidth": 1, "tickcolor": "#404055",
                     "tickfont": {"size": 9, "color": "#6b6b80"},
                     "nticks": 10},
            "bar":  {"color": color, "thickness": 0.22},
            "bgcolor":    "#18181f",
            "borderwidth": 0,
            "steps": [
                {"range": [300, 487], "color": "#1a0a0a"},
                {"range": [487, 583], "color": "#1a1200"},
                {"range": [583, 681], "color": "#0a1a12"},
                {"range": [681, 767], "color": "#0a1020"},
                {"range": [767, 850], "color": "#0a1a14"},
            ],
            "threshold": {
                "line":  {"color": "#4a9eff", "width": 2},
                "thickness": 0.8,
                "value": cutoff,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#111116",
        font=dict(family="IBM Plex Mono", color="#e8e8f0"),
        height=240,
        margin=dict(l=20, r=20, t=10, b=10),
    )
    return fig


def roc_chart(fpr, tpr, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color="#00d4aa", width=2),
        name=f"AUC = {auc:.4f}",
        fill="tozeroy", fillcolor="rgba(0,212,170,0.06)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#404055", width=1, dash="dot"),
        name="Random", showlegend=False,
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="ROC CURVE", font=dict(size=10, color="#6b6b80")),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=300,
    )
    return fig


def ks_chart(ks_df: pd.DataFrame, ks_stat: float) -> go.Figure:
    fig = go.Figure()
    x   = ks_df.index / len(ks_df)
    fig.add_trace(go.Scatter(x=x, y=ks_df["cum_good"], mode="lines",
                             line=dict(color="#4a9eff", width=2),
                             name="Cumulative Good"))
    fig.add_trace(go.Scatter(x=x, y=ks_df["cum_bad"], mode="lines",
                             line=dict(color="#ff5555", width=2),
                             name="Cumulative Bad"))
    ks_idx = (ks_df["cum_good"] - ks_df["cum_bad"]).abs().idxmax()
    xv = ks_idx / len(ks_df)
    fig.add_vline(x=xv, line_color="#ffb347", line_width=1, line_dash="dot",
                  annotation_text=f"KS={ks_stat:.4f}",
                  annotation_font=dict(color="#ffb347", size=10))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="KS CHART", font=dict(size=10, color="#6b6b80")),
        xaxis_title="Population Fraction",
        yaxis_title="Cumulative Rate",
        height=300,
    )
    return fig


def pd_dist_chart(pd_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pd_series, nbinsx=40,
        marker_color="#4a9eff", opacity=0.7,
        marker_line=dict(color="#111116", width=0.5),
        name="PD Distribution",
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="PORTFOLIO PD DISTRIBUTION", font=dict(size=10, color="#6b6b80")),
        xaxis_title="Probability of Default",
        yaxis_title="Count",
        height=260,
    )
    return fig


def el_by_grade_chart(df: pd.DataFrame, el_col: str, grade_col: str) -> go.Figure:
    grp = df.groupby(grade_col)[el_col].sum().reset_index()
    grp = grp.sort_values(grade_col)
    fig = go.Figure(go.Bar(
        x=grp[grade_col], y=grp[el_col],
        marker_color="#00d4aa", opacity=0.85,
        marker_line=dict(color="#111116", width=0.5),
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="EXPECTED LOSS BY GRADE", font=dict(size=10, color="#6b6b80")),
        xaxis_title="Grade",
        yaxis_title="Expected Loss",
        height=260,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def header():
    st.markdown("""
    <div class="bb-header">
        <span class="bb-logo">NEXUS RISK</span>
        <div class="bb-dot"></div>
        <span class="bb-title">Basel II Credit Risk Intelligence Platform</span>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "", color: str = "teal",
                delta: str = None, delta_pos: bool = True):
    delta_html = ""
    if delta:
        cls = "metric-delta-pos" if delta_pos else "metric-delta-neg"
        delta_html = f'<div class="{cls}">{delta}</div>'
    st.markdown(f"""
    <div class="metric-card {color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-sub">{sub}</div>' if sub else ""}
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# WOE TRANSFORM (inline — avoids import dependency issues)
# ══════════════════════════════════════════════════════════════════════════════

def woe_transform_single(input_dict: dict, woe_mappings: dict,
                          feature_names: list) -> np.ndarray:
    """
    Transform a single applicant's raw values into WOE-encoded feature vector.
    Unmatched values get WOE = 0 (population average — conservative).
    """
    result = []
    for feat in feature_names:
        val = input_dict.get(feat, None)
        if feat not in woe_mappings or val is None:
            result.append(0.0)
            continue
        tbl = woe_mappings[feat]
        if isinstance(tbl, pd.DataFrame) and "bin" in tbl.columns and "woe" in tbl.columns:
            woe_map = dict(zip(tbl["bin"].astype(str), tbl["woe"]))
        elif isinstance(tbl, dict):
            woe_map = {str(k): float(v) if not hasattr(v, "__len__") else 0.0
                       for k, v in tbl.items()}
        else:
            result.append(0.0)
            continue
        # Find matching bin
        matched = False
        for bin_label, woe_val in woe_map.items():
            if bin_label == "Missing":
                continue
            try:
                # Try interval matching
                import re
                m = re.match(r"[\(\[](.+),\s*(.+)[\)\]]", str(bin_label))
                if m and isinstance(val, (int, float)):
                    lo = float(m.group(1).replace("-inf", "-999999999"))
                    hi = float(m.group(2).replace("inf",  "999999999"))
                    if lo < val <= hi:
                        result.append(float(woe_val))
                        matched = True
                        break
                elif str(val).strip().lower() == bin_label.strip().lower():
                    result.append(float(woe_val))
                    matched = True
                    break
            except Exception:
                continue
        if not matched:
            # fallback: most common bin WOE or 0
            result.append(0.0)
    return np.array(result, dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def page_inference(arts: dict):
    section("APPLICANT ASSESSMENT")

    model    = arts.get("model")
    features = arts.get("features", [])
    woe_maps = arts.get("woe_mappings", {})
    lgd_mdl  = arts.get("lgd_model", {}) or {}

    if model is None:
        st.error("⚠  model.pkl not found. Ensure scorecard_outputs/model.pkl exists.")
        return

    features = align_features(model, features)

    cutoff = st.sidebar.slider("Approval cut-off score", 480, 700, SA_APPROVAL, 5)

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("applicant_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Loan Details**")
            loan_amnt  = st.number_input("Loan Amount (R)",    value=25000.0, step=1000.0)
            int_rate   = st.number_input("Interest Rate (%)",  value=14.5,    step=0.5)
            term       = st.selectbox("Term",                  ["36 months", "60 months"])
            installment= st.number_input("Installment (R/mo)", value=850.0,   step=50.0)
            sub_grade  = st.selectbox("Sub Grade",
                         ["A1","A2","A3","A4","A5","B1","B2","B3","B4","B5",
                          "C1","C2","C3","C4","C5","D1","D2","D3","D4","D5",
                          "E1","E2","E3","E4","E5","F1","F2","F3","F4","F5",
                          "G1","G2","G3","G4","G5"])

        with c2:
            st.markdown("**Borrower Profile**")
            annual_inc       = st.number_input("Annual Income (R)",      value=480000.0, step=10000.0)
            dti              = st.number_input("Debt-to-Income Ratio",   value=18.5,     step=0.5)
            emp_length       = st.selectbox("Employment Length",
                              ["< 1 year","1 year","2 years","3 years","4 years",
                               "5 years","6 years","7 years","8 years","9 years","10+ years"])
            home_ownership   = st.selectbox("Home Ownership",            ["RENT","OWN","MORTGAGE","OTHER"])
            verification_status = st.selectbox("Verification Status",   ["Verified","Source Verified","Not Verified"])
            purpose          = st.selectbox("Purpose",
                              ["debt_consolidation","credit_card","home_improvement",
                               "other","major_purchase","medical","small_business",
                               "car","vacation","moving","house","wedding","renewable_energy","educational"])

        with c3:
            st.markdown("**Credit Bureau**")
            fico_range_high  = st.number_input("FICO High",          value=700.0, step=1.0)
            fico_range_low   = st.number_input("FICO Low",           value=695.0, step=1.0)
            revol_bal        = st.number_input("Revolving Balance",   value=12000.0, step=500.0)
            revol_util       = st.number_input("Revolving Util (%)",  value=42.0, step=1.0)
            open_acc         = st.number_input("Open Accounts",       value=8, step=1)
            delinq_2yrs      = st.number_input("Delinquencies (2yr)", value=0, step=1)
            inq_last_6mths   = st.number_input("Inquiries (6mo)",     value=1, step=1)
            pub_rec          = st.number_input("Public Records",       value=0, step=1)
            total_acc        = st.number_input("Total Accounts",       value=18, step=1)

        submitted = st.form_submit_button("▶  RUN ASSESSMENT", use_container_width=True)

    if not submitted:
        st.markdown('<p style="color:#6b6b80; font-size:12px; font-family:IBM Plex Mono;">↑  Complete the form and click RUN ASSESSMENT</p>', unsafe_allow_html=True)
        return

    # ── Compute PD ────────────────────────────────────────────────────────────
    raw_vals = {
        "loan_amnt": loan_amnt, "int_rate": int_rate,
        "installment": installment, "sub_grade": sub_grade,
        "emp_length": emp_length, "home_ownership": home_ownership,
        "annual_inc": annual_inc, "verification_status": verification_status,
        "purpose": purpose, "dti": dti,
        "fico_range_high": fico_range_high, "fico_range_low": fico_range_low,
        "delinq_2yrs": delinq_2yrs, "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc, "pub_rec": pub_rec,
        "revol_bal": revol_bal, "revol_util": revol_util,
        "total_acc": total_acc, "term": term,
    }

    woe_vec = woe_transform_single(raw_vals, woe_maps, features)
    pd_val  = compute_pd(model, woe_vec)
    score   = score_from_pd(pd_val)
    band, band_css = get_band(score)
    decision, dec_css = get_decision(score, cutoff)

    # LGD from model or Basel constant
    grade   = sub_grade[0] if sub_grade else "C"
    lgd_map = lgd_mdl.get("lgd_by_grade", {}) if lgd_mdl else {}
    lgd_val = float(lgd_map.get(grade, 0.45)) if hasattr(lgd_map, "get") else 0.45

    rwa_res = compute_rwa(pd_val, lgd_val, loan_amnt)

    # ── Results row ───────────────────────────────────────────────────────────
    st.markdown("---")
    section("ASSESSMENT RESULTS")

    r1, r2, r3, r4, r5 = st.columns([2, 1, 1, 1, 1])

    with r1:
        st.plotly_chart(gauge_chart(score, cutoff), use_container_width=True)
        st.markdown(f"""
        <div style="text-align:center;">
            <span class="band-pill band-{band_css}">{band.upper()}</span>
            &nbsp;
            <span class="badge badge-{dec_css}">{decision}</span>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        metric_card("Credit Score", str(score), f"cutoff {cutoff}",
                    "good" if score >= cutoff + 15 else "warn" if score >= cutoff - 10 else "danger")
        st.markdown("<br>", unsafe_allow_html=True)
        metric_card("Probability of Default", f"{pd_val*100:.2f}%",
                    "P(Bad | application)", "danger" if pd_val > 0.20 else "warn" if pd_val > 0.10 else "teal")

    with r3:
        metric_card("LGD", f"{lgd_val*100:.1f}%", f"Grade {grade}", "blue")
        st.markdown("<br>", unsafe_allow_html=True)
        metric_card("Expected Loss", f"R{rwa_res['EL']:,.0f}",
                    "PD × LGD × EAD", "warn" if rwa_res["EL"] > loan_amnt * 0.05 else "teal")

    with r4:
        metric_card("Reg. Capital (K)", f"{rwa_res['K']*100:.2f}%",
                    "Basel II IRB formula", "blue")
        st.markdown("<br>", unsafe_allow_html=True)
        metric_card("RWA", f"R{rwa_res['RWA']:,.0f}",
                    "K × EAD × 12.5", "warn")

    with r5:
        corr = basel_correlation(pd_val)
        metric_card("Basel Correlation (R)", f"{corr*100:.1f}%",
                    "Asset correlation", "blue")
        st.markdown("<br>", unsafe_allow_html=True)
        metric_card("EAD", f"R{loan_amnt:,.0f}", "Funded amount", "teal")

    # ── Explanation ───────────────────────────────────────────────────────────
    section("DECISION RATIONALE")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Output</div>
            <div style="font-family: IBM Plex Mono; font-size: 12px; color: #e8e8f0; line-height: 1.8;">
                <span style="color:#6b6b80">Score         :</span> {score}<br>
                <span style="color:#6b6b80">PD            :</span> {pd_val*100:.3f}%<br>
                <span style="color:#6b6b80">Score band    :</span> {band}<br>
                <span style="color:#6b6b80">Cutoff        :</span> {cutoff}<br>
                <span style="color:#6b6b80">Decision      :</span> {decision}<br>
                <span style="color:#6b6b80">PDO           :</span> {SA_PDO}<br>
                <span style="color:#6b6b80">Base odds     :</span> {SA_BASE_ODDS}:1
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        top_feats = sorted(
            zip(features, woe_vec, model.coef_[0]),
            key=lambda x: abs(x[2] * x[1]), reverse=True
        )[:6]
        rows_html = "".join([
            f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
            f'border-bottom:1px solid #252530;">'
            f'<span style="color:#6b6b80;font-size:11px">{f}</span>'
            f'<span style="color:{"#50fa7b" if w*c > 0 else "#ff5555"};font-size:11px">'
            f'{w:.3f} × {c:.4f}</span></div>'
            for f, w, c in top_feats
        ])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Top Feature Contributions (WOE × Coef)</div>
            <div style="font-family: IBM Plex Mono; margin-top: 8px;">{rows_html}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════

def page_portfolio(arts: dict):
    section("PORTFOLIO ANALYTICS")

    portfolio = arts.get("portfolio")
    lgd_mdl   = arts.get("lgd_model") or {}

    if portfolio is None:
        st.warning("⚠  Portfolio data not found. Expected: scorecard_outputs/el_results.pkl")
        st.markdown("Place your `el_results.pkl` in `scorecard_outputs/` and restart.")
        return

    df = portfolio.copy() if isinstance(portfolio, pd.DataFrame) else pd.DataFrame(portfolio)
    df.columns = [c.strip() for c in df.columns]

    # ── Flexible column detection ──────────────────────────────────────────────
    pd_col  = get_col(df, "pd", "PD", "prob_default", "probability_of_default",
                      "predicted_pd", default=0.12)
    lgd_col = get_col(df, "lgd", "LGD", "loss_given_default", default=0.45)
    ead_col = get_col(df, "ead", "EAD", "exposure", "funded_amnt",
                      "loan_amnt", "exposure_at_default", default=10000)
    el_col  = get_col(df, "el", "EL", "expected_loss", default=None)

    df["_pd"]  = pd.to_numeric(pd_col,  errors="coerce").fillna(0.12).clip(1e-6, 1-1e-6)
    df["_lgd"] = pd.to_numeric(lgd_col, errors="coerce").fillna(0.45).clip(0, 1)
    df["_ead"] = pd.to_numeric(ead_col, errors="coerce").fillna(10000).clip(0)
    df["_el"]  = df["_pd"] * df["_lgd"] * df["_ead"] if el_col is None else \
                 pd.to_numeric(el_col, errors="coerce").fillna(df["_pd"] * df["_lgd"] * df["_ead"])

    # Compute RWA per row
    df["_K"]   = df.apply(lambda r: regulatory_capital(r["_pd"], r["_lgd"]), axis=1)
    df["_RWA"] = df["_K"] * df["_ead"] * 12.5

    # ── Stress test toggle ─────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    stress_on = st.sidebar.toggle("🔴 Stress Test (LGD +200bps)", value=False)
    stress_lgd_shift = 0.02 if stress_on else 0.0

    df["_lgd_stress"] = (df["_lgd"] + stress_lgd_shift).clip(0, 1)
    df["_el_stress"]  = df["_pd"] * df["_lgd_stress"] * df["_ead"]
    df["_RWA_stress"] = df.apply(
        lambda r: regulatory_capital(r["_pd"], r["_lgd_stress"]) * r["_ead"] * 12.5, axis=1
    )

    total_ead    = df["_ead"].sum()
    wa_pd        = (df["_pd"] * df["_ead"]).sum() / max(total_ead, 1)
    wa_lgd       = (df["_lgd"] * df["_ead"]).sum() / max(total_ead, 1)
    total_el     = df["_el"].sum()
    total_rwa    = df["_RWA"].sum()
    capital_req  = total_rwa * 0.08   # Basel II minimum 8% capital ratio

    total_el_s   = df["_el_stress"].sum()
    total_rwa_s  = df["_RWA_stress"].sum()
    el_delta     = total_el_s - total_el
    rwa_delta    = total_rwa_s - total_rwa

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        metric_card("Total Exposure", f"R{total_ead/1e6:.1f}M",
                    f"{len(df):,} loans", "blue")
    with k2:
        metric_card("WA PD", f"{wa_pd*100:.2f}%", "Exposure-weighted", "danger")
    with k3:
        metric_card("WA LGD", f"{wa_lgd*100:.1f}%", "Exposure-weighted", "warn")
    with k4:
        metric_card("Total EL", f"R{total_el/1e6:.2f}M",
                    f"+R{el_delta/1e6:.2f}M stress" if stress_on else "Baseline",
                    "warn" if not stress_on else "danger",
                    delta=f"+R{el_delta/1e6:.2f}M" if stress_on else None,
                    delta_pos=False)
    with k5:
        metric_card("Total RWA", f"R{total_rwa/1e6:.1f}M",
                    f"+R{rwa_delta/1e6:.1f}M stress" if stress_on else "Baseline",
                    "warn",
                    delta=f"+R{rwa_delta/1e6:.1f}M" if stress_on else None,
                    delta_pos=False)
    with k6:
        cap_ratio = (total_ead * 0.08) / max(total_rwa, 1)
        metric_card("Capital Adequacy", f"{cap_ratio*100:.1f}%",
                    "Min. 8% Basel II",
                    "good" if cap_ratio >= 0.08 else "danger")

    # ── Charts ────────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        section("PD DISTRIBUTION")
        st.plotly_chart(pd_dist_chart(df["_pd"]), use_container_width=True)

    with ch2:
        # EL by grade if grade column exists
        grade_col = next(
            (c for c in df.columns if c.lower() in ["grade", "sub_grade"]), None
        )
        if grade_col:
            section("EXPECTED LOSS BY GRADE")
            st.plotly_chart(
                el_by_grade_chart(df, "_el_stress" if stress_on else "_el",
                                  grade_col),
                use_container_width=True
            )
        else:
            section("EL vs EAD SCATTER")
            fig = go.Figure(go.Scatter(
                x=df["_ead"], y=df["_el"],
                mode="markers",
                marker=dict(color=df["_pd"], colorscale="RdYlGn_r",
                            size=4, opacity=0.6,
                            colorbar=dict(title="PD", thickness=10,
                                          tickfont=dict(size=9))),
                text=df["_pd"].map(lambda x: f"PD={x:.2%}"),
            ))
            fig.update_layout(
                **DARK_LAYOUT,
                title=dict(text="EL vs EAD (coloured by PD)",
                           font=dict(size=10, color="#6b6b80")),
                xaxis_title="EAD (R)",
                yaxis_title="Expected Loss (R)",
                height=260,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Stress test detail table ───────────────────────────────────────────────
    if stress_on:
        section("STRESS TEST IMPACT — LGD +200bps")
        st.markdown(f"""
        <div class="metric-card warn">
            <div class="metric-label">Stress scenario: LGD shifted +200bps across portfolio</div>
            <div style="font-family: IBM Plex Mono; font-size: 13px; margin-top: 8px; line-height: 2.0;">
                <span style="color:#6b6b80">Baseline EL      :</span> R{total_el:>16,.0f}<br>
                <span style="color:#6b6b80">Stressed EL      :</span> R{total_el_s:>16,.0f}<br>
                <span style="color:#ffb347">EL delta         :</span> +R{el_delta:>15,.0f}
                  ({el_delta/total_el*100:.1f}% increase)<br>
                <span style="color:#6b6b80">Baseline RWA     :</span> R{total_rwa:>16,.0f}<br>
                <span style="color:#6b6b80">Stressed RWA     :</span> R{total_rwa_s:>16,.0f}<br>
                <span style="color:#ffb347">RWA delta        :</span> +R{rwa_delta:>15,.0f}
                  ({rwa_delta/total_rwa*100:.1f}% increase)
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Portfolio table preview ────────────────────────────────────────────────
    section("PORTFOLIO DETAIL (TOP 50)")
    display_cols = {c: c for c in df.columns if not c.startswith("_")}
    display_cols.update({"_pd": "PD", "_lgd": "LGD", "_ead": "EAD",
                         "_el": "EL", "_RWA": "RWA"})
    show_df = df.rename(columns=display_cols).head(50)
    st.dataframe(
        show_df[[c for c in display_cols.values() if c in show_df.columns]],
        use_container_width=True, height=280
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MONITORING
# ══════════════════════════════════════════════════════════════════════════════

def page_monitoring(arts: dict):
    section("MODEL MONITORING")

    metrics = arts.get("metrics")

    if metrics is None:
        st.warning("⚠  model_metrics.pkl not found. Run evaluate.py first.")
        return

    # ── Basel threshold checks ─────────────────────────────────────────────────
    auc  = metrics.get("auc",  0.0)
    gini = metrics.get("gini", 0.0)
    ks   = metrics.get("ks",   0.0)
    psi  = metrics.get("psi",  0.0)

    BASEL_AUC  = 0.70
    BASEL_GINI = 0.40
    BASEL_KS   = 0.30
    BASEL_PSI  = 0.10

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("AUC",
                    f"{auc:.4f}",
                    f"{'✓ PASS' if auc >= BASEL_AUC else '✗ FAIL'} (≥ {BASEL_AUC})",
                    "good" if auc >= BASEL_AUC else "danger")
    with m2:
        metric_card("GINI",
                    f"{gini:.4f}",
                    f"{'✓ PASS' if gini >= BASEL_GINI else '✗ FAIL'} (≥ {BASEL_GINI})",
                    "good" if gini >= BASEL_GINI else "danger")
    with m3:
        metric_card("KS STATISTIC",
                    f"{ks:.4f}",
                    f"{'✓ PASS' if ks >= BASEL_KS else '✗ FAIL'} (≥ {BASEL_KS})",
                    "good" if ks >= BASEL_KS else "danger")
    with m4:
        psi_status = metrics.get("psi_status", "Unknown")
        metric_card("PSI",
                    f"{psi:.4f}",
                    f"{'✓ Stable' if psi < BASEL_PSI else '✗ ' + psi_status}",
                    "good" if psi < BASEL_PSI else "danger")

    # ── Charts row ────────────────────────────────────────────────────────────
    p_good = metrics.get("y_prob_test")
    y_true = metrics.get("y_test")

    ch1, ch2 = st.columns(2)

    with ch1:
        section("ROC CURVE")
        if p_good is not None and y_true is not None:
            from sklearn.metrics import roc_curve as _roc
            fpr, tpr, _ = _roc(y_true, p_good)
            st.plotly_chart(roc_chart(fpr, tpr, auc), use_container_width=True)
        else:
            st.info("ROC data not in model_metrics.pkl. Ensure evaluate.py saves y_prob_test and y_test.")

    with ch2:
        section("KS CHART")
        if p_good is not None and y_true is not None:
            ks_df = pd.DataFrame({"y": y_true, "p_good": p_good})
            ks_df = ks_df.sort_values("p_good", ascending=False).reset_index(drop=True)
            ks_df["cum_good"] = (ks_df["y"] == 1).cumsum() / max((ks_df["y"] == 1).sum(), 1)
            ks_df["cum_bad"]  = (ks_df["y"] == 0).cumsum() / max((ks_df["y"] == 0).sum(), 1)
            st.plotly_chart(ks_chart(ks_df, ks), use_container_width=True)
        else:
            st.info("KS data not in model_metrics.pkl.")

    # ── Threshold sweep table ──────────────────────────────────────────────────
    sweep = metrics.get("threshold_sweep", [])
    if sweep:
        section("THRESHOLD SWEEP — BAD LOAN DETECTION")
        sweep_df = pd.DataFrame(sweep)

        # Colour recall column
        def _colour_recall(val):
            if val >= 0.75:
                return "color: #50fa7b"
            elif val >= 0.60:
                return "color: #ffb347"
            return "color: #ff5555"

        fmt = {
            "recall_bad":    "{:.4f}",
            "precision_bad": "{:.4f}",
            "f1_bad":        "{:.4f}",
            "threshold":     "{:.2f}",
        }
        st.dataframe(
            sweep_df.style.format(fmt).applymap(
                _colour_recall, subset=["recall_bad"]
            ),
            use_container_width=True, height=240,
        )

    # ── Calibration note ──────────────────────────────────────────────────────
    section("MODEL METADATA")
    meta_cols = ["best_threshold", "recall_bad", "precision_bad",
                 "f1_bad", "recall_target", "psi_status"]
    meta_html = "".join([
        f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
        f'border-bottom:1px solid #252530;">'
        f'<span style="font-family:IBM Plex Mono;font-size:11px;color:#6b6b80">{k}</span>'
        f'<span style="font-family:IBM Plex Mono;font-size:11px;color:#e8e8f0">{metrics.get(k, "—")}</span>'
        f'</div>'
        for k in meta_cols if k in metrics
    ])
    st.markdown(f"""
    <div class="metric-card blue" style="max-width: 480px;">
        <div class="metric-label">Stored metrics</div>
        <div style="margin-top: 8px;">{meta_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    header()
    arts = load_artifacts()

    # ── Sidebar navigation ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div style="font-family:IBM Plex Mono; font-size:10px; color:#6b6b80; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:12px;">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "", ["Inference", "Portfolio", "Monitoring"],
            label_visibility="collapsed"
        )
        st.markdown("---")

        # File status
        st.markdown('<div style="font-family:IBM Plex Mono; font-size:10px; color:#6b6b80; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px;">Artifact Status</div>', unsafe_allow_html=True)
        checks = [
            ("model.pkl",        arts.get("model") is not None),
            ("feature_names.pkl",arts.get("features") is not None),
            ("woe_mappings.pkl", arts.get("woe_mappings") is not None),
            ("model_metrics.pkl",arts.get("metrics") is not None),
            ("lgd_model.pkl",    arts.get("lgd_model") is not None),
            ("el_results.pkl",   arts.get("portfolio") is not None),
        ]
        for name, ok in checks:
            icon  = "🟢" if ok else "🔴"
            color = "#50fa7b" if ok else "#ff5555"
            st.markdown(
                f'<div style="font-family:IBM Plex Mono;font-size:10px;'
                f'color:{color};margin:3px 0">{icon} {name}</div>',
                unsafe_allow_html=True
            )

    # ── Route ─────────────────────────────────────────────────────────────────
    if page == "Inference":
        page_inference(arts)
    elif page == "Portfolio":
        page_portfolio(arts)
    elif page == "Monitoring":
        page_monitoring(arts)


if __name__ == "__main__":
    main()
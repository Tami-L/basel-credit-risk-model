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

FIXES v3:
  - Navigation labels now visible (was var(--muted), now var(--text) with accent on selected)
  - prepare_portfolio accepts DataFrame directly, eliminating pickle serialisation round trip
  - Portfolio prep stored in st.session_state so it runs exactly once per session
  - Vectorised Basel II capital formula (eliminates row-by-row .apply bottleneck)
  - @st.cache_data on portfolio prep (computes once per session, not per page switch)
  - WoE transform updated for v2 pipeline mapping format {rules, edges, is_numeric}
  - pandas Styler.applymap -> map compatibility shim
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from scipy.special import expit
from scipy.stats import norm as _norm

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Basel II Credit Risk",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bloomberg-style CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

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

html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background-color: var(--bg); }

.bb-header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 10px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin: -1rem -1rem 1.5rem -1rem;
}
.bb-logo  { font-family: var(--mono); font-size: 15px; font-weight: 600; color: var(--accent); letter-spacing: 0.08em; }
.bb-title { font-family: var(--sans);  font-size: 13px; font-weight: 500; color: var(--muted); letter-spacing: 0.04em; text-transform: uppercase; }
.bb-dot   { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

.metric-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 4px; padding: 16px 18px;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
}
.metric-card.blue::before   { background: var(--accent2); }
.metric-card.teal::before   { background: var(--accent); }
.metric-card.warn::before   { background: var(--warn); }
.metric-card.danger::before { background: var(--danger); }
.metric-card.good::before   { background: var(--good); }

.metric-label { font-family: var(--mono); font-size: 9px; color: var(--muted); letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 6px; }
.metric-value { font-family: var(--mono); font-size: 26px; font-weight: 600; line-height: 1; letter-spacing: -0.02em; }
.metric-sub   { font-family: var(--mono); font-size: 10px; color: var(--muted); margin-top: 6px; }
.metric-delta-pos { color: var(--good);   font-size: 11px; }
.metric-delta-neg { color: var(--danger); font-size: 11px; }

.section-title {
    font-family: var(--mono); font-size: 10px; font-weight: 500;
    color: var(--muted); letter-spacing: 0.18em; text-transform: uppercase;
    border-left: 2px solid var(--accent); padding-left: 10px;
    margin-bottom: 14px; margin-top: 24px;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    background: var(--surface2) !important; border: 1px solid var(--border2) !important;
    color: var(--text) !important; font-family: var(--mono) !important;
    font-size: 13px !important; border-radius: 3px !important;
}
.stSelectbox > div > div > div:hover { border-color: var(--accent) !important; }

.stButton > button {
    background: transparent !important; border: 1px solid var(--accent) !important;
    color: var(--accent) !important; font-family: var(--mono) !important;
    font-size: 12px !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; border-radius: 3px !important;
    padding: 8px 20px !important; transition: all 0.15s !important;
}
.stButton > button:hover { background: var(--accent) !important; color: var(--bg) !important; }

section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }

/* FIX: replaced radio with session_state buttons for persistent navigation */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: var(--text) !important;
    text-align: left !important;
    padding: 6px 10px !important;
    text-transform: none !important;
    letter-spacing: 0.04em !important;
    font-size: 13px !important;
    width: 100% !important;
    border-radius: 3px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    color: var(--accent) !important;
    background: rgba(0,212,170,0.06) !important;
    border: none !important;
}

.stDataFrame { background: var(--surface) !important; }
hr { border-color: var(--border) !important; }

.badge { display: inline-block; font-family: var(--mono); font-size: 10px; font-weight: 600; padding: 3px 10px; border-radius: 2px; letter-spacing: 0.08em; }
.badge-approve { background: rgba(80,250,123,0.12); color: var(--good);   border: 1px solid var(--good); }
.badge-refer   { background: rgba(255,179,71,0.12);  color: var(--warn);   border: 1px solid var(--warn); }
.badge-decline { background: rgba(255,85,85,0.12);   color: var(--danger); border: 1px solid var(--danger); }

.band-pill { font-family: var(--mono); font-size: 11px; font-weight: 600; padding: 4px 14px; border-radius: 2px; display: inline-block; margin-top: 6px; }
.band-exc  { background: rgba(80,250,123,0.15); color: #50fa7b; border: 1px solid #50fa7b44; }
.band-good { background: rgba(74,158,255,0.15); color: #4a9eff; border: 1px solid #4a9eff44; }
.band-fav  { background: rgba(0,212,170,0.15);  color: #00d4aa; border: 1px solid #00d4aa44; }
.band-avg  { background: rgba(255,179,71,0.15); color: #ffb347; border: 1px solid #ffb34744; }
.band-bad  { background: rgba(255,85,85,0.15);  color: #ff5555; border: 1px solid #ff555544; }

.stToggle label { font-family: var(--mono) !important; font-size: 12px !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — cached, error-handled
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load all model artifacts once per process lifetime."""
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
    _load("lgd_model",    SRC_DIR     / "lgd_model.pkl")

    for el_path in [
        OUTPUTS_DIR / "el_results.pkl",
        OUTPUTS_DIR / "el_results.csv",
        SRC_DIR     / "el_results.pkl",
    ]:
        if el_path.exists():
            try:
                arts["portfolio"] = (
                    pd.read_csv(el_path)
                    if el_path.suffix == ".csv"
                    else pickle.load(open(el_path, "rb"))
                )
            except Exception:
                pass
            break

    if "portfolio" not in arts:
        arts["portfolio"] = None

    return arts


# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR     = Path(__file__).resolve().parent
OUTPUTS_DIR = SRC_DIR.parent / "scorecard_outputs"


def align_features(model, feature_names: list) -> list:
    if model is None or feature_names is None:
        return feature_names or []
    n_coef = model.coef_.shape[1]
    n_feat = len(feature_names)
    if n_coef == n_feat:
        return feature_names
    if n_coef < n_feat:
        return feature_names[:n_coef]
    return feature_names + [f"_pad_{i}" for i in range(n_coef - n_feat)]


def get_col(df: pd.DataFrame, *candidates: str, default=None):
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
SA_APPROVAL   = 550
SA_MIN_SCORE  = 300
SA_MAX_SCORE  = 850

SA_BANDS = [
    (767, 850, "Excellent",     "exc"),
    (681, 766, "Good",          "good"),
    (614, 680, "Favourable",    "fav"),
    (583, 613, "Average",       "avg"),
    (527, 582, "Below Average", "bad"),
    (487, 526, "Unfavourable",  "bad"),
    (300, 486, "Poor",          "bad"),
]


def score_from_pd(pd_val: float) -> int:
    factor = SA_PDO / np.log(2)
    offset = SA_BASE_SCORE - factor * np.log(SA_BASE_ODDS)
    odds   = max((1 - pd_val) / max(pd_val, 1e-9), 0.001)
    return int(np.clip(round(offset + factor * np.log(odds)), SA_MIN_SCORE, SA_MAX_SCORE))


def get_band(score: int) -> tuple:
    for lo, hi, label, css in SA_BANDS:
        if lo <= score <= hi:
            return label, css
    return "Poor", "bad"


def get_decision(score: int, cutoff: int = SA_APPROVAL) -> tuple:
    if score >= cutoff + 15:   return "APPROVE", "approve"
    elif score >= cutoff - 10: return "REFER",   "refer"
    else:                      return "DECLINE",  "decline"


def compute_pd(model, woe_vec: np.ndarray) -> float:
    if model is None:
        return 0.15
    logit  = model.intercept_[0] + np.dot(model.coef_[0], woe_vec)
    p_good = expit(logit)
    return float(1 - p_good)


# ══════════════════════════════════════════════════════════════════════════════
# BASEL II RWA — FULLY VECTORISED
# ══════════════════════════════════════════════════════════════════════════════

def basel_correlation(pd_val: float) -> float:
    e50 = np.exp(-50)
    r1  = (1 - np.exp(-50 * pd_val)) / (1 - e50)
    return 0.12 * r1 + 0.24 * (1 - r1)


def regulatory_capital(pd_val: float, lgd: float, maturity: float = 2.5) -> float:
    if pd_val <= 0 or pd_val >= 1:
        return 0.0
    R   = basel_correlation(pd_val)
    sr  = np.sqrt(R)
    sr1 = np.sqrt(1 - R)
    b   = (0.11852 - 0.05478 * np.log(max(pd_val, 1e-6))) ** 2
    ma  = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
    K   = lgd * (_norm.cdf((_norm.ppf(pd_val) + sr * _norm.ppf(0.999)) / sr1) - pd_val) * ma
    return max(float(K), 0.0)


def _basel_correlation_vec(pd_arr: np.ndarray) -> np.ndarray:
    r1 = (1 - np.exp(-50 * pd_arr)) / (1 - np.exp(-50))
    return 0.12 * r1 + 0.24 * (1 - r1)


def _regulatory_capital_vec(
    pd_arr: np.ndarray,
    lgd_arr: np.ndarray,
    maturity: float = 2.5,
) -> np.ndarray:
    pd_arr  = np.clip(pd_arr,  1e-6, 1 - 1e-6)
    lgd_arr = np.clip(lgd_arr, 0.0,  1.0)
    R   = _basel_correlation_vec(pd_arr)
    sr  = np.sqrt(R)
    sr1 = np.sqrt(1 - R)
    b   = (0.11852 - 0.05478 * np.log(pd_arr)) ** 2
    ma  = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
    K   = lgd_arr * (
        _norm.cdf((_norm.ppf(pd_arr) + sr * _norm.ppf(0.999)) / sr1) - pd_arr
    ) * ma
    return np.maximum(K, 0.0)


def compute_rwa(pd_val: float, lgd: float, ead: float) -> dict:
    K = regulatory_capital(pd_val, lgd)
    return {"K": K, "RWA": K * ead * 12.5, "EL": pd_val * lgd * ead}


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO PREP
# FIX: accepts DataFrame directly instead of serialised bytes, eliminating the
#      pickle.dumps / pickle.loads round trip on every page render.
#      Result is stored in st.session_state so it runs exactly once per session.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Preparing portfolio…")
def prepare_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    def _num(series, fallback):
        return pd.to_numeric(series, errors="coerce").fillna(fallback)

    def _resolve(*names, fallback):
        for n in names:
            if n in df.columns:
                return _num(df[n], fallback)
            low   = n.lower()
            match = next((c for c in df.columns if c.lower() == low), None)
            if match:
                return _num(df[match], fallback)
        return pd.Series(fallback, index=df.index)

    df["_pd"]  = _resolve("pd", "PD", "prob_default", "predicted_pd", fallback=0.12).clip(1e-6, 1 - 1e-6)
    df["_lgd"] = _resolve("lgd", "LGD", "loss_given_default",          fallback=0.45).clip(0, 1)
    df["_ead"] = _resolve("ead", "EAD", "funded_amnt", "loan_amnt",    fallback=10_000).clip(0)

    K_vec       = _regulatory_capital_vec(df["_pd"].values, df["_lgd"].values)
    df["_K"]    = K_vec
    df["_RWA"]  = K_vec * df["_ead"].values * 12.5
    df["_el"]   = df["_pd"] * df["_lgd"] * df["_ead"]

    lgd_stress        = (df["_lgd"] + 0.02).clip(0, 1)
    K_stress          = _regulatory_capital_vec(df["_pd"].values, lgd_stress.values)
    df["_lgd_stress"] = lgd_stress
    df["_RWA_stress"] = K_stress * df["_ead"].values * 12.5
    df["_el_stress"]  = df["_pd"] * lgd_stress * df["_ead"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# WOE TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════

def woe_transform_single(
    input_dict: dict,
    woe_mappings: dict,
    feature_names: list,
) -> np.ndarray:
    result = []

    for feat in feature_names:
        val = input_dict.get(feat)
        if feat not in woe_mappings or val is None:
            result.append(0.0)
            continue

        meta = woe_mappings[feat]

        if isinstance(meta, dict) and "rules" in meta:
            rules      = meta["rules"]
            edges      = meta.get("edges")
            is_numeric = meta.get("is_numeric", True)

            if is_numeric and edges is not None and isinstance(val, (int, float)) and not np.isnan(float(val)):
                bin_idx = int(np.digitize(float(val), edges[1:-1]))
                woe_val = rules.get(str(bin_idx), rules.get(bin_idx, 0.0))
            else:
                woe_val = rules.get(str(val), rules.get(val, 0.0))

            result.append(float(woe_val) if woe_val is not None else 0.0)
            continue

        if isinstance(meta, pd.DataFrame) and "bin" in meta.columns and "woe" in meta.columns:
            woe_map = dict(zip(meta["bin"].astype(str), meta["woe"]))
            result.append(float(woe_map.get(str(val), 0.0)))
            continue

        if isinstance(meta, dict):
            result.append(float(meta.get(str(val), meta.get(val, 0.0))))
            continue

        result.append(0.0)

    return np.array(result, dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

DARK_LAYOUT = dict(
    paper_bgcolor="#111116",
    plot_bgcolor="#111116",
    font=dict(family="IBM Plex Mono", color="#e8e8f0", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="#18181f", bordercolor="#252530", borderwidth=1, font=dict(size=10)),
    xaxis=dict(gridcolor="#252530", zerolinecolor="#252530", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#252530", zerolinecolor="#252530", tickfont=dict(size=10)),
)


def gauge_chart(score: int, cutoff: int = SA_APPROVAL) -> go.Figure:
    color = "#50fa7b" if score >= cutoff + 15 else "#ffb347" if score >= cutoff - 10 else "#ff5555"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"font": {"family": "IBM Plex Mono", "size": 40, "color": color}},
        gauge={
            "axis": {"range": [SA_MIN_SCORE, SA_MAX_SCORE], "tickwidth": 1,
                     "tickcolor": "#404055", "tickfont": {"size": 9, "color": "#6b6b80"}, "nticks": 10},
            "bar":  {"color": color, "thickness": 0.22},
            "bgcolor": "#18181f", "borderwidth": 0,
            "steps": [
                {"range": [300, 487], "color": "#1a0a0a"},
                {"range": [487, 583], "color": "#1a1200"},
                {"range": [583, 681], "color": "#0a1a12"},
                {"range": [681, 767], "color": "#0a1020"},
                {"range": [767, 850], "color": "#0a1a14"},
            ],
            "threshold": {"line": {"color": "#4a9eff", "width": 2}, "thickness": 0.8, "value": cutoff},
        },
    ))
    fig.update_layout(paper_bgcolor="#111116", font=dict(family="IBM Plex Mono", color="#e8e8f0"),
                      height=240, margin=dict(l=20, r=20, t=10, b=10))
    return fig


def roc_chart(fpr, tpr, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             line=dict(color="#00d4aa", width=2), name=f"AUC = {auc:.4f}",
                             fill="tozeroy", fillcolor="rgba(0,212,170,0.06)"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="#404055", width=1, dash="dot"),
                             name="Random", showlegend=False))
    fig.update_layout(**DARK_LAYOUT,
                      title=dict(text="ROC CURVE", font=dict(size=10, color="#6b6b80")),
                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=300)
    return fig


def ks_chart(ks_df: pd.DataFrame, ks_stat: float) -> go.Figure:
    fig = go.Figure()
    x   = ks_df.index / len(ks_df)
    fig.add_trace(go.Scatter(x=x, y=ks_df["cum_good"], mode="lines",
                             line=dict(color="#4a9eff", width=2), name="Cumulative Good"))
    fig.add_trace(go.Scatter(x=x, y=ks_df["cum_bad"],  mode="lines",
                             line=dict(color="#ff5555", width=2), name="Cumulative Bad"))
    ks_idx = (ks_df["cum_good"] - ks_df["cum_bad"]).abs().idxmax()
    fig.add_vline(x=ks_idx / len(ks_df), line_color="#ffb347", line_width=1, line_dash="dot",
                  annotation_text=f"KS={ks_stat:.4f}",
                  annotation_font=dict(color="#ffb347", size=10))
    fig.update_layout(**DARK_LAYOUT,
                      title=dict(text="KS CHART", font=dict(size=10, color="#6b6b80")),
                      xaxis_title="Population Fraction", yaxis_title="Cumulative Rate", height=300)
    return fig


def pd_dist_chart(pd_series: pd.Series) -> go.Figure:
    fig = go.Figure(go.Histogram(x=pd_series, nbinsx=40, marker_color="#4a9eff", opacity=0.7,
                                 marker_line=dict(color="#111116", width=0.5)))
    fig.update_layout(**DARK_LAYOUT,
                      title=dict(text="PORTFOLIO PD DISTRIBUTION", font=dict(size=10, color="#6b6b80")),
                      xaxis_title="Probability of Default", yaxis_title="Count", height=260)
    return fig


def el_by_grade_chart(df: pd.DataFrame, el_col: str, grade_col: str) -> go.Figure:
    grp = df.groupby(grade_col)[el_col].sum().reset_index().sort_values(grade_col)
    fig = go.Figure(go.Bar(x=grp[grade_col], y=grp[el_col],
                           marker_color="#00d4aa", opacity=0.85,
                           marker_line=dict(color="#111116", width=0.5)))
    fig.update_layout(**DARK_LAYOUT,
                      title=dict(text="EXPECTED LOSS BY GRADE", font=dict(size=10, color="#6b6b80")),
                      xaxis_title="Grade", yaxis_title="Expected Loss", height=260)
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


def metric_card(label, value, sub="", color="teal", delta=None, delta_pos=True):
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
# PAGE 1 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def page_inference(arts: dict):
    section("APPLICANT ASSESSMENT")

    model    = arts.get("model")
    features = arts.get("features", [])
    woe_maps = arts.get("woe_mappings", {})
    lgd_mdl  = arts.get("lgd_model") or {}

    if model is None:
        st.error("⚠  model.pkl not found. Ensure scorecard_outputs/model.pkl exists.")
        return

    features = align_features(model, features)
    cutoff   = st.sidebar.slider("Approval cut-off score", 480, 700, SA_APPROVAL, 5)

    with st.form("applicant_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Loan Details**")
            loan_amnt   = st.number_input("Loan Amount (R)",    value=25000.0,  step=1000.0)
            int_rate    = st.number_input("Interest Rate (%)",  value=14.5,     step=0.5)
            term        = st.selectbox("Term", ["36 months", "60 months"])
            installment = st.number_input("Installment (R/mo)", value=850.0,    step=50.0)
            sub_grade   = st.selectbox("Sub Grade",
                          ["A1","A2","A3","A4","A5","B1","B2","B3","B4","B5",
                           "C1","C2","C3","C4","C5","D1","D2","D3","D4","D5",
                           "E1","E2","E3","E4","E5","F1","F2","F3","F4","F5",
                           "G1","G2","G3","G4","G5"])

        with c2:
            st.markdown("**Borrower Profile**")
            annual_inc          = st.number_input("Annual Income (R)",    value=480000.0, step=10000.0)
            dti                 = st.number_input("Debt-to-Income Ratio", value=18.5,     step=0.5)
            emp_length          = st.selectbox("Employment Length",
                                  ["< 1 year","1 year","2 years","3 years","4 years",
                                   "5 years","6 years","7 years","8 years","9 years","10+ years"])
            home_ownership      = st.selectbox("Home Ownership", ["RENT","OWN","MORTGAGE","OTHER"])
            verification_status = st.selectbox("Verification Status",
                                  ["Verified","Source Verified","Not Verified"])
            purpose             = st.selectbox("Purpose",
                                  ["debt_consolidation","credit_card","home_improvement","other",
                                   "major_purchase","medical","small_business","car","vacation",
                                   "moving","house","wedding","renewable_energy","educational"])

        with c3:
            st.markdown("**Credit Bureau**")
            fico_range_high = st.number_input("FICO High",          value=700.0,   step=1.0)
            fico_range_low  = st.number_input("FICO Low",           value=695.0,   step=1.0)
            revol_bal       = st.number_input("Revolving Balance",   value=12000.0, step=500.0)
            revol_util      = st.number_input("Revolving Util (%)",  value=42.0,    step=1.0)
            open_acc        = st.number_input("Open Accounts",       value=8,       step=1)
            delinq_2yrs     = st.number_input("Delinquencies (2yr)", value=0,       step=1)
            inq_last_6mths  = st.number_input("Inquiries (6mo)",     value=1,       step=1)
            pub_rec         = st.number_input("Public Records",       value=0,       step=1)
            total_acc       = st.number_input("Total Accounts",       value=18,      step=1)

        submitted = st.form_submit_button("▶  RUN ASSESSMENT", use_container_width=True)

    if not submitted:
        st.markdown('<p style="color:#6b6b80;font-size:12px;font-family:IBM Plex Mono;">↑  Complete the form and click RUN ASSESSMENT</p>',
                    unsafe_allow_html=True)
        return

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

    woe_vec           = woe_transform_single(raw_vals, woe_maps, features)
    pd_val            = compute_pd(model, woe_vec)
    score             = score_from_pd(pd_val)
    band, band_css    = get_band(score)
    decision, dec_css = get_decision(score, cutoff)

    grade   = sub_grade[0] if sub_grade else "C"
    lgd_map = lgd_mdl.get("lgd_by_grade", {}) if lgd_mdl else {}
    lgd_val = float(lgd_map.get(grade, 0.45)) if hasattr(lgd_map, "get") else 0.45
    rwa_res = compute_rwa(pd_val, lgd_val, loan_amnt)

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
        </div>""", unsafe_allow_html=True)

    with r2:
        metric_card("Credit Score", str(score), f"cutoff {cutoff}",
                    "good" if score >= cutoff + 15 else "warn" if score >= cutoff - 10 else "danger")
        st.markdown("<br>", unsafe_allow_html=True)
        metric_card("Probability of Default", f"{pd_val*100:.2f}%", "P(Bad | application)",
                    "danger" if pd_val > 0.20 else "warn" if pd_val > 0.10 else "teal")

    with r3:
        metric_card("LGD", f"{lgd_val*100:.1f}%", f"Grade {grade}", "blue")
        st.markdown("<br>", unsafe_allow_html=True)
        metric_card("Expected Loss", f"R{rwa_res['EL']:,.0f}", "PD x LGD x EAD",
                    "warn" if rwa_res["EL"] > loan_amnt * 0.05 else "teal")

    with r4:
        metric_card("Reg. Capital (K)", f"{rwa_res['K']*100:.2f}%", "Basel II IRB formula", "blue")
        st.markdown("<br>", unsafe_allow_html=True)
        metric_card("RWA", f"R{rwa_res['RWA']:,.0f}", "K x EAD x 12.5", "warn")

    with r5:
        corr = basel_correlation(pd_val)
        metric_card("Basel Correlation (R)", f"{corr*100:.1f}%", "Asset correlation", "blue")
        st.markdown("<br>", unsafe_allow_html=True)
        metric_card("EAD", f"R{loan_amnt:,.0f}", "Funded amount", "teal")

    section("DECISION RATIONALE")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Output</div>
            <div style="font-family:IBM Plex Mono;font-size:12px;color:#e8e8f0;line-height:1.8;">
                <span style="color:#6b6b80">Score      :</span> {score}<br>
                <span style="color:#6b6b80">PD         :</span> {pd_val*100:.3f}%<br>
                <span style="color:#6b6b80">Score band :</span> {band}<br>
                <span style="color:#6b6b80">Cutoff     :</span> {cutoff}<br>
                <span style="color:#6b6b80">Decision   :</span> {decision}<br>
                <span style="color:#6b6b80">PDO        :</span> {SA_PDO}<br>
                <span style="color:#6b6b80">Base odds  :</span> {SA_BASE_ODDS}:1
            </div>
        </div>""", unsafe_allow_html=True)

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
            f'{w:.3f} x {c:.4f}</span></div>'
            for f, w, c in top_feats
        ])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Top Feature Contributions (WOE x Coef)</div>
            <div style="font-family:IBM Plex Mono;margin-top:8px;">{rows_html}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════

def page_portfolio(arts: dict):
    section("PORTFOLIO ANALYTICS")

    portfolio = arts.get("portfolio")
    if portfolio is None:
        st.warning("⚠  Portfolio data not found. Expected: scorecard_outputs/el_results.pkl")
        st.markdown("Place your `el_results.pkl` in `scorecard_outputs/` and restart.")
        return

    # FIX: store prepared portfolio in session_state so prepare_portfolio
    # runs exactly once per session regardless of how many times the user
    # navigates to this page. Eliminates the pickle bytes round trip entirely.
    if "prepared_portfolio" not in st.session_state:
        st.session_state["prepared_portfolio"] = prepare_portfolio(portfolio)
    df = st.session_state["prepared_portfolio"]

    stress_on = st.sidebar.toggle("🔴 Stress Test (LGD +200bps)", value=False)

    el_col  = "_el_stress"  if stress_on else "_el"
    rwa_col = "_RWA_stress" if stress_on else "_RWA"

    total_ead    = df["_ead"].sum()
    wa_pd        = (df["_pd"]  * df["_ead"]).sum() / max(total_ead, 1)
    wa_lgd       = (df["_lgd"] * df["_ead"]).sum() / max(total_ead, 1)
    total_el     = df[el_col].sum()
    total_rwa    = df[rwa_col].sum()
    baseline_el  = df["_el"].sum()
    baseline_rwa = df["_RWA"].sum()
    el_delta     = total_el  - baseline_el
    rwa_delta    = total_rwa - baseline_rwa

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        metric_card("Total Exposure", f"R{total_ead/1e6:.1f}M", f"{len(df):,} loans", "blue")
    with k2:
        metric_card("WA PD",  f"{wa_pd*100:.2f}%",  "Exposure-weighted", "danger")
    with k3:
        metric_card("WA LGD", f"{wa_lgd*100:.1f}%", "Exposure-weighted", "warn")
    with k4:
        metric_card("Total EL", f"R{total_el/1e6:.2f}M",
                    f"+R{el_delta/1e6:.2f}M stress" if stress_on else "Baseline",
                    "danger" if stress_on else "warn",
                    delta=f"+R{el_delta/1e6:.2f}M" if stress_on else None, delta_pos=False)
    with k5:
        metric_card("Total RWA", f"R{total_rwa/1e6:.1f}M",
                    f"+R{rwa_delta/1e6:.1f}M stress" if stress_on else "Baseline",
                    "warn",
                    delta=f"+R{rwa_delta/1e6:.1f}M" if stress_on else None, delta_pos=False)
    with k6:
        cap_ratio = (total_ead * 0.08) / max(total_rwa, 1)
        metric_card("Capital Adequacy", f"{cap_ratio*100:.1f}%", "Min. 8% Basel II",
                    "good" if cap_ratio >= 0.08 else "danger")

    ch1, ch2 = st.columns(2)

    with ch1:
        section("PD DISTRIBUTION")
        st.plotly_chart(pd_dist_chart(df["_pd"]), use_container_width=True)

    with ch2:
        grade_col = next((c for c in df.columns if c.lower() in ["grade", "sub_grade"]), None)
        if grade_col:
            section("EXPECTED LOSS BY GRADE")
            st.plotly_chart(el_by_grade_chart(df, el_col, grade_col), use_container_width=True)
        else:
            section("EL vs EAD SCATTER")
            fig = go.Figure(go.Scatter(
                x=df["_ead"], y=df[el_col], mode="markers",
                marker=dict(color=df["_pd"], colorscale="RdYlGn_r", size=4, opacity=0.6,
                            colorbar=dict(title="PD", thickness=10, tickfont=dict(size=9))),
            ))
            fig.update_layout(**DARK_LAYOUT,
                              title=dict(text="EL vs EAD (coloured by PD)", font=dict(size=10, color="#6b6b80")),
                              xaxis_title="EAD (R)", yaxis_title="Expected Loss (R)", height=260)
            st.plotly_chart(fig, use_container_width=True)

    if stress_on:
        section("STRESS TEST IMPACT — LGD +200bps")
        st.markdown(f"""
        <div class="metric-card warn">
            <div class="metric-label">Stress scenario: LGD shifted +200bps across portfolio</div>
            <div style="font-family:IBM Plex Mono;font-size:13px;margin-top:8px;line-height:2.0;">
                <span style="color:#6b6b80">Baseline EL  :</span> R{baseline_el:>16,.0f}<br>
                <span style="color:#6b6b80">Stressed EL  :</span> R{total_el:>16,.0f}<br>
                <span style="color:#ffb347">EL delta     :</span> +R{el_delta:>15,.0f} ({el_delta/max(baseline_el,1)*100:.1f}%)<br>
                <span style="color:#6b6b80">Baseline RWA :</span> R{baseline_rwa:>16,.0f}<br>
                <span style="color:#6b6b80">Stressed RWA :</span> R{total_rwa:>16,.0f}<br>
                <span style="color:#ffb347">RWA delta    :</span> +R{rwa_delta:>15,.0f} ({rwa_delta/max(baseline_rwa,1)*100:.1f}%)
            </div>
        </div>""", unsafe_allow_html=True)

    section("PORTFOLIO DETAIL (TOP 50)")
    rename   = {"_pd": "PD", "_lgd": "LGD", "_ead": "EAD", "_el": "EL", "_RWA": "RWA"}
    raw_cols = [c for c in df.columns if not c.startswith("_")]
    show_df  = df[raw_cols + ["_pd","_lgd","_ead","_el","_RWA"]].rename(columns=rename).head(50)
    st.dataframe(show_df, use_container_width=True, height=280)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MONITORING
# ══════════════════════════════════════════════════════════════════════════════

def page_monitoring(arts: dict):
    section("MODEL MONITORING")

    metrics = arts.get("metrics")
    if metrics is None:
        st.warning("⚠  model_metrics.pkl not found. Run evaluate.py first.")
        return

    auc  = metrics.get("auc",  0.0)
    gini = metrics.get("gini", 0.0)
    ks   = metrics.get("ks",   0.0)
    psi  = metrics.get("psi",  0.0)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, thresh in [
        (m1, "AUC",          auc,  0.70),
        (m2, "GINI",         gini, 0.40),
        (m3, "KS STATISTIC", ks,   0.30),
        (m4, "PSI",          psi,  0.10),
    ]:
        with col:
            passed = val >= thresh if label != "PSI" else val < thresh
            metric_card(label, f"{val:.4f}",
                        f"{'✓ PASS' if passed else '✗ FAIL'} ({'≥' if label != 'PSI' else '<'} {thresh})",
                        "good" if passed else "danger")

    sweep = metrics.get("threshold_sweep", [])
    if sweep:
        section("THRESHOLD SWEEP — BAD LOAN DETECTION")
        sweep_df = pd.DataFrame(sweep)

        def _colour_recall(val):
            if val >= 0.75:   return "color: #50fa7b"
            elif val >= 0.60: return "color: #ffb347"
            return "color: #ff5555"

        fmt = {"recall_bad": "{:.4f}", "precision_bad": "{:.4f}",
               "f1_bad": "{:.4f}", "threshold": "{:.2f}"}

        styled = sweep_df.style.format(fmt)
        if "recall_bad" in sweep_df.columns:
            try:
                styled = styled.map(_colour_recall, subset=["recall_bad"])
            except AttributeError:
                styled = styled.applymap(_colour_recall, subset=["recall_bad"])

        st.dataframe(styled, use_container_width=True, height=240)

    section("MODEL METADATA")
    meta_keys = ["best_threshold", "recall_bad", "precision_bad",
                 "f1_bad", "recall_target", "psi_status"]
    meta_html = "".join([
        f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
        f'border-bottom:1px solid #252530;">'
        f'<span style="font-family:IBM Plex Mono;font-size:11px;color:#6b6b80">{k}</span>'
        f'<span style="font-family:IBM Plex Mono;font-size:11px;color:#e8e8f0">{metrics.get(k,"—")}</span>'
        f'</div>'
        for k in meta_keys if k in metrics
    ])
    st.markdown(f"""
    <div class="metric-card blue" style="max-width:480px;">
        <div class="metric-label">Stored metrics</div>
        <div style="margin-top:8px;">{meta_html}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    header()
    arts = load_artifacts()

    with st.sidebar:
        st.markdown('<div style="font-family:IBM Plex Mono;font-size:10px;color:#6b6b80;'
                    'letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">Navigation</div>',
                    unsafe_allow_html=True)
        if "page" not in st.session_state:
            st.session_state["page"] = "Inference"

        for p in ["Inference", "Portfolio", "Monitoring"]:
            active = st.session_state["page"] == p
            label  = f"{'> ' if active else '  '}{p}"
            if st.sidebar.button(label, key=f"nav_{p}", use_container_width=True):
                st.session_state["page"] = p

        page = st.session_state["page"]
        st.markdown("---")

        st.markdown('<div style="font-family:IBM Plex Mono;font-size:10px;color:#6b6b80;'
                    'letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;">Artifact Status</div>',
                    unsafe_allow_html=True)
        checks = [
            ("model.pkl",         arts.get("model")        is not None),
            ("feature_names.pkl", arts.get("features")     is not None),
            ("woe_mappings.pkl",  arts.get("woe_mappings") is not None),
            ("model_metrics.pkl", arts.get("metrics")      is not None),
            ("lgd_model.pkl",     arts.get("lgd_model")    is not None),
            ("el_results.pkl",    arts.get("portfolio")    is not None),
        ]
        for name, ok in checks:
            color = "#50fa7b" if ok else "#ff5555"
            icon  = "🟢" if ok else "🔴"
            st.markdown(f'<div style="font-family:IBM Plex Mono;font-size:10px;color:{color};margin:3px 0">'
                        f'{icon} {name}</div>', unsafe_allow_html=True)

    if   page == "Inference":  page_inference(arts)
    elif page == "Portfolio":  page_portfolio(arts)
    elif page == "Monitoring": page_monitoring(arts)


if __name__ == "__main__":
    main()
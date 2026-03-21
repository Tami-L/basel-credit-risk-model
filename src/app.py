import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm as _norm

sys.path.append(os.path.dirname(__file__))

from ETL import transform_inference, load_woe_artifacts
from Scorecard import build_scorecard, calculate_credit_score, get_risk_band, get_approval_decision

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Basel Credit Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    letter-spacing: -0.02em;
}

section[data-testid="stSidebar"] {
    background: #0f1923;
    border-right: 1px solid #1e2d3d;
}
section[data-testid="stSidebar"] * {
    color: #c9d6e3 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem;
    padding: 6px 0;
}

div[data-testid="metric-container"] {
    background: #0f1923;
    border: 1px solid #1e2d3d;
    border-radius: 10px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label {
    color: #7a9bb5 !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e8f0f7 !important;
    font-size: 1.6rem;
    font-weight: 500;
}

.section-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a7fa5;
    margin-bottom: 6px;
    margin-top: 20px;
}

hr { border-color: #1e2d3d; }

div[data-testid="stFormSubmitButton"] button {
    background: #1a6baa !important;
    color: white !important;
    border: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: #1580cc !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Artefact loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pd_model():
    # New pipeline: scorecard_outputs/model.pkl  (trained by train.py on WoE features)
    path = os.path.join(os.path.dirname(__file__), "..", "scorecard_outputs", "model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_features():
    # New pipeline: scorecard_outputs/feature_names.pkl  (saved by train.py)
    path = os.path.join(os.path.dirname(__file__), "..", "scorecard_outputs", "feature_names.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_el_results():
    path = os.path.join(os.path.dirname(__file__), "el_results.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_lgd_model():
    path = os.path.join(os.path.dirname(__file__), "lgd_model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_threshold():
    # Try to read best_threshold from model_metrics.pkl (set by evaluate.py)
    for base in [
        os.path.join(os.path.dirname(__file__), "..", "scorecard_outputs"),
        os.path.dirname(__file__),
    ]:
        path = os.path.join(base, "model_metrics.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                m = pickle.load(f)
            return m.get("best_threshold", 0.5)
    # Legacy fallback
    path = os.path.join(os.path.dirname(__file__), "pd_model_threshold.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return 0.5


@st.cache_resource
def load_model_metrics():
    # Try new pipeline location first, fall back to src/
    for base in [
        os.path.join(os.path.dirname(__file__), "..", "scorecard_outputs"),
        os.path.dirname(__file__),
    ]:
        path = os.path.join(base, "model_metrics.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


@st.cache_resource
def load_woe_mappings():
    # Loads woe_mappings.pkl and selected_features.pkl saved by woe_etl.py
    root = os.path.join(os.path.dirname(__file__), "..", "scorecard_outputs")
    woe_path  = os.path.join(root, "woe_mappings.pkl")
    feat_path = os.path.join(root, "selected_features.pkl")
    if not os.path.exists(woe_path) or not os.path.exists(feat_path):
        return None, None
    with open(woe_path, "rb") as f:
        woe_tables = pickle.load(f)
    with open(feat_path, "rb") as f:
        selected_features = pickle.load(f)
    return woe_tables, selected_features


# ---------------------------------------------------------------------------
# Capital requirement helpers (Basel II IRB retail — §328)
# ---------------------------------------------------------------------------
def _basel_correlation(pd_arr):
    e50 = np.exp(-50)
    return (
        0.12 * (1 - np.exp(-50 * pd_arr)) / (1 - e50)
        + 0.24 * (1 - (1 - np.exp(-50 * pd_arr)) / (1 - e50))
    )


def _capital_requirement(pd_arr, lgd_arr, ead_arr, maturity_adj=1.06):
    pd_arr  = np.clip(pd_arr,  1e-6, 1 - 1e-6)
    lgd_arr = np.clip(lgd_arr, 0, 1)
    R       = _basel_correlation(pd_arr)
    G_pd    = _norm.ppf(pd_arr)
    G_999   = _norm.ppf(0.999)
    N_arg   = G_pd / np.sqrt(1 - R) + np.sqrt(R / (1 - R)) * G_999
    K       = lgd_arr * (_norm.cdf(N_arg) - pd_arr) * maturity_adj
    rwa     = K * ead_arr * 12.5
    return K, rwa


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏦 Basel Credit Risk")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["🔍 Applicant Assessment", "📊 Portfolio Dashboard", "🧪 Model Performance"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#4a7fa5; line-height:1.6'>"
        "Basel II compliant credit risk framework.<br>"
        "EL = PD × LGD × EAD<br><br>"
        "LGD: 45% regulatory constant<br>"
        "EAD: funded_amnt − total_pymnt<br>"
        "Score scale: 300–850 (ZA / TransUnion)<br>"
        f"PD threshold: {load_threshold():.2f}"
        "</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 1 — APPLICANT ASSESSMENT
# ===========================================================================
if page == "🔍 Applicant Assessment":

    st.title("Applicant Credit Assessment")
    st.markdown(
        "<div style='color:#7a9bb5; margin-bottom:24px'>"
        "Enter loan applicant details to calculate probability of default, "
        "credit score, and expected loss exposure."
        "</div>",
        unsafe_allow_html=True,
    )

    model                    = load_pd_model()
    feature_names            = load_features()
    lgd_model                = load_lgd_model()
    threshold                = load_threshold()
    woe_tables, woe_features = load_woe_mappings()

    if model is None:
        st.error("model.pkl not found in scorecard_outputs/. Run train.py first.")
        st.stop()
    if feature_names is None:
        st.error("feature_names.pkl not found in scorecard_outputs/. Run train.py first.")
        st.stop()
    if woe_tables is None:
        st.error("woe_mappings.pkl not found in scorecard_outputs/. Run woe_etl.py first.")
        st.stop()

    # active_features: WoE feature list from ETL (must match model.pkl exactly)
    active_features = woe_features if woe_features is not None else feature_names

    scorecard, base_points = build_scorecard(
        model,
        feature_names=active_features,
        pdo=20,           # SA market: tighter PDO for high-default environment
        base_score=620,   # anchors 8:1 odds borrower at mid-Favourable tier
        base_odds=8,      # ~11% default rate (NCR SA unsecured benchmark)
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='section-label'>Loan Details</div>", unsafe_allow_html=True)
            loan_amnt      = st.number_input("Loan Amount ($)", 1000, 50000, 15000)
            term           = st.selectbox("Term", ["36 months", "60 months"])
            int_rate       = st.slider("Interest Rate (%)", 5.0, 30.0, 12.5, 0.1)
            installment    = st.number_input("Monthly Installment ($)", 100, 2000, 350)
            grade          = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
            sub_grade      = st.selectbox("Sub Grade", [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)])

            st.markdown("<div class='section-label'>Employment</div>", unsafe_allow_html=True)
            emp_length     = st.selectbox("Employment Length",
                                ["< 1 year","1 year","2 years","3 years","4 years",
                                 "5 years","6 years","7 years","8 years","9 years","10+ years"])
            home_ownership = st.selectbox("Home Ownership", ["RENT","MORTGAGE","OWN","OTHER"])
            annual_inc     = st.number_input("Annual Income ($)", 20000, 500000, 60000)

        with col2:
            st.markdown("<div class='section-label'>Credit History</div>", unsafe_allow_html=True)
            dti            = st.slider("Debt-to-Income Ratio", 0.0, 50.0, 15.0, 0.1)
            delinq_2yrs    = st.number_input("Delinquencies (2 yrs)", 0, 10, 0)
            inq_last_6mths = st.number_input("Credit Inquiries (6 mths)", 0, 10, 1)
            open_acc       = st.number_input("Open Credit Lines", 1, 30, 8)
            pub_rec        = st.number_input("Public Records", 0, 5, 0)
            revol_bal      = st.number_input("Revolving Balance ($)", 0, 100000, 5000)
            revol_util     = st.slider("Revolving Utilisation (%)", 0.0, 100.0, 45.5, 0.1)
            total_acc      = st.number_input("Total Credit Lines", 5, 50, 15)

            st.markdown("<div class='section-label'>EAD Inputs</div>", unsafe_allow_html=True)
            total_pymnt    = st.number_input("Payments Made to Date ($)", 0, 50000, 0)

        with col3:
            st.markdown("<div class='section-label'>Other Details</div>", unsafe_allow_html=True)
            addr_state = st.selectbox("State",
                ["CA","NY","TX","FL","IL","PA","OH","GA","NC","MI","NJ","VA","WA",
                 "AZ","MA","TN","IN","MO","MD","WI","CO","MN","SC","AL","LA","KY",
                 "OR","OK","CT","IA","MS","AR","KS","UT","NV","NM","WV","NE","ID",
                 "NH","ME","MT","RI","VT","AK","WY","DE","DC","SD","ND"])
            verification_status = st.selectbox("Income Verification",
                ["Verified","Source Verified","Not Verified"])
            purpose = st.selectbox("Loan Purpose",
                ["debt_consolidation","credit_card","home_improvement","major_purchase",
                 "medical","car","educational","vacation","moving","house",
                 "renewable_energy","wedding","small_business"])
            initial_list_status = st.selectbox("List Status", ["f","w"])

            st.markdown("<div class='section-label'>Dates</div>", unsafe_allow_html=True)
            issue_d          = st.date_input("Issue Date", datetime(2017, 12, 1))
            earliest_cr_line = st.date_input("Earliest Credit Line", datetime(2005, 1, 1))

        submitted = st.form_submit_button(
            "Calculate Credit Score & Risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        with st.spinner("Processing application..."):
            try:
                funded_amnt = loan_amnt
                ead = max(funded_amnt - total_pymnt, 0)
                lgd = lgd_model["lgd_by_grade"].get(grade, lgd_model["lgd_default"]) if lgd_model else 0.45

                input_df = pd.DataFrame([{
                    "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
                    "installment": installment, "grade": grade, "sub_grade": sub_grade,
                    "emp_length": emp_length, "home_ownership": home_ownership,
                    "annual_inc": annual_inc, "dti": dti, "delinq_2yrs": delinq_2yrs,
                    "inq_last_6mths": inq_last_6mths, "open_acc": open_acc,
                    "pub_rec": pub_rec, "revol_bal": revol_bal, "revol_util": revol_util,
                    "total_acc": total_acc, "addr_state": addr_state,
                    "verification_status": verification_status, "purpose": purpose,
                    "initial_list_status": initial_list_status,
                    "issue_d": issue_d.strftime("%b-%y"),
                    "earliest_cr_line": earliest_cr_line.strftime("%b-%y"),
                    "funded_amnt": funded_amnt, "funded_amnt_inv": funded_amnt,
                    "mths_since_last_delinq": np.nan, "mths_since_last_record": np.nan,
                    "collection_recovery_fee": 0, "tot_coll_amt": 0,
                    "tot_cur_bal": revol_bal,
                    "total_rev_hi_lim": revol_bal / (revol_util / 100) if revol_util > 0 else revol_bal,
                }])

                # transform_inference applies the saved WoE bin mappings
                # to the raw input — no manual feature engineering needed
                X_final = transform_inference(
                    input_df,
                    woe_tables=woe_tables,
                    selected_features=active_features,
                )

                # Ensure all active features are present (fill 0 = neutral WoE)
                for col in active_features:
                    if col not in X_final.columns:
                        X_final[col] = 0.0
                X_final = X_final[active_features]

                X_arr      = X_final.values.astype(np.float64)
                proba_arr  = model.predict_proba(X_arr)[0]
                p_good     = proba_arr[1]
                pd_val     = 1 - p_good
                el         = pd_val * lgd * ead
                # Score-based decision (TransUnion SA bands — threshold=614 Favourable)
                credit_score = calculate_credit_score(
                    X_final, scorecard, base_points, min_score=300, max_score=850
                )[0]
                risk_band = get_risk_band(credit_score)
                decision  = get_approval_decision(credit_score)

                K_loan, rwa_loan = _capital_requirement(
                    np.array([pd_val]), np.array([lgd]), np.array([ead])
                )
                cap_req_loan = float(K_loan[0] * ead)

                st.markdown("---")
                st.markdown("### Assessment Results")

                m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
                m1.metric("Credit Score",    f"{credit_score}")
                m2.metric("Risk Band",       risk_band.split(" ")[0])
                m3.metric("PD",              f"{pd_val:.2%}")
                m4.metric("P(Good)",         f"{p_good:.2%}")
                m5.metric("Expected Loss",   f"${el:,.0f}")
                m6.metric("Score Cut-off",   "614 (Favourable)")
                m7.metric("Decision",
                          "✅ Approve" if decision == "Approve"
                          else "🔄 Refer" if decision == "Refer"
                          else "❌ Decline")

                score_color = (
                    "#22c55e" if credit_score >= 681 else
                    "#84cc16" if credit_score >= 614 else
                    "#eab308" if credit_score >= 583 else
                    "#f97316" if credit_score >= 487 else "#ef4444"
                )
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=credit_score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Credit Score (TransUnion SA)", "font": {"size": 16}},
                    number={"font": {"color": score_color, "size": 48}},
                    gauge={
                        "axis": {"range": [300, 850], "tickcolor": "#4a7fa5"},
                        "bar":  {"color": score_color, "thickness": 0.25},
                        "bgcolor": "#0f1923",
                        "bordercolor": "#1e2d3d",
                        "steps": [
                            {"range": [300, 487], "color": "#2d1515"},   # Poor
                            {"range": [487, 527], "color": "#2d1e0f"},   # Unfavourable
                            {"range": [527, 583], "color": "#2d2a0f"},   # Below Average
                            {"range": [583, 614], "color": "#1e2d1e"},   # Average
                            {"range": [614, 681], "color": "#152d15"},   # Favourable
                            {"range": [681, 767], "color": "#0f2a0f"},   # Good
                            {"range": [767, 850], "color": "#0a1f0a"},   # Excellent
                        ],
                    },
                ))
                fig_gauge.update_layout(
                    height=280, paper_bgcolor="#080e14",
                    font_color="#c9d6e3",
                    margin=dict(t=40, b=10, l=20, r=20),
                )

                gc1, gc2 = st.columns([1, 1])
                with gc1:
                    st.plotly_chart(fig_gauge, use_container_width=True)
                with gc2:
                    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                    st.markdown(f"""
| Component | Value |
|-----------|-------|
| EAD (Exposure at Default) | ${ead:,.0f} |
| LGD (Loss Given Default) | {lgd:.0%} |
| PD (Probability of Default) | {pd_val:.4%} |
| **EL = PD × LGD × EAD** | **${el:,.2f}** |
| Capital Requirement (K × EAD) | ${cap_req_loan:,.2f} |
| RWA | ${float(rwa_loan[0]):,.2f} |
""")
                    # SA TransUnion band messaging
                    if credit_score >= 767:
                        st.success(f"**Excellent** (767–850) — Very low risk. Best available rates.")
                    elif credit_score >= 681:
                        st.success(f"**Good** (681–766) — Low risk. Standard prime rates.")
                    elif credit_score >= 614:
                        st.info(f"**Favourable** (614–680) — Moderate risk. NCA affordability check required.")
                    elif credit_score >= 583:
                        st.warning(f"**Average** (583–613) — Elevated risk. Higher rates may apply.")
                    elif credit_score >= 527:
                        st.warning(f"**Below Average** (527–582) — High risk. Secured options recommended.")
                    elif credit_score >= 487:
                        st.error(f"**Unfavourable** (487–526) — Very high risk.")
                    else:
                        st.error(f"**Poor** (300–486) — Decline recommended.")

                with st.expander("View Score Factors"):
                    contribs = [
                        {"Feature": f, "Points Contribution": round(X_final[f].values[0] * r, 2)}
                        for f, r in zip(scorecard["feature"], scorecard["raw_points"])
                        if X_final[f].values[0] * r != 0
                    ]
                    if contribs:
                        cdf = pd.DataFrame(contribs).sort_values("Points Contribution", ascending=False)
                        fig_bar = px.bar(
                            cdf, x="Points Contribution", y="Feature",
                            orientation="h", color="Points Contribution",
                            color_continuous_scale=["#ef4444", "#eab308", "#22c55e"],
                            template="plotly_dark",
                        )
                        fig_bar.update_layout(
                            height=max(300, len(cdf) * 22),
                            paper_bgcolor="#080e14", plot_bgcolor="#080e14",
                            coloraxis_showscale=False,
                            margin=dict(l=0, r=0, t=10, b=0),
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.exception(e)


# ===========================================================================
# PAGE 2 — PORTFOLIO DASHBOARD
# ===========================================================================
elif page == "📊 Portfolio Dashboard":

    st.title("Portfolio Risk Dashboard")
    st.markdown(
        "<div style='color:#7a9bb5; margin-bottom:24px'>"
        "Portfolio-level Expected Loss and Capital Requirement metrics "
        "computed from the test set. EL = PD × LGD × EAD."
        "</div>",
        unsafe_allow_html=True,
    )

    el = load_el_results()

    if el is None:
        st.warning("el_results.pkl not found. Run expected_loss.py to generate portfolio metrics.", icon="⚠️")
        st.stop()

    portfolio  = el["portfolio"]
    loan_level = el["loan_level"]
    summary    = el["summary"]

    st.markdown("### Portfolio Overview")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Loans",            f"{portfolio['n_loans']:,}")
    k2.metric("Total EAD",              f"${portfolio['total_ead']:,.0f}")
    k3.metric("Total Expected Loss",     f"${portfolio['total_el']:,.0f}")
    k4.metric("EL Rate",                 f"{portfolio['el_rate']:.3%}")
    k5.metric("Mean PD",                 f"{portfolio['mean_pd']:.3%}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    k6, k7, k8 = st.columns(3)
    k6.metric("Mean LGD",               f"{portfolio['mean_lgd']:.1%}")
    k7.metric("Mean EAD per Loan",      f"${portfolio['mean_ead']:,.0f}")
    k8.metric("High-Risk Loans (PD≥50%)",
              f"{portfolio['n_high_risk']:,} ({portfolio['n_high_risk']/portfolio['n_loans']:.1%})")

    st.markdown("---")
    st.markdown("### Expected Loss by Grade")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig_el = px.bar(
            summary.reset_index(), x="grade", y="total_el",
            color="total_el",
            color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
            labels={"total_el": "Total EL ($)", "grade": "Grade"},
            template="plotly_dark", title="Total Expected Loss by Grade",
        )
        fig_el.update_layout(paper_bgcolor="#080e14", plot_bgcolor="#080e14",
                             coloraxis_showscale=False, margin=dict(t=40,b=20,l=0,r=0))
        st.plotly_chart(fig_el, use_container_width=True)

    with r1c2:
        fig_rate = px.line(
            summary.reset_index(), x="grade", y="el_rate", markers=True,
            labels={"el_rate": "EL Rate (EL/EAD)", "grade": "Grade"},
            template="plotly_dark", title="EL Rate by Grade",
        )
        fig_rate.update_traces(line_color="#f97316", marker_color="#f97316", line_width=2.5)
        fig_rate.update_layout(paper_bgcolor="#080e14", plot_bgcolor="#080e14",
                               margin=dict(t=40,b=20,l=0,r=0), yaxis_tickformat=".2%")
        st.plotly_chart(fig_rate, use_container_width=True)

    st.markdown("### Risk Distribution")
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig_pd = px.histogram(
            loan_level, x="pd", nbins=50,
            color_discrete_sequence=["#1a6baa"],
            labels={"pd": "Predicted PD"},
            template="plotly_dark", title="Distribution of Predicted PD",
        )
        fig_pd.update_layout(paper_bgcolor="#080e14", plot_bgcolor="#080e14",
                             margin=dict(t=40,b=20,l=0,r=0), bargap=0.05)
        st.plotly_chart(fig_pd, use_container_width=True)

    with r2c2:
        sample = loan_level.sample(min(5000, len(loan_level)), random_state=42)
        fig_scatter = px.scatter(
            sample, x="ead", y="el", color="pd",
            color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
            opacity=0.5,
            labels={"ead": "EAD ($)", "el": "EL ($)", "pd": "PD"},
            template="plotly_dark", title="EAD vs EL (coloured by PD)",
        )
        fig_scatter.update_layout(paper_bgcolor="#080e14", plot_bgcolor="#080e14",
                                  margin=dict(t=40,b=20,l=0,r=0))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Portfolio Composition")
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        grade_counts = loan_level["grade"].value_counts().reset_index()
        grade_counts.columns = ["grade", "count"]
        fig_pie = px.pie(
            grade_counts, values="count", names="grade",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            template="plotly_dark", title="Loan Count by Grade", hole=0.45,
        )
        fig_pie.update_layout(paper_bgcolor="#080e14", margin=dict(t=40,b=20,l=0,r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with r3c2:
        ll_sorted = loan_level.sort_values("el", ascending=False).reset_index(drop=True)
        ll_sorted["cumulative_el_share"] = ll_sorted["el"].cumsum() / ll_sorted["el"].sum()
        ll_sorted["pct_of_portfolio"]    = (ll_sorted.index + 1) / len(ll_sorted)
        fig_conc = px.area(
            ll_sorted, x="pct_of_portfolio", y="cumulative_el_share",
            labels={"pct_of_portfolio": "% of Loans (ranked by EL)",
                    "cumulative_el_share": "Cumulative EL Share"},
            template="plotly_dark", title="EL Concentration Curve",
            color_discrete_sequence=["#f97316"],
        )
        fig_conc.add_scatter(x=[0,1], y=[0,1], mode="lines",
                             line=dict(color="#4a7fa5", dash="dash", width=1.5),
                             name="Equal distribution", showlegend=True)
        fig_conc.update_layout(paper_bgcolor="#080e14", plot_bgcolor="#080e14",
                               margin=dict(t=40,b=20,l=0,r=0),
                               yaxis_tickformat=".0%", xaxis_tickformat=".0%")
        st.plotly_chart(fig_conc, use_container_width=True)

    st.markdown("### Grade-Level Summary Table")
    display_summary = summary.copy()
    display_summary["mean_pd"]   = display_summary["mean_pd"].map("{:.3%}".format)
    display_summary["mean_lgd"]  = display_summary["mean_lgd"].map("{:.1%}".format)
    display_summary["el_rate"]   = display_summary["el_rate"].map("{:.3%}".format)
    display_summary["total_ead"] = display_summary["total_ead"].map("${:,.0f}".format)
    display_summary["total_el"]  = display_summary["total_el"].map("${:,.0f}".format)
    display_summary["mean_el"]   = display_summary["mean_el"].map("${:,.2f}".format)
    display_summary = display_summary.rename(columns={
        "n_loans": "Loans", "mean_pd": "Mean PD", "mean_lgd": "Mean LGD",
        "total_ead": "Total EAD", "total_el": "Total EL",
        "mean_el": "Mean EL", "el_rate": "EL Rate",
    })
    st.dataframe(display_summary, use_container_width=True)

    st.markdown("---")
    st.markdown("### Capital Requirements (Basel II IRB §328)")
    st.markdown(
        "<div style='color:#7a9bb5; margin-bottom:16px; font-size:0.88rem'>"
        "K = LGD × [N(G(PD)/√(1−R) + √(R/(1−R)) × G(0.999)) − PD] × 1.06 &nbsp;·&nbsp; "
        "RWA = K × EAD × 12.5"
        "</div>",
        unsafe_allow_html=True,
    )

    pd_arr  = loan_level["pd"].values
    lgd_arr = loan_level["lgd"].values
    ead_arr = loan_level["ead"].values
    K_arr, rwa_arr = _capital_requirement(pd_arr, lgd_arr, ead_arr)
    total_rwa     = rwa_arr.sum()
    total_cap_req = (K_arr * ead_arr).sum()

    cr1, cr2, cr3, cr4 = st.columns(4)
    cr1.metric("Total RWA",                f"${total_rwa:,.0f}")
    cr2.metric("Total Capital Required",   f"${total_cap_req:,.0f}")
    cr3.metric("Min Capital (Tier 1, 6%)", f"${total_rwa * 0.06:,.0f}")
    cr4.metric("Min Capital (Total, 8%)",  f"${total_rwa * 0.08:,.0f}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    loan_level_cap = loan_level.copy()
    loan_level_cap["K"]   = K_arr
    loan_level_cap["rwa"] = rwa_arr
    loan_level_cap["capital_required"] = K_arr * ead_arr

    cap_by_grade = (
        loan_level_cap.groupby("grade")
        .agg(total_rwa=("rwa","sum"), total_capital_req=("capital_required","sum"), mean_K=("K","mean"))
        .assign(rwa_density=lambda d: d["total_rwa"] / loan_level_cap.groupby("grade")["ead"].sum())
        .sort_index()
    )

    cr_l, cr_r = st.columns(2)
    with cr_l:
        fig_rwa = px.bar(cap_by_grade.reset_index(), x="grade", y="total_rwa",
                         color="total_rwa",
                         color_continuous_scale=["#1a6baa","#f97316","#ef4444"],
                         labels={"total_rwa":"Total RWA ($)","grade":"Grade"},
                         template="plotly_dark", title="RWA by Grade")
        fig_rwa.update_layout(paper_bgcolor="#080e14", plot_bgcolor="#080e14",
                              coloraxis_showscale=False, margin=dict(t=40,b=20,l=0,r=0))
        st.plotly_chart(fig_rwa, use_container_width=True)

    with cr_r:
        fig_k = px.bar(cap_by_grade.reset_index(), x="grade", y="mean_K",
                       color="mean_K",
                       color_continuous_scale=["#1a6baa","#f97316","#ef4444"],
                       labels={"mean_K":"Mean K","grade":"Grade"},
                       template="plotly_dark", title="Mean Capital Requirement (K) by Grade")
        fig_k.update_layout(paper_bgcolor="#080e14", plot_bgcolor="#080e14",
                            coloraxis_showscale=False, yaxis_tickformat=".2%",
                            margin=dict(t=40,b=20,l=0,r=0))
        st.plotly_chart(fig_k, use_container_width=True)

    cap_display = cap_by_grade.copy()
    cap_display["total_rwa"]         = cap_display["total_rwa"].map("${:,.0f}".format)
    cap_display["total_capital_req"] = cap_display["total_capital_req"].map("${:,.0f}".format)
    cap_display["mean_K"]            = cap_display["mean_K"].map("{:.3%}".format)
    cap_display["rwa_density"]       = cap_display["rwa_density"].map("{:.2f}x".format)
    cap_display = cap_display.rename(columns={
        "total_rwa":"Total RWA","total_capital_req":"Capital Required",
        "mean_K":"Mean K","rwa_density":"RWA Density (RWA/EAD)",
    })
    st.dataframe(cap_display, use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.75rem; color:#4a7fa5'>"
        f"Data: test split ({portfolio['n_loans']:,} loans) · "
        f"LGD: Basel II 45% constant · EAD: funded_amnt − total_pymnt · "
        f"Generated: {datetime.now().strftime('%Y-%m-%d')}</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 3 — MODEL PERFORMANCE
# ===========================================================================
else:
    st.title("Model Performance")
    st.markdown(
        "<div style='color:#7a9bb5; margin-bottom:24px'>"
        "WoE logistic regression scorecard — discrimination, bad loan detection, "
        "and threshold analysis. Run evaluate.py to refresh these metrics."
        "</div>",
        unsafe_allow_html=True,
    )

    metrics = load_model_metrics()

    if metrics is None:
        st.warning(
            "model_metrics.pkl not found. Run `evaluate.py` first to generate metrics.",
            icon="⚠️",
        )
        st.stop()

    # ── Section 1: Discrimination (threshold-independent) ────────────────────
    st.markdown("### Discrimination Metrics")
    st.markdown(
        "<div style='color:#7a9bb5; font-size:0.85rem; margin-bottom:16px'>"
        "These metrics do not depend on the classification threshold — "
        "they measure the model's ability to rank borrowers by risk."
        "</div>",
        unsafe_allow_html=True,
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric(
        "ROC-AUC",
        f"{metrics['auc']:.4f}",
        help="Area under the ROC curve. 0.5 = random, 1.0 = perfect.",
    )
    d2.metric(
        "Gini Coefficient",
        f"{metrics['gini']:.4f}",
        delta="Good" if metrics["gini"] >= 0.3 else "Below 0.30 threshold",
        delta_color="normal" if metrics["gini"] >= 0.3 else "inverse",
        help="Gini = 2×AUC − 1. Industry minimum: 0.30 (30%).",
    )
    d3.metric(
        "KS Statistic",
        f"{metrics['ks']:.4f}",
        delta="Good" if metrics["ks"] >= 0.2 else "Below 0.20 threshold",
        delta_color="normal" if metrics["ks"] >= 0.2 else "inverse",
        help="Max separation between cumulative good/bad distributions. Minimum: 0.20.",
    )
    d4.metric(
        "PSI",
        f"{metrics['psi']:.4f}",
        delta=metrics["psi_status"],
        delta_color="normal" if metrics["psi"] < 0.10 else ("off" if metrics["psi"] < 0.25 else "inverse"),
        help="Population Stability Index. <0.10 stable, 0.10–0.25 monitor, >0.25 retrain.",
    )

    st.markdown("---")

    # ── Section 2: Bad Loan Detection ────────────────────────────────────────
    st.markdown("### Bad Loan Detection")
    st.markdown(
        f"<div style='color:#7a9bb5; font-size:0.85rem; margin-bottom:16px'>"
        f"Metrics at the best recall threshold "
        f"(<b style='color:#e8f0f7'>t = {metrics['best_threshold']}</b>). "
        f"Bad = Default (class 0). Recall is the primary metric — a missed "
        f"default is an unexpected loss."
        f"</div>",
        unsafe_allow_html=True,
    )

    b1, b2, b3 = st.columns(3)
    b1.metric(
        "Recall (Bad)",
        f"{metrics['recall_bad']:.4f}",
        delta=f"{metrics['recall_bad']*100:.1f}% of defaults caught",
        delta_color="normal" if metrics["recall_bad"] >= 0.6 else "inverse",
        help="Of all actual defaults, what % did the model flag? Maximise this.",
    )
    b2.metric(
        "Precision (Bad)",
        f"{metrics['precision_bad']:.4f}",
        delta=f"{metrics['precision_bad']*100:.1f}% of flagged are truly bad",
        delta_color="normal" if metrics["precision_bad"] >= 0.4 else "off",
        help="Of all loans flagged as bad, what % are actually bad?",
    )
    b3.metric(
        "F1 (Bad)",
        f"{metrics['f1_bad']:.4f}",
        help="Harmonic mean of recall and precision for the bad class.",
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Defaults Caught (TP)",
        f"{metrics['true_positives']:,}",
        help="Correctly flagged defaults.",
    )
    c2.metric(
        "Defaults Missed (FN)",
        f"{metrics['false_negatives']:,}",
        delta="Costly — unexpected loss",
        delta_color="inverse",
        help="Defaults the model failed to catch. These become unexpected losses.",
    )
    c3.metric(
        "Good Loans Rejected (FP)",
        f"{metrics['false_positives']:,}",
        delta="Lost revenue",
        delta_color="off",
        help="Good borrowers incorrectly declined. These are lost interest revenue.",
    )

    st.markdown("---")

    # ── Section 3: Threshold sweep table + chart ─────────────────────────────
    st.markdown("### Threshold Analysis")
    st.markdown(
        "<div style='color:#7a9bb5; font-size:0.85rem; margin-bottom:16px'>"
        "Each row is the same model evaluated at a different decision threshold. "
        "Lowering the threshold flags more loans as bad — recall rises, precision falls. "
        "Pick the threshold that matches your risk appetite."
        "</div>",
        unsafe_allow_html=True,
    )

    sweep_df = pd.DataFrame(metrics["threshold_sweep"])

    # Chart: recall, precision, f1 vs threshold
    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["threshold"], y=sweep_df["recall_bad"],
        mode="lines+markers", name="Recall (Bad)",
        line=dict(color="#ef4444", width=2.5),
        marker=dict(size=8),
    ))
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["threshold"], y=sweep_df["precision_bad"],
        mode="lines+markers", name="Precision (Bad)",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=8),
    ))
    fig_sweep.add_trace(go.Scatter(
        x=sweep_df["threshold"], y=sweep_df["f1_bad"],
        mode="lines+markers", name="F1 (Bad)",
        line=dict(color="#22c55e", width=2, dash="dash"),
        marker=dict(size=8),
    ))
    # Mark the best threshold
    fig_sweep.add_vline(
        x=metrics["best_threshold"],
        line_dash="dot", line_color="#f59e0b", line_width=1.5,
        annotation_text=f"Best recall (t={metrics['best_threshold']})",
        annotation_font_color="#f59e0b",
        annotation_position="top left",
    )
    fig_sweep.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080e14",
        plot_bgcolor="#080e14",
        xaxis_title="Threshold (min p_good to approve)",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(t=40, b=30, l=0, r=0),
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    # Table
    sweep_display = sweep_df[[
        "threshold","recall_bad","precision_bad","f1_bad",
        "true_positives","false_negatives","false_positives"
    ]].copy()
    sweep_display["recall_bad"]    = sweep_display["recall_bad"].map("{:.4f}".format)
    sweep_display["precision_bad"] = sweep_display["precision_bad"].map("{:.4f}".format)
    sweep_display["f1_bad"]        = sweep_display["f1_bad"].map("{:.4f}".format)
    sweep_display = sweep_display.rename(columns={
        "threshold":       "Threshold",
        "recall_bad":      "Recall (Bad)",
        "precision_bad":   "Precision (Bad)",
        "f1_bad":          "F1 (Bad)",
        "true_positives":  "TP (Caught)",
        "false_negatives": "FN (Missed)",
        "false_positives": "FP (Good Rejected)",
    })
    st.dataframe(sweep_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#4a7fa5'>"
        "AUC, Gini, KS are threshold-independent. "
        "Recall / Precision / F1 shown at the best-recall threshold from the sweep. "
        "Re-run evaluate.py to update all metrics after retraining."
        "</div>",
        unsafe_allow_html=True,
    )
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

from ETL import prepare_inputs
from Scorecard import build_scorecard, calculate_credit_score, get_risk_tier, get_approval_decision

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

/* Sidebar */
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

/* Metric cards */
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

/* Section headers */
.section-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a7fa5;
    margin-bottom: 6px;
    margin-top: 20px;
}

/* Dividers */
hr { border-color: #1e2d3d; }

/* Tables */
.dataframe { font-size: 0.85rem; }

/* Buttons */
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
    path = os.path.join(os.path.dirname(__file__), "pd_model.sav")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_features():
    path = os.path.join(os.path.dirname(__file__), "pd_model_features.pkl")
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
    path = os.path.join(os.path.dirname(__file__), "pd_model_threshold.pkl")
    if not os.path.exists(path):
        return 0.5   # fall back to sklearn default if not found
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Capital requirement helpers (Basel II IRB retail — §328)
# ---------------------------------------------------------------------------
def _basel_correlation(pd_arr):
    """Asset correlation R for retail exposures (Basel II §328)."""
    e50 = np.exp(-50)
    return (
        0.12 * (1 - np.exp(-50 * pd_arr)) / (1 - e50)
        + 0.24 * (1 - (1 - np.exp(-50 * pd_arr)) / (1 - e50))
    )


def _capital_requirement(pd_arr, lgd_arr, ead_arr, maturity_adj=1.06):
    """
    Basel II IRB Retail capital requirement (§328).

    K   = LGD × [N(G(PD)/√(1−R) + √(R/(1−R)) × G(0.999)) − PD] × 1.06
    RWA = K × EAD × 12.5
    """
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
        ["🔍 Applicant Assessment", "📊 Portfolio Dashboard"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#4a7fa5; line-height:1.6'>"
        "Basel II compliant credit risk framework.<br>"
        "EL = PD × LGD × EAD<br><br>"
        "LGD: 45% regulatory constant<br>"
        "EAD: funded_amnt − total_pymnt<br>"
        "Score scale: 300–850 (ZA)<br>"
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

    # Load model artefacts
    model                = load_pd_model()
    significant_features = load_features()
    lgd_model            = load_lgd_model()
    threshold            = load_threshold()

    if model is None:
        st.error("pd_model.sav not found. Run fit_model.py first.")
        st.stop()
    if significant_features is None:
        st.error("pd_model_features.pkl not found. Run fit_model.py first.")
        st.stop()

    # Build scorecard — uses absolute PDO formula, no min-max scaling
    scorecard, base_points = build_scorecard(
        model,
        feature_names=significant_features,
        pdo=50,
        base_score=600,
        base_odds=50,
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

                X_prepared, _, _, _ = prepare_inputs(
                    input_df, reference_date="2017-12-01",
                    target_col=None, fit_ohe=False, fit_train_params=False,
                    train_params={"annual_inc_mean": annual_inc,
                                  "max_mths_issue_d": 84,
                                  "max_mths_earliest_cr_line": 400},
                )

                # Align to saved significant features
                X_final = pd.DataFrame(0, index=[0], columns=significant_features)
                for col in significant_features:
                    if col in X_prepared.columns:
                        X_final[col] = X_prepared[col].values[0]

                X_arr      = X_final.values.astype(np.float64)
                proba_arr  = model.predict_proba(X_arr)[0]
                p_good     = proba_arr[1]
                pd_val     = 1 - p_good    # PD = P(default)
                el         = pd_val * lgd * ead

                # Apply the saved optimal threshold (not sklearn default 0.5)
                # 1 = good loan, 0 = default
                model_decision = int(p_good >= threshold)

                # Absolute PDO credit score — approval decision driven by score
                credit_score = calculate_credit_score(
                    X_final, scorecard, base_points, min_score=300, max_score=850
                )[0]
                risk_tier = get_risk_tier(credit_score)
                decision  = get_approval_decision(credit_score)

                # Capital requirement for this single loan
                K_loan, rwa_loan = _capital_requirement(
                    np.array([pd_val]), np.array([lgd]), np.array([ead])
                )
                cap_req_loan = float(K_loan[0] * ead)

                # ----------------------------------------------------------
                # Results
                # ----------------------------------------------------------
                st.markdown("---")
                st.markdown("### Assessment Results")

                m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
                m1.metric("Credit Score",        f"{credit_score}")
                m2.metric("Risk Tier",           risk_tier.split(" - ")[0])
                m3.metric("PD",                  f"{pd_val:.2%}")
                m4.metric("P(Good)",             f"{p_good:.2%}")
                m5.metric("Expected Loss",       f"${el:,.0f}")
                m6.metric("Model Threshold",     f"{threshold:.2f}")
                m7.metric("Decision",
                          "✅ Approve" if decision == "Approve" else "❌ Decline")

                # Gauge
                score_color = (
                    "#22c55e" if credit_score >= 670 else
                    "#84cc16" if credit_score >= 592 else
                    "#eab308" if credit_score >= 560 else
                    "#f97316" if credit_score >= 505 else "#ef4444"
                )
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=credit_score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Credit Score", "font": {"size": 18}},
                    number={"font": {"color": score_color, "size": 48}},
                    gauge={
                        "axis": {"range": [300, 850], "tickcolor": "#4a7fa5"},
                        "bar":  {"color": score_color, "thickness": 0.25},
                        "bgcolor": "#0f1923",
                        "bordercolor": "#1e2d3d",
                        "steps": [
                            {"range": [300, 505], "color": "#2d1515"},
                            {"range": [505, 560], "color": "#2d1e0f"},
                            {"range": [560, 592], "color": "#2d2a0f"},
                            {"range": [592, 670], "color": "#152d15"},
                            {"range": [670, 850], "color": "#0f2d0f"},
                        ],
                    },
                ))
                fig_gauge.update_layout(
                    height=280,
                    paper_bgcolor="#080e14",
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
                    if credit_score >= 670:
                        st.success("**Excellent Credit** — Low risk, favourable terms available.")
                    elif credit_score >= 592:
                        st.success("**Good Credit** — Moderate risk, standard terms.")
                    elif credit_score >= 560:
                        st.warning("**Average Credit** — Medium risk, higher rates may apply.")
                    elif credit_score >= 505:
                        st.error("**Below Average** — High risk, limited options.")
                    else:
                        st.error("**High Risk** — Very high risk, consider secured options.")

                # Score factors
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
else:
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
        st.warning(
            "el_results.pkl not found. Run `expected_loss.py` to generate portfolio metrics.",
            icon="⚠️",
        )
        st.stop()

    portfolio  = el["portfolio"]
    loan_level = el["loan_level"]
    summary    = el["summary"]

    # ------------------------------------------------------------------
    # KPI row
    # ------------------------------------------------------------------
    st.markdown("### Portfolio Overview")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Loans",         f"{portfolio['n_loans']:,}")
    k2.metric("Total EAD",           f"${portfolio['total_ead']:,.0f}")
    k3.metric("Total Expected Loss",  f"${portfolio['total_el']:,.0f}")
    k4.metric("EL Rate",              f"{portfolio['el_rate']:.3%}")
    k5.metric("Mean PD",              f"{portfolio['mean_pd']:.3%}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    k6, k7, k8 = st.columns(3)
    k6.metric("Mean LGD",              f"{portfolio['mean_lgd']:.1%}")
    k7.metric("Mean EAD per Loan",     f"${portfolio['mean_ead']:,.0f}")
    k8.metric("High-Risk Loans (PD≥50%)",
              f"{portfolio['n_high_risk']:,} ({portfolio['n_high_risk']/portfolio['n_loans']:.1%})")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Row 1: EL by grade bar + EL rate by grade line
    # ------------------------------------------------------------------
    st.markdown("### Expected Loss by Grade")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig_el = px.bar(
            summary.reset_index(),
            x="grade", y="total_el",
            color="total_el",
            color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
            labels={"total_el": "Total EL ($)", "grade": "Grade"},
            template="plotly_dark",
            title="Total Expected Loss by Grade",
        )
        fig_el.update_layout(
            paper_bgcolor="#080e14", plot_bgcolor="#080e14",
            coloraxis_showscale=False,
            margin=dict(t=40, b=20, l=0, r=0),
        )
        st.plotly_chart(fig_el, use_container_width=True)

    with r1c2:
        fig_rate = px.line(
            summary.reset_index(),
            x="grade", y="el_rate", markers=True,
            labels={"el_rate": "EL Rate (EL/EAD)", "grade": "Grade"},
            template="plotly_dark",
            title="EL Rate by Grade",
        )
        fig_rate.update_traces(line_color="#f97316", marker_color="#f97316", line_width=2.5)
        fig_rate.update_layout(
            paper_bgcolor="#080e14", plot_bgcolor="#080e14",
            margin=dict(t=40, b=20, l=0, r=0),
            yaxis_tickformat=".2%",
        )
        st.plotly_chart(fig_rate, use_container_width=True)

    # ------------------------------------------------------------------
    # Row 2: PD distribution + EAD vs EL scatter
    # ------------------------------------------------------------------
    st.markdown("### Risk Distribution")
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig_pd = px.histogram(
            loan_level, x="pd", nbins=50,
            color_discrete_sequence=["#1a6baa"],
            labels={"pd": "Predicted PD"},
            template="plotly_dark",
            title="Distribution of Predicted PD",
        )
        fig_pd.update_layout(
            paper_bgcolor="#080e14", plot_bgcolor="#080e14",
            margin=dict(t=40, b=20, l=0, r=0),
            bargap=0.05,
        )
        st.plotly_chart(fig_pd, use_container_width=True)

    with r2c2:
        sample = loan_level.sample(min(5000, len(loan_level)), random_state=42)
        fig_scatter = px.scatter(
            sample, x="ead", y="el", color="pd",
            color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
            opacity=0.5,
            labels={"ead": "EAD ($)", "el": "EL ($)", "pd": "PD"},
            template="plotly_dark",
            title="EAD vs EL (coloured by PD)",
        )
        fig_scatter.update_layout(
            paper_bgcolor="#080e14", plot_bgcolor="#080e14",
            margin=dict(t=40, b=20, l=0, r=0),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ------------------------------------------------------------------
    # Row 3: Grade composition + EL concentration curve
    # ------------------------------------------------------------------
    st.markdown("### Portfolio Composition")
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        grade_counts = loan_level["grade"].value_counts().reset_index()
        grade_counts.columns = ["grade", "count"]
        fig_pie = px.pie(
            grade_counts, values="count", names="grade",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            template="plotly_dark",
            title="Loan Count by Grade",
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="#080e14",
            margin=dict(t=40, b=20, l=0, r=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with r3c2:
        loan_level_sorted = loan_level.sort_values("el", ascending=False).reset_index(drop=True)
        loan_level_sorted["cumulative_el_share"] = loan_level_sorted["el"].cumsum() / loan_level_sorted["el"].sum()
        loan_level_sorted["pct_of_portfolio"] = (loan_level_sorted.index + 1) / len(loan_level_sorted)

        fig_conc = px.area(
            loan_level_sorted,
            x="pct_of_portfolio", y="cumulative_el_share",
            labels={"pct_of_portfolio": "% of Loans (ranked by EL)",
                    "cumulative_el_share": "Cumulative EL Share"},
            template="plotly_dark",
            title="EL Concentration Curve",
            color_discrete_sequence=["#f97316"],
        )
        fig_conc.add_scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color="#4a7fa5", dash="dash", width=1.5),
            name="Equal distribution",
            showlegend=True,
        )
        fig_conc.update_layout(
            paper_bgcolor="#080e14", plot_bgcolor="#080e14",
            margin=dict(t=40, b=20, l=0, r=0),
            yaxis_tickformat=".0%", xaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_conc, use_container_width=True)

    # ------------------------------------------------------------------
    # Grade summary table
    # ------------------------------------------------------------------
    st.markdown("### Grade-Level Summary Table")
    display_summary = summary.copy()
    display_summary["mean_pd"]   = display_summary["mean_pd"].map("{:.3%}".format)
    display_summary["mean_lgd"]  = display_summary["mean_lgd"].map("{:.1%}".format)
    display_summary["el_rate"]   = display_summary["el_rate"].map("{:.3%}".format)
    display_summary["total_ead"] = display_summary["total_ead"].map("${:,.0f}".format)
    display_summary["total_el"]  = display_summary["total_el"].map("${:,.0f}".format)
    display_summary["mean_el"]   = display_summary["mean_el"].map("${:,.2f}".format)
    display_summary = display_summary.rename(columns={
        "n_loans":  "Loans",
        "mean_pd":  "Mean PD",
        "mean_lgd": "Mean LGD",
        "total_ead":"Total EAD",
        "total_el": "Total EL",
        "mean_el":  "Mean EL",
        "el_rate":  "EL Rate",
    })
    st.dataframe(display_summary, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Capital Requirements (Basel II IRB retail — §328)
    # ------------------------------------------------------------------
    st.markdown("### Capital Requirements")
    st.markdown(
        "<div style='color:#7a9bb5; margin-bottom:16px; font-size:0.88rem'>"
        "Basel II IRB retail formula (§328): "
        "K = LGD × [N(G(PD)/√(1−R) + √(R/(1−R)) × G(0.999)) − PD] × 1.06 &nbsp;·&nbsp; "
        "RWA = K × EAD × 12.5 &nbsp;·&nbsp; "
        "Min Tier 1 = 6% of RWA &nbsp;·&nbsp; Min Total = 8% of RWA"
        "</div>",
        unsafe_allow_html=True,
    )

    pd_arr  = loan_level["pd"].values
    lgd_arr = loan_level["lgd"].values
    ead_arr = loan_level["ead"].values

    K_arr, rwa_arr = _capital_requirement(pd_arr, lgd_arr, ead_arr)

    total_rwa     = rwa_arr.sum()
    total_cap_req = (K_arr * ead_arr).sum()
    min_cap_t1    = total_rwa * 0.06
    min_cap_total = total_rwa * 0.08

    cr1, cr2, cr3, cr4 = st.columns(4)
    cr1.metric("Total RWA",                f"${total_rwa:,.0f}")
    cr2.metric("Total Capital Required",   f"${total_cap_req:,.0f}")
    cr3.metric("Min Capital (Tier 1, 6%)", f"${min_cap_t1:,.0f}")
    cr4.metric("Min Capital (Total, 8%)",  f"${min_cap_total:,.0f}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Capital by grade
    loan_level_cap               = loan_level.copy()
    loan_level_cap["K"]          = K_arr
    loan_level_cap["rwa"]        = rwa_arr
    loan_level_cap["capital_required"] = K_arr * ead_arr

    cap_by_grade = (
        loan_level_cap.groupby("grade")
        .agg(
            total_rwa         = ("rwa",              "sum"),
            total_capital_req = ("capital_required", "sum"),
            mean_K            = ("K",                "mean"),
        )
        .assign(rwa_density=lambda d: d["total_rwa"] /
                loan_level_cap.groupby("grade")["ead"].sum())
        .sort_index()
    )

    cr_l, cr_r = st.columns(2)

    with cr_l:
        fig_rwa = px.bar(
            cap_by_grade.reset_index(),
            x="grade", y="total_rwa",
            color="total_rwa",
            color_continuous_scale=["#1a6baa", "#f97316", "#ef4444"],
            labels={"total_rwa": "Total RWA ($)", "grade": "Grade"},
            template="plotly_dark",
            title="Risk-Weighted Assets (RWA) by Grade",
        )
        fig_rwa.update_layout(
            paper_bgcolor="#080e14", plot_bgcolor="#080e14",
            coloraxis_showscale=False,
            margin=dict(t=40, b=20, l=0, r=0),
        )
        st.plotly_chart(fig_rwa, use_container_width=True)

    with cr_r:
        fig_k = px.bar(
            cap_by_grade.reset_index(),
            x="grade", y="mean_K",
            color="mean_K",
            color_continuous_scale=["#1a6baa", "#f97316", "#ef4444"],
            labels={"mean_K": "Mean Capital Requirement (K)", "grade": "Grade"},
            template="plotly_dark",
            title="Mean Capital Requirement (K) by Grade",
        )
        fig_k.update_layout(
            paper_bgcolor="#080e14", plot_bgcolor="#080e14",
            coloraxis_showscale=False,
            yaxis_tickformat=".2%",
            margin=dict(t=40, b=20, l=0, r=0),
        )
        st.plotly_chart(fig_k, use_container_width=True)

    # Capital requirements table
    st.markdown("### Capital Requirements by Grade")
    cap_display = cap_by_grade.copy()
    cap_display["total_rwa"]         = cap_display["total_rwa"].map("${:,.0f}".format)
    cap_display["total_capital_req"] = cap_display["total_capital_req"].map("${:,.0f}".format)
    cap_display["mean_K"]            = cap_display["mean_K"].map("{:.3%}".format)
    cap_display["rwa_density"]       = cap_display["rwa_density"].map("{:.2f}x".format)
    cap_display = cap_display.rename(columns={
        "total_rwa":         "Total RWA",
        "total_capital_req": "Capital Required",
        "mean_K":            "Mean K",
        "rwa_density":       "RWA Density (RWA/EAD)",
    })
    st.dataframe(cap_display, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#4a7fa5'>"
        f"Data: test split ({portfolio['n_loans']:,} loans) · "
        f"LGD: Basel II 45% constant · "
        f"EAD: funded_amnt − total_pymnt · "
        f"Capital: Basel II IRB retail §328 · "
        f"Generated: {datetime.now().strftime('%Y-%m-%d')}"
        "</div>",
        unsafe_allow_html=True,
    )
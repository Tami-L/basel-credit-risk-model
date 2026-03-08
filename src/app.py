import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime

# Add the src directory to path so we can import modules
sys.path.append(os.path.dirname(__file__))
from ETL import prepare_inputs
from Scorecard import build_scorecard, calculate_credit_score, get_risk_tier

# Page config
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Credit Risk Assessment Tool")
st.markdown("Enter loan applicant details to predict probability of default and credit score.")

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "pd_model.sav")
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

if model is None:
    st.stop()

# Get the internal sklearn model (handle custom class)
if hasattr(model, 'model') and hasattr(model.model, 'coef_'):
    internal_model = model.model
    st.sidebar.success(f"✅ Custom model loaded (expects {len(internal_model.coef_[0])} features)")
elif hasattr(model, 'coef_'):
    internal_model = model
    st.sidebar.success(f"✅ Sklearn model loaded (expects {len(internal_model.coef_[0])} features)")
else:
    st.sidebar.error("Model format not recognized")
    st.stop()

# Define ALL 86 feature columns (including the ones that were mistakenly dropped)
feature_columns = [
    # Grade dummies (7 features - ALL grades including G)
    'grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F', 'grade:G',
    
    # Home ownership (3 features - including RENT_OTHER_NONE_ANY)
    'home_ownership:RENT_OTHER_NONE_ANY', 'home_ownership:OWN', 'home_ownership:MORTGAGE',
    
    # Address state groupings (13 features - ND_NE_IA_NV_FL_HI_AL is reference)
    'addr_state:NM_VA', 'addr_state:NY',
    'addr_state:OK_TN_MO_LA_MD_NC', 'addr_state:CA', 'addr_state:UT_KY_AZ_NJ',
    'addr_state:AR_MI_PA_OH_MN', 'addr_state:RI_MA_DE_SD_IN', 'addr_state:GA_WA_OR',
    'addr_state:WI_MT', 'addr_state:TX', 'addr_state:IL_CT', 
    'addr_state:KS_SC_CO_VT_AK_MS', 'addr_state:WV_NH_WY_DC_ME_ID',
    
    # Verification status (2 features - Verified is reference)
    'verification_status:Not Verified', 'verification_status:Source Verified',
    
    # Purpose groupings (4 features - educ__sm_b__wedd__ren_en__mov__house is reference)
    'purpose:credit_card', 'purpose:debt_consolidation', 
    'purpose:oth__med__vacation', 'purpose:major_purch__car__home_impr',
    
    # Initial list status (1 feature - f is reference)
    'initial_list_status:w',
    
    # Term (1 feature - 60 is reference)
    'term:36',
    
    # Employment length (5 features - 0 is reference)
    'emp_length:1', 'emp_length:2-4', 'emp_length:5-6', 'emp_length:7-9', 'emp_length:10',
    
    # Months since issue d (7 features - >84 is reference)
    'mths_since_issue_d:<38', 'mths_since_issue_d:38-39', 'mths_since_issue_d:40-41',
    'mths_since_issue_d:42-48', 'mths_since_issue_d:49-52', 'mths_since_issue_d:53-64',
    'mths_since_issue_d:65-84',
    
    # Interest rate buckets (4 features - >20.281 is reference)
    'int_rate:<9.548', 'int_rate:9.548-12.025', 'int_rate:12.025-15.74',
    'int_rate:15.74-20.281',
    
    # Months since earliest credit line (5 features - <140 is reference)
    'mths_since_earliest_cr_line:141-164', 'mths_since_earliest_cr_line:165-247',
    'mths_since_earliest_cr_line:248-270', 'mths_since_earliest_cr_line:271-352',
    'mths_since_earliest_cr_line:>352',
    
    # Inquiries last 6 months (3 features - >6 is reference)
    'inq_last_6mths:0', 'inq_last_6mths:1-2', 'inq_last_6mths:3-6',
    
    # Accounts now delinquent (1 feature - 0 is reference)
    'acc_now_delinq:>=1',
    
    # Annual income buckets (11 features - <20K is reference)
    'annual_inc:20K-30K', 'annual_inc:30K-40K', 'annual_inc:40K-50K',
    'annual_inc:50K-60K', 'annual_inc:60K-70K', 'annual_inc:70K-80K', 'annual_inc:80K-90K',
    'annual_inc:90K-100K', 'annual_inc:100K-120K', 'annual_inc:120K-140K', 'annual_inc:>140K',
    
    # DTI buckets (9 features - >35 is reference)
    'dti:<=1.4', 'dti:1.4-3.5', 'dti:3.5-7.7', 'dti:7.7-10.5', 'dti:10.5-16.1',
    'dti:16.1-20.3', 'dti:20.3-21.7', 'dti:21.7-22.4', 'dti:22.4-35',
    
    # Months since last delinquency (4 features - 0-3 is reference)
    'mths_since_last_delinq:Missing', 'mths_since_last_delinq:4-30', 
    'mths_since_last_delinq:31-56', 'mths_since_last_delinq:>=57',
    
    # Months since last record (6 features - 0-2 is reference)
    'mths_since_last_record:Missing', 'mths_since_last_record:3-20', 
    'mths_since_last_record:21-31', 'mths_since_last_record:32-80', 
    'mths_since_last_record:81-86', 'mths_since_last_record:>=86'
]

# Verify we have 86 features
assert len(feature_columns) == 86, f"Expected 86 features, got {len(feature_columns)}"

# Build scorecard (no caching needed, it's fast)
scorecard, base_points_raw = build_scorecard(
    internal_model,
    feature_names=feature_columns,
    pdo=50,
    base_score=600,
    base_odds=50
)

# Create input form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Loan Details")
        loan_amnt = st.number_input("Loan Amount ($)", 1000, 50000, 15000)
        term = st.selectbox("Term", ["36 months", "60 months"])
        int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.5, 0.1)
        installment = st.number_input("Monthly Installment ($)", 100, 2000, 350)
        grade = st.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
        sub_grade = st.selectbox("Sub Grade", [f"{g}{i}" for g in ["A","B","C","D","E","F","G"] for i in range(1,6)])
        
        st.subheader("🏠 Employment")
        emp_length = st.selectbox(
            "Employment Length",
            ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", 
             "6 years", "7 years", "8 years", "9 years", "10+ years"]
        )
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        annual_inc = st.number_input("Annual Income ($)", 20000, 500000, 60000)
    
    with col2:
        st.subheader("💳 Credit History")
        dti = st.slider("Debt-to-Income Ratio", 0.0, 50.0, 15.0, 0.1)
        delinq_2yrs = st.number_input("Delinquencies (2 years)", 0, 10, 0)
        inq_last_6mths = st.number_input("Recent Credit Inquiries", 0, 10, 1)
        open_acc = st.number_input("Open Credit Lines", 1, 30, 8)
        pub_rec = st.number_input("Public Records", 0, 5, 0)
        revol_bal = st.number_input("Revolving Balance ($)", 0, 100000, 5000)
        revol_util = st.slider("Revolving Utilization %", 0.0, 100.0, 45.5, 0.1)
        total_acc = st.number_input("Total Credit Lines", 5, 50, 15)
    
    with col3:
        st.subheader("📍 Other Information")
        addr_state = st.selectbox(
            "State",
            ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "NJ", 
             "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI", "CO", "MN", 
             "SC", "AL", "LA", "KY", "OR", "OK", "CT", "IA", "MS", "AR", "KS", 
             "UT", "NV", "NM", "WV", "NE", "ID", "NH", "ME", "MT", "RI", "VT", 
             "AK", "WY", "DE", "DC", "SD", "ND"]
        )
        
        verification_status = st.selectbox(
            "Income Verification",
            ["Verified", "Source Verified", "Not Verified"]
        )
        
        purpose = st.selectbox(
            "Loan Purpose",
            ["debt_consolidation", "credit_card", "home_improvement", 
             "major_purchase", "medical", "car", "educational", "vacation",
             "moving", "house", "renewable_energy", "wedding", "small_business"]
        )
        
        initial_list_status = st.selectbox("List Status", ["f", "w"])
        
        # Default values for required fields
        st.subheader("📅 Dates")
        issue_d = st.date_input("Issue Date", datetime(2017, 12, 1))
        earliest_cr_line = st.date_input("Earliest Credit Line", datetime(2005, 1, 1))
    
    submitted = st.form_submit_button("🚀 Calculate Credit Score & Risk", type="primary", use_container_width=True)

if submitted:
    with st.spinner("Processing application and calculating risk..."):
        try:
            # Create a single-row DataFrame with all inputs
            input_df = pd.DataFrame([{
                'loan_amnt': loan_amnt,
                'term': term,
                'int_rate': int_rate,
                'installment': installment,
                'grade': grade,
                'sub_grade': sub_grade,
                'emp_length': emp_length,
                'home_ownership': home_ownership,
                'annual_inc': annual_inc,
                'dti': dti,
                'delinq_2yrs': delinq_2yrs,
                'inq_last_6mths': inq_last_6mths,
                'open_acc': open_acc,
                'pub_rec': pub_rec,
                'revol_bal': revol_bal,
                'revol_util': revol_util,
                'total_acc': total_acc,
                'addr_state': addr_state,
                'verification_status': verification_status,
                'purpose': purpose,
                'initial_list_status': initial_list_status,
                'issue_d': issue_d.strftime('%b-%y'),
                'earliest_cr_line': earliest_cr_line.strftime('%b-%y'),
                # Add required columns with default values
                'funded_amnt': loan_amnt,
                'funded_amnt_inv': loan_amnt,
                'mths_since_last_delinq': np.nan,
                'mths_since_last_record': np.nan,
                'collection_recovery_fee': 0,
                'tot_coll_amt': 0,
                'tot_cur_bal': revol_bal,
                'total_rev_hi_lim': revol_bal / (revol_util/100) if revol_util > 0 else revol_bal
            }])
            
            # Apply the same preprocessing as training
            with st.spinner("Applying feature engineering..."):
                X_prepared, _, _ = prepare_inputs(
                    input_df, 
                    reference_date="2017-12-01",
                    target_col=None,
                    fit_ohe=False
                )
            
            # Create a DataFrame with all 86 features initialized to 0
            X_final = pd.DataFrame(0, index=[0], columns=feature_columns)
            
            # Update with values from X_prepared where they exist
            for col in feature_columns:
                if col in X_prepared.columns:
                    X_final[col] = X_prepared[col].values[0]
            
            # Make probability prediction
            X = X_final.values.astype(np.float64)
            probability = internal_model.predict_proba(X)[0][1]
            prediction = internal_model.predict(X)[0]
            
            # Calculate credit score using scorecard
            credit_score = calculate_credit_score(
                X_final,
                scorecard,
                base_points_raw,
                min_score=300,
                max_score=850
            )[0]  # Get first (and only) score
            
            # Get risk tier
            risk_tier = get_risk_tier(credit_score)
            
            # Display results in a nice format
            st.markdown("---")
            st.subheader("📊 Assessment Results")
            
            # Create metrics in two rows
            row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
            
            with row1_col1:
                st.metric("Credit Score", f"{credit_score}", delta=None)
            
            with row1_col2:
                st.metric("Risk Tier", risk_tier.split('-')[0].strip())
            
            with row1_col3:
                st.metric("Default Probability", f"{probability:.1%}")
            
            with row1_col4:
                decision = "✅ Approve" if prediction == 0 else "❌ Reject"
                st.metric("Decision", decision)
            
            # Create a gauge chart for credit score
            import plotly.graph_objects as go
            
            # Define color ranges for South African credit scores
            if credit_score >= 670:
                color = "darkgreen"
                bar_color = "green"
            elif credit_score >= 592:
                color = "green"
                bar_color = "lightgreen"
            elif credit_score >= 560:
                color = "orange"
                bar_color = "orange"
            elif credit_score >= 505:
                color = "red"
                bar_color = "red"
            else:
                color = "darkred"
                bar_color = "darkred"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=credit_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Credit Score"},
                gauge={
                    'axis': {'range': [300, 850]},
                    'bar': {'color': bar_color},
                    'steps': [
                        {'range': [300, 505], 'color': "salmon"},
                        {'range': [505, 560], 'color': "lightcoral"},
                        {'range': [560, 592], 'color': "yellow"},
                        {'range': [592, 670], 'color': "lightgreen"},
                        {'range': [670, 850], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': credit_score
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed risk assessment
            st.subheader("💡 Risk Assessment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Credit Score:** {credit_score}")
                st.markdown(f"**Risk Tier:** {risk_tier}")
                st.markdown(f"**Default Probability:** {probability:.1%}")
            
            with col2:
                if credit_score >= 670:
                    st.success("✅ **Excellent Credit** - Low risk, favorable terms available")
                elif credit_score >= 592:
                    st.success("🟢 **Good Credit** - Moderate risk, standard terms")
                elif credit_score >= 560:
                    st.warning("🟡 **Average Credit** - Medium risk, higher rates may apply")
                elif credit_score >= 505:
                    st.error("🔴 **Below Average** - High risk, limited options")
                else:
                    st.error("⚠️ **High Risk** - Very high risk, consider secured options")
            
            # Show score factors
            with st.expander("📈 View Score Factors"):
                # Get top positive and negative factors
                feature_contributions = []
                for i, feature in enumerate(scorecard["feature"]):
                    points = scorecard.loc[i, "raw_points"]
                    contribution = X_final[feature].values[0] * points
                    if contribution != 0:
                        feature_contributions.append({
                            "feature": feature,
                            "contribution": contribution
                        })
                
                if feature_contributions:
                    contrib_df = pd.DataFrame(feature_contributions)
                    contrib_df = contrib_df.sort_values("contribution", ascending=False)
                    
                    st.markdown("**Top Positive Factors:**")
                    st.dataframe(contrib_df.head(5))
                    
                    st.markdown("**Top Negative Factors:**")
                    st.dataframe(contrib_df.tail(5))
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)

# Add information about the model
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    This model predicts credit scores and probability of default using:
    - Logistic Regression
    - 86 engineered features
    - Data from 2007-2014
    
    **Credit Score Ranges:**
    - 🟢 670-850: Excellent
    - 🟢 592-669: Good  
    - 🟡 560-591: Average
    - 🔴 505-559: Below Average
    - ⚠️ 300-504: High Risk
    """)
    
    st.markdown("---")
    st.subheader("📊 Scorecard Preview")
    st.dataframe(scorecard[['feature', 'raw_points']].head(10))
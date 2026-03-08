import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
import scipy.stats as stat


class LogisticRegression_with_p_values:

    def __init__(self,*args,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)

    def fit(self,X,y):
        self.model.fit(X,y)

        denom = (2.0*(1.0+np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X/denom).T,X)
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))

        z_scores = self.model.coef_[0]/sigma_estimates
        p_values = [stat.norm.sf(abs(x))*2 for x in z_scores]

        self.p_values = p_values

# =====================================================
# SCORECARD FUNCTIONS
# =====================================================

def build_scorecard(model, feature_names, 
                   pdo=50,  # Points to Double Odds
                   base_score=600,  # Base score
                   base_odds=50,    # Odds at base score
                   min_score=300,    # Minimum possible score
                   max_score=850):   # Maximum possible score
    
    """
    Build a scorecard with scaling appropriate for South African context
    
    Parameters:
    - pdo: Points to Double Odds 
    - base_score: Score at base odds 
    - base_odds: Odds at base score 
    - min_score: Minimum credit score 
    - max_score: Maximum credit score 
    """
    
    # Calculate factor and offset
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    
    coefs = model.coef_[0]
    intercept = model.intercept_[0]
    
    # Calculate raw points
    scorecard = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs
    })
    
    # Points for each feature (negative coefficient = higher risk = lower points)
    scorecard["raw_points"] = -coefs * factor
    
    # Base points (from intercept)
    base_points_raw = offset - factor * intercept
    
    print(f"Raw base points: {base_points_raw:.2f}")
    print(f"Raw points range: {scorecard['raw_points'].min():.2f} to {scorecard['raw_points'].max():.2f}")
    
    return scorecard, base_points_raw


def calculate_credit_score(X, scorecard, base_points_raw, 
                          min_score=300, max_score=850,
                          scaling_factor=1.0):
    
    
    # Calculate raw scores
    raw_scores = np.zeros(len(X))
    
    for i, feature in enumerate(scorecard["feature"]):
        points = scorecard.loc[i, "raw_points"]
        raw_scores += X[feature] * points
    
    raw_scores += base_points_raw
    
    # Print raw score statistics for debugging
    print(f"Raw score stats - Min: {raw_scores.min():.2f}, Max: {raw_scores.max():.2f}, Mean: {raw_scores.mean():.2f}")
    
    # Scale to desired range using min-max scaling
    if raw_scores.min() != raw_scores.max():
        
        scaled_scores = min_score + (raw_scores - raw_scores.min()) * (max_score - min_score) / (raw_scores.max() - raw_scores.min())
        
        
        # scaled_scores = base_score + (raw_scores - raw_scores.mean()) * (max_score - min_score) / (raw_scores.max() - raw_scores.min())
    else:
        scaled_scores = np.full_like(raw_scores, base_score)
    
   
    scaled_scores = np.clip(scaled_scores, min_score, max_score)
    
    
    scaled_scores = np.round(scaled_scores).astype(int)
    
    return scaled_scores


def get_risk_tier(score):
    """Assign South African credit risk tiers"""
    if score >= 670:
        return "Tier 1 - Excellent (Low Risk)"
    elif score >= 592:
        return "Tier 2 - Good"
    elif score >= 560:
        return "Tier 3 - Average"
    elif score >= 505:
        return "Tier 4 - Below Average"
    else:
        return "Tier 5 - High Risk"


# =====================================================
# LOAD MODEL
# =====================================================

model_path = "/Users/lindokuhletami/Desktop/Space/basel-credit-risk-model/src/pd_model.sav"

with open(model_path, "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully")
print("Model type:", type(model))


# =====================================================
# LOAD DATA
# =====================================================

data_path = "/Users/lindokuhletami/Desktop/Space/basel-credit-risk-model/data/loan_data_inputs_train.csv"


X_test = pd.read_csv(data_path)
X_test = X_test.loc[: , ['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'home_ownership:OWN',
'home_ownership:MORTGAGE',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'addr_state:NM_VA',
'addr_state:NY',
'addr_state:OK_TN_MO_LA_MD_NC',
'addr_state:CA',
'addr_state:UT_KY_AZ_NJ',
'addr_state:AR_MI_PA_OH_MN',
'addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR',
'addr_state:WI_MT',
'addr_state:TX',
'addr_state:IL_CT',
'addr_state:KS_SC_CO_VT_AK_MS',
'addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified',
'verification_status:Source Verified',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:oth__med__vacation',
'purpose:major_purch__car__home_impr',
'initial_list_status:f',
'initial_list_status:w',
'term:36',
'term:60',
'emp_length:0',
'emp_length:1',
'emp_length:2-4',
'emp_length:5-6',
'emp_length:7-9',
'emp_length:10',
'mths_since_issue_d:<38',
'mths_since_issue_d:38-39',
'mths_since_issue_d:40-41',
'mths_since_issue_d:42-48',
'mths_since_issue_d:49-52',
'mths_since_issue_d:53-64',
'mths_since_issue_d:65-84',
'mths_since_issue_d:>84',
'int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'mths_since_earliest_cr_line:141-164',
'mths_since_earliest_cr_line:165-247',
'mths_since_earliest_cr_line:248-270',
'mths_since_earliest_cr_line:271-352',
'mths_since_earliest_cr_line:>352',
'delinq_2yrs:0',
'delinq_2yrs:1-3',
'delinq_2yrs:>=4',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'inq_last_6mths:>6',
'open_acc:0',
'open_acc:1-3',
'open_acc:4-12',
'open_acc:13-17',
'open_acc:18-22',
'open_acc:23-25',
'open_acc:26-30',
'open_acc:>=31',
'pub_rec:0-2',
'pub_rec:3-4',
'pub_rec:>=5',
'total_acc:<=27',
'total_acc:28-51',
'total_acc:>=52',
'acc_now_delinq:0',
'acc_now_delinq:>=1',
'total_rev_hi_lim:<=5K',
'total_rev_hi_lim:5K-10K',
'total_rev_hi_lim:10K-20K',
'total_rev_hi_lim:20K-30K',
'total_rev_hi_lim:30K-40K',
'total_rev_hi_lim:40K-55K',
'total_rev_hi_lim:55K-95K',
'total_rev_hi_lim:>95K',
'annual_inc:<20K',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'dti:>35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:0-3',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:0-2',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31',
'mths_since_last_record:32-80',
'mths_since_last_record:81-86',
'mths_since_last_record:>=86']]

print("Data loaded successfully")
print("Original shape:", X_test.shape)


# =====================================================
# ALIGN DATA WITH MODEL FEATURES
# =====================================================

# Check if model is custom class or sklearn model
if hasattr(model, 'model') and hasattr(model.model, 'coef_'):
    internal_model = model.model
    n_features = len(internal_model.coef_[0])
    print("Using internal sklearn model from custom class")
elif hasattr(model, 'coef_'):
    internal_model = model
    n_features = len(internal_model.coef_[0])
    print("Using sklearn model directly")
else:
    raise TypeError("Model format not recognized")

X_test = X_test.iloc[:, :n_features]

print("Filtered shape (matching model features):", X_test.shape)


# =====================================================
# BUILD SCORECARD WITH SOUTH AFRICAN SCALING
# =====================================================

# South African credit score parameters
ZA_MIN_SCORE = 300
ZA_MAX_SCORE = 850
ZA_BASE_SCORE = 600
ZA_BASE_ODDS = 50  
ZA_PDO = 50 

scorecard, base_points_raw = build_scorecard(
    internal_model,
    feature_names=X_test.columns,
    pdo=ZA_PDO,
    base_score=ZA_BASE_SCORE,
    base_odds=ZA_BASE_ODDS
)

print("\nScorecard created")
print(f"Base Points (raw): {base_points_raw:.2f}")

print("\nScorecard Preview:")
print(scorecard.head())


# =====================================================
# CALCULATE CREDIT SCORES WITH SCALING
# =====================================================

scores = calculate_credit_score(
    X_test,
    scorecard,
    base_points_raw,
    min_score=ZA_MIN_SCORE,
    max_score=ZA_MAX_SCORE
)

print("\nFirst 10 Scaled Scores (South African scale):")
print(scores[:10])

# Add risk tiers
risk_tiers = [get_risk_tier(score) for score in scores]


# =====================================================
# ADD SCORES TO DATASET
# =====================================================

X_test["credit_score"] = scores
X_test["risk_tier"] = risk_tiers


# =====================================================
# CHECK PD VS SCORE (should be inverse relationship)
# =====================================================

pd_values = internal_model.predict_proba(X_test.drop(columns=["credit_score", "risk_tier"]))[:, 1]

print("\nScore vs PD Check (first 10):")

check_df = pd.DataFrame({
    "score": scores[:10],
    "pd": pd_values[:10],
    "risk_tier": risk_tiers[:10]
})

print(check_df)

# Verify inverse relationship (higher score = lower PD)
correlation = np.corrcoef(scores, pd_values)[0, 1]
print(f"\nCorrelation between score and PD: {correlation:.3f} (should be negative)")


# =====================================================
# SCORE DISTRIBUTION
# =====================================================

print("\nScore Distribution (South African scale):")
print(X_test["credit_score"].describe())

print("\nRisk Tier Distribution:")
print(X_test["risk_tier"].value_counts())

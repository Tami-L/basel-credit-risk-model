"""
fit_lgd_model.py
================
Loss Given Default (LGD) Model — Beta Regression Edition
---------------------------------------------------------
Estimates LGD at the loan level using Beta regression.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, expit, logit, digamma
from scipy.stats import beta as beta_dist

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
# This assumes your script is in /src/ and your data is in /data/
SRC_DIR     = Path(__file__).resolve().parent.parent
ROOT_DIR    = SRC_DIR.parent
DATA_DIR    = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "scorecard_outputs"
LGD_SAVE_PATH = SRC_DIR / "lgd_model.pkl"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BASEL_LGD_UNSECURED = 0.45
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]

BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
    "Late (31-120 days)",
}

LGD_COVARIATES = [
    "int_rate", "dti", "annual_inc", "loan_amnt", 
    "revol_util", "open_acc", "pub_rec", "fico_range_low"
]

LGD_CAT_COVARIATES = [
    "grade", "term", "home_ownership", "purpose"
]

# ── Stress scenarios ───────────────────────────────────────────────────────────
SCENARIOS = {
    "Baseline": {
        "logit_scale": 1.00, "phi_scale": 1.00, "pd_multiplier": 1.00,
        "description": "Fitted parameters — no stress applied", "color": "#1D9E75",
    },
    "Mild Stress": {
        "logit_scale": 1.20, "phi_scale": 0.80, "pd_multiplier": 1.30,
        "description": "Moderate recession — LGD worsens ~15%, PDs rise 30%", "color": "#BA7517",
    },
    "Severe Stress": {
        "logit_scale": 1.50, "phi_scale": 0.55, "pd_multiplier": 2.00,
        "description": "2008-style systemic shock — LGD worsens ~35%, PDs double", "color": "#D85A30",
    },
}

# ============================================================================
# SECTION 1 — DATA LOADING & LGD DERIVATION
# ============================================================================

def load_raw_data(data_path: Path) -> pd.DataFrame:
    print(f"Loading raw data from {data_path.name}...")
    
    # We use engine='c' for speed on the large 2007-2014 file.
    # index_col=False ensures we don't accidentally treat a column as an index.
    try:
        df = pd.read_csv(data_path, index_col=False, low_memory=False)
    except Exception as e:
        print(f"Standard load failed, trying auto-delimiter detection: {e}")
        df = pd.read_csv(data_path, index_col=False, sep=None, engine='python')
    
    # Cleanup: Remove index columns like 'Unnamed: 0' if they exist
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
    print(f"  Loaded: {len(df):,} rows x {df.shape[1]} columns")
    return df

def derive_lgd(df: pd.DataFrame) -> pd.DataFrame:
    print("Deriving LGD for defaulted loans...")
    
    # Standardize column names (lowercase and strip whitespace)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Check for the status column with some flexibility
    status_col = None
    for col in ['loan_status', 'status']:
        if col in df.columns:
            status_col = col
            break
            
    if status_col is None:
        print(f"CRITICAL ERROR: Could not find 'loan_status'. Available: {df.columns.tolist()[:10]}...")
        raise KeyError("Target column 'loan_status' missing from CSV.")

    # Filter for defaults
    defaults = df[df[status_col].isin(BAD_STATUSES)].copy()
    print(f"  Defaulted loans found: {len(defaults):,}")

    # Ensure required columns exist for math
    if "recoveries" not in defaults.columns or "funded_amnt" not in defaults.columns:
        raise ValueError("Dataset must contain 'recoveries' and 'funded_amnt' to calculate LGD.")

    # LGD Calculation
    rec_fee = defaults["collection_recovery_fee"].fillna(0) if "collection_recovery_fee" in defaults.columns else 0
    defaults["net_recoveries"] = (defaults["recoveries"].fillna(0) - rec_fee).clip(lower=0)
    defaults["ead"] = defaults["funded_amnt"].fillna(0)
    
    # Filter out any non-positive EAD to avoid division by zero
    defaults = defaults[defaults["ead"] > 0].copy()
    
    defaults["recovery_rate"] = (defaults["net_recoveries"] / defaults["ead"]).clip(0, 1)
    # Beta regression requires values strictly between 0 and 1
    defaults["lgd"] = (1 - defaults["recovery_rate"]).clip(1e-4, 1 - 1e-4)

    # Map sub_grade to grade if grade is missing
    if "grade" not in defaults.columns and "sub_grade" in defaults.columns:
        defaults["grade"] = defaults["sub_grade"].str[0]

    print(f"  Valid LGD observations for training: {len(defaults):,}")
    return defaults

# ============================================================================
# SECTION 2 — FEATURE PREPARATION
# ============================================================================

def prepare_features(df: pd.DataFrame):
    df = df.copy()
    num_cols = [c for c in LGD_COVARIATES if c in df.columns]
    cat_cols = [c for c in LGD_CAT_COVARIATES if c in df.columns]

    # Imputation and Conversion
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna("Missing")

    # Scaling Numerics
    scales = {}
    X_num  = pd.DataFrame(index=df.index)
    for col in num_cols:
        mu, std = df[col].mean(), max(df[col].std(), 1e-8)
        X_num[col] = (df[col] - mu) / std
        scales[col] = (mu, std)

    # Encoding Categoricals
    X_cat = (pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)
             if cat_cols else pd.DataFrame(index=df.index))

    X_df = pd.concat([X_num, X_cat], axis=1).fillna(0)
    colnames = X_df.columns.tolist()
    X = np.column_stack([np.ones(len(X_df)), X_df.values])
    y = df["lgd"].values

    return X, y, colnames, scales

# ============================================================================
# SECTION 3 — MODEL CORE
# ============================================================================

def _neg_log_lik(params, X, y):
    beta_coef = params[:-1]
    phi = np.exp(params[-1])
    mu = np.clip(expit(X @ beta_coef), 1e-6, 1 - 1e-6)
    a, b = mu * phi, (1 - mu) * phi
    ll = ((a-1)*np.log(y) + (b-1)*np.log(1-y) + gammaln(a+b) - gammaln(a) - gammaln(b))
    return -ll.sum()

def _neg_log_lik_grad(params, X, y):
    beta_coef = params[:-1]
    phi = np.exp(params[-1])
    mu = np.clip(expit(X @ beta_coef), 1e-6, 1 - 1e-6)
    a, b = mu * phi, (1 - mu) * phi
    psi_ab, psi_a, psi_b = digamma(a + b), digamma(a), digamma(b)
    dmu_deta = mu * (1 - mu)
    dL_dmu = phi * (np.log(y) - np.log(1-y) + psi_b - psi_a)
    grad_beta = -(X.T @ (dL_dmu * dmu_deta))
    dL_dphi = (mu*(np.log(y) - psi_a + psi_ab) + (1-mu)*(np.log(1-y) - psi_b + psi_ab))
    grad_logphi = -(dL_dphi.sum() * phi)
    return np.append(grad_beta, grad_logphi)

def fit_beta_regression(X, y, colnames):
    print("Fitting Beta regression (MLE)...")
    init = np.zeros(X.shape[1] + 1)
    init[0], init[-1] = logit(np.clip(y.mean(), 0.01, 0.99)), np.log(10.0)
    res = minimize(_neg_log_lik, init, args=(X, y), method="L-BFGS-B", jac=_neg_log_lik_grad)
    return {"coef": res.x[:-1], "phi": np.exp(res.x[-1]), "colnames": colnames, "converged": res.success}

def predict_lgd(X, beta_reg, logit_scale=1.0):
    return np.clip(expit((X @ beta_reg["coef"]) * logit_scale), 1e-6, 1 - 1e-6)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # 1. Search for the specific user file
    filename = "loan_data_2007_2014(1).csv"
    data_path = DATA_DIR / filename
    
    if not data_path.exists():
        print(f"Warning: {filename} not found in {DATA_DIR}. Searching for any CSV...")
        candidates = list(DATA_DIR.glob("*.csv"))
        if not candidates: raise FileNotFoundError("No CSV data files found.")
        data_path = candidates[0]

    # 2. Process
    df_raw = load_raw_data(data_path)
    defaults = derive_lgd(df_raw)
    X, y, colnames, scales = prepare_features(defaults)
    beta_reg = fit_beta_regression(X, y, colnames)
    
    # 3. Save Model
    lgd_model = {
        "beta_regression": beta_reg,
        "feature_scales": scales,
        "lgd_default": float(y.mean()),
        "method": "beta_regression_v2"
    }

    with open(LGD_SAVE_PATH, "wb") as f:
        pickle.dump(lgd_model, f)
    
    print(f"\nSUCCESS: LGD model fitted on {len(y):,} defaults.")
    print(f"Model saved to: {LGD_SAVE_PATH}")

if __name__ == "__main__":
    main()
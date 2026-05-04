import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, expit, logit

warnings.filterwarnings("ignore")

# ── VERIFIED PATHS ────────────────────────────────────────────────────────────
# We are using the absolute path you provided to ensure it loads the right data
DATA_PATH = Path("/Users/lindokuhletami/Desktop/Space/data/loan_data_2007_2014(1).csv")
SRC_DIR   = Path(__file__).resolve().parent
LGD_SAVE_PATH = SRC_DIR / "lgd_model.pkl"

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
BAD_STATUSES = {
    "Charged Off", "Default", "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
}
LGD_COVARIATES = ["int_rate", "dti", "annual_inc", "loan_amnt", "revol_util", "open_acc", "pub_rec", "fico_range_low"]
LGD_CAT_COVARIATES = ["grade", "term", "home_ownership", "purpose"]

# ── FUNCTIONS ──────────────────────────────────────────────────────────────────

def derive_lgd(df: pd.DataFrame):
    df.columns = [c.lower().strip() for c in df.columns]
    status_col = next((c for c in ['loan_status', 'status'] if c in df.columns), None)
    
    if not status_col:
        raise KeyError(f"Could not find status column. Found: {df.columns.tolist()[:10]}")

    defaults = df[df[status_col].isin(BAD_STATUSES)].copy()
    
    # Calculate EAD and LGD components
    defaults["ead"] = defaults["funded_amnt"].fillna(0)
    defaults = defaults[defaults["ead"] > 0].copy()
    
    rec = defaults["recoveries"].fillna(0)
    fee = defaults["collection_recovery_fee"].fillna(0) if "collection_recovery_fee" in defaults.columns else 0
    
    # recovery_rate = (Recoveries - Fees) / EAD
    defaults["recovery_rate"] = ((rec - fee) / defaults["ead"]).clip(0, 1)
    # LGD = 1 - Recovery Rate (Clipped for Beta Regression 0 < y < 1)
    defaults["lgd"] = (1 - defaults["recovery_rate"]).clip(1e-4, 1 - 1e-4)
    return defaults

def prepare_features(df: pd.DataFrame):
    df = df.copy()
    scales = {}
    X_num = pd.DataFrame(index=df.index)
    
    for col in LGD_COVARIATES:
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
            mu, std = val.mean(), max(val.std(), 1e-8)
            X_num[col] = (val - mu) / std
            scales[col] = {"mean": mu, "std": std}
    
    cat_cols = [c for c in LGD_CAT_COVARIATES if c in df.columns]
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)
    X_df = pd.concat([X_num, X_cat], axis=1).fillna(0)
    
    return np.column_stack([np.ones(len(X_df)), X_df.values]), df["lgd"].values, X_df.columns.tolist(), scales

def _neg_log_lik(params, X, y):
    beta_coef, phi = params[:-1], np.exp(params[-1])
    mu = np.clip(expit(X @ beta_coef), 1e-6, 1 - 1e-6)
    a, b = mu * phi, (1 - mu) * phi
    ll = ((a-1)*np.log(y) + (b-1)*np.log(1-y) + gammaln(a+b) - gammaln(a) - gammaln(b))
    return -ll.sum()

def main():
    if not DATA_PATH.exists():
        print(f"Error: File not found at {DATA_PATH}")
        return

    print(f"Loading raw data: {DATA_PATH.name}...")
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    
    defaults = derive_lgd(df_raw)
    X, y, colnames, scales = prepare_features(defaults)
    
    print(f"Fitting Beta regression (MLE) on {len(y):,} defaulted loans...")
    init = np.zeros(X.shape[1] + 1)
    init[0], init[-1] = logit(np.clip(y.mean(), 0.01, 0.99)), np.log(10.0)
    res = minimize(_neg_log_lik, init, args=(X, y), method="L-BFGS-B")
    
    lgd_model = {
        "beta_regression": {"coef": res.x[:-1], "phi": np.exp(res.x[-1])},
        "feature_scales": scales,
        "training_columns": colnames,
        "lgd_default": float(y.mean()),
        "method": "Beta Regression v2 (MLE)"
    }

    with open(LGD_SAVE_PATH, "wb") as f:
        pickle.dump(lgd_model, f)
    
    print(f"SUCCESS: Model saved to {LGD_SAVE_PATH}")

if __name__ == "__main__":
    main()
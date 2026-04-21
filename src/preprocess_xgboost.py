"""
preprocess_xgboost.py
---------------------
Minimal preprocessing of raw loan data for XGBoost.

Philosophy
----------
XGBoost does not need WOE encoding, binning, or monotonicity enforcement.
It discovers non-linear patterns and interactions in raw data autonomously
via tree splits. All we need to do is:

  1. Remove post-origination columns (data leakage — same whitelist as woe_etl.py)
  2. Encode string categoricals as integers (XGBoost cannot handle strings)
  3. Leave all numeric columns as-is — including missing values, which XGBoost
     handles natively by learning an optimal default split direction per feature
  4. Align to the EXACT same train/test row split as the LR model so AUC
     comparisons are on the same held-out observations

No binning. No WOE. No feature selection. XGBoost will rank feature importance
itself; weak features get zero splits and cause no harm.

Run order
---------
    python src/woe_etl.py               
    python src/preprocess_xgboost.py   
    python src/train_xgboost.py

Outputs written to scorecard_outputs/
    X_raw_train.csv        Train split — raw numerics + integer-encoded cats
    X_raw_test.csv         Test split
    y_xgb_train.csv        Target for train split (Bad=1, Good=0)
    y_xgb_test.csv         Target for test split
    cat_encoders_xgb.pkl   {col: {category_string: integer}} for inference
"""

import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent.parent
DATA    = "/Users/lindokuhletami/Desktop/Space/data/loan_data_2007_2014(1).csv"
OUTPUTS = ROOT / "scorecard_outputs"

# ── Post-origination leakage columns — identical to woe_etl.py ───────────────
POST_ORIGINATION_COLS = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low", "collections_12_mths_ex_med", "policy_code",
    "application_type", "annual_inc_joint", "dti_joint",
    "verification_status_joint", "acc_now_delinq", "tot_coll_amt",
    "tot_cur_bal", "open_acc_6m", "open_il_6m", "open_il_12m",
    "open_il_24m", "mths_since_rcnt_il", "total_bal_il", "il_util",
    "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util",
    "total_rev_hi_lim", "inq_fi", "total_cu_tl", "inq_last_12m",
    "loan_status",
]

BAD_STATUSES = {"Charged Off", "Default", "Late (31-120 days)"}


def build_target(df: pd.DataFrame) -> pd.Series:
    # Bad=1, Good=0  (XGBoost positive class = the minority default class)
    return df["loan_status"].isin(BAD_STATUSES).astype(int)


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all object/category columns as integers.
    NaN stays NaN — XGBoost handles missing values natively.
    Returns the encoded DataFrame and a {col: {label: code}} dict for inference.
    """
    encoders = {}
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        # Build mapping from unique non-null string values to integers starting at 0
        unique_vals = df[col].dropna().unique()
        mapping = {v: i for i, v in enumerate(sorted(unique_vals))}
        encoders[col] = mapping
        df[col] = df[col].map(mapping)   # unmapped (NaN) stays NaN
    return df, encoders


def drop_high_cardinality(df: pd.DataFrame, threshold: int = 200) -> pd.DataFrame:
    """
    Drop columns with more unique values than threshold after encoding.
    High-cardinality ID-like columns (emp_title, url, desc) add noise.
    """
    to_drop = [c for c in df.select_dtypes(include=["object"]).columns
               if df[c].nunique() > threshold]
    if to_drop:
        print(f"  Dropping {len(to_drop)} high-cardinality columns: {to_drop}")
    return df.drop(columns=to_drop)


def main():
    print("Loading raw data...")
    df = pd.read_csv(DATA, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # ── Build target before dropping loan_status ──────────────────────────────
    print("Building target (Bad=1, Good=0)...")
    y_full = build_target(df)

    # ── Drop leakage + target column ──────────────────────────────────────────
    drop_cols = [c for c in POST_ORIGINATION_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"  After dropping {len(drop_cols)} post-origination columns: {df.shape}")

    # ── Drop high-cardinality text columns ────────────────────────────────────
    df = drop_high_cardinality(df)

    # ── Drop columns that are >60% missing ───────────────────────────────────
    missing_pct = df.isnull().mean()
    mostly_missing = missing_pct[missing_pct > 0.60].index.tolist()
    if mostly_missing:
        print(f"  Dropping {len(mostly_missing)} columns with >60% missing: {mostly_missing}")
    df = df.drop(columns=mostly_missing)

    # ── Encode categoricals ───────────────────────────────────────────────────
    print("Encoding categorical columns as integers...")
    df, cat_encoders = encode_categoricals(df)
    n_cat = len(cat_encoders)
    print(f"  Encoded {n_cat} categorical columns")

    # ── Align with woe_etl.py train/test split ────────────────────────────────
    # woe_etl.py dropped rows (missing targets, etc.) and saved X_train.csv /
    # X_test.csv with the surviving row indices. We use those exact indices so
    # the XGBoost model is evaluated on the identical held-out observations as LR.
    print("Loading train/test indices from woe_etl.py outputs...")
    if not (OUTPUTS / "X_train.csv").exists():
        raise FileNotFoundError(
            "X_train.csv not found. Run woe_etl.py first:\n"
            "    python src/woe_etl.py"
        )
    train_idx = pd.read_csv(OUTPUTS / "X_train.csv", index_col=0, usecols=[0]).index
    test_idx  = pd.read_csv(OUTPUTS / "X_test.csv",  index_col=0, usecols=[0]).index

    # Filter raw df to only the rows woe_etl.py kept
    all_idx    = train_idx.union(test_idx)
    df_clean   = df.loc[df.index.intersection(all_idx)]
    y_clean    = y_full.loc[df_clean.index]

    X_train = df_clean.loc[df_clean.index.intersection(train_idx)]
    X_test  = df_clean.loc[df_clean.index.intersection(test_idx)]
    y_train = y_clean.loc[X_train.index]
    y_test  = y_clean.loc[X_test.index]

    assert len(X_train) == len(y_train), "Train X/y row mismatch"
    assert len(X_test)  == len(y_test),  "Test  X/y row mismatch"

    # ── Variance check ────────────────────────────────────────────────────────
    zero_var = [c for c in X_train.columns if X_train[c].nunique(dropna=True) <= 1]
    if zero_var:
        print(f"  Dropping {len(zero_var)} zero-variance columns: {zero_var}")
        X_train = X_train.drop(columns=zero_var)
        X_test  = X_test.drop(columns=zero_var)

    print(f"\n  Final feature count : {X_train.shape[1]}")
    print(f"  Train rows          : {len(X_train):,}")
    print(f"  Test  rows          : {len(X_test):,}")
    print(f"  Bad rate (train)    : {float(y_train.mean()):.2%}")
    print(f"  Bad rate (test)     : {float(y_test.mean()):.2%}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nSaving outputs to scorecard_outputs/...")
    X_train.to_csv(OUTPUTS / "X_raw_train.csv", index=True)
    X_test.to_csv( OUTPUTS / "X_raw_test.csv",  index=True)
    y_train.to_csv(OUTPUTS / "y_xgb_train.csv", index=True, header=True)
    y_test.to_csv( OUTPUTS / "y_xgb_test.csv",  index=True, header=True)

    with open(OUTPUTS / "cat_encoders_xgb.pkl", "wb") as f:
        pickle.dump(cat_encoders, f)

    print("\n── preprocess_xgboost.py complete ──────────────────────────────")
    print("  scorecard_outputs/X_raw_train.csv")
    print("  scorecard_outputs/X_raw_test.csv")
    print("  scorecard_outputs/y_xgb_train.csv")
    print("  scorecard_outputs/y_xgb_test.csv")
    print("  scorecard_outputs/cat_encoders_xgb.pkl")
    print("\n  Next step: python src/train_xgboost.py")


if __name__ == "__main__":
    main()
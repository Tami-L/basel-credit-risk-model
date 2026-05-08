from __future__ import annotations

import logging
import pickle
import warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

# ============================================================================
# IRB CONFIGURATION (RELAXED FOR DISCRIMINATION)
# ============================================================================
TARGET_COL = "good_bad"
# Relaxed thresholds to recover "missing" signal
IV_DROP_BELOW = 0.005         # Recover marginal predictors
CORR_THRESHOLD = 0.9         # Allow more complementary signal
VIF_THRESHOLD = 15         # Standard bank-grade limit
DEFAULT_N_BINS = 8            # Slightly fewer bins for higher stability/volatility balance
MIN_BIN_PCT = 0.03            # Smaller bins allowed to capture tail-end risk
WOE_CLIP = 4               # Allow slightly more leverage in extreme bins
SMOOTHING = 0.2               # Laplacian smoothing for WoE stability

OUTPUT_DIR = Path("scorecard_outputs")

RAW_INPUT_FEATURES = {
    "loan_amnt", "term", "int_rate", "installment", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status", 
    "purpose", "addr_state", "dti", "fico_range_high", "fico_range_low",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", 
    "revol_bal", "revol_util", "total_acc", "mths_since_last_delinq"
}

# ============================================================================
# SECTION 1 — SUPERVISED FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced interactions to capture non-linear risk surfaces."""
    df = df.copy()
    cols = df.columns.tolist()
    
    # 1. Basel Affordability Interactions (The 'Payment Stress' indicator)
    if 'annual_inc' in cols and 'installment' in cols:
        safe_inc = df['annual_inc'].replace(0, np.nan)
        df['inst_to_inc_ratio'] = (df['installment'] * 12) / safe_inc
    
    # 2. Risk Layering: DTI vs Credit Quality
    if 'dti' in cols and 'fico_range_low' in cols:
        # High DTI is more dangerous at Low FICO
        df['dti_x_fico_risk'] = df['dti'] / (df['fico_range_low'] + 1)
        
    # 3. Utilization Velocity (revol_util x inq_last_6mths)
    if 'revol_util' in cols and 'inq_last_6mths' in cols:
        util = pd.to_numeric(df['revol_util'].astype(str).str.replace('%', ''), errors='coerce') / 100
        df['credit_hunger_index'] = util * (df['inq_last_6mths'] + 1)

    # 4. Credit Age Proxy
    if 'total_acc' in cols and 'open_acc' in cols:
        df['acc_util_ratio'] = df['open_acc'] / (df['total_acc'] + 1)

    return df

# ============================================================================
# SECTION 2 — SUPERVISED BINNING (THE GINI BOOST)
# ============================================================================

def get_tree_bins(series: pd.Series, y: pd.Series, n_bins: int) -> np.ndarray:
    """Use Decision Trees to find split points that maximize KS separation."""
    df = pd.DataFrame({'x': series, 'y': y}).dropna()
    if df['x'].nunique() <= 2: return np.array([-np.inf, np.inf])
    
    tree = DecisionTreeClassifier(max_leaf_nodes=n_bins, min_samples_leaf=MIN_BIN_PCT)
    tree.fit(df[['x']], df['y'])
    
    # Extract thresholds from tree
    thresholds = np.sort(tree.tree_.threshold[tree.tree_.threshold != -2])
    return np.concatenate([[-np.inf], thresholds, [np.inf]])

def bin_variable(series: pd.Series, y: pd.Series, is_numeric: bool) -> dict:
    """Supervised WoE calculation with stability smoothing."""
    if is_numeric:
        edges = get_tree_bins(series, y, DEFAULT_N_BINS)
        binned = np.digitize(series.fillna(-999), edges[1:-1])
        df_tmp = pd.DataFrame({"bin": np.where(series.isna(), -1, binned), "target": y})
    else:
        mapped = series.astype(str).replace('nan', 'Missing')
        df_tmp = pd.DataFrame({"bin": mapped, "target": y})

    res = df_tmp.groupby("bin")["target"].agg(
        count="count", 
        n_bad=lambda s: (s == 0).sum(), 
        n_good=lambda s: (s == 1).sum()
    ).reset_index()
    
    # Laplacian Smoothing: prevents infinite WoE in small bins
    n_g_total, n_b_total = y.sum(), (y == 0).sum()
    res["woe"] = np.log(((res["n_good"] + SMOOTHING) / n_g_total) / 
                        ((res["n_bad"] + SMOOTHING) / n_b_total)).clip(-WOE_CLIP, WOE_CLIP)
    
    res["iv"] = (((res["n_good"] / n_g_total) - (res["n_bad"] / n_b_total)) * res["woe"])
    
    return {
        "rules": res.set_index("bin")["woe"].to_dict(), 
        "iv": res["iv"].sum(), 
        "edges": edges if is_numeric else None, 
        "is_numeric": is_numeric
    }

# ============================================================================
# SECTION 3 — PIPELINE EXECUTION
# ============================================================================

def _apply_woe_transform(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    X_woe = pd.DataFrame(index=df.index)
    for col, meta in mappings.items():
        if col not in df.columns: continue
        if meta["is_numeric"]:
            binned = np.where(df[col].isna(), -1, np.digitize(df[col].fillna(-999), meta["edges"][1:-1]))
            X_woe[col] = pd.Series(binned, index=df.index).map(meta["rules"]).fillna(0.0)
        else:
            X_woe[col] = df[col].astype(str).map(meta["rules"]).fillna(0.0)
    return X_woe

def run_pipeline(raw_path: str):
    log.info("Starting High-Resolution IRB Pipeline...")
    df_raw = pd.read_csv(raw_path, index_col=0, low_memory=False)
    
    # Create Target
    bad_statuses = {"Charged Off", "Default", "Late (31-120 days)"}
    df_raw[TARGET_COL] = np.where(df_raw["loan_status"].isin(bad_statuses), 0, 1)
    
    train_df, test_df = train_test_split(df_raw, test_size=0.2, random_state=42, stratify=df_raw[TARGET_COL])
    
    # Engineering
    X_train_eng = engineer_features(train_df)
    X_test_eng = engineer_features(test_df)
    
    # Binning & Mapping
    woe_mappings = {}
    iv_results = []
    
    # Dynamically select columns to process
    potential_cols = [c for c in X_train_eng.columns if c in RAW_INPUT_FEATURES or 
                      c in ['inst_to_inc_ratio', 'dti_x_fico_risk', 'credit_hunger_index', 'acc_util_ratio']]

    for col in potential_cols:
        is_num = X_train_eng[col].dtype != 'object'
        try:
            meta = bin_variable(X_train_eng[col], train_df[TARGET_COL], is_numeric=is_num)
            woe_mappings[col] = meta
            iv_results.append({"feature": col, "iv": meta["iv"]})
        except Exception as e:
            log.warning(f"Skipping {col} due to error: {e}")

    iv_summary = pd.DataFrame(iv_results).sort_values("iv", ascending=False)
    
    # Transform
    X_train_woe_full = _apply_woe_transform(X_train_eng, woe_mappings)
    
    # Refined Selection (Correlation & VIF)
    candidates = iv_summary[iv_summary["iv"] >= IV_DROP_BELOW]["feature"].tolist()
    X_final = X_train_woe_full[candidates]
    
    # Drop highly correlated features
    corr_matrix = X_final.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > CORR_THRESHOLD)]
    selected = [c for c in candidates if c not in to_drop]
    
    # Final VIF check
    while True:
        vifs = [variance_inflation_factor(X_final[selected].values, i) for i in range(len(selected))]
        if max(vifs) <= VIF_THRESHOLD: break
        selected.pop(np.argmax(vifs))

    log.info(f"Retained {len(selected)} high-signal features.")
    
    # Save Artifacts
    OUTPUT_DIR.mkdir(exist_ok=True)
    X_final[selected].to_csv(OUTPUT_DIR / "X_train_woe.csv", index=False)
    train_df[TARGET_COL].to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    
    X_test_woe = _apply_woe_transform(X_test_eng, woe_mappings)[selected]
    X_test_woe.to_csv(OUTPUT_DIR / "X_test_woe.csv", index=False)
    test_df[TARGET_COL].to_csv(OUTPUT_DIR / "y_test.csv", index=False)
    
    with open(OUTPUT_DIR / "woe_mappings.pkl", "wb") as f:
        pickle.dump(woe_mappings, f)
        
    print("\nTop Predictors by Information Value (IV):")
    print(iv_summary[iv_summary['feature'].isin(selected)].head(15))

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/lindokuhletami/Desktop/Space/data/loan_data_2007_2014(1).csv"
    run_pipeline(path)
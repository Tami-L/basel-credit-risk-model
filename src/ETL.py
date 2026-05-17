from __future__ import annotations

import logging
import pickle
import warnings
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
# CONFIGURATION
# ============================================================================
TARGET_COL     = "good_bad"
IV_DROP_BELOW  = 0.005
CORR_THRESHOLD = 0.90
VIF_THRESHOLD  = 15.0
DEFAULT_N_BINS = 8
MIN_BIN_SIZE   = 30        # FIX: absolute floor — prevents tiny bins on small segments
MIN_BIN_PCT    = 0.03      # used alongside MIN_BIN_SIZE (whichever is larger wins)
WOE_CLIP       = 4.0
SMOOTHING      = 0.2

OUTPUT_DIR = Path("scorecard_outputs")

RAW_INPUT_FEATURES = {
    "loan_amnt", "term", "int_rate", "installment", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "addr_state", "dti", "fico_range_high", "fico_range_low",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mths_since_last_delinq",
}

ENGINEERED_FEATURES = {
    "inst_to_inc_ratio", "dti_x_fico_risk", "credit_hunger_index", "acc_util_ratio",
}

# ============================================================================
# SECTION 1 — FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    cols = set(df.columns)

    if "annual_inc" in cols and "installment" in cols:
        safe_inc = df["annual_inc"].replace(0, np.nan)
        df["inst_to_inc_ratio"] = (df["installment"] * 12) / safe_inc

    if "dti" in cols and "fico_range_low" in cols:
        df["dti_x_fico_risk"] = df["dti"] / (df["fico_range_low"] + 1)

    if "revol_util" in cols and "inq_last_6mths" in cols:
        util = (
            pd.to_numeric(
                df["revol_util"].astype(str).str.replace("%", "", regex=False),
                errors="coerce",
            ) / 100
        )
        df["credit_hunger_index"] = util * (df["inq_last_6mths"] + 1)

    if "total_acc" in cols and "open_acc" in cols:
        df["acc_util_ratio"] = df["open_acc"] / (df["total_acc"] + 1)

    return df


# ============================================================================
# SECTION 2 — SUPERVISED BINNING
# ============================================================================

def get_tree_bins(series: pd.Series, y: pd.Series, n_bins: int) -> np.ndarray:
    """Decision-tree supervised split-point discovery."""
    df_fit = pd.DataFrame({"x": series, "y": y}).dropna()

    if len(df_fit) < MIN_BIN_SIZE * 2 or df_fit["x"].nunique() <= 2:
        return np.array([-np.inf, np.inf])

    # FIX: min_samples_leaf needs an absolute floor, not just a fraction.
    # A pure float (0.03) on a 200-row segment → 6 obs/leaf — too noisy.
    min_leaf = max(MIN_BIN_SIZE, int(MIN_BIN_PCT * len(df_fit)))

    tree = DecisionTreeClassifier(
        max_leaf_nodes=n_bins,
        min_samples_leaf=min_leaf,   
        max_depth=6,
        criterion="gini",
        random_state=42,
    )
    tree.fit(df_fit[["x"]], df_fit["y"])

    thresholds = np.sort(tree.tree_.threshold[tree.tree_.threshold != -2])
    return np.concatenate([[-np.inf], thresholds, [np.inf]])


def bin_variable(series: pd.Series, y: pd.Series, is_numeric: bool) -> dict:
    """
    Supervised WoE binning with corrected Laplacian smoothing.

    Fixes applied vs original:
      1. Missing values encoded as string "Missing" (not integer -1) so the
         rules dict key is consistent between fit and transform.
      2. Smoothing denominator now accounts for the added pseudo-counts so
         bin percentages sum to 1.
      3. IV uses the same smoothed percentages as WoE — no more mixing raw
         counts with smoothed log-odds.
    """
    edges = None  # will be set for numeric variables

    if is_numeric:
        edges  = get_tree_bins(series, y, DEFAULT_N_BINS)
        valid  = ~series.isna()

        # FIX: encode missing as "Missing" string — not integer -1.
        # Integer -1 may or may not appear as a rules key depending on whether
        # any training rows were missing, causing silent NaN at inference time.
        binned = pd.Series("Missing", index=series.index, dtype=object)
        binned[valid] = np.digitize(
            series[valid].values, edges[1:-1]
        ).astype(str)

        df_tmp = pd.DataFrame({"bin": binned, "target": y})
    else:
        mapped = (
            series.astype(str)
            .replace({"nan": "Missing", "None": "Missing", "<NA>": "Missing"})
        )
        df_tmp = pd.DataFrame({"bin": mapped, "target": y})

    res = (
        df_tmp.groupby("bin")["target"]
        .agg(
            count="count",
            n_bad=lambda s: (s == 0).sum(),
            n_good=lambda s: (s == 1).sum(),
        )
        .reset_index()
    )

    n_g_total = int(y.sum())
    n_b_total = int((y == 0).sum())
    n_bins    = len(res)

    # FIX 1 — Corrected smoothing denominators.
    # Original divided by n_g_total / n_b_total with no adjustment for the
    # added pseudo-counts, so smoothed bin percentages didn't sum to 1 and
    # all WoE values were systematically biased.
    denom_g = n_g_total + SMOOTHING * n_bins
    denom_b = n_b_total + SMOOTHING * n_bins

    pct_good = (res["n_good"] + SMOOTHING) / denom_g
    pct_bad  = (res["n_bad"]  + SMOOTHING) / denom_b

    res["woe"] = np.log(pct_good / pct_bad).clip(-WOE_CLIP, WOE_CLIP)

    # FIX 2 — IV uses the same smoothed percentages as WoE.
    # Original used raw (n_good / n_g_total) for IV but smoothed WoE, so
    # zero-bad/zero-good bins had understated IV and got filtered too aggressively.
    res["iv_component"] = (pct_good - pct_bad) * res["woe"]

    return {
        "rules":      res.set_index("bin")["woe"].to_dict(),
        "iv":         float(res["iv_component"].sum()),
        "edges":      edges,          # None for categoricals
        "is_numeric": is_numeric,
        "bin_stats":  res,            # retained for diagnostics
    }


# ============================================================================
# SECTION 3 — WoE TRANSFORM
# ============================================================================

def _apply_woe_transform(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """
    Apply pre-fitted WoE mappings to a dataframe.
    Unseen bins → 0.0 (population average, conservative).

    FIX: numeric missing values now look up "Missing" string key, consistent
    with how bin_variable encodes them at fit time.
    """
    X_woe = pd.DataFrame(index=df.index)

    for col, meta in mappings.items():
        if col not in df.columns:
            continue

        edges = meta.get("edges")

        if meta["is_numeric"] and edges is not None:
            valid  = ~df[col].isna()
            # FIX: use "Missing" string key, not integer -1
            binned = pd.Series("Missing", index=df.index, dtype=object)
            binned[valid] = np.digitize(
                df.loc[valid, col].values, edges[1:-1]
            ).astype(str)
        else:
            binned = (
                df[col].astype(str)
                .replace({"nan": "Missing", "None": "Missing", "<NA>": "Missing"})
            )

        X_woe[col] = binned.map(meta["rules"]).fillna(0.0)

    return X_woe


# ============================================================================
# SECTION 4 — FEATURE SELECTION (IV-PREFERENTIAL)
# ============================================================================

def select_features(
    X_woe: pd.DataFrame,
    iv_summary: pd.DataFrame,
) -> list[str]:
    """
    Three-stage feature selection: IV filter → correlation → VIF.

    FIX (correlation): original dropped the last column in any correlated pair
    regardless of predictive value. Now always drops the lower-IV partner.

    FIX (VIF): original popped by argmax index which could remove a high-IV
    feature when a low-IV feature had equal VIF. Now drops the lowest-IV
    violator among those exceeding the threshold.
    """
    iv_dict = iv_summary.set_index("feature")["iv"].to_dict()

    # Stage 1 — IV filter
    candidates = [
        f for f in iv_summary[iv_summary["iv"] >= IV_DROP_BELOW]["feature"]
        if f in X_woe.columns
    ]

    if not candidates:
        log.warning("No features survived IV filter — returning all WoE columns.")
        return X_woe.columns.tolist()

    # Stage 2 — Correlation (drop lower-IV partner)
    corr  = X_woe[candidates].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop: set = set()
    for col in upper.columns:
        partners = upper.index[upper[col] > CORR_THRESHOLD].tolist()
        for partner in partners:
            if col in to_drop or partner in to_drop:
                continue
            # FIX: keep whichever has higher IV
            loser = col if iv_dict.get(col, 0) < iv_dict.get(partner, 0) else partner
            to_drop.add(loser)

    selected = [c for c in candidates if c not in to_drop]

    # Stage 3 — VIF (drop lowest-IV violator)
    for _ in range(50):
        if len(selected) <= 1:
            break
        X_sel = X_woe[selected].fillna(0.0)
        try:
            vifs = [variance_inflation_factor(X_sel.values, i) for i in range(len(selected))]
        except Exception:
            break

        if max(vifs) <= VIF_THRESHOLD:
            break

        # FIX: among all features above threshold, drop the one with lowest IV
        violators = [selected[i] for i, v in enumerate(vifs) if v > VIF_THRESHOLD]
        loser = min(violators, key=lambda f: iv_dict.get(f, 0))
        log.debug(f"VIF pruning: removing {loser}")
        selected.remove(loser)

    return selected


# ============================================================================
# SECTION 5 — PIPELINE EXECUTION
# ============================================================================

def run_pipeline(raw_path: str) -> None:
    log.info("Starting IRB ETL Pipeline...")
    df_raw = pd.read_csv(raw_path, index_col=0, low_memory=False)
    log.info(f"Loaded {len(df_raw):,} rows")

    # Target: 0 = bad, 1 = good
    bad_statuses    = {"Charged Off", "Default", "Late (31-120 days)"}
    df_raw[TARGET_COL] = np.where(df_raw["loan_status"].isin(bad_statuses), 0, 1)
    log.info(f"Bad rate: {(df_raw[TARGET_COL] == 0).mean():.2%}")

    train_df, test_df = train_test_split(
        df_raw, test_size=0.2, random_state=42, stratify=df_raw[TARGET_COL]
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    y_train = train_df[TARGET_COL]
    y_test  = test_df[TARGET_COL]

    # Feature engineering
    X_train_eng = engineer_features(train_df)
    X_test_eng  = engineer_features(test_df)

    # WoE binning — fit on train only
    potential_cols = [
        c for c in X_train_eng.columns
        if c in RAW_INPUT_FEATURES or c in ENGINEERED_FEATURES
    ]

    woe_mappings: dict = {}
    iv_results:   list = []

    for col in potential_cols:
        is_num = X_train_eng[col].dtype != object
        try:
            meta = bin_variable(X_train_eng[col], y_train, is_numeric=is_num)
            woe_mappings[col] = meta
            iv_results.append({"feature": col, "iv": meta["iv"]})
        except Exception as e:
            log.warning(f"Skipping {col}: {e}")

    iv_summary = pd.DataFrame(iv_results).sort_values("iv", ascending=False)

    # Transform
    X_train_woe = _apply_woe_transform(X_train_eng, woe_mappings)
    X_test_woe  = _apply_woe_transform(X_test_eng,  woe_mappings)

    # Feature selection
    selected = select_features(X_train_woe, iv_summary)
    log.info(f"Selected {len(selected)} features after IV / corr / VIF filtering")

    # Save artifacts
    OUTPUT_DIR.mkdir(exist_ok=True)

    X_train_woe[selected].to_csv(OUTPUT_DIR / "X_train_woe.csv", index=False)
    X_test_woe[selected].to_csv( OUTPUT_DIR / "X_test_woe.csv",  index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv( OUTPUT_DIR / "y_test.csv",  index=False)

    with open(OUTPUT_DIR / "woe_mappings.pkl", "wb") as f:
        pickle.dump(woe_mappings, f)

    iv_display = iv_summary[iv_summary["feature"].isin(selected)]
    iv_display.to_csv(OUTPUT_DIR / "iv_summary.csv", index=False)

    print("\nTop Predictors by Information Value (IV):")
    print(iv_display.head(15).to_string(index=False))


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/lindokuhletami/Desktop/Space/data/loan_data_2007_2014(1).csv"
    run_pipeline(path)
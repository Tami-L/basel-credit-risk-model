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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
log = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
TARGET_COL      = "good_bad"
IV_DROP_BELOW   = 0.02
CORR_THRESHOLD  = 0.90
VIF_THRESHOLD   = 15.0
DEFAULT_N_BINS  = 8
MIN_BIN_SIZE    = 50
MIN_BIN_PCT     = 0.05
RARE_CAT_THRESH = 0.02
WOE_CLIP        = 4.0
SMOOTHING       = 0.05
# Add this constant near the top of your config
PROTECTED_FEATURES = {"int_rate", "fico_range_low", "dti"}

OUTPUT_DIR = Path("scorecard_outputs")

# sub_grade excluded — lender-assigned risk grade, near-direct proxy for target.
RAW_INPUT_FEATURES = {
    "loan_amnt", "term", "int_rate", "installment",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "addr_state", "dti", "fico_range_high", "fico_range_low",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mths_since_last_delinq",
}

ENGINEERED_FEATURES = {
    "inst_to_inc_ratio",
    "dti_x_fico_risk",
    "credit_hunger_index",
    "acc_util_ratio",
}

BAD_STATUSES = {"Charged Off", "Default", "Late (31-120 days)"}


# ============================================================================
# SECTION 1 — FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct economically motivated interaction features.

    Called after train/test split to prevent test-set information
    from influencing training features.

    Parameters
    ----------
    df : pd.DataFrame — raw loan data.

    Returns
    -------
    pd.DataFrame — copy with additional engineered columns.
    """
    df   = df.copy()
    cols = set(df.columns)

    if {"annual_inc", "installment"} <= cols:
        safe_inc = df["annual_inc"].replace(0, np.nan)
        df["inst_to_inc_ratio"] = (df["installment"] * 12) / safe_inc

    if {"dti", "fico_range_low"} <= cols:
        df["dti_x_fico_risk"] = df["dti"] / (df["fico_range_low"] + 1)

    if {"revol_util", "inq_last_6mths"} <= cols:
        util = (
            pd.to_numeric(
                df["revol_util"].astype(str).str.replace("%", "", regex=False),
                errors="coerce",
            ) / 100
        )
        df["credit_hunger_index"] = util * (df["inq_last_6mths"] + 1)

    if {"total_acc", "open_acc"} <= cols:
        df["acc_util_ratio"] = df["open_acc"] / (df["total_acc"] + 1)

    # Encode emp_length ordinally to preserve natural order.
    if "emp_length" in cols:
        emp_map = {
            "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
            "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
            "8 years": 8, "9 years": 9, "10+ years": 10,
        }
        df["emp_length"] = df["emp_length"].map(emp_map)

    return df


def group_rare_categories(
    series: pd.Series,
    y: pd.Series,
    threshold: float = RARE_CAT_THRESH,
) -> pd.Series:
    """
    Collapse infrequent categories into an 'Other' group.

    Rare categories produce noisy WoE estimates that do not generalise.

    Parameters
    ----------
    series    : pd.Series — categorical feature.
    y         : pd.Series — binary target (unused here, reserved for future
                            bad-rate-based grouping strategies).
    threshold : float — minimum frequency to retain a category.

    Returns
    -------
    pd.Series with rare categories replaced by 'Other'.
    """
    freq = series.value_counts(normalize=True)
    rare = freq[freq < threshold].index
    return series.where(~series.isin(rare), other="Other")


# ============================================================================
# SECTION 2 — SUPERVISED BINNING WITH MONOTONICITY ENFORCEMENT
# ============================================================================

def get_tree_bins(series: pd.Series, y: pd.Series, n_bins: int) -> np.ndarray:
    """
    Discover split points using a supervised decision tree.

    Parameters
    ----------
    series : pd.Series — numeric feature.
    y      : pd.Series — binary target.
    n_bins : int — maximum number of leaf nodes.

    Returns
    -------
    np.ndarray of bin edges including -inf and +inf sentinels.
    """
    df_fit = pd.DataFrame({"x": series, "y": y}).dropna()

    if len(df_fit) < MIN_BIN_SIZE * 2 or df_fit["x"].nunique() <= 2:
        return np.array([-np.inf, np.inf])

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


def compute_bin_woe(
    res: pd.DataFrame,
    n_good_total: int,
    n_bad_total: int,
) -> pd.DataFrame:
    """
    Compute WoE and IV component for a bin stats DataFrame.

    Separated from bin_variable so both bin_variable and
    enforce_monotonic_woe can call it cleanly without circular dependency.

    Parameters
    ----------
    res          : pd.DataFrame — bin stats with columns [bin, count, n_bad, n_good].
    n_good_total : int — total good count in the training set.
    n_bad_total  : int — total bad count in the training set.

    Returns
    -------
    pd.DataFrame — input with woe and iv_component columns added.
    """
    res     = res.copy()
    n_bins  = len(res)
    denom_g = n_good_total + SMOOTHING * n_bins
    denom_b = n_bad_total  + SMOOTHING * n_bins
    pct_g   = (res["n_good"] + SMOOTHING) / denom_g
    pct_b   = (res["n_bad"]  + SMOOTHING) / denom_b
    res["woe"]          = np.log(pct_g / pct_b).clip(-WOE_CLIP, WOE_CLIP)
    res["iv_component"] = (pct_g - pct_b) * res["woe"]
    return res


def enforce_monotonic_woe(
    bin_stats: pd.DataFrame,
    n_good_total: int,
    n_bad_total: int,
) -> pd.DataFrame:
    """
    Merge adjacent bins until WoE is monotonic.

    Regulators require WoE to move consistently in one direction across bins.
    Non-monotonic WoE indicates noise-driven binning and cannot be explained
    to a credit committee.

    The function determines the dominant direction and iteratively merges the
    pair of adjacent bins most responsible for the violation (smallest WoE
    difference) until monotonicity is achieved or only two bins remain.

    Parameters
    ----------
    bin_stats    : pd.DataFrame — raw bin counts [bin, count, n_bad, n_good].
    n_good_total : int — total good count in the training set.
    n_bad_total  : int — total bad count in the training set.

    Returns
    -------
    pd.DataFrame — merged bin stats with monotonic WoE.
    """
    stats = bin_stats.copy()

    missing_mask = stats["bin"].astype(str) == "Missing"
    missing_row  = stats[missing_mask].copy()
    stats        = stats[~missing_mask].reset_index(drop=True)

    if len(stats) <= 2:
        return pd.concat([stats, missing_row]).reset_index(drop=True)

    # Determine dominant direction from initial WoE values.
    stats = compute_bin_woe(stats, n_good_total, n_bad_total)
    diffs = np.diff(stats["woe"].values)
    dominant_increasing = (diffs > 0).sum() >= (diffs < 0).sum()

    for _ in range(50):
        stats      = compute_bin_woe(stats, n_good_total, n_bad_total)
        diffs      = np.diff(stats["woe"].values)
        violations = (
            np.where(diffs < 0)[0] if dominant_increasing
            else np.where(diffs > 0)[0]
        )

        if len(violations) == 0:
            break

        # Merge the pair with the smallest absolute WoE difference.
        idx  = violations[np.argmin(np.abs(diffs[violations]))]
        i, j = idx, idx + 1

        merged = {
            "bin":    f"{stats.loc[i, 'bin']}-{stats.loc[j, 'bin']}",
            "count":  stats.loc[i, "count"]  + stats.loc[j, "count"],
            "n_bad":  stats.loc[i, "n_bad"]  + stats.loc[j, "n_bad"],
            "n_good": stats.loc[i, "n_good"] + stats.loc[j, "n_good"],
        }
        stats = pd.concat(
            [stats.iloc[:i], pd.DataFrame([merged]), stats.iloc[j + 1:]],
            ignore_index=True,
        )

        if len(stats) <= 2:
            break

    stats = compute_bin_woe(stats, n_good_total, n_bad_total)
    return pd.concat([stats, missing_row]).reset_index(drop=True)


def bin_variable(series: pd.Series, y: pd.Series, is_numeric: bool) -> dict:
    """
    Supervised WoE binning with monotonicity enforcement and Laplacian smoothing.

    Parameters
    ----------
    series     : pd.Series — feature column.
    y          : pd.Series — binary target (1 = good, 0 = bad).
    is_numeric : bool — True for numeric features, False for categoricals.

    Returns
    -------
    dict with keys:
        rules      — {bin_label: woe_value} for transform step.
        iv         — total Information Value for this feature.
        edges      — bin edges (numeric only, else None).
        is_numeric — passed through for use in transform.
        bin_stats  — full bin-level statistics for diagnostics.
    """
    n_g_total = int(y.sum())
    n_b_total = int((y == 0).sum())
    edges     = None

    if is_numeric:
        edges  = get_tree_bins(series, y, DEFAULT_N_BINS)
        valid  = ~series.isna()
        binned = pd.Series("Missing", index=series.index, dtype=object)
        binned[valid] = np.digitize(
            series[valid].values, edges[1:-1]
        ).astype(str)
        df_tmp = pd.DataFrame({"bin": binned, "target": y})
    else:
        series = group_rare_categories(series.astype(str), y)
        mapped = series.replace({"nan": "Missing", "None": "Missing", "<NA>": "Missing"})
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

    # Enforce monotonicity for numeric features with more than 3 non-missing bins.
    if is_numeric and (res["bin"] != "Missing").sum() > 3:
        res = enforce_monotonic_woe(res, n_good_total=n_g_total, n_bad_total=n_b_total)

    # Final WoE computation — covers categoricals and short numeric bin cases.
    res = compute_bin_woe(res, n_g_total, n_b_total)

    return {
        "rules":      res.set_index("bin")["woe"].to_dict(),
        "iv":         float(res["iv_component"].sum()),
        "edges":      edges,
        "is_numeric": is_numeric,
        "bin_stats":  res,
    }


# ============================================================================
# SECTION 3 — WoE TRANSFORM
# ============================================================================

def apply_woe_transform(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """
    Apply pre-fitted WoE mappings to a dataframe.

    Unseen bins are assigned WoE = 0.0 (population log-odds — conservative).

    Parameters
    ----------
    df       : pd.DataFrame — feature data (train or test).
    mappings : dict — output of bin_variable keyed by column name.

    Returns
    -------
    pd.DataFrame of WoE-transformed features.
    """
    X_woe = pd.DataFrame(index=df.index)

    for col, meta in mappings.items():
        if col not in df.columns:
            continue

        edges = meta.get("edges")

        if meta["is_numeric"] and edges is not None:
            valid  = ~df[col].isna()
            binned = pd.Series("Missing", index=df.index, dtype=object)
            binned[valid] = np.digitize(
                df.loc[valid, col].values, edges[1:-1]
            ).astype(str)
        else:
            series = group_rare_categories(
                df[col].astype(str),
                pd.Series(np.zeros(len(df)), index=df.index),
            )
            binned = series.replace(
                {"nan": "Missing", "None": "Missing", "<NA>": "Missing"}
            )

        X_woe[col] = binned.map(meta["rules"]).fillna(0.0)

    return X_woe


# ============================================================================
# SECTION 4 — FEATURE SELECTION
# ============================================================================

def select_features(
    X_woe: pd.DataFrame,
    iv_summary: pd.DataFrame,
) -> list[str]:
    """
    Three-stage feature selection: IV filter → correlation → VIF.

    Parameters
    ----------
    X_woe      : pd.DataFrame — WoE-transformed training features.
    iv_summary : pd.DataFrame — columns [feature, iv].

    Returns
    -------
    list[str] of selected feature names.
    """
    iv_dict = iv_summary.set_index("feature")["iv"].to_dict()

    # Stage 1 — IV filter
    candidates = [
        f for f in iv_summary[iv_summary["iv"] >= IV_DROP_BELOW]["feature"]
        if f in X_woe.columns
    ]

    if not candidates:
        log.warning("No features survived IV filter — check target definition and feature set.")
        return X_woe.columns.tolist()

    # Stage 2 — Correlation (retain higher-IV partner)
    corr  = X_woe[candidates].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop: set = set()
    for col in upper.columns:
        partners = upper.index[upper[col] > CORR_THRESHOLD].tolist()
        for partner in partners:
            if col in to_drop or partner in to_drop:
                continue
            # Never drop a protected feature regardless of IV
            if col in PROTECTED_FEATURES:
                to_drop.add(partner)
            elif partner in PROTECTED_FEATURES:
                to_drop.add(col)
            else:
                loser = col if iv_dict.get(col, 0) < iv_dict.get(partner, 0) else partner
                to_drop.add(loser)

    selected = [c for c in candidates if c not in to_drop]

    # Stage 3 — VIF (remove lowest-IV violator at each iteration)
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

        violators = [selected[i] for i, v in enumerate(vifs) if v > VIF_THRESHOLD]
        loser = min(violators, key=lambda f: iv_dict.get(f, 0))
        log.info(f"VIF pruning: removing '{loser}' (VIF={max(vifs):.1f})")
        selected.remove(loser)

    return selected


# ============================================================================
# SECTION 5 — POPULATION STABILITY INDEX
# ============================================================================

def compute_psi(
    train_woe: pd.Series,
    test_woe: pd.Series,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index between train and test distributions.

    PSI < 0.10  — stable.
    PSI 0.10–0.25 — moderate shift, monitor.
    PSI > 0.25  — significant shift, investigate.

    Parameters
    ----------
    train_woe : pd.Series — WoE values from training set.
    test_woe  : pd.Series — WoE values from test set.
    n_bins    : int — number of bins for PSI calculation.

    Returns
    -------
    float — PSI value.
    """
    min_val = min(train_woe.min(), test_woe.min())
    max_val = max(train_woe.max(), test_woe.max())
    bins    = np.linspace(min_val, max_val, n_bins + 1)

    train_counts = np.histogram(train_woe, bins=bins)[0]
    test_counts  = np.histogram(test_woe,  bins=bins)[0]

    train_pct = (train_counts + 0.5) / len(train_woe)
    test_pct  = (test_counts  + 0.5) / len(test_woe)

    return float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))


# ============================================================================
# SECTION 6 — PIPELINE EXECUTION
# ============================================================================

def run_pipeline(raw_path: str) -> None:
    """
    Execute the full IRB scorecard preprocessing pipeline.

    Steps:
        1. Load raw data and define target variable.
        2. Stratified train/test split.
        3. Feature engineering (post-split).
        4. Supervised WoE binning with monotonicity enforcement (train only).
        5. WoE transform applied to train and test.
        6. IV / correlation / VIF feature selection.
        7. PSI stability check.
        8. Export artifacts.

    Parameters
    ----------
    raw_path : str — path to raw loan CSV file.
    """
    log.info("Starting IRB Scorecard Preprocessing Pipeline...")
    df_raw = pd.read_csv(raw_path, index_col=0, low_memory=False)
    log.info(f"Loaded {len(df_raw):,} rows, {df_raw.shape[1]} columns.")

    df_raw[TARGET_COL] = np.where(df_raw["loan_status"].isin(BAD_STATUSES), 0, 1)
    bad_rate = (df_raw[TARGET_COL] == 0).mean()
    log.info(f"Bad rate: {bad_rate:.2%} ({(df_raw[TARGET_COL] == 0).sum():,} bads)")

    train_df, test_df = train_test_split(
        df_raw, test_size=0.2, random_state=42, stratify=df_raw[TARGET_COL]
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    y_train = train_df[TARGET_COL]
    y_test  = test_df[TARGET_COL]

    X_train_eng = engineer_features(train_df)
    X_test_eng  = engineer_features(test_df)

    potential_cols = [
        c for c in X_train_eng.columns
        if c in RAW_INPUT_FEATURES or c in ENGINEERED_FEATURES
    ]

    woe_mappings: dict = {}
    iv_results:   list = []

    for col in potential_cols:
        is_num = pd.api.types.is_numeric_dtype(X_train_eng[col])
        try:
            meta = bin_variable(X_train_eng[col], y_train, is_numeric=is_num)
            woe_mappings[col] = meta
            iv_results.append({"feature": col, "iv": meta["iv"]})
            log.debug(f"Binned '{col}': IV={meta['iv']:.4f}, bins={len(meta['bin_stats'])}")
        except Exception as exc:
            log.warning(f"Skipping '{col}': {exc}")

    iv_summary = (
        pd.DataFrame(iv_results)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )

    X_train_woe = apply_woe_transform(X_train_eng, woe_mappings)
    X_test_woe  = apply_woe_transform(X_test_eng,  woe_mappings)

    selected = select_features(X_train_woe, iv_summary)
    log.info(f"Selected {len(selected)} features after IV / correlation / VIF filtering.")

    # PSI stability check
    psi_results = []
    for col in selected:
        psi = compute_psi(X_train_woe[col], X_test_woe[col])
        psi_results.append({"feature": col, "psi": psi})
        if psi > 0.25:
            log.warning(f"High PSI for '{col}': {psi:.3f} — review bin stability.")

    psi_df = pd.DataFrame(psi_results).sort_values("psi", ascending=False)

    # Export artifacts
    OUTPUT_DIR.mkdir(exist_ok=True)

    X_train_woe[selected].to_csv(OUTPUT_DIR / "X_train_woe.csv", index=False)
    X_test_woe[selected].to_csv( OUTPUT_DIR / "X_test_woe.csv",  index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv( OUTPUT_DIR / "y_test.csv",  index=False)
    psi_df.to_csv( OUTPUT_DIR / "psi_summary.csv", index=False)

    with open(OUTPUT_DIR / "woe_mappings.pkl", "wb") as f:
        pickle.dump(woe_mappings, f)

    iv_display = iv_summary[iv_summary["feature"].isin(selected)]
    iv_display.to_csv(OUTPUT_DIR / "iv_summary.csv", index=False)

    print("\n" + "=" * 60)
    print(f"{'TOP PREDICTORS BY IV':^60}")
    print("-" * 60)
    print(iv_display.head(15).to_string(index=False))
    print("\n" + "=" * 60)
    print(f"{'POPULATION STABILITY INDEX':^60}")
    print("-" * 60)
    print(psi_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/lindokuhletami/Desktop/Space/data/loan_data_2007_2014(1).csv"
    run_pipeline(path)
"""
woe_etl.py
==========
Credit Scorecard ETL Pipeline — WoE / IV Edition
-------------------------------------------------
Transforms raw Lending Club loan data into a WoE-encoded, model-ready dataset
for logistic regression scorecard development.

Pipeline stages
---------------
1.  Data cleaning & target creation
2.  Variable binning  (numerical quantile / monotonic; categorical → grouped)
3.  WoE & IV computation
4.  Feature selection  (IV thresholds + correlation filter)
5.  WoE transformation of the dataset
6.  Artifact persistence
7.  Output validation & reporting

Does NOT train any model.
Does NOT scale or normalise features (WoE replaces this).
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

TARGET_COL = "good_bad"
REFERENCE_DATE = "2017-12-01"

# IV thresholds
IV_DROP_BELOW = 0.02        # useless predictors
IV_WEAK_BELOW = 0.1         # flagged but optionally kept
IV_STRONG_GTE = 0.1         # strong — always kept

# Binning params
DEFAULT_N_BINS = 10         # quantile bins before monotonicity merging
MIN_BIN_OBS = 50            # minimum observations per bin (absolute)
MIN_BIN_PCT = 0.05          # minimum observations per bin (fraction of total)
RARE_CATEGORY_THRESHOLD = 0.02   # group categories below this frequency → "Other"

# Correlation threshold
CORR_THRESHOLD = 0.70

# WoE clipping (avoid ±inf from empty event / non-event bins)
WOE_CLIP = 3.0

# Output directory — set OUTPUT_DIR to your preferred path before running
OUTPUT_DIR = Path("scorecard_outputs")

# ── Origination-time feature whitelist ────────────────────────────────────────
# Only these columns are available at the time a loan application is submitted.
# Post-origination columns (payment history, last_pymnt_d, total_rec_prncp etc.)
# are excluded here to prevent data leakage into the scorecard model.
# Under Basel II IRB and the SA NCA, a PD model must be based solely on
# information available at origination.
ORIGINATION_FEATURES = {
    # Loan terms agreed at origination
    "loan_amnt", "funded_amnt", "funded_amnt_inv", "term", "int_rate",
    "installment", "grade", "sub_grade",
    # Borrower profile at application
    "emp_title", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "title", "desc",
    # Bureau / credit file at application
    "dti", "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq",
    "mths_since_last_record", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "initial_list_status",
    "mths_since_last_major_derog", "collections_12_mths_ex_med",
    "acc_now_delinq", "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
    # Dates available at origination
    "issue_d", "earliest_cr_line",
    # Geography
    "addr_state",
}


# ============================================================================
# SECTION 1 — DATA LOADING & CLEANING
# ============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV; expects an index column at position 0."""
    log.info("Loading data from %s", path)
    df = pd.read_csv(path, index_col=0, low_memory=False)
    log.info("Loaded: %d rows × %d columns", *df.shape)
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive binary target:  1 = Good (repaid),  0 = Bad (default / charged-off).

    Bad statuses include Charged Off, Default, credit-policy violations, and
    seriously delinquent accounts (31-120 days late).
    """
    bad_statuses = {
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
        "Late (31-120 days)",
    }
    df = df.copy()
    df[TARGET_COL] = np.where(df["loan_status"].isin(bad_statuses), 0, 1)
    n_bad  = (df[TARGET_COL] == 0).sum()
    n_good = (df[TARGET_COL] == 1).sum()
    log.info(
        "Target created — Good: %d (%.1f%%)  Bad: %d (%.1f%%)",
        n_good, 100 * n_good / len(df),
        n_bad,  100 * n_bad  / len(df),
    )
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Perform light cleaning and return (feature_df, target_series).

      Numeric columns: left as-is; missing values are handled at binning time
       (they receive their own "Missing" bin).
     Categorical columns: missing values filled with the string "Missing" so
       they are treated as a distinct category.
      Columns with > 95% missing are dropped entirely.
     Constant columns are dropped.
     The original loan_status column is dropped (it was used to build the target).
    """
    df = df.copy()

    # ------------------------------------------------------------------ target
    y = df[TARGET_COL].copy().astype(int)
    assert set(y.unique()).issubset({0, 1}), "Target must be binary 0/1."

    # Drop target + source column so they never appear in features
    cols_to_drop = [TARGET_COL, "loan_status"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # --------------------------------------------------- drop near-empty columns
    missing_rate = df.isnull().mean()
    high_missing = missing_rate[missing_rate > 0.95].index.tolist()
    if high_missing:
        log.info("Dropping %d columns with >95%% missing: %s", len(high_missing), high_missing)
    df = df.drop(columns=high_missing)

    # --------------------------------------------------------- drop constants
    nunique = df.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        log.info("Dropping %d constant columns: %s", len(constant_cols), constant_cols)
    df = df.drop(columns=constant_cols)

    # ─────────────── keep only origination-time features (no leakage) ──────────
    available = [c for c in df.columns if c in ORIGINATION_FEATURES]
    dropped_leakage = [c for c in df.columns if c not in ORIGINATION_FEATURES]
    if dropped_leakage:
        log.info(
            "Dropping %d post-origination / leakage columns: %s",
            len(dropped_leakage), dropped_leakage,
        )
    df = df[available]

    # --------------------------------- fill categorical NaN with "Missing" string
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df[cat_cols] = df[cat_cols].fillna("Missing")

    log.info("After cleaning: %d rows × %d features", *df.shape)
    return df, y


# ============================================================================
# SECTION 2 — VARIABLE BINNING
# ============================================================================

def _coerce_numeric(series: pd.Series) -> pd.Series:
    """ Try to parse a column as numeric; return original if it fails. """
    try:
        return pd.to_numeric(series, errors="raise")
    except (ValueError, TypeError):
        return series


def _group_rare_categories(series: pd.Series, threshold: float = RARE_CATEGORY_THRESHOLD) -> pd.Series:
    """ Replace infrequent categories (below threshold) with 'Other' . """
    freq = series.value_counts(normalize=True, dropna=False)
    rare = freq[freq < threshold].index
    return series.where(~series.isin(rare), other="Other")


def _quantile_bin(series: pd.Series, n_bins: int = DEFAULT_N_BINS) -> pd.Series:
    """
    Assign each observation to a quantile bin label (string).
    Missing values are assigned to a dedicated "Missing" bin.
    """
    missing_mask = series.isnull()
    result = pd.Series("Missing", index=series.index, dtype=object)

    non_missing = series[~missing_mask]
    if non_missing.nunique() <= 1:
        result[~missing_mask] = "bin_0"
        return result

    # pd.qcut can fail with duplicate edges — fallback to fewer bins
    for nb in range(n_bins, 1, -1):
        try:
            binned = pd.qcut(non_missing, q=nb, duplicates="drop", precision=4)
            result[~missing_mask] = binned.astype(str)
            return result
        except Exception:
            continue

    # Last resort: single bin
    result[~missing_mask] = "bin_0"
    return result


def _enforce_monotonic_bins(
    series_binned: pd.Series,
    y: pd.Series,
    min_obs: int = MIN_BIN_OBS,
    min_pct: float = MIN_BIN_PCT,
) -> pd.Series:
    """
    Attempt to merge adjacent bins so the default rate is monotone.
    Also merges bins that are too small.

    Returns the (possibly merged) binned series.
    """
    df_tmp = pd.DataFrame({"bin": series_binned, "target": y})
    non_missing_mask = series_binned != "Missing"
    df_nm = df_tmp[non_missing_mask].copy()

    if df_nm.empty:
        return series_binned

    stats = (
        df_nm.groupby("bin", observed=True)["target"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "bad_rate"})
        .sort_index()
    )

    n_total = len(df_nm)

    def _needs_merge(stats_df: pd.DataFrame) -> bool:
        """Return True if any bin violates size or monotonicity."""
        if len(stats_df) <= 1:
            return False
        too_small = (
            (stats_df["count"] < min_obs) |
            (stats_df["count"] / n_total < min_pct)
        ).any()
        if too_small:
            return True
        rates = stats_df["bad_rate"].values
        diffs = np.diff(rates)
        monotone_up   = (diffs >= 0).all()
        monotone_down = (diffs <= 0).all()
        return not (monotone_up or monotone_down)

    max_passes = 20
    for _ in range(max_passes):
        if not _needs_merge(stats):
            break

        # Find the pair of adjacent bins with the smallest bad_rate difference
        # and merge them
        rates = stats["bad_rate"].values
        idx = stats.index.tolist()

        if len(idx) < 2:
            break

        diffs = np.abs(np.diff(rates))
        merge_pos = int(np.argmin(diffs))

        old_bin = idx[merge_pos + 1]
        new_bin = idx[merge_pos]

        df_nm.loc[df_nm["bin"] == old_bin, "bin"] = new_bin

        stats = (
            df_nm.groupby("bin", observed=True)["target"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "bad_rate"})
            .sort_index()
        )

    # Write merged labels back to the full series
    result = series_binned.copy()
    result[non_missing_mask] = df_nm["bin"]
    return result


def bin_variable(
    series: pd.Series,
    y: pd.Series,
    is_numeric: bool,
    n_bins: int = DEFAULT_N_BINS,
) -> pd.Series:
    """
    Full binning pipeline for one variable.
    Returns a Series of string bin labels.
    """
    if is_numeric:
        binned = _quantile_bin(series, n_bins=n_bins)
        binned = _enforce_monotonic_bins(binned, y)
    else:
        binned = _group_rare_categories(series.astype(str)).astype(str)

    return binned


def bin_all_variables(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """
    Bin every column in X.

    Returns
    -------
    X_binned   : DataFrame of string bin labels
    is_numeric : dict mapping column name → True if numeric
    """
    X_binned   = pd.DataFrame(index=X.index)
    is_numeric: dict[str, bool] = {}

    for col in X.columns:
        series = _coerce_numeric(X[col])
        numeric = pd.api.types.is_numeric_dtype(series)
        is_numeric[col] = numeric
        X_binned[col] = bin_variable(series, y, is_numeric=numeric)

    log.info("Binning complete: %d variables processed", len(X.columns))
    return X_binned, is_numeric


# ============================================================================
# SECTION 3 — WoE & IV COMPUTATION
# ============================================================================

def compute_woe_iv(
    binned_series: pd.Series,
    y: pd.Series,
) -> tuple[pd.DataFrame, float]:
    """
    Compute WoE and IV for a single binned variable.

    Convention used here:
        y = 1    Good  (non-default)
        y = 0   Bad   (default)

    WoE_bin = ln( P(Good | bin) / P(Bad | bin) )
            = ln( (n_good_bin / N_good) / (n_bad_bin / N_bad) )

    (pct_good_bin − pct_bad_bin) × WoE_bin

    Returns
    -------
    woe_table : DataFrame with columns [bin, n_obs, n_good, n_bad,
                                        pct_good, pct_bad, woe, iv_contrib]
    iv        : float — total Information Value for the variable
    """
    df = pd.DataFrame({"bin": binned_series, "target": y})
    stats = (
        df.groupby("bin", observed=True)["target"]
        .agg(
            n_obs="count",
            n_good=lambda s: (s == 1).sum(),
            n_bad =lambda s: (s == 0).sum(),
        )
        .reset_index()
    )

    n_good_total = max((y == 1).sum(), 1)
    n_bad_total  = max((y == 0).sum(), 1)

    # Add small epsilon to avoid log(0)
    eps = 0.5
    stats["pct_good"] = (stats["n_good"] + eps) / (n_good_total + eps * len(stats))
    stats["pct_bad"]  = (stats["n_bad"]  + eps) / (n_bad_total  + eps * len(stats))

    stats["woe"] = np.log(stats["pct_good"] / stats["pct_bad"]).clip(-WOE_CLIP, WOE_CLIP)
    stats["iv_contrib"] = (stats["pct_good"] - stats["pct_bad"]) * stats["woe"]

    iv = stats["iv_contrib"].sum()
    return stats, iv


def compute_all_woe_iv(
    X_binned: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Run WoE / IV computation for every variable.

    Returns
    -------
    woe_tables : {col: woe_table DataFrame}
    iv_summary : DataFrame[feature, iv]  sorted descending
    """
    woe_tables: dict[str, pd.DataFrame] = {}
    iv_records: list[dict] = []

    for col in X_binned.columns:
        tbl, iv = compute_woe_iv(X_binned[col], y)
        tbl.insert(0, "feature", col)
        woe_tables[col] = tbl
        iv_records.append({"feature": col, "iv": round(iv, 6)})

    iv_summary = (
        pd.DataFrame(iv_records)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )

    log.info("WoE / IV computed for %d variables", len(iv_summary))
    return woe_tables, iv_summary


# ============================================================================
# SECTION 4 — FEATURE SELECTION
# ============================================================================

def _iv_selection(
    iv_summary: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split features into three groups based on IV thresholds.

    Returns
    -------
    strong_features : IV >= 0.1
    weak_features   : 0.02 <= IV < 0.1  (flagged, optionally kept)
    dropped_features: IV < 0.02
    """
    strong   = iv_summary[iv_summary["iv"] >= IV_STRONG_GTE]["feature"].tolist()
    weak     = iv_summary[
        (iv_summary["iv"] >= IV_DROP_BELOW) & (iv_summary["iv"] < IV_STRONG_GTE)
    ]["feature"].tolist()
    dropped  = iv_summary[iv_summary["iv"] < IV_DROP_BELOW]["feature"].tolist()
    return strong, weak, dropped


def _correlation_filter(
    X_woe_candidate: pd.DataFrame,
    iv_summary: pd.DataFrame,
    threshold: float = CORR_THRESHOLD,
) -> list[str]:
    """
    Remove correlated features, keeping the one with higher IV.

    Uses Spearman rank correlation on the WoE-encoded values.
    Returns the list of features to keep.
    """
    iv_map = iv_summary.set_index("feature")["iv"].to_dict()
    features = X_woe_candidate.columns.tolist()

    if len(features) < 2:
        return features

    corr_matrix = X_woe_candidate.corr(method="spearman").abs()

    to_drop: set[str] = set()
    for i, fi in enumerate(features):
        if fi in to_drop:
            continue
        for fj in features[i + 1:]:
            if fj in to_drop:
                continue
            if corr_matrix.loc[fi, fj] > threshold:
                # Drop the one with lower IV
                iv_i = iv_map.get(fi, 0)
                iv_j = iv_map.get(fj, 0)
                drop_col = fj if iv_i >= iv_j else fi
                to_drop.add(drop_col)
                log.debug(
                    "Corr(%.2f) between %s and %s → dropping %s",
                    corr_matrix.loc[fi, fj], fi, fj, drop_col,
                )

    selected = [f for f in features if f not in to_drop]
    log.info(
        "Correlation filter removed %d features (threshold=%.2f)",
        len(to_drop), threshold,
    )
    return selected


def select_features(
    X_binned: pd.DataFrame,
    woe_tables: dict[str, pd.DataFrame],
    iv_summary: pd.DataFrame,
    keep_weak: bool = True,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """
    Full feature selection pipeline.

    Parameters
    ----------
    keep_weak   : if True, include weak (0.02–0.1 IV) features after flagging

    Returns
    -------
    selected_features : final list of feature names
    flagged_features  : weak-IV features that were kept (flagged)
    iv_summary        : updated with 'iv_band' column
    """
    strong, weak, dropped = _iv_selection(iv_summary)

    log.info(
        "IV bands — Strong (≥%.2f): %d | Weak (%.2f–%.2f): %d | Dropped (<%.2f): %d",
        IV_STRONG_GTE, len(strong),
        IV_DROP_BELOW, IV_STRONG_GTE, len(weak),
        IV_DROP_BELOW, len(dropped),
    )

    if dropped:
        log.info("Dropped (IV<%.2f): %s", IV_DROP_BELOW, dropped[:20])

    candidate_features = strong + (weak if keep_weak else [])

    # Build a temporary WoE dataset for the candidates only
    X_woe_candidate = _apply_woe_transform(
        X_binned[candidate_features], woe_tables
    )

    # Correlation filter
    selected_features = _correlation_filter(X_woe_candidate, iv_summary)
    flagged = [f for f in selected_features if f in weak]

    # Annotate iv_summary
    iv_band_map: dict[str, str] = {}
    for f in strong:   iv_band_map[f] = "strong"
    for f in weak:     iv_band_map[f] = "weak"
    for f in dropped:  iv_band_map[f] = "dropped"
    iv_summary = iv_summary.copy()
    iv_summary["iv_band"] = iv_summary["feature"].map(iv_band_map)

    log.info(
        "Feature selection complete: %d features selected (%d flagged as weak)",
        len(selected_features), len(flagged),
    )
    return selected_features, flagged, iv_summary


# ============================================================================
# SECTION 5 — WoE TRANSFORMATION
# ============================================================================

def _apply_woe_transform(
    X_binned: pd.DataFrame,
    woe_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Replace each bin label with its WoE value.

    Unknown bins (e.g. from unseen categories) receive WoE = 0.
    """
    X_woe = pd.DataFrame(index=X_binned.index)

    for col in X_binned.columns:
        if col not in woe_tables:
            log.warning("No WoE table found for %s — skipping", col)
            continue
        woe_map = woe_tables[col].set_index("bin")["woe"].to_dict()
        X_woe[col] = X_binned[col].map(woe_map).fillna(0.0)

    return X_woe


def transform_to_woe(
    X_binned: pd.DataFrame,
    woe_tables: dict[str, pd.DataFrame],
    selected_features: list[str],
) -> pd.DataFrame:
    """
    Apply WoE transformation to the selected features only.

    Returns a DataFrame where every column is a WoE-encoded numeric feature,
    with column names preserved (no renaming needed; values speak for themselves).
    """
    X_binned_selected = X_binned[selected_features]
    X_woe = _apply_woe_transform(X_binned_selected, woe_tables)

    # Guarantee numeric dtype and no NaNs
    X_woe = X_woe.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    log.info("WoE transformation applied: shape %s", X_woe.shape)
    return X_woe


# ============================================================================
# SECTION 6 — ARTIFACT PERSISTENCE
# ============================================================================

def save_artifacts(
    X_woe: pd.DataFrame,
    y: pd.Series,
    woe_tables: dict[str, pd.DataFrame],
    iv_summary: pd.DataFrame,
    selected_features: list[str],
    out_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Persist all pipeline artifacts to disk.

    Files saved
    -----------
    X_woe.csv             — WoE-encoded feature matrix
    y.csv                 — aligned binary target
    woe_mappings.pkl      — {feature: woe_table DataFrame}
    iv_summary.csv        — feature × IV × iv_band
    selected_features.pkl — list of selected feature names
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. WoE dataset
    xwoe_path = out_dir / "X_woe.csv"
    X_woe.to_csv(xwoe_path, index=False)
    log.info("Saved X_woe      → %s  (%d rows × %d cols)", xwoe_path, *X_woe.shape)

    # 2. Target
    y_path = out_dir / "y.csv"
    y.to_frame(name=TARGET_COL).to_csv(y_path, index=False)
    log.info("Saved y          → %s", y_path)

    # 3. WoE mappings (full woe_tables dict)
    woe_path = out_dir / "woe_mappings.pkl"
    with open(woe_path, "wb") as fh:
        pickle.dump(woe_tables, fh)
    log.info("Saved woe_mappings → %s", woe_path)

    # 4. IV summary
    iv_path = out_dir / "iv_summary.csv"
    iv_summary.to_csv(iv_path, index=False)
    log.info("Saved iv_summary → %s", iv_path)

    # 5. Selected features
    sf_path = out_dir / "selected_features.pkl"
    with open(sf_path, "wb") as fh:
        pickle.dump(selected_features, fh)
    log.info("Saved selected_features → %s  (%d features)", sf_path, len(selected_features))


# ============================================================================
# SECTION 7 — OUTPUT VALIDATION & REPORTING
# ============================================================================

def validate_and_report(
    X_woe: pd.DataFrame,
    y: pd.Series,
    iv_summary: pd.DataFrame,
    original_n_features: int,
    flagged_features: list[str],
) -> None:
    """
    Run post-pipeline checks and print a human-readable summary.
    Raises AssertionError if critical checks fail.
    """
    print("\n" + "=" * 65)
    print("  PIPELINE VALIDATION REPORT")
    print("=" * 65)

    # --- Dimensionality
    print(f"\n  Features: {original_n_features} raw  →  {X_woe.shape[1]} selected")
    print(f"  Observations : {X_woe.shape[0]:,}")

    # --- NaN check
    nan_counts = X_woe.isnull().sum()
    assert nan_counts.sum() == 0, f"NaNs found in X_woe:\n{nan_counts[nan_counts > 0]}"
    print("\n   No NaN values in X_woe")

    # --- Numeric check
    non_numeric = [c for c in X_woe.columns if not pd.api.types.is_numeric_dtype(X_woe[c])]
    assert len(non_numeric) == 0, f"Non-numeric columns found: {non_numeric}"
    print("   All features are numeric")

    # --- Target alignment
    assert len(X_woe) == len(y), "X_woe and y row counts do not match!"
    assert set(y.unique()).issubset({0, 1}), "Target is not binary!"
    print("    Target aligned and binary")

    # --- Flagged features
    if flagged_features:
        print(f"\n   Flagged (weak IV, kept): {flagged_features}")

    # --- Top 10 by IV
    top10 = iv_summary[iv_summary["feature"].isin(X_woe.columns)].head(10)
    print("\n  Top 10 features by IV (selected only):")
    print(
        top10[["feature", "iv", "iv_band"]]
        .to_string(index=False, float_format="%.4f")
    )
    print("\n" + "=" * 65 + "\n")

    log.info("Validation passed ")


# ============================================================================
# INFERENCE HELPERS  (reuse mappings on new data)
# ============================================================================

def load_woe_artifacts(
    out_dir: Path = OUTPUT_DIR,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    Load persisted WoE mappings and selected features for inference.

    Returns
    -------
    woe_tables        : {feature: woe_table}
    selected_features: list[str]
    """
    with open(out_dir / "woe_mappings.pkl", "rb") as fh:
        woe_tables: dict[str, pd.DataFrame] = pickle.load(fh)
    with open(out_dir / "selected_features.pkl", "rb") as fh:
        selected_features: list[str] = pickle.load(fh)
    log.info(
        "Loaded %d WoE tables and %d selected features from %s",
        len(woe_tables), len(selected_features), out_dir,
    )
    return woe_tables, selected_features


def transform_inference(
    X_new: pd.DataFrame,
    woe_tables: dict[str, pd.DataFrame],
    selected_features: list[str],
    is_numeric: dict[str, bool] | None = None,
    y_new: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Apply saved WoE mappings to a new dataset (inference / test set).

    Parameters
    ----------
    X_new            : raw feature DataFrame (same schema as training data)
    woe_tables       : loaded via load_woe_artifacts()
    selected_features : loaded via load_woe_artifacts()
    is_numeric      : optional dict from training-time bin_all_variables()
                        (if None, dtype is inferred from X_new)
    y_new            : optional target; only used for binning monotonicity if
                        is_numeric is provided

    Returns
    -------
    X_woe_new : WoE-transformed DataFrame aligned to selected_features
    """
    # Fill categoricals with "Missing"
    X_new = X_new.copy()
    cat_cols = X_new.select_dtypes(include=["object", "category"]).columns
    X_new[cat_cols] = X_new[cat_cols].fillna("Missing")

    # Bin each selected feature using the stored WoE table bin labels as a
    # look-up (we don't re-fit bins; we map via the woe_table directly)
    X_binned_new = pd.DataFrame(index=X_new.index)

    for col in selected_features:
        if col not in X_new.columns:
            log.warning("Column %s not found in new data — filling WoE=0", col)
            X_binned_new[col] = "Missing"
            continue

        tbl = woe_tables[col]
        # Determine whether this column is numeric
        if is_numeric is not None:
            numeric = is_numeric.get(col, False)
        else:
            numeric = pd.api.types.is_numeric_dtype(X_new[col])

        if numeric:
            # Rebuild the bin cuts from the WoE table bin labels (IntervalIndex strings)
            X_binned_new[col] = _bin_numeric_inference(
                _coerce_numeric(X_new[col]), tbl
            )
        else:
            X_binned_new[col] = _group_rare_categories(X_new[col].astype(str))

    X_woe_new = _apply_woe_transform(X_binned_new, woe_tables)
    X_woe_new = X_woe_new.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X_woe_new


def _bin_numeric_inference(
    series: pd.Series,
    woe_table: pd.DataFrame,
) -> pd.Series:
    """
    Assign bins to new numeric data using the cuts stored in the WoE table.

    Bin labels from training are Interval strings such as "(1.23, 4.56]".
    We parse those back into cut-points and use pd.cut to assign bins.
    Unknown or missing values fall into "Missing".
    """
    import re

    missing_mask = series.isnull()
    result = pd.Series("Missing", index=series.index, dtype=object)

    # Extract bin boundaries from stored labels
    known_bins = [b for b in woe_table["bin"].unique() if b != "Missing"]
    # Try to parse "(lo, hi]" or "(-inf, hi]" or "(lo, inf]" style labels
    cuts: list[float] = []
    for label in known_bins:
        m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|inf", label)
        cuts.extend(float(v) for v in m)

    if not cuts:
        # fallback: no parseable cuts, assign first non-missing bin
        if known_bins:
            result[~missing_mask] = known_bins[0]
        return result

    unique_cuts = sorted(set(cuts))
    unique_cuts = [-np.inf] + [c for c in unique_cuts if not np.isinf(c)] + [np.inf]

    try:
        binned = pd.cut(
            series[~missing_mask],
            bins=unique_cuts,
            include_lowest=True,
            right=True,
        )
        result[~missing_mask] = binned.astype(str)
    except Exception:
        result[~missing_mask] = known_bins[0] if known_bins else "bin_0"

    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(
    raw_path: str,
    out_dir: str | Path = OUTPUT_DIR,
    keep_weak_iv: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Execute the full ETL pipeline end-to-end.

    Parameters
    ----------
    raw_path      : path to the raw CSV file
    out_dir       : directory where all artifacts will be saved
    keep_weak_iv  : whether to include 0.02–0.1 IV features (flagged)

    Returns
    -------
    X_woe : WoE-encoded feature DataFrame
    y     : aligned binary target Series
    """
    out_dir = Path(out_dir)

    # -------------------------------------------------------------- 1. Load
    df_raw = load_data(raw_path)

    # -------------------------------------------------------------- 1. Target
    df = create_target(df_raw)

    # -------------------------------------------------------------- 1. Clean
    X, y = clean_data(df)
    original_n_features = X.shape[1]

    # -------------------------------------------------------------- 2. Bin
    log.info("Binning all variables ...")
    X_binned, is_numeric = bin_all_variables(X, y)

    # -------------------------------------------------------------- 3. WoE/IV
    log.info("Computing WoE and IV ...")
    woe_tables, iv_summary = compute_all_woe_iv(X_binned, y)

    # -------------------------------------------------------------- 4. Select
    log.info("Selecting features ...")
    selected_features, flagged, iv_summary = select_features(
        X_binned, woe_tables, iv_summary, keep_weak=keep_weak_iv
    )

    # -------------------------------------------------------------- 5. Transform
    log.info("Transforming dataset to WoE space ...")
    X_woe = transform_to_woe(X_binned, woe_tables, selected_features)

    # -------------------------------------------------------------- 6. Save
    log.info("Saving artifacts to %s ...", out_dir)
    save_artifacts(X_woe, y, woe_tables, iv_summary, selected_features, out_dir)

    # -------------------------------------------------------------- 7. Validate
    validate_and_report(
        X_woe, y, iv_summary, original_n_features, flagged
    )

    return X_woe, y


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys

    DATA_PATH = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/Users/lindokuhletami/Desktop/Space/data/loan_data_2007_2014(1).csv"
    )

    X_woe, y = run_pipeline(
        raw_path=DATA_PATH,
        out_dir=OUTPUT_DIR,
        keep_weak_iv=True,
    )

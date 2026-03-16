"""
Expected Loss Script
====================
Combines PD, LGD, and EAD to compute Expected Loss (EL) at loan level
and produces a portfolio-level summary.

    EL = PD × LGD × EAD

Inputs  (all produced by earlier pipeline steps)
-------
data/loan_data_inputs_test.csv   — preprocessed model features
data/ead_test.csv                — EAD per loan (funded_amnt - total_pymnt)
src/pd_model.sav                 — trained PD logistic regression
src/pd_model_features.pkl        — significant feature list
src/lgd_model.pkl                — LGD lookup table by grade

Output
------
src/el_results.pkl  — dict with keys:
    "loan_level"  : pd.DataFrame  — one row per loan with PD, LGD, EAD, EL
    "summary"     : pd.DataFrame  — EL aggregated by grade
    "portfolio"   : dict          — scalar portfolio-level metrics
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR       = "/Users/lindokuhletami/Desktop/Space/basel-credit-risk-model/data"
MODEL_DIR      = "/Users/lindokuhletami/Desktop/Space/basel-credit-risk-model/src"

INPUTS_PATH    = f"{DATA_DIR}/loan_data_inputs_test.csv"
EAD_PATH       = f"{DATA_DIR}/ead_test.csv"
PD_MODEL_PATH  = f"{MODEL_DIR}/pd_model.sav"
FEATURES_PATH  = f"{MODEL_DIR}/pd_model_features.pkl"
LGD_MODEL_PATH = f"{MODEL_DIR}/lgd_model.pkl"
EL_SAVE_PATH   = f"{MODEL_DIR}/el_results.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_artifacts() -> tuple:
    """Load all saved model artefacts."""
    with open(PD_MODEL_PATH, "rb") as f:
        pd_model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        significant_features = pickle.load(f)
    with open(LGD_MODEL_PATH, "rb") as f:
        lgd_model = pickle.load(f)
    return pd_model, significant_features, lgd_model


def compute_pd(pd_model, X_features: pd.DataFrame) -> np.ndarray:
    """Return predicted PD (probability of default = class 0) for each loan."""
    # Model predicts P(good=1); PD = P(default=0) = 1 - P(good=1)
    p_good = pd_model.predict_proba(X_features)[:, 1]
    return 1 - p_good


def compute_lgd(grade_series: pd.Series, lgd_model: dict) -> pd.Series:
    """Look up LGD for each loan from the grade-based table."""
    return grade_series.map(lgd_model["lgd_by_grade"]).fillna(
        lgd_model["lgd_default"]
    )


def portfolio_summary(df: pd.DataFrame) -> dict:
    """Compute scalar portfolio-level EL metrics."""
    return {
        "total_ead":              df["ead"].sum(),
        "total_el":               df["el"].sum(),
        "el_rate":                df["el"].sum() / df["ead"].sum() if df["ead"].sum() > 0 else 0,
        "mean_pd":                df["pd"].mean(),
        "mean_lgd":               df["lgd"].mean(),
        "mean_ead":               df["ead"].mean(),
        "mean_el":                df["el"].mean(),
        "n_loans":                len(df),
        "n_high_risk":            int((df["pd"] >= 0.5).sum()),
        "high_risk_el_share":     (
            df.loc[df["pd"] >= 0.5, "el"].sum() / df["el"].sum()
            if df["el"].sum() > 0 else 0
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # 1. Load artefacts
    # ------------------------------------------------------------------
    print("Loading model artefacts...")
    pd_model, significant_features, lgd_model = load_artifacts()
    print(f"  PD model:   {type(pd_model)}")
    print(f"  Features:   {len(significant_features)}")
    print(f"  LGD method: {lgd_model['method'][:60]}...")

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    print("\nLoading data...")
    inputs = pd.read_csv(INPUTS_PATH)
    ead    = pd.read_csv(EAD_PATH).squeeze()   # single-column CSV → Series

    print(f"  Inputs shape: {inputs.shape}")
    print(f"  EAD shape:    {ead.shape}")

    if len(inputs) != len(ead):
        raise ValueError(
            f"Row count mismatch: inputs has {len(inputs)} rows, "
            f"EAD has {len(ead)} rows. Re-run ETL.py to regenerate both."
        )

    # ------------------------------------------------------------------
    # 3. Compute PD
    # ------------------------------------------------------------------
    print("\nComputing PD...")
    missing = [c for c in significant_features if c not in inputs.columns]
    if missing:
        raise ValueError(f"Missing significant features in inputs: {missing}")

    X = inputs[significant_features]
    pd_values = compute_pd(pd_model, X)
    print(f"  PD — mean: {pd_values.mean():.4f}, "
          f"min: {pd_values.min():.4f}, max: {pd_values.max():.4f}")

    # ------------------------------------------------------------------
    # 4. Compute LGD
    # ------------------------------------------------------------------
    print("\nComputing LGD...")
    # Extract raw grade from the one-hot encoded inputs
    # grade columns are named "grade:A", "grade:B", etc.
    grade_cols = [c for c in inputs.columns if c.startswith("grade:")]

    if grade_cols:
        # Reconstruct the grade label from one-hot columns
        grade_series = (
            inputs[grade_cols]
            .idxmax(axis=1)                   # e.g. "grade:A"
            .str.replace("grade:", "", regex=False)  # → "A"
        )
    else:
        print("  WARNING: no grade columns found — applying default LGD to all loans.")
        grade_series = pd.Series(["UNKNOWN"] * len(inputs))

    lgd_values = compute_lgd(grade_series, lgd_model)
    print(f"  LGD — mean: {lgd_values.mean():.4f}  "
          f"(Basel II constant: {lgd_model['lgd_default']:.2%})")

    # ------------------------------------------------------------------
    # 5. Compute EL = PD × LGD × EAD
    # ------------------------------------------------------------------
    print("\nComputing EL = PD × LGD × EAD...")
    el_values = pd_values * lgd_values.values * ead.values

    print(f"  EL   — mean: {el_values.mean():.2f}, "
          f"total: {el_values.sum():,.2f}")

    # ------------------------------------------------------------------
    # 6. Assemble loan-level results DataFrame
    # ------------------------------------------------------------------
    loan_level = pd.DataFrame({
        "pd":    pd_values,
        "lgd":   lgd_values.values,
        "ead":   ead.values,
        "el":    el_values,
        "grade": grade_series.values,
    })

    # ------------------------------------------------------------------
    # 7. Grade-level summary
    # ------------------------------------------------------------------
    grade_summary = (
        loan_level
        .groupby("grade")
        .agg(
            n_loans        = ("el",  "count"),
            mean_pd        = ("pd",  "mean"),
            mean_lgd       = ("lgd", "mean"),
            total_ead      = ("ead", "sum"),
            total_el       = ("el",  "sum"),
            mean_el        = ("el",  "mean"),
        )
        .assign(el_rate=lambda d: d["total_el"] / d["total_ead"])
        .sort_index()
    )

    print("\nGrade-level EL summary:")
    print(grade_summary.to_string())

    # ------------------------------------------------------------------
    # 8. Portfolio-level metrics
    # ------------------------------------------------------------------
    portfolio = portfolio_summary(loan_level)

    print("\n" + "=" * 50)
    print("PORTFOLIO SUMMARY")
    print("=" * 50)
    print(f"  Total loans:          {portfolio['n_loans']:,}")
    print(f"  Total EAD:            ${portfolio['total_ead']:>15,.2f}")
    print(f"  Total EL:             ${portfolio['total_el']:>15,.2f}")
    print(f"  EL Rate (EL/EAD):     {portfolio['el_rate']:.4%}")
    print(f"  Mean PD:              {portfolio['mean_pd']:.4%}")
    print(f"  Mean LGD:             {portfolio['mean_lgd']:.4%}")
    print(f"  High-risk loans (PD≥50%): {portfolio['n_high_risk']:,} "
          f"({portfolio['n_high_risk']/portfolio['n_loans']:.2%} of portfolio)")
    print(f"  High-risk EL share:   {portfolio['high_risk_el_share']:.2%}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    el_results = {
        "loan_level": loan_level,
        "summary":    grade_summary,
        "portfolio":  portfolio,
    }

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    with open(EL_SAVE_PATH, "wb") as f:
        pickle.dump(el_results, f)
    print(f"\nEL results saved to: {EL_SAVE_PATH}")


if __name__ == "__main__":
    main()

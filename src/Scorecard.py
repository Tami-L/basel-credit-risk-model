"""
Credit Risk Scorecard Script
=============================
Loads the saved PD model and its significant feature list (produced by
fit_model.py), applies identical feature selection to the data, then
builds and calculates a South-African-scaled credit scorecard.

Scoring method
--------------
Uses the standard PDO (Points to Double Odds) formula:

    Score = Offset - Factor x (intercept + b1*x1 + b2*x2 + ...)

where:
    Factor = PDO / ln(2)
    Offset = Base Score - Factor x ln(Base Odds)

This gives each loan an ABSOLUTE score on the 300-850 scale based on
its actual log-odds -- NOT relative to other loans in the batch.
The same loan always produces the same score regardless of dataset size
or composition.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths — relative to this file so they work on any machine / Streamlit Cloud
# ---------------------------------------------------------------------------
SRC_DIR   = Path(__file__).resolve().parent   # .../src/
ROOT_DIR  = SRC_DIR.parent                    # .../basel-credit-risk-model/
DATA_DIR  = ROOT_DIR / "data"
MODEL_DIR = SRC_DIR

DATA_PATH          = DATA_DIR  / "loan_data_inputs_train.csv"
MODEL_SAVE_PATH    = MODEL_DIR / "pd_model.sav"
FEATURES_SAVE_PATH = MODEL_DIR / "pd_model_features.pkl"

# South African credit score scale parameters
ZA_MIN_SCORE  = 300
ZA_MAX_SCORE  = 850
ZA_BASE_SCORE = 600   # score at base odds
ZA_BASE_ODDS  = 50    # good:bad odds at base score (50:1 = ~2% default rate)
ZA_PDO        = 50    # points to double the odds


# ---------------------------------------------------------------------------
# Scorecard functions — available at import time (used by app.py)
# ---------------------------------------------------------------------------

def build_scorecard(model, feature_names, pdo=50, base_score=600, base_odds=50):
    """
    Build a scorecard using the PDO (Points to Double Odds) scaling method.

    Each feature is assigned a fixed point value:
        raw_points_i = -coefficient_i x Factor

    The intercept contribution is captured in base_points so that:
        score = base_points + sum(raw_points_i * x_i)

    This produces an ABSOLUTE score — the same loan always gets the same
    score, independent of any other loans in the dataset.

    Parameters
    ----------
    model        : fitted sklearn LogisticRegression (or ThresholdClassifier
                   wrapping one — coef_ and intercept_ are read via properties)
    feature_names: list/Index of feature names matching model.coef_ order
    pdo          : points to double the odds (default 50)
    base_score   : score at base_odds (default 600)
    base_odds    : good:bad odds at the base score (default 50)

    Returns
    -------
    scorecard   : DataFrame — one row per feature with its raw_points value
    base_points : scalar — intercept + offset contribution added to every score
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    coefs     = model.coef_[0]
    intercept = model.intercept_[0]

    # coef_ may be shorter than feature_names if zero-variance columns were
    # dropped inside LogisticRegressionWithPValues.fit() — align by slicing.
    feature_names_aligned = list(feature_names)[: len(coefs)]
    if len(feature_names_aligned) < len(coefs):
        raise ValueError(
            f"Cannot align: {len(coefs)} coefficients but only "
            f"{len(list(feature_names))} feature names available."
        )

    scorecard = pd.DataFrame({
        "feature":     feature_names_aligned,
        "coefficient": coefs,
        # Higher coefficient (more risky) = fewer points
        "raw_points":  -coefs * factor,
    })

    # Intercept absorbs the base log-odds; offset shifts to the desired scale
    base_points = offset - factor * intercept

    print(f"Factor:       {factor:.4f}")
    print(f"Offset:       {offset:.4f}")
    print(f"Base points:  {base_points:.2f}")
    print(f"Points range: {scorecard['raw_points'].min():.2f} "
          f"to {scorecard['raw_points'].max():.2f}")

    return scorecard, base_points


def calculate_credit_score(X, scorecard, base_points, min_score=300, max_score=850):
    """
    Calculate credit scores using the absolute PDO formula.

    Score = base_points + sum(raw_points_i * x_i)

    Scores are clipped to [min_score, max_score] and rounded to integers.
    No min-max normalisation — scores are absolute and stable across datasets.

    Parameters
    ----------
    X           : DataFrame aligned to scorecard["feature"] columns
    scorecard   : DataFrame produced by build_scorecard()
    base_points : scalar from build_scorecard()
    min_score   : floor for output scores (default 300)
    max_score   : ceiling for output scores (default 850)

    Returns
    -------
    scores : integer ndarray of credit scores
    """
    # Matrix multiply: (n_loans x n_features) · (n_features,) = (n_loans,)
    feature_matrix = X[scorecard["feature"].tolist()].values
    raw_scores     = feature_matrix @ scorecard["raw_points"].values + base_points

    print(f"Raw score stats — "
          f"Min: {raw_scores.min():.1f}, "
          f"Max: {raw_scores.max():.1f}, "
          f"Mean: {raw_scores.mean():.1f}")

    return np.round(np.clip(raw_scores, min_score, max_score)).astype(int)


def get_risk_tier(score):
    """Map a credit score to a South African risk tier."""
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


def get_approval_decision(score, threshold=592):
    """
    Return an approval decision based on credit score.

    Default threshold is 592 (Tier 2 boundary — Good credit and above
    are approved; Average and below are declined).
    """
    return "Approve" if score >= threshold else "Decline"


# ---------------------------------------------------------------------------
# Standalone execution only — nothing below here runs when app.py imports
# this file. Only the functions above are exposed at import time.
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. Load model and feature list
    print("Loading model and feature list...")
    with open(MODEL_SAVE_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_SAVE_PATH, "rb") as f:
        significant_features = pickle.load(f)
    print(f"Model loaded:    {type(model)}")
    print(f"Features loaded: {len(significant_features)} significant features")

    # 2. Load data and apply the saved feature selection
    print("\nLoading data...")
    X = pd.read_csv(DATA_PATH)
    print(f"Raw data shape: {X.shape}")

    missing = [c for c in significant_features if c not in X.columns]
    if missing:
        raise ValueError(
            f"The following significant features are missing from the data:\n{missing}"
        )
    X = X[significant_features].copy()
    print(f"Filtered shape (significant features only): {X.shape}")

    # 3. Build scorecard
    print("\nBuilding scorecard...")
    scorecard, base_points = build_scorecard(
        model,
        feature_names=significant_features,
        pdo=ZA_PDO,
        base_score=ZA_BASE_SCORE,
        base_odds=ZA_BASE_ODDS,
    )
    print("\nScorecard preview (top 10 by absolute points):")
    print(scorecard.reindex(
        scorecard["raw_points"].abs().sort_values(ascending=False).index
    ).head(10).to_string(index=False))

    # 4. Calculate credit scores
    print("\nCalculating credit scores...")
    scores     = calculate_credit_score(X, scorecard, base_points,
                                        min_score=ZA_MIN_SCORE, max_score=ZA_MAX_SCORE)
    risk_tiers = [get_risk_tier(s)        for s in scores]
    decisions  = [get_approval_decision(s) for s in scores]

    print("\nFirst 10 scores:")
    print(pd.DataFrame({
        "score":     scores[:10],
        "risk_tier": risk_tiers[:10],
        "decision":  decisions[:10],
    }).to_string(index=False))

    # 5. Validate inverse relationship: higher score → lower PD
    pd_values   = model.predict_proba(X)[:, 1]
    correlation = np.corrcoef(scores, pd_values)[0, 1]
    print("\nScore vs PD check (first 10 rows):")
    print(pd.DataFrame({
        "score":     scores[:10],
        "pd":        pd_values[:10].round(4),
        "risk_tier": risk_tiers[:10],
    }).to_string(index=False))
    print(f"\nCorrelation between score and PD: {correlation:.3f}  (should be negative)")

    # 6. Attach scores and report distribution
    X["credit_score"] = scores
    X["risk_tier"]    = risk_tiers
    X["decision"]     = decisions
    print("\nScore distribution (South African scale):")
    print(X["credit_score"].describe())
    print("\nRisk tier distribution:")
    print(X["risk_tier"].value_counts())
    print("\nApproval rate:")
    print(X["decision"].value_counts(normalize=True).map("{:.1%}".format))
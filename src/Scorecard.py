"""
Credit Risk Scorecard Script
=============================
Loads the saved PD model and its significant feature list (produced by
train_pd_model.py), applies identical feature selection to the data,
then builds and calculates a South-African-scaled credit scorecard.
"""

import pickle

import numpy as np
import pandas as pd
import scipy.stats as stat
from sklearn import linear_model


# ---------------------------------------------------------------------------
# Paths — update these to match your environment
# ---------------------------------------------------------------------------

from pathlib import Path

SRC_DIR   = Path(__file__).resolve().parent   # .../src/
ROOT_DIR  = SRC_DIR.parent                    # .../basel-credit-risk-model/
DATA_DIR  = ROOT_DIR / "data"
MODEL_DIR = SRC_DIR

DATA_PATH          = f"{DATA_DIR}/loan_data_inputs_train.csv"
MODEL_SAVE_PATH    = f"{MODEL_DIR}/pd_model.sav"
# Feature list saved by train_pd_model.py — contains only the columns
# that survived the p-value filter, in the exact order the model expects
FEATURES_SAVE_PATH = f"{MODEL_DIR}/pd_model_features.pkl"

# South African credit score scale parameters
ZA_MIN_SCORE  = 300
ZA_MAX_SCORE  = 850
ZA_BASE_SCORE = 600
ZA_BASE_ODDS  = 50
ZA_PDO        = 50


# ---------------------------------------------------------------------------
# Scorecard functions
# ---------------------------------------------------------------------------

def build_scorecard(model, feature_names,
                    pdo=50, base_score=600, base_odds=50):
    """
    Build a scorecard using the PDO (Points to Double Odds) scaling method.

    Parameters
    ----------
    model        : fitted sklearn LogisticRegression
    feature_names: list/Index of feature names matching model.coef_ order
    pdo          : points needed to double the odds (default 50)
    base_score   : score assigned at base_odds (default 600)
    base_odds    : odds (good/bad) at the base score (default 50)

    Returns
    -------
    scorecard      : DataFrame with one row per feature and its raw points
    base_points_raw: scalar — points contributed by the intercept
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    coefs     = model.coef_[0]
    intercept = model.intercept_[0]

    # The sklearn model may have internally dropped zero-variance columns
    # during training (inside LogisticRegressionWithPValues.fit()), so
    # coef_ can be shorter than the full significant_features list.
    # Align by slicing feature names to match the number of coefficients.
    feature_names_aligned = list(feature_names)[: len(coefs)]
    if len(feature_names_aligned) < len(coefs):
        raise ValueError(
            f"Cannot align: {len(coefs)} coefficients but only "
            f"{len(list(feature_names))} feature names available."
        )

    scorecard = pd.DataFrame({
        "feature":     feature_names_aligned,
        "coefficient": coefs,
    })

    # Higher risk (positive coefficient toward bad) → fewer points
    scorecard["raw_points"] = -coefs * factor

    base_points_raw = offset - factor * intercept

    print(f"Raw base points:  {base_points_raw:.2f}")
    print(f"Raw points range: {scorecard['raw_points'].min():.2f} "
          f"to {scorecard['raw_points'].max():.2f}")

    return scorecard, base_points_raw


def calculate_credit_score(X, scorecard, base_points_raw,
                            min_score=300, max_score=850):
    """
    Calculate min-max scaled credit scores from a feature matrix.

    Parameters
    ----------
    X              : DataFrame aligned to the scorecard feature list
    scorecard      : DataFrame produced by build_scorecard()
    base_points_raw: scalar intercept contribution from build_scorecard()
    min_score      : lower bound of the output score range
    max_score      : upper bound of the output score range

    Returns
    -------
    scaled_scores  : integer ndarray of credit scores
    """
    raw_scores = np.zeros(len(X))

    for _, row in scorecard.iterrows():
        raw_scores += X[row["feature"]].values * row["raw_points"]

    raw_scores += base_points_raw

    print(f"Raw score stats — "
          f"Min: {raw_scores.min():.2f}, "
          f"Max: {raw_scores.max():.2f}, "
          f"Mean: {raw_scores.mean():.2f}")

    # Min-max scale to [min_score, max_score]
    r_min, r_max = raw_scores.min(), raw_scores.max()
    if r_min != r_max:
        scaled = min_score + (raw_scores - r_min) * (max_score - min_score) / (r_max - r_min)
    else:
        scaled = np.full(len(raw_scores), (min_score + max_score) // 2, dtype=float)

    return np.round(np.clip(scaled, min_score, max_score)).astype(int)


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


# ---------------------------------------------------------------------------
# 1. Load model and feature list
# ---------------------------------------------------------------------------
print("Loading model and feature list...")

with open(MODEL_SAVE_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURES_SAVE_PATH, "rb") as f:
    significant_features = pickle.load(f)

print(f"Model loaded:    {type(model)}")
print(f"Features loaded: {len(significant_features)} significant features")


# ---------------------------------------------------------------------------
# 2. Load data and apply the saved feature selection
#    — uses exactly the columns that survived the p-value filter,
#      in the exact order the model expects. No manual column lists needed.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 3. Build scorecard
# ---------------------------------------------------------------------------
print("\nBuilding scorecard...")
scorecard, base_points_raw = build_scorecard(
    model,
    feature_names=significant_features,
    pdo=ZA_PDO,
    base_score=ZA_BASE_SCORE,
    base_odds=ZA_BASE_ODDS,
)

print("\nScorecard preview:")
print(scorecard.head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# 4. Calculate credit scores
# ---------------------------------------------------------------------------
print("\nCalculating credit scores...")
scores     = calculate_credit_score(X, scorecard, base_points_raw,
                                    min_score=ZA_MIN_SCORE, max_score=ZA_MAX_SCORE)
risk_tiers = [get_risk_tier(s) for s in scores]

print("\nFirst 10 scores (South African scale):")
print(scores[:10])


# ---------------------------------------------------------------------------
# 5. Validate inverse relationship: higher score → lower PD
# ---------------------------------------------------------------------------
pd_values   = model.predict_proba(X)[:, 1]
correlation = np.corrcoef(scores, pd_values)[0, 1]

print("\nScore vs PD check (first 10 rows):")
print(pd.DataFrame({
    "score":     scores[:10],
    "pd":        pd_values[:10].round(4),
    "risk_tier": risk_tiers[:10],
}).to_string(index=False))

print(f"\nCorrelation between score and PD: {correlation:.3f}  (should be negative)")


# ---------------------------------------------------------------------------
# 6. Attach scores and report distribution
# ---------------------------------------------------------------------------
X["credit_score"] = scores
X["risk_tier"]    = risk_tiers

print("\nScore distribution (South African scale):")
print(X["credit_score"].describe())

print("\nRisk tier distribution:")
print(X["risk_tier"].value_counts())

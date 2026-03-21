"""
scorecard.py
============
Builds and calculates a credit scorecard calibrated to the South African
market, using the FICO Score 6 / TransUnion SA 300-850 scale and band
definitions, with PDO parameters grounded in the SA unsecured credit
default environment.

South African market context
-----------------------------
- Scale       : 300–850  (FICO Score 6 / TransUnion SA standard)
- Default rate: ~10–11%  (NCR data; our training set is 10.9%)
  → Base odds ≈ 8:1  (8 good for every 1 bad at the anchor score)
- PDO         : 20 points  (tighter than the generic 50; appropriate for a
                high-default-rate market where score bands are narrower)
- Base score  : 620  (mid-Favourable tier; where a 8:1 odds borrower sits)

Score bands (TransUnion SA — aligned to FICO Score 6)
------------------------------------------------------
  Excellent    : 767 – 850
  Good         : 681 – 766
  Favourable   : 614 – 680   ← approval cut-off default: 614
  Average      : 583 – 613
  Below Average: 527 – 582
  Unfavourable : 487 – 526
  Poor         : 300 – 486

NCR / National Credit Act compliance note
------------------------------------------
Under the NCA (Act 34 of 2005), lenders must conduct an affordability
assessment before granting credit. A credit score alone is not sufficient
for approval — it is one input into the decision. The approval threshold
and risk tiers here reflect credit risk only; affordability, income, and
debt-service ratio must be assessed separately as required by the NCA.

Artifacts consumed (produced by train.py)
------------------------------------------
    scorecard_outputs/model.pkl           — fitted LogisticRegression
    scorecard_outputs/feature_names.pkl   — WoE feature names
    scorecard_outputs/X_train.csv         — WoE-encoded features
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR   = Path(__file__).resolve().parent   # .../src/
ROOT_DIR  = SRC_DIR.parent                    # .../basel-credit-risk-model/
MODEL_DIR = ROOT_DIR / "scorecard_outputs"

MODEL_PATH    = MODEL_DIR / "model.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.pkl"
DATA_PATH     = MODEL_DIR / "X_train.csv"

# ── SA market PDO parameters ───────────────────────────────────────────────────
# Base odds of 8:1 reflects ~11% default rate (NCR SA unsecured credit market).
# PDO of 20 is standard for high-default-rate markets — each band is meaningful.
# Base score of 620 anchors the 8:1 borrower in the middle of the Favourable tier.
SA_MIN_SCORE  = 300
SA_MAX_SCORE  = 850
SA_BASE_SCORE = 620
SA_BASE_ODDS  = 8     # ~11% default rate  →  8 goods per 1 bad
SA_PDO        = 20

# ── TransUnion SA / FICO Score 6 bands ────────────────────────────────────────
# Source: TransUnion South Africa (transunion.co.za/education/credit-score)
SA_BANDS = [
    (767, 850, "Excellent"),
    (681, 766, "Good"),
    (614, 680, "Favourable"),
    (583, 613, "Average"),
    (527, 582, "Below Average"),
    (487, 526, "Unfavourable"),
    (300, 486, "Poor"),
]

# Approval cut-off: Favourable and above are considered for approval.
# Lenders typically approve 614+ for unsecured credit in the SA market.
# This must still be combined with an NCA affordability assessment.
SA_APPROVAL_THRESHOLD = 614


# ── Scorecard functions (importable by app.py) ─────────────────────────────────

def build_scorecard(model, feature_names, pdo=SA_PDO, base_score=SA_BASE_SCORE, base_odds=SA_BASE_ODDS):
    """
    Build a scorecard using the PDO (Points to Double Odds) method, scaled
    to the South African FICO Score 6 / TransUnion 300-850 range.

    Formula
    -------
        Factor      = PDO / ln(2)
        Offset      = Base Score - Factor × ln(Base Odds)
        raw_points_i = -coefficient_i × Factor
        base_points  = Offset - Factor × intercept

    The minus sign on the coefficient means:
        positive coef (good signal) → fewer raw_points deducted → higher score
        negative coef (bad signal)  → more raw_points deducted → lower score

    With SA_BASE_ODDS=8 and SA_PDO=20:
        - A borrower at exactly 8:1 odds scores ~620 (mid-Favourable)
        - Doubling the odds to 16:1 adds 20 points → ~640
        - Halving the odds to 4:1 subtracts 20 points → ~600

    Parameters
    ----------
    model         : fitted sklearn LogisticRegression from train.py
    feature_names : list of WoE feature names matching model.coef_ order
    pdo           : points to double the odds (default: SA_PDO = 20)
    base_score    : anchor score at base_odds (default: 620)
    base_odds     : good:bad odds at anchor score (default: 8)

    Returns
    -------
    scorecard   : DataFrame[feature, coefficient, raw_points]
    base_points : scalar added to every loan's score
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    coefs     = model.coef_[0]
    intercept = model.intercept_[0]

    if len(coefs) != len(feature_names):
        raise ValueError(
            f"Mismatch: {len(coefs)} coefficients vs {len(feature_names)} feature names. "
            f"Reload feature_names.pkl from the same train.py run."
        )

    scorecard = pd.DataFrame({
        "feature":     list(feature_names),
        "coefficient": coefs,
        "raw_points":  -coefs * factor,
    })

    base_points = offset - factor * intercept

    print(f"\nScorecard scaling (SA market parameters)")
    print(f"  PDO         : {pdo} pts to double odds")
    print(f"  Base odds   : {base_odds}:1  (~{100 / (base_odds + 1):.0f}% default rate)")
    print(f"  Base score  : {base_score}  (mid-Favourable tier)")
    print(f"  Factor      : {factor:.4f}")
    print(f"  Offset      : {offset:.4f}")
    print(f"  Base points : {base_points:.2f}")
    print(f"  Points range: {scorecard['raw_points'].min():.2f}  to  {scorecard['raw_points'].max():.2f}")

    return scorecard, base_points


def calculate_credit_score(X, scorecard, base_points, min_score=SA_MIN_SCORE, max_score=SA_MAX_SCORE):
    """
    Calculate an absolute SA credit score for each row in X.

        Score = base_points + Σ(raw_points_i × woe_i)

    X must already be WoE-encoded (output of woe_etl.py).
    Scores are clipped to [300, 850] and rounded to integers.

    Parameters
    ----------
    X           : WoE-encoded DataFrame
    scorecard   : from build_scorecard()
    base_points : from build_scorecard()
    min_score   : floor  (300 — SA scale minimum)
    max_score   : ceiling (850 — SA scale maximum)

    Returns
    -------
    scores : integer ndarray on the 300–850 scale
    """
    features = scorecard["feature"].tolist()
    missing  = [f for f in features if f not in X.columns]
    if missing:
        raise ValueError(f"Features missing from X:\n{missing}")

    raw_scores = X[features].values @ scorecard["raw_points"].values + base_points

    print(f"\nRaw score stats (before clipping):")
    print(f"  Min  : {raw_scores.min():.1f}")
    print(f"  Max  : {raw_scores.max():.1f}")
    print(f"  Mean : {raw_scores.mean():.1f}")

    clipped = np.round(np.clip(raw_scores, min_score, max_score)).astype(int)
    n_clipped = ((raw_scores < min_score) | (raw_scores > max_score)).sum()
    if n_clipped > 0:
        print(f"  Note : {n_clipped:,} scores clipped to [{min_score}, {max_score}]")

    return clipped


def get_risk_band(score):
    """
    Map a score to the TransUnion SA / FICO Score 6 risk band.

    Bands sourced from TransUnion South Africa (transunion.co.za).
    """
    for low, high, label in SA_BANDS:
        if low <= score <= high:
            return label
    return "Poor"   # fallback for any score below 300


def get_approval_decision(score, threshold=SA_APPROVAL_THRESHOLD):
    """
    Return an approval recommendation based on the SA credit score.

    Default threshold is 614 (bottom of the Favourable tier). Lenders
    typically consider Favourable and above for unsecured credit.

    IMPORTANT: Under the NCA, this score-based decision must be combined
    with a mandatory affordability assessment before credit is granted.

    Parameters
    ----------
    score     : integer credit score (300–850)
    threshold : minimum score for approval (default: 614)

    Returns
    -------
    "Approve" | "Decline" | "Refer"
        Refer = borderline (within 10 points of the threshold — manual review)
    """
    if score >= threshold:
        return "Approve"
    elif score >= threshold - 10:
        return "Refer"     # borderline — manual underwriting review
    else:
        return "Decline"


# ── Standalone execution ───────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load model and feature names
    print("Loading model and feature names...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_names = pickle.load(f)
    print(f"  Model    : {type(model).__name__}")
    print(f"  Features : {len(feature_names)}")

    # 2. Load WoE-encoded data
    print(f"\nLoading WoE data from {DATA_PATH.name}...")
    X = pd.read_csv(DATA_PATH, index_col=0)
    missing_cols = [f for f in feature_names if f not in X.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    X = X[feature_names].copy()
    print(f"  Shape: {X.shape}")

    # 3. Build scorecard with SA parameters
    print("\nBuilding SA scorecard...")
    scorecard, base_points = build_scorecard(
        model,
        feature_names=feature_names,
        pdo=SA_PDO,
        base_score=SA_BASE_SCORE,
        base_odds=SA_BASE_ODDS,
    )

    print("\nTop 10 features by absolute point contribution:")
    print(
        scorecard
        .reindex(scorecard["raw_points"].abs().sort_values(ascending=False).index)
        .head(10)
        .to_string(index=False)
    )

    # 4. Calculate scores
    print("\nCalculating credit scores...")
    scores    = calculate_credit_score(X, scorecard, base_points)
    risk_bands = [get_risk_band(s)         for s in scores]
    decisions  = [get_approval_decision(s) for s in scores]

    print("\nFirst 10 scores:")
    print(pd.DataFrame({
        "score":     scores[:10],
        "risk_band": risk_bands[:10],
        "decision":  decisions[:10],
    }).to_string(index=False))

    # 5. Validate: score must correlate negatively with P(Default)
    pd_values   = model.predict_proba(X)[:, 0]    # P(Bad) = P(Default)
    correlation = np.corrcoef(scores, pd_values)[0, 1]
    print(f"\nCorrelation(score, PD): {correlation:.4f}  (must be negative)")
    if correlation >= 0:
        print("  WARNING: positive correlation — check sign convention in build_scorecard()")
    else:
        print("  OK — higher score correctly indicates lower default probability")

    # 6. Distribution across SA risk bands
    results = X.copy()
    results["credit_score"] = scores
    results["risk_band"]    = risk_bands
    results["decision"]     = decisions

    print("\nScore distribution (SA 300–850 scale):")
    print(results["credit_score"].describe().round(1))

    print("\nTransUnion SA risk band breakdown:")
    band_order = ["Excellent", "Good", "Favourable", "Average", "Below Average", "Unfavourable", "Poor"]
    counts = results["risk_band"].value_counts().reindex(band_order, fill_value=0)
    pcts   = (counts / len(results) * 100).round(1)
    print(pd.concat([counts, pcts], axis=1, keys=["count", "%"]).to_string())

    print(f"\nApproval breakdown (threshold = {SA_APPROVAL_THRESHOLD}):")
    print(results["decision"].value_counts(normalize=True).mul(100).round(1).map("{}%".format).to_string())

    print(f"\nSA market context:")
    print(f"  Approval threshold : {SA_APPROVAL_THRESHOLD}  (bottom of Favourable tier)")
    print(f"  Base odds          : {SA_BASE_ODDS}:1  (≈{100/(SA_BASE_ODDS+1):.0f}% default rate — NCR SA benchmark)")
    print(f"  PDO                : {SA_PDO}  (standard for high-default-rate unsecured lending)")
    print(f"  NCA compliance     : score is one input only; affordability assessment required separately")
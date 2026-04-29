"""
train.py
--------
Trains a logistic regression scorecard model on WoE features.

Changes from original to target Basel II IRB thresholds
(AUC ≥ 0.70, Gini ≥ 0.40, KS ≥ 0.30):

1. Finer C grid focused on [0.01, 2.0] — where optimal C lives for WOE data
2. Both L1 and L2 penalty searched — L1 can give sparser, more stable models
3. solver='saga' — handles class imbalance + both penalty types reliably
4. CV scores Gini directly (= 2×AUC−1) so we optimise the Basel metric
5. Correct Basel thresholds in evaluation printout (Gini≥0.40, KS≥0.30)
6. Added train-set Gini to detect overfitting (if train >> test, model is overfit)

Run: python src/train.py
"""

import pickle
import warnings
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = Path("scorecard_outputs")
OUTPUT_DIR = Path("scorecard_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5

# Finer grid focused on the range that works for WOE-encoded data.
# Original grid was too sparse between 0.01 and 1.0 — that's exactly where
# the optimal C for regularised WOE logistic regression typically lives.
C_GRID = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

# Search both penalties — L1 gives sparsity (fewer features, more interpretable)
# L2 gives stability (handles correlated WOE features better).
# Basel model risk management prefers sparser models.
PENALTY_GRID = ["l1", "l2"]

# Basel II IRB minimum thresholds
BASEL_AUC  = 0.70
BASEL_GINI = 0.40
BASEL_KS   = 0.30

mlflow.set_experiment("credit_scorecard")


# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")

X = pd.read_csv(INPUT_DIR / "X_woe.csv", index_col=0)
y = pd.read_csv(INPUT_DIR / "y.csv").iloc[:, -1]

with open(INPUT_DIR / "selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

X = X[[col for col in selected_features if col in X.columns]]

print(f"  Rows     : {len(X):,}")
print(f"  Features : {X.shape[1]}")
print(f"  Bad rate : {(y == 0).mean():.1%}")


# ── 2. Train / test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)
print(f"\nSplit — Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# ── 3. Cross-validation over C × penalty ──────────────────────────────────────
print(f"\nTuning C and penalty with {CV_FOLDS}-fold stratified CV...")
print(f"{'Penalty':<8} {'C':<8} {'Gini (CV)':>12} {'±':>8}")
print("-" * 42)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_results = []

for penalty in PENALTY_GRID:
    for c in C_GRID:
        with mlflow.start_run(run_name=f"cv_{penalty}_C={c}"):

            model_cv = LogisticRegression(
                C=c,
                penalty=penalty,
                solver="saga",          # handles both l1 and l2, good with imbalance
                class_weight="balanced",
                max_iter=2000,          # saga needs more iterations than lbfgs
                random_state=RANDOM_STATE,
            )

            # Score on AUC — Gini = 2×AUC−1, so maximising AUC = maximising Gini
            auc_scores = cross_val_score(
                model_cv, X_train, y_train,
                cv=cv, scoring="roc_auc", n_jobs=-1,
            )
            mean_gini = round(2 * auc_scores.mean() - 1, 4)
            std_gini  = round(2 * auc_scores.std(),      4)

            mlflow.log_param("C",            c)
            mlflow.log_param("penalty",      penalty)
            mlflow.log_param("cv_folds",     CV_FOLDS)
            mlflow.log_param("class_weight", "balanced")
            mlflow.log_metric("cv_mean_gini", mean_gini)
            mlflow.log_metric("cv_std_gini",  std_gini)
            mlflow.log_metric("cv_mean_auc",  round(auc_scores.mean(), 4))

            cv_results.append({
                "penalty":   penalty,
                "C":         c,
                "mean_gini": mean_gini,
                "std_gini":  std_gini,
                "mean_auc":  round(auc_scores.mean(), 4),
            })
            print(f"{penalty:<8} {c:<8}  {mean_gini:>10.4f}  ±{std_gini:.4f}")

cv_df    = pd.DataFrame(cv_results).sort_values("mean_gini", ascending=False).reset_index(drop=True)
best_row = cv_df.iloc[0]
best_c       = float(best_row["C"])
best_penalty = str(best_row["penalty"])

print(f"\nBest: penalty={best_penalty}  C={best_c}  (CV Gini = {best_row['mean_gini']:.4f})")

# Basel pre-check on CV Gini
if best_row["mean_gini"] < BASEL_GINI:
    print(f"\n  ⚠  CV Gini {best_row['mean_gini']:.4f} < Basel threshold {BASEL_GINI}.")
    print("     Consider: relaxing IV threshold in woe_etl.py (try IV ≥ 0.02),")
    print("     adding interaction features, or reviewing bin quality.")
else:
    print(f"\n  ✓  CV Gini {best_row['mean_gini']:.4f} ≥ Basel threshold {BASEL_GINI}.")


# ── 4. Fit final model ─────────────────────────────────────────────────────────
print("\nFitting final model on full training set...")

with mlflow.start_run(run_name="best_model"):

    model = LogisticRegression(
        C=best_c,
        penalty=best_penalty,
        solver="saga",
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # ── Evaluate on train AND test — large gap = overfit
    p_train = model.predict_proba(X_train)[:, 1]
    p_test  = model.predict_proba(X_test)[:, 1]

    train_auc  = roc_auc_score(y_train, p_train)
    test_auc   = roc_auc_score(y_test,  p_test)
    train_gini = 2 * train_auc - 1
    test_gini  = 2 * test_auc  - 1

    # KS on test set
    ks_df = pd.DataFrame({"y": y_test.values, "p_good": p_test})
    ks_df = ks_df.sort_values("p_good", ascending=False).reset_index(drop=True)
    ks_df["cum_good"] = (ks_df["y"] == 1).cumsum() / (y_test == 1).sum()
    ks_df["cum_bad"]  = (ks_df["y"] == 0).cumsum() / (y_test == 0).sum()
    ks_stat = float((ks_df["cum_good"] - ks_df["cum_bad"]).abs().max())

    # MLflow logging
    mlflow.log_param("best_C",       best_c)
    mlflow.log_param("best_penalty", best_penalty)
    mlflow.log_param("solver",       "saga")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("n_features",   X_train.shape[1])
    mlflow.log_param("n_train",      len(X_train))
    mlflow.log_param("n_test",       len(X_test))

    mlflow.log_metric("train_auc",   round(train_auc,  4))
    mlflow.log_metric("test_auc",    round(test_auc,   4))
    mlflow.log_metric("train_gini",  round(train_gini, 4))
    mlflow.log_metric("test_gini",   round(test_gini,  4))
    mlflow.log_metric("ks_stat",     round(ks_stat,    4))
    mlflow.log_metric("gini_gap",    round(train_gini - test_gini, 4))

    mlflow.sklearn.log_model(model, artifact_path="model")

    # ── Coefficient table
    coef_df = pd.DataFrame({
        "feature":     X_train.columns,
        "coefficient": model.coef_[0],
    })
    coef_df = (
        coef_df
        .assign(abs_coef=coef_df["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )
    coef_path = OUTPUT_DIR / "coef_summary.csv"
    coef_df.to_csv(coef_path, index=False)
    mlflow.log_artifact(str(coef_path))

    cv_path = OUTPUT_DIR / "cv_results.csv"
    cv_df.to_csv(cv_path, index=False)
    mlflow.log_artifact(str(cv_path))

    print("\nTop 10 features by coefficient magnitude:")
    print(coef_df.head(10).to_string(index=False))

    # ── Basel threshold check
    print(f"\n{'=' * 55}")
    print("  BASEL II IRB THRESHOLD CHECK")
    print(f"{'=' * 55}")
    print(f"  {'Metric':<10} {'Value':>8}  {'Threshold':>10}  {'Status':>8}")
    print(f"  {'-' * 44}")
    for label, val, thresh in [
        ("AUC",  test_auc,   BASEL_AUC),
        ("Gini", test_gini,  BASEL_GINI),
        ("KS",   ks_stat,    BASEL_KS),
    ]:
        status = "✓ PASS" if val >= thresh else "✗ FAIL"
        print(f"  {label:<10} {val:>8.4f}  {thresh:>10.2f}  {status:>8}")
    print(f"{'=' * 55}")

    if train_gini - test_gini > 0.05:
        print(f"\n  ⚠  Gini gap (train={train_gini:.4f}, test={test_gini:.4f}) > 0.05")
        print("     Model may be overfit. Try increasing regularisation (lower C).")


# ── 5. Save artifacts ─────────────────────────────────────────────────────────
with open(OUTPUT_DIR / "model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(OUTPUT_DIR / "feature_names.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)

X_train.to_csv(OUTPUT_DIR / "X_train.csv")
X_test.to_csv(OUTPUT_DIR  / "X_test.csv")
y_train.to_frame().to_csv(OUTPUT_DIR / "y_train.csv")
y_test.to_frame().to_csv(OUTPUT_DIR  / "y_test.csv")
cv_df.to_csv(OUTPUT_DIR   / "cv_results.csv",   index=False)
coef_df.to_csv(OUTPUT_DIR / "coef_summary.csv", index=False)

print(f"\nAll files saved to: {OUTPUT_DIR.resolve()}")
print("=== TRAINING COMPLETE ===")
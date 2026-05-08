"""
train.py
--------
Optimized Basel II IRB Scorecard Training Pipeline.
FIXED: Corrected target variable loading to prevent StratifiedKFold index errors.
"""

import pickle
import warnings
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = Path("scorecard_outputs")
OUTPUT_DIR = Path("scorecard_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Basel Settings ────────────────────────────────────────────────────────────
RANDOM_STATE = 42
CV_FOLDS     = 5
C_GRID       = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0]

BASEL_AUC  = 0.70
BASEL_GINI = 0.40
BASEL_KS   = 0.30

mlflow.set_experiment("credit_scorecard_irb")

# ── 1. Load Data with Index Correction ────────────────────────────────────────
print("Loading WoE-transformed data...")

X_train = pd.read_csv(INPUT_DIR / "X_train_woe.csv")
X_test  = pd.read_csv(INPUT_DIR / "X_test_woe.csv")

# FIX: Load target and explicitly drop the index column if it exists
y_train_df = pd.read_csv(INPUT_DIR / "y_train.csv")
y_test_df  = pd.read_csv(INPUT_DIR / "y_test.csv")

# If 'good_bad' is in columns, use it; otherwise, take the last column (ignoring index)
target_col = 'good_bad' if 'good_bad' in y_train_df.columns else y_train_df.columns[-1]
y_train = y_train_df[target_col].astype(int)
y_test  = y_test_df[target_col].astype(int)

# ── 2. Basel Sanity Check ─────────────────────────────────────────────────────
n_bads = (y_train == 0).sum()
bad_rate = (n_bads / len(y_train)) * 100

print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"  Features: {X_train.shape[1]}")
print(f"  Bads in Train: {n_bads} ({bad_rate:.2f}%)")

if n_bads < CV_FOLDS:
    print(f"\nCRITICAL ERROR: Only {n_bads} 'Bad' samples found in training set.")
    print(f"Stratified {CV_FOLDS}-fold CV requires at least {CV_FOLDS} samples per class.")
    print("Check your ETL script to ensure 'loan_status' is being mapped correctly.")
    exit()

# ── 3. Hyperparameter Tuning (Stability-Based) ────────────────────────────────
print(f"\nTuning L2 Regularization ({CV_FOLDS}-fold CV)...")
print(f"{'C':<10} {'Gini (Mean)':>12} {'Gini (Std)':>10} {'Stability Index':>18}")
print("-" * 55)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
tuning_results = []

for c in C_GRID:
    model_cv = LogisticRegression(
        C=c, penalty='l2', solver='lbfgs', 
        max_iter=1000, random_state=RANDOM_STATE
    )
    
    cv_out = cross_validate(
        model_cv, X_train, y_train, 
        cv=cv, scoring='roc_auc', return_estimator=True, n_jobs=-1
    )
    
    aucs = cv_out['test_score']
    ginis = 2 * aucs - 1
    mean_gini, std_gini = np.mean(ginis), np.std(ginis)
    stability_score = mean_gini - std_gini 
    
    tuning_results.append({
        "C": c, "mean_gini": mean_gini, "std_gini": std_gini, "stability_score": stability_score
    })
    print(f"{c:<10.3f} {mean_gini:>12.4f} {std_gini:>10.4f} {stability_score:>18.4f}")

tuning_df = pd.DataFrame(tuning_results).sort_values("stability_score", ascending=False)
best_c = tuning_df.iloc[0]["C"]

# ── 4. Final Model Fit & Validation ───────────────────────────────────────────
with mlflow.start_run(run_name="final_irb_scorecard"):
    model = LogisticRegression(C=best_c, penalty='l2', solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    p_test = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, p_test)
    test_gini = 2 * test_auc - 1
    
    # Standard Basel KS Logic
    def get_ks(y_true, y_prob):
        df = pd.DataFrame({'target': y_true, 'prob': y_prob}).sort_values('prob')
        df['cum_good'] = (df['target'] == 1).cumsum() / (df['target'] == 1).sum()
        df['cum_bad'] = (df['target'] == 0).cumsum() / (df['target'] == 0).sum()
        return (df['cum_bad'] - df['cum_good']).max()

    ks_stat = get_ks(y_test, p_test)
    
    print(f"\n{'=' * 60}\n  BASEL II IRB VALIDATION REPORT\n{'=' * 60}")
    for label, val, threshold in [("AUC", test_auc, BASEL_AUC), ("Gini", test_gini, BASEL_GINI), ("KS Stat", ks_stat, BASEL_KS)]:
        status = "PASS" if val >= threshold else "FAIL"
        print(f"  {label:<12} | Value: {val:.4f} | Threshold: {threshold:.2f} | [{status}]")
    print(f"{'=' * 60}")

# ── 5. Persistence ────────────────────────────────────────────────────────────
with open(OUTPUT_DIR / "model.pkl", "wb") as f:
    pickle.dump(model, f)

pd.DataFrame({"Feature": X_train.columns, "Coef": model.coef_[0]}).to_csv(OUTPUT_DIR / "coef_summary.csv", index=False)
print(f"\nModel artifacts saved to {OUTPUT_DIR}")
"""
Model Validation Script for Credit Risk PD Model
=================================================
Loads the saved model and its significant feature list (produced by
train_pd_model.py), applies identical feature selection to the test
data, then runs a full suite of validation metrics.
"""

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)
from scipy.stats import ks_2samp

sns.set()


# ---------------------------------------------------------------------------
# Paths — update these to match your environment
# ---------------------------------------------------------------------------
from pathlib import Path

SRC_DIR   = Path(__file__).resolve().parent   # .../src/
ROOT_DIR  = SRC_DIR.parent                    # .../basel-credit-risk-model/
DATA_DIR  = ROOT_DIR / "data"
MODEL_DIR = SRC_DIR

INPUTS_TEST_PATH   = f"{DATA_DIR}/loan_data_inputs_test.csv"
TARGETS_TEST_PATH  = f"{DATA_DIR}/loan_data_targets_test.csv"
MODEL_SAVE_PATH    = f"{MODEL_DIR}/pd_model.sav"
# Feature list saved by train_pd_model.py — contains only the columns
# that survived the p-value filter, in the exact order the model expects
FEATURES_SAVE_PATH = f"{MODEL_DIR}/pd_model_features.pkl"

# Classification threshold (adjust as needed based on threshold-optimisation
# output from the training script)
THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# 1. Load model and feature list
# ---------------------------------------------------------------------------
print("Loading model and feature list...")

with open(MODEL_SAVE_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURES_SAVE_PATH, "rb") as f:
    significant_features = pickle.load(f)

print(f"Model loaded successfully")
print(f"Significant features loaded: {len(significant_features)} features")


# ---------------------------------------------------------------------------
# 2. Load test data
# ---------------------------------------------------------------------------
print("\nLoading test data...")
loan_data_inputs_test  = pd.read_csv(INPUTS_TEST_PATH)
loan_data_targets_test = pd.read_csv(TARGETS_TEST_PATH)

print(f"Inputs  test shape:  {loan_data_inputs_test.shape}")
print(f"Targets test shape:  {loan_data_targets_test.shape}")


# ---------------------------------------------------------------------------
# 3. Select features
#    Use exactly the significant columns the training script saved — this
#    automatically excludes both reference categories AND the features that
#    were dropped for failing the p-value filter (p > 0.05).
# ---------------------------------------------------------------------------
missing = [c for c in significant_features if c not in loan_data_inputs_test.columns]
if missing:
    raise ValueError(
        f"The following significant features are missing from the test data:\n{missing}"
    )

inputs_test = loan_data_inputs_test[significant_features]
print(f"\nTest features selected: {inputs_test.shape[1]} columns")


# ---------------------------------------------------------------------------
# 4. Generate predictions
# ---------------------------------------------------------------------------
y_hat_test       = model.predict(inputs_test)
y_hat_test_proba = model.predict_proba(inputs_test)[:, 1]

print(f"Class predictions shape:       {y_hat_test.shape}")
print(f"Probability predictions shape: {y_hat_test_proba.shape}")


# ---------------------------------------------------------------------------
# 5. Build actual-vs-predicted DataFrame
# ---------------------------------------------------------------------------
targets_reset = loan_data_targets_test.copy().reset_index(drop=True)

df_actual_predicted_probs = pd.concat(
    [targets_reset, pd.Series(y_hat_test_proba, name="y_hat_test_proba")],
    axis=1,
)
df_actual_predicted_probs.columns = ["loan_data_targets_test", "y_hat_test_proba"]
df_actual_predicted_probs.index   = loan_data_inputs_test.index

print(f"\nActual vs Predicted DataFrame shape: {df_actual_predicted_probs.shape}")


# ---------------------------------------------------------------------------
# 6. Accuracy metrics with threshold
# ---------------------------------------------------------------------------
df_actual_predicted_probs["y_hat_test"] = np.where(
    df_actual_predicted_probs["y_hat_test_proba"] > THRESHOLD, 1, 0
)

cm = confusion_matrix(
    df_actual_predicted_probs["loan_data_targets_test"],
    df_actual_predicted_probs["y_hat_test"],
)
print(f"\nConfusion Matrix (threshold={THRESHOLD}):")
print(cm)

cm_norm = confusion_matrix(
    df_actual_predicted_probs["loan_data_targets_test"],
    df_actual_predicted_probs["y_hat_test"],
    normalize="all",
)
print(f"\nNormalized Confusion Matrix (threshold={THRESHOLD}):")
print(cm_norm)

accuracy = accuracy_score(
    df_actual_predicted_probs["loan_data_targets_test"],
    df_actual_predicted_probs["y_hat_test"],
)
print(f"\nAccuracy (threshold={THRESHOLD}): {accuracy:.4f}")


# ---------------------------------------------------------------------------
# 7. ROC curve and AUROC
# ---------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(
    df_actual_predicted_probs["loan_data_targets_test"],
    df_actual_predicted_probs["y_hat_test_proba"],
)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, linewidth=2)
plt.plot(fpr, fpr, linestyle="--", color="k", alpha=0.7)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.grid(True, alpha=0.3)
plt.show()

AUROC = roc_auc_score(
    df_actual_predicted_probs["loan_data_targets_test"],
    df_actual_predicted_probs["y_hat_test_proba"],
)
print(f"AUROC: {AUROC:.4f}")


# ---------------------------------------------------------------------------
# 8. Gini coefficient
# ---------------------------------------------------------------------------
Gini = AUROC * 2 - 1
print(f"Gini Coefficient: {Gini:.4f}")


# ---------------------------------------------------------------------------
# 9. Kolmogorov-Smirnov statistic
# ---------------------------------------------------------------------------
good_probs = df_actual_predicted_probs[
    df_actual_predicted_probs["loan_data_targets_test"] == 0
]["y_hat_test_proba"]

bad_probs = df_actual_predicted_probs[
    df_actual_predicted_probs["loan_data_targets_test"] == 1
]["y_hat_test_proba"]

ks_statistic, _ = ks_2samp(good_probs, bad_probs)
print(f"KS Statistic: {ks_statistic:.4f}")

# KS chart
df_sorted   = df_actual_predicted_probs.sort_values("y_hat_test_proba").reset_index(drop=True)
total_pop   = df_sorted.shape[0]
total_good  = (df_sorted["loan_data_targets_test"] == 0).sum()
total_bad   = (df_sorted["loan_data_targets_test"] == 1).sum()

df_sorted["Cumulative Perc Good"] = (
    (df_sorted["loan_data_targets_test"] == 0).cumsum() / total_good
)
df_sorted["Cumulative Perc Bad"] = (
    (df_sorted["loan_data_targets_test"] == 1).cumsum() / total_bad
)

plt.figure(figsize=(10, 6))
plt.plot(df_sorted["y_hat_test_proba"], df_sorted["Cumulative Perc Bad"],
         color="r", linewidth=2, label="Bad")
plt.plot(df_sorted["y_hat_test_proba"], df_sorted["Cumulative Perc Good"],
         color="b", linewidth=2, label="Good")
plt.xlabel("Estimated Probability for being Good")
plt.ylabel("Cumulative %")
plt.title("Kolmogorov-Smirnov Chart")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("MODEL VALIDATION SUMMARY")
print("=" * 50)
print(f"Test Set Size:          {total_pop}")
print(f"Number of Features:     {inputs_test.shape[1]}")
print(f"Good Loans (0):         {total_good}")
print(f"Bad Loans  (1):         {total_bad}")
print(f"Bad Rate:               {total_bad / total_pop:.4f}")
print("-" * 50)
print(f"AUROC:                  {AUROC:.4f}")
print(f"Gini Coefficient:       {Gini:.4f}")
print(f"KS Statistic:           {ks_statistic:.4f}")
print(f"Accuracy (thr={THRESHOLD}):   {accuracy:.4f}")
print("=" * 50)

"""
Credit Risk PD Model Training Script
=====================================
Trains a Probability of Default (PD) model using Logistic Regression
and Random Forest, with MLflow tracking and threshold optimisation.
"""

import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stat

import mlflow
import mlflow.sklearn

from sklearn import linear_model, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths  — update these to match your environment
# ---------------------------------------------------------------------------
DATA_DIR  = "/Users/lindokuhletami/Desktop/Space/basel-credit-risk-model/data"
MODEL_DIR = "/Users/lindokuhletami/Desktop/Space/basel-credit-risk-model/src"

INPUTS_TRAIN_PATH  = f"{DATA_DIR}/loan_data_inputs_train.csv"
TARGETS_TRAIN_PATH = f"{DATA_DIR}/loan_data_targets_train.csv"
MODEL_SAVE_PATH    = f"{MODEL_DIR}/pd_model.sav"
# Significant feature list — loaded by the validation script to apply
# the same column selection that was determined during training
FEATURES_SAVE_PATH = f"{MODEL_DIR}/pd_model_features.pkl"


# ---------------------------------------------------------------------------
# Feature columns selected for the model
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "grade:A", "grade:B", "grade:C", "grade:D", "grade:E", "grade:F", "grade:G",
    "home_ownership:RENT_OTHER_NONE_ANY", "home_ownership:OWN", "home_ownership:MORTGAGE",
    "addr_state:ND_NE_IA_NV_FL_HI_AL", "addr_state:NM_VA", "addr_state:NY",
    "addr_state:OK_TN_MO_LA_MD_NC", "addr_state:CA", "addr_state:UT_KY_AZ_NJ",
    "addr_state:AR_MI_PA_OH_MN", "addr_state:RI_MA_DE_SD_IN", "addr_state:GA_WA_OR",
    "addr_state:WI_MT", "addr_state:TX", "addr_state:IL_CT",
    "addr_state:KS_SC_CO_VT_AK_MS", "addr_state:WV_NH_WY_DC_ME_ID",
    "verification_status:Not Verified", "verification_status:Source Verified",
    "verification_status:Verified",
    "purpose:educ__sm_b__wedd__ren_en__mov__house", "purpose:credit_card",
    "purpose:debt_consolidation", "purpose:oth__med__vacation",
    "purpose:major_purch__car__home_impr",
    "initial_list_status:f", "initial_list_status:w",
    "term:36", "term:60",
    "emp_length:0", "emp_length:1", "emp_length:2-4", "emp_length:5-6",
    "emp_length:7-9", "emp_length:10",
    "mths_since_issue_d:<38", "mths_since_issue_d:38-39", "mths_since_issue_d:40-41",
    "mths_since_issue_d:42-48", "mths_since_issue_d:49-52", "mths_since_issue_d:53-64",
    "mths_since_issue_d:65-84", "mths_since_issue_d:>84",
    "int_rate:<9.548", "int_rate:9.548-12.025", "int_rate:12.025-15.74",
    "int_rate:15.74-20.281", "int_rate:>20.281",
    "mths_since_earliest_cr_line:<140", "mths_since_earliest_cr_line:141-164",
    "mths_since_earliest_cr_line:165-247", "mths_since_earliest_cr_line:248-270",
    "mths_since_earliest_cr_line:271-352", "mths_since_earliest_cr_line:>352",
    "delinq_2yrs:0", "delinq_2yrs:1-3", "delinq_2yrs:>=4",
    "inq_last_6mths:0", "inq_last_6mths:1-2", "inq_last_6mths:3-6", "inq_last_6mths:>6",
    "open_acc:0", "open_acc:1-3", "open_acc:4-12", "open_acc:13-17",
    "open_acc:18-22", "open_acc:23-25", "open_acc:26-30", "open_acc:>=31",
    "pub_rec:0-2", "pub_rec:3-4", "pub_rec:>=5",
    "total_acc:<=27", "total_acc:28-51", "total_acc:>=52",
    "acc_now_delinq:0", "acc_now_delinq:>=1",
    "total_rev_hi_lim:<=5K", "total_rev_hi_lim:5K-10K", "total_rev_hi_lim:10K-20K",
    "total_rev_hi_lim:20K-30K", "total_rev_hi_lim:30K-40K", "total_rev_hi_lim:40K-55K",
    "total_rev_hi_lim:55K-95K", "total_rev_hi_lim:>95K",
    "annual_inc:<20K", "annual_inc:20K-30K", "annual_inc:30K-40K", "annual_inc:40K-50K",
    "annual_inc:50K-60K", "annual_inc:60K-70K", "annual_inc:70K-80K",
    "annual_inc:80K-90K", "annual_inc:90K-100K", "annual_inc:100K-120K",
    "annual_inc:120K-140K", "annual_inc:>140K",
    "dti:<=1.4", "dti:1.4-3.5", "dti:3.5-7.7", "dti:7.7-10.5", "dti:10.5-16.1",
    "dti:16.1-20.3", "dti:20.3-21.7", "dti:21.7-22.4", "dti:22.4-35", "dti:>35",
    "mths_since_last_delinq:Missing", "mths_since_last_delinq:0-3",
    "mths_since_last_delinq:4-30", "mths_since_last_delinq:31-56",
    "mths_since_last_delinq:>=57",
    "mths_since_last_record:Missing", "mths_since_last_record:0-2",
    "mths_since_last_record:3-20", "mths_since_last_record:21-31",
    "mths_since_last_record:32-80", "mths_since_last_record:81-86",
    "mths_since_last_record:>=86",
]

# Reference categories to drop (one per dummy group) to avoid multicollinearity
REF_CATEGORIES = [
    "grade:G",
    "home_ownership:RENT_OTHER_NONE_ANY",
    "addr_state:ND_NE_IA_NV_FL_HI_AL",
    "verification_status:Verified",
    "purpose:educ__sm_b__wedd__ren_en__mov__house",
    "initial_list_status:f",
    "term:60",
    "emp_length:0",
    "mths_since_issue_d:>84",
    "int_rate:>20.281",
    "mths_since_earliest_cr_line:<140",
    "delinq_2yrs:>=4",
    "inq_last_6mths:>6",
    "open_acc:0",
    "pub_rec:0-2",
    "total_acc:<=27",
    "acc_now_delinq:0",
    "total_rev_hi_lim:<=5K",
    "annual_inc:<20K",
    "dti:>35",
    "mths_since_last_delinq:0-3",
    "mths_since_last_record:0-2",
]


# ---------------------------------------------------------------------------
# Logistic Regression wrapper that also computes p-values
# ---------------------------------------------------------------------------
class LogisticRegressionWithPValues:
    """Wraps sklearn LogisticRegression and attaches Wald-test p-values."""

    def __init__(self, *args, **kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)

    def fit(self, X, y):
        # Keep track of the original column names/index before any filtering
        if hasattr(X, 'columns'):
            original_names = list(X.columns)
        else:
            original_names = list(range(np.asarray(X).shape[1]))

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Drop zero-variance (constant) columns to avoid a singular matrix.
        # Record the mask so callers can retrieve the surviving feature names.
        var = X.var(axis=0)
        self.feature_mask_ = var > 0
        if not self.feature_mask_.all():
            n_dropped = int((~self.feature_mask_).sum())
            print(f'  [LogisticRegressionWithPValues] dropping {n_dropped} '
                  f'zero-variance column(s) before fitting.')
            X = X[:, self.feature_mask_]
        self.feature_names_in_ = [
            name for name, keep in zip(original_names, self.feature_mask_) if keep
        ]

        self.model.fit(X, y)

        denom = 2.0 * (1.0 + np.cosh(self.model.decision_function(X)))
        denom = np.tile(denom, (X.shape[1], 1)).T
        F_ij = np.dot((X / denom).T, X).astype(float)

        try:
            Cramer_Rao = np.linalg.inv(F_ij)
        except np.linalg.LinAlgError:
            Cramer_Rao = np.linalg.pinv(F_ij)

        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        self.p_values = [stat.norm.sf(abs(z)) * 2 for z in z_scores]
        self.coef_      = self.model.coef_
        self.intercept_ = self.model.intercept_


# ---------------------------------------------------------------------------
# Helper — build coefficient summary table
# ---------------------------------------------------------------------------
def build_summary_table(model_wrapper, feature_names):
    coefs = model_wrapper.coef_.ravel()
    names = feature_names[: coefs.size]

    summary = pd.DataFrame({"Feature name": names, "Coefficients": coefs})
    summary.index = range(1, len(summary) + 1)
    summary.loc[0] = ["Intercept", model_wrapper.intercept_[0]]
    summary.sort_index(inplace=True)

    p_values = np.append(np.nan, np.array(model_wrapper.p_values))
    summary["p_values"] = p_values
    return summary


# ---------------------------------------------------------------------------
# Threshold optimisation via cross-validation
# ---------------------------------------------------------------------------
def optimise_threshold(model_wrapper, X, y):
    print("\n" + "=" * 50)
    print("OPTIMIZING THRESHOLD VIA CROSS-VALIDATION")
    print("=" * 50)

    y_arr = np.array(y).ravel()

    unique_classes, class_counts = np.unique(y_arr, return_counts=True)
    print("\nClass distribution in training data:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples ({count / len(y_arr) * 100:.2f}%)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    try:
        y_proba_cv = cross_val_predict(
            model_wrapper.model, X, y_arr, cv=cv, method="predict_proba"
        )
        print(f"\nCV predictions shape: {y_proba_cv.shape}")
        if y_proba_cv.shape[1] == 2:
            y_proba_cv = y_proba_cv[:, 1]
        else:
            y_proba_cv = y_proba_cv.ravel()
    except Exception as exc:
        print(f"Error in cross_val_predict: {exc}\nFalling back to manual CV...")
        y_proba_cv = np.zeros(len(y_arr))
        for train_idx, val_idx in cv.split(X, y_arr):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y_arr[train_idx]
            model_wrapper.model.fit(X_tr, y_tr)
            fold_proba = model_wrapper.model.predict_proba(X_val)
            y_proba_cv[val_idx] = (
                fold_proba[:, 1] if fold_proba.shape[1] == 2 else fold_proba.ravel()
            )
        print("Manual CV completed")

    thresholds       = np.arange(0.1, 0.9, 0.05)
    f1_scores        = []
    precision_scores = []
    recall_scores    = []
    accuracy_scores  = []

    for t in thresholds:
        y_pred = (y_proba_cv > t).astype(int)
        if len(np.unique(y_pred)) == 2:
            f1_scores.append(f1_score(y_arr, y_pred))
            precision_scores.append(precision_score(y_arr, y_pred))
            recall_scores.append(recall_score(y_arr, y_pred))
        else:
            f1_scores.append(0)
            precision_scores.append(0)
            recall_scores.append(0)
        accuracy_scores.append(np.mean(y_pred == y_arr))

    opt_idx_f1  = int(np.argmax(f1_scores))
    opt_thr_f1  = thresholds[opt_idx_f1]
    opt_idx_acc = int(np.argmax(accuracy_scores))
    opt_thr_acc = thresholds[opt_idx_acc]

    print(f"\nOptimal threshold (based on F1 score): {opt_thr_f1:.2f}")
    print(f"F1 score at optimal threshold:         {f1_scores[opt_idx_f1]:.4f}")
    print(f"Precision at optimal threshold:        {precision_scores[opt_idx_f1]:.4f}")
    print(f"Recall at optimal threshold:           {recall_scores[opt_idx_f1]:.4f}")
    print(f"Accuracy at optimal threshold:         {accuracy_scores[opt_idx_f1]:.4f}")
    print(f"\nOptimal threshold (based on Accuracy): {opt_thr_acc:.2f}")
    print(f"Accuracy at optimal threshold:         {accuracy_scores[opt_idx_acc]:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(thresholds, f1_scores,        marker="o", label="F1 Score",  linewidth=2)
    axes[0].plot(thresholds, precision_scores, marker="s", label="Precision", linewidth=2)
    axes[0].plot(thresholds, recall_scores,    marker="^", label="Recall",    linewidth=2)
    axes[0].axvline(x=opt_thr_f1, color="r", linestyle="--",
                    label=f"Optimal F1: {opt_thr_f1:.2f}")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Threshold Optimisation — Classification Metrics")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(thresholds, accuracy_scores, marker="o", color="green", linewidth=2)
    axes[1].axvline(x=opt_thr_acc, color="r", linestyle="--",
                    label=f"Optimal Acc: {opt_thr_acc:.2f}")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Threshold Optimisation — Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    loan_data_inputs_train  = pd.read_csv(INPUTS_TRAIN_PATH)
    loan_data_targets_train = pd.read_csv(TARGETS_TRAIN_PATH)

    print(f"Inputs shape:  {loan_data_inputs_train.shape}")
    print(f"Targets shape: {loan_data_targets_train.shape}")

    # ------------------------------------------------------------------
    # 2. Select & prepare features
    # ------------------------------------------------------------------
    inputs_train_with_ref_cat = loan_data_inputs_train[FEATURE_COLUMNS]
    inputs_train = inputs_train_with_ref_cat.drop(
        columns=[c for c in REF_CATEGORIES if c in inputs_train_with_ref_cat.columns]
    )

    X = inputs_train.values
    y = loan_data_targets_train["good_bad"].values

    # ------------------------------------------------------------------
    # 3. MLflow experiment
    # ------------------------------------------------------------------
    mlflow.set_experiment("credit_risk_pd_model")

    # ------------------------------------------------------------------
    # 4. Logistic Regression (standard sklearn)
    # ------------------------------------------------------------------
    print("\nTraining Logistic Regression...")
    with mlflow.start_run(run_name="logistic_regression"):
        lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr_model.fit(X, y)

        preds_proba = lr_model.predict_proba(X)[:, 1]
        preds       = lr_model.predict(X)

        auc       = roc_auc_score(y, preds_proba)
        precision = precision_score(y, preds)
        recall    = recall_score(y, preds)
        f1        = f1_score(y, preds)

        mlflow.log_param("model_type",    "logistic_regression")
        mlflow.log_param("max_iter",      1000)
        mlflow.log_param("class_weight",  "balanced")
        mlflow.log_metric("AUC",          auc)
        mlflow.log_metric("precision",    precision)
        mlflow.log_metric("recall",       recall)
        mlflow.log_metric("f1_score",     f1)
        mlflow.sklearn.log_model(lr_model, "logistic_model")

        print(f"AUC:                        {auc}")
        print(f"Precision (macro):          {precision}")
        print(f"Recall    (macro):          {recall}")
        print(f"F1        (macro):          {f1}")
        # Minority-class (bad loan = 0) metrics
        print(f"Precision (bad loan / 0):   {precision_score(y, preds, pos_label=0):.4f}")
        print(f"Recall    (bad loan / 0):   {recall_score(y, preds, pos_label=0):.4f}")
        print(f"F1        (bad loan / 0):   {f1_score(y, preds, pos_label=0):.4f}")

    # ------------------------------------------------------------------
    # 5. Random Forest
    # ------------------------------------------------------------------
    print("\nTraining Random Forest...")
    with mlflow.start_run(run_name="random_forest"):
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
        )
        rf_model.fit(X, y)

        rf_preds_proba = rf_model.predict_proba(X)[:, 1]
        rf_preds       = rf_model.predict(X)

        rf_auc       = roc_auc_score(y, rf_preds_proba)
        rf_precision = precision_score(y, rf_preds)
        rf_recall    = recall_score(y, rf_preds)
        rf_f1        = f1_score(y, rf_preds)

        mlflow.log_param("model_type",   "random_forest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth",    8)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metric("AUC",         rf_auc)
        mlflow.log_metric("precision",   rf_precision)
        mlflow.log_metric("recall",      rf_recall)
        mlflow.log_metric("f1_score",    rf_f1)
        mlflow.sklearn.log_model(rf_model, "rf_model")

        print(f"Random Forest AUC:                        {rf_auc}")
        print(f"Precision (macro):                        {rf_precision}")
        print(f"Recall    (macro):                        {rf_recall}")
        print(f"F1        (macro):                        {rf_f1}")
        # Minority-class (bad loan = 0) metrics
        print(f"Precision (bad loan / 0):                 {precision_score(y, rf_preds, pos_label=0):.4f}")
        print(f"Recall    (bad loan / 0):                 {recall_score(y, rf_preds, pos_label=0):.4f}")
        print(f"F1        (bad loan / 0):                 {f1_score(y, rf_preds, pos_label=0):.4f}")

    # ------------------------------------------------------------------
    # 6. Logistic Regression with p-values — initial fit
    # ------------------------------------------------------------------
    print("\nTraining Logistic Regression with p-values (initial fit)...")
    reg2 = LogisticRegressionWithPValues(class_weight="balanced")
    reg2.fit(inputs_train, loan_data_targets_train)

    summary_table = build_summary_table(reg2, inputs_train.columns)
    print("\nInitial model summary:")
    print(summary_table.to_string())

    # ------------------------------------------------------------------
    # 7. Drop statistically insignificant features (p-value > 0.05)
    # ------------------------------------------------------------------
    P_VALUE_THRESHOLD = 0.05

    # Row 0 is the intercept (p-value = NaN), so skip it
    insignificant_features = summary_table[
        (summary_table.index > 0) & (summary_table["p_values"] > P_VALUE_THRESHOLD)
    ]["Feature name"].tolist()

    print(f"\nDropping {len(insignificant_features)} insignificant feature(s) "
          f"(p-value > {P_VALUE_THRESHOLD}):")
    for feat in insignificant_features:
        pval = summary_table.loc[
            summary_table["Feature name"] == feat, "p_values"
        ].values[0]
        print(f"  {feat:50s}  p = {pval:.4f}")

    inputs_train_significant = inputs_train.drop(
        columns=[f for f in insignificant_features if f in inputs_train.columns]
    )
    print(f"\nFeatures remaining: {inputs_train_significant.shape[1]} "
          f"(was {inputs_train.shape[1]})")

    # ------------------------------------------------------------------
    # 8. Refit on significant features only
    # ------------------------------------------------------------------
    print("\nRefitting on statistically significant features only...")
    reg2_final = LogisticRegressionWithPValues(class_weight="balanced")
    reg2_final.fit(inputs_train_significant, loan_data_targets_train)

    summary_table_final = build_summary_table(reg2_final, inputs_train_significant.columns)
    print("\nFinal model summary (significant features only):")
    print(summary_table_final.to_string())

    # ------------------------------------------------------------------
    # 9. Threshold optimisation via cross-validation (final model)
    # ------------------------------------------------------------------
    optimise_threshold(reg2_final, inputs_train_significant, loan_data_targets_train)

    # ------------------------------------------------------------------
    # 10. Save the final model and its significant feature list
    # ------------------------------------------------------------------
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(reg2_final.model, f)
    print(f"\nModel successfully saved to: {MODEL_SAVE_PATH}")

    significant_features = reg2_final.feature_names_in_
    with open(FEATURES_SAVE_PATH, "wb") as f:
        pickle.dump(significant_features, f)
    print(f"Feature list ({len(significant_features)} features) saved to: {FEATURES_SAVE_PATH}")


if __name__ == "__main__":
    main()

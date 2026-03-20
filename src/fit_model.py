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
import mlflow
import mlflow.sklearn

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
from pathlib import Path

SRC_DIR   = Path(__file__).resolve().parent        # .../src/
ROOT_DIR  = SRC_DIR.parent                          # .../basel-credit-risk-model/
DATA_DIR  = ROOT_DIR / "data"
MODEL_DIR = SRC_DIR

INPUTS_TRAIN_PATH  = DATA_DIR  / "loan_data_inputs_train.csv"
TARGETS_TRAIN_PATH = DATA_DIR  / "loan_data_targets_train.csv"
MODEL_SAVE_PATH    = MODEL_DIR / "pd_model.sav"
FEATURES_SAVE_PATH  = MODEL_DIR / "pd_model_features.pkl"
THRESHOLD_SAVE_PATH = MODEL_DIR / "pd_model_threshold.pkl"


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
# LogisticRegressionWithPValues — defined here, only used during training.
# We never pickle this class — only the inner sklearn model is saved.
# ---------------------------------------------------------------------------
import scipy.stats as stat
from sklearn import linear_model


class LogisticRegressionWithPValues:
    """Wraps sklearn LogisticRegression and attaches Wald-test p-values.
    Only used during training. Never serialised to disk.
    """

    def __init__(self, *args, **kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)

    def fit(self, X, y):
        if hasattr(X, 'columns'):
            original_names = list(X.columns)
        else:
            original_names = list(range(np.asarray(X).shape[1]))

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

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
        F_ij  = np.dot((X / denom).T, X).astype(float)

        try:
            Cramer_Rao = np.linalg.inv(F_ij)
        except np.linalg.LinAlgError:
            Cramer_Rao = np.linalg.pinv(F_ij)

        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores        = self.model.coef_[0] / sigma_estimates
        self.p_values   = [stat.norm.sf(abs(z)) * 2 for z in z_scores]
        self.coef_      = self.model.coef_
        self.intercept_ = self.model.intercept_


# ---------------------------------------------------------------------------
# Helper — build coefficient summary table
# ---------------------------------------------------------------------------
def build_summary_table(model_wrapper, feature_names):
    coefs = model_wrapper.coef_.ravel()
    names = list(feature_names)[: coefs.size]

    summary = pd.DataFrame({"Feature name": names, "Coefficients": coefs})
    summary.index = range(1, len(summary) + 1)
    summary.loc[0] = ["Intercept", model_wrapper.intercept_[0]]
    summary.sort_index(inplace=True)

    p_values = np.append(np.nan, np.array(model_wrapper.p_values))
    summary["p_values"] = p_values

    # Format p-values as readable decimals — clip at 0.0001 so very small
    # values show as "< 0.0001" rather than scientific notation
    summary["p_values_fmt"] = summary["p_values"].apply(
        lambda p: "—" if pd.isna(p) else ("< 0.0001" if p < 0.0001 else f"{p:.4f}")
    )
    summary["significant"] = summary["p_values"].apply(
        lambda p: "" if pd.isna(p) else ("✓" if p <= 0.05 else "✗")
    )
    return summary


# ---------------------------------------------------------------------------
# Threshold optimisation using sklearn's StratifiedKFold + cross_val_predict
# ---------------------------------------------------------------------------
def optimise_threshold(model, X, y, cv_splits: int = 5) -> float:
    """
    Find the probability threshold that maximises F1 score using
    sklearn's cross_val_predict with StratifiedKFold — no manual
    loop required.

    Parameters
    ----------
    model     : fitted sklearn model with predict_proba()
    X         : feature DataFrame
    y         : target array/Series
    cv_splits : number of CV folds (default 5)

    Returns
    -------
    optimal_threshold : float — best threshold based on F1 score
    """
    print("\n" + "=" * 50)
    print("OPTIMISING THRESHOLD VIA CROSS-VALIDATION")
    print("=" * 50)

    y_arr = np.array(y).ravel()

    unique_classes, class_counts = np.unique(y_arr, return_counts=True)
    print("\nClass distribution in training data:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples ({count / len(y_arr) * 100:.2f}%)")

    # sklearn's cross_val_predict handles splitting, fitting on each fold,
    # and collecting out-of-fold predictions automatically.
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    y_proba_cv = cross_val_predict(model, X, y_arr, cv=cv, method="predict_proba")
    y_proba_cv = y_proba_cv[:, 1]   # probability of class 1 (good loan)

    print(f"\nCV out-of-fold predictions collected: {len(y_proba_cv)} samples")

    # Sweep thresholds and record metrics.
    # IMPORTANT: we optimise on the MINORITY CLASS (bad loans = 0) F1,
    # not the macro/majority F1. Optimising on macro F1 with an 89/11
    # imbalance always picks the lowest threshold (predict everything
    # as good), which is useless for credit risk.
    thresholds            = np.arange(0.1, 0.91, 0.01)
    f1_bad_scores         = []   # F1 for class 0 (bad loans) — our target
    precision_bad_scores  = []
    recall_bad_scores     = []
    f1_macro_scores       = []
    accuracy_scores       = []

    for t in thresholds:
        y_pred = (y_proba_cv >= t).astype(int)
        if len(np.unique(y_pred)) == 2:
            f1_bad_scores.append(f1_score(y_arr, y_pred, pos_label=0))
            precision_bad_scores.append(precision_score(y_arr, y_pred, pos_label=0))
            recall_bad_scores.append(recall_score(y_arr, y_pred, pos_label=0))
            f1_macro_scores.append(f1_score(y_arr, y_pred))
        else:
            f1_bad_scores.append(0.0)
            precision_bad_scores.append(0.0)
            recall_bad_scores.append(0.0)
            f1_macro_scores.append(0.0)
        accuracy_scores.append(np.mean(y_pred == y_arr))

    # ------------------------------------------------------------------
    # Objective: MAXIMISE RECALL of bad loans (catch as many defaulters
    # as possible), subject to a minimum precision floor.
    #
    # Why constrained? Pure recall maximisation always picks the lowest
    # threshold (flag everything as bad → recall=1.0, precision=~11%).
    # The precision floor ensures the model is still actionable — at least
    # MIN_PRECISION of flagged loans must actually be bad.
    #
    # Given the 89/11 class split in this dataset, 20% precision means
    # the model is ~2x more accurate than random at flagging defaulters.
    # ------------------------------------------------------------------
    MIN_PRECISION = 0.20   # tune this — lower = catch more, reject more good loans

    precision_arr = np.array(precision_bad_scores)
    recall_arr    = np.array(recall_bad_scores)
    f1_arr        = np.array(f1_bad_scores)

    valid_mask = precision_arr >= MIN_PRECISION
    if valid_mask.any():
        # Among thresholds that meet the precision floor,
        # pick the one with the highest recall
        valid_recalls     = np.where(valid_mask, recall_arr, 0.0)
        opt_idx_primary   = int(np.argmax(valid_recalls))
        primary_method    = f"max recall (precision >= {MIN_PRECISION:.0%})"
    else:
        # Fallback: no threshold meets the floor — use best F1 instead
        opt_idx_primary   = int(np.argmax(f1_arr))
        primary_method    = "best F1 (precision floor not met)"

    opt_thr_primary = thresholds[opt_idx_primary]

    # Also compute F1-optimal and accuracy-optimal for reference
    opt_idx_f1  = int(np.argmax(f1_arr))
    opt_thr_f1  = thresholds[opt_idx_f1]
    opt_idx_acc = int(np.argmax(accuracy_scores))
    opt_thr_acc = thresholds[opt_idx_acc]

    print(f"\n--- Primary objective: {primary_method} ---")
    print(f"Optimal threshold:           {opt_thr_primary:.2f}")
    print(f"  Recall    (bad loan / 0):  {recall_arr[opt_idx_primary]:.4f}")
    print(f"  Precision (bad loan / 0):  {precision_arr[opt_idx_primary]:.4f}")
    print(f"  F1        (bad loan / 0):  {f1_arr[opt_idx_primary]:.4f}")
    print(f"  F1        (macro):         {f1_macro_scores[opt_idx_primary]:.4f}")
    print(f"  Accuracy:                  {accuracy_scores[opt_idx_primary]:.4f}")
    print(f"\n--- Reference: best bad-loan F1 ---")
    print(f"Threshold: {opt_thr_f1:.2f}  |  "
          f"F1: {f1_arr[opt_idx_f1]:.4f}  |  "
          f"Rec: {recall_arr[opt_idx_f1]:.4f}  |  "
          f"Pre: {precision_arr[opt_idx_f1]:.4f}")
    print(f"\n--- Reference: best accuracy ---")
    print(f"Threshold: {opt_thr_acc:.2f}  |  "
          f"Accuracy: {accuracy_scores[opt_idx_acc]:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(thresholds, recall_arr,    linewidth=2.5, label="Recall (bad loan)")
    axes[0].plot(thresholds, precision_arr, linewidth=2.5, label="Precision (bad loan)")
    axes[0].plot(thresholds, f1_arr,        linewidth=2,   label="F1 (bad loan)", linestyle="--")
    axes[0].axhline(MIN_PRECISION, color="grey", linestyle=":", linewidth=1.5,
                    label=f"Precision floor ({MIN_PRECISION:.0%})")
    axes[0].axvline(opt_thr_primary, color="r", linestyle="--", linewidth=2,
                    label=f"Selected: {opt_thr_primary:.2f}")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Threshold Optimisation — Bad Loan (Class 0)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(thresholds, accuracy_scores, color="green", linewidth=2)
    axes[1].axvline(opt_thr_acc, color="r", linestyle="--",
                    label=f"Best Accuracy: {opt_thr_acc:.2f}")
    axes[1].axvline(opt_thr_primary, color="blue", linestyle="--",
                    label=f"Selected: {opt_thr_primary:.2f}")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Threshold Optimisation — Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return float(opt_thr_primary)


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

        mlflow.log_param("model_type",   "logistic_regression")
        mlflow.log_param("max_iter",     1000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metric("AUC",         auc)
        mlflow.log_metric("precision",   precision)
        mlflow.log_metric("recall",      recall)
        mlflow.log_metric("f1_score",    f1)
        mlflow.sklearn.log_model(lr_model, "logistic_model")

        print(f"AUC:                      {auc:.4f}")
        print(f"Precision (macro):        {precision:.4f}")
        print(f"Recall    (macro):        {recall:.4f}")
        print(f"F1        (macro):        {f1:.4f}")
        print(f"Precision (bad loan / 0): {precision_score(y, preds, pos_label=0):.4f}")
        print(f"Recall    (bad loan / 0): {recall_score(y, preds, pos_label=0):.4f}")
        print(f"F1        (bad loan / 0): {f1_score(y, preds, pos_label=0):.4f}")

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

        print(f"Random Forest AUC:        {rf_auc:.4f}")
        print(f"Precision (macro):        {rf_precision:.4f}")
        print(f"Recall    (macro):        {rf_recall:.4f}")
        print(f"F1        (macro):        {rf_f1:.4f}")
        print(f"Precision (bad loan / 0): {precision_score(y, rf_preds, pos_label=0):.4f}")
        print(f"Recall    (bad loan / 0): {recall_score(y, rf_preds, pos_label=0):.4f}")
        print(f"F1        (bad loan / 0): {f1_score(y, rf_preds, pos_label=0):.4f}")

    # ------------------------------------------------------------------
    # 6. Logistic Regression with p-values — initial fit
    # ------------------------------------------------------------------
    print("\nTraining Logistic Regression with p-values (initial fit)...")
    reg2 = LogisticRegressionWithPValues(class_weight="balanced")
    reg2.fit(inputs_train, loan_data_targets_train)

    summary_table = build_summary_table(reg2, inputs_train.columns)
    print("\nInitial model summary:")
    print(summary_table[["Feature name", "Coefficients", "p_values_fmt", "significant"]].to_string())

    # ------------------------------------------------------------------
    # 7. Drop statistically insignificant features (p-value > 0.05)
    # ------------------------------------------------------------------
    P_VALUE_THRESHOLD = 0.05

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
    print("\nSecond pass model summary:")
    print(summary_table_final[["Feature name", "Coefficients", "p_values_fmt", "significant"]].to_string())

    # ------------------------------------------------------------------
    # 8b. ONE additional p-value pass (third pass total).
    #
    #     We do exactly one more pass — not an unbounded loop. Iterative
    #     stepwise elimination cascades: each drop shifts p-values of
    #     correlated features, triggering further drops, until the model
    #     is stripped to a skeleton (15 features, AUC 0.65 vs 0.69).
    #     Two passes is the right balance — cleans up the obvious noise
    #     without over-pruning genuinely useful predictors.
    # ------------------------------------------------------------------
    insig_pass3 = summary_table_final[
        (summary_table_final.index > 0) & (summary_table_final["p_values"] > P_VALUE_THRESHOLD)
    ]["Feature name"].tolist()

    if insig_pass3:
        print(f"\nPass 3: dropping {len(insig_pass3)} insignificant feature(s):")
        for feat in insig_pass3:
            pval = summary_table_final.loc[
                summary_table_final["Feature name"] == feat, "p_values"
            ].values[0]
            print(f"  {feat:50s}  p = {pval:.4f}")

        inputs_train_significant = inputs_train_significant.drop(
            columns=[f for f in insig_pass3 if f in inputs_train_significant.columns]
        )
        print(f"  Features remaining after pass 3: {inputs_train_significant.shape[1]}")

        reg2_final = LogisticRegressionWithPValues(class_weight="balanced")
        reg2_final.fit(inputs_train_significant, loan_data_targets_train)

        summary_table_final = build_summary_table(reg2_final, inputs_train_significant.columns)
        print("\nFinal model summary (pass 3 — stopping here to preserve predictive power):")
        print(summary_table_final[
            ["Feature name", "Coefficients", "p_values_fmt", "significant"]
        ].to_string())
    else:
        print("\nPass 3: no further insignificant features — model is clean.")

    # ------------------------------------------------------------------
    # 9. Find optimal threshold using sklearn CV (no manual loop).
    #
    #    Pass reg2_final.model (the raw sklearn LR) and only the columns
    #    it was actually fitted on (feature_names_in_), NOT the full
    #    inputs_train_significant which still contains the zero-variance
    #    columns that were dropped inside the wrapper's fit().
    # ------------------------------------------------------------------
    model_features_for_cv = reg2_final.feature_names_in_
    optimal_threshold = optimise_threshold(
        reg2_final.model,
        inputs_train_significant[model_features_for_cv],
        loan_data_targets_train,
    )

    # ------------------------------------------------------------------
    # 10. Evaluate the final model at the optimal threshold and log to
    #     MLflow. We use the raw sklearn model + apply threshold manually
    #     — no custom wrapper class, so pickle works everywhere.
    # ------------------------------------------------------------------
    model_features = reg2_final.feature_names_in_
    X_model        = inputs_train_significant[model_features]
    y_arr          = np.array(loan_data_targets_train).ravel()

    final_preds_proba = reg2_final.model.predict_proba(X_model)[:, 1]
    final_preds       = (final_preds_proba >= optimal_threshold).astype(int)

    final_auc = roc_auc_score(y_arr, final_preds_proba)
    final_f1  = f1_score(y_arr, final_preds)

    with mlflow.start_run(run_name="pd_model_final"):
        mlflow.log_param("model_type",         "logistic_regression_pvalues")
        mlflow.log_param("class_weight",        "balanced")
        mlflow.log_param("p_value_threshold",   P_VALUE_THRESHOLD)
        mlflow.log_param("n_features",          len(model_features))
        mlflow.log_param("optimal_threshold",   optimal_threshold)
        mlflow.log_metric("AUC",                final_auc)
        mlflow.log_metric("f1_score",           final_f1)
        mlflow.log_metric("precision_bad_loan", precision_score(y_arr, final_preds, pos_label=0))
        mlflow.log_metric("recall_bad_loan",    recall_score(y_arr, final_preds, pos_label=0))
        mlflow.log_metric("f1_bad_loan",        f1_score(y_arr, final_preds, pos_label=0))

    print(f"\nFinal model at threshold {optimal_threshold:.2f}:")
    print(f"  AUC:                  {final_auc:.4f}")
    print(f"  F1 (macro):           {final_f1:.4f}")
    print(f"  Precision (bad loan): {precision_score(y_arr, final_preds, pos_label=0):.4f}")
    print(f"  Recall    (bad loan): {recall_score(y_arr, final_preds, pos_label=0):.4f}")
    print(f"  F1        (bad loan): {f1_score(y_arr, final_preds, pos_label=0):.4f}")

    # ------------------------------------------------------------------
    # 11. Save artefacts:
    #     - pd_model.sav            : plain sklearn LogisticRegression
    #     - pd_model_features.pkl   : exact feature list the model expects
    #     - pd_model_threshold.pkl  : optimal threshold as a plain float
    #
    #     Nothing custom is pickled — any script can load these without
    #     importing any project-specific classes first.
    # ------------------------------------------------------------------
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(reg2_final.model, f)
    print(f"\nModel saved to:     {MODEL_SAVE_PATH}")

    significant_features = reg2_final.feature_names_in_
    with open(FEATURES_SAVE_PATH, "wb") as f:
        pickle.dump(significant_features, f)
    print(f"Feature list saved: {FEATURES_SAVE_PATH} ({len(significant_features)} features)")

    with open(THRESHOLD_SAVE_PATH, "wb") as f:
        pickle.dump(optimal_threshold, f)
    print(f"Threshold saved:    {THRESHOLD_SAVE_PATH} (threshold = {optimal_threshold:.4f})")


if __name__ == "__main__":
    main()
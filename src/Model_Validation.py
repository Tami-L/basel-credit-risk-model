"""
evaluate.py
-----------
Evaluates the trained scorecard model with a focus on bad loan detection.
Loops over a grid of thresholds and logs each one as its own MLflow run
so you can compare them in the UI and pick the threshold that best fits
your risk appetite (e.g. maximise recall while keeping precision above some floor).

Run: python evaluate.py
Then: mlflow ui   (visit http://localhost:5000 → experiment "credit_scorecard")
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import CalibrationDisplay

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR  = Path("scorecard_outputs")
REPORT_DIR = Path("scorecard_outputs/evaluation")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Threshold grid to sweep ───────────────────────────────────────────────────
# Each value is the minimum p_good required to approve a loan.
# Lower threshold = more loans flagged as bad = higher recall, lower precision.
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Stop the sweep as soon as recall for bad loans reaches this value.
# Pushing beyond 0.75 typically causes precision to collapse and
# good borrowers to be rejected at an unacceptable rate.
RECALL_TARGET = 0.75

# Scorecard scaling
PDO             = 20
REFERENCE_SCORE = 600
REFERENCE_ODDS  = 1.0

mlflow.set_experiment("credit_scorecard")


# ── 1. Load artifacts ─────────────────────────────────────────────────────────
print("Loading artifacts...")

with open(MODEL_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODEL_DIR / "feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

X_test  = pd.read_csv(MODEL_DIR / "X_test.csv",  index_col=0)
X_train = pd.read_csv(MODEL_DIR / "X_train.csv", index_col=0)
y_test  = pd.read_csv(MODEL_DIR / "y_test.csv",  index_col=0).squeeze()
y_train = pd.read_csv(MODEL_DIR / "y_train.csv", index_col=0).squeeze()

print(f"  Test rows: {len(X_test):,}  |  Bad rate: {(y_test == 0).mean():.1%}")


# ── 2. Probabilities (computed once, reused across all threshold runs) ─────────
p_good_test  = model.predict_proba(X_test)[:, 1]
pd_est_test  = 1 - p_good_test
p_good_train = model.predict_proba(X_train)[:, 1]

# Treat bad (0) as the positive class for all bad-loan metrics
y_true_bad = (y_test == 0).astype(int)

# AUC and Gini are threshold-independent — compute once
auc  = roc_auc_score(y_test, p_good_test)
gini = 2 * auc - 1

# KS statistic
ks_df = pd.DataFrame({"y": y_test.values, "p_good": p_good_test})
ks_df = ks_df.sort_values("p_good", ascending=False).reset_index(drop=True)
ks_df["cum_good"] = (ks_df["y"] == 1).cumsum() / (y_test == 1).sum()
ks_df["cum_bad"]  = (ks_df["y"] == 0).cumsum() / (y_test == 0).sum()
ks = (ks_df["cum_good"] - ks_df["cum_bad"]).abs().max()

# Full precision-recall curve (used for the sweep plot)
prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_true_bad, pd_est_test)
thresh_curve = np.append(thresh_curve, 1.0)
f1_curve     = 2 * prec_curve * rec_curve / np.maximum(prec_curve + rec_curve, 1e-9)

# Scorecard scores
factor = PDO / np.log(2)
offset = REFERENCE_SCORE - factor * np.log(REFERENCE_ODDS)
scores_test  = (offset + factor * model.decision_function(X_test)).round(0).astype(int)
scores_train = (offset + factor * model.decision_function(X_train)).round(0).astype(int)

# PSI
bins_psi   = np.linspace(min(scores_train.min(), scores_test.min()),
                         max(scores_train.max(), scores_test.max()), 11)
train_pct  = pd.cut(pd.Series(scores_train), bins=bins_psi).value_counts(sort=False, normalize=True).clip(lower=1e-4)
test_pct   = pd.cut(pd.Series(scores_test),  bins=bins_psi).value_counts(sort=False, normalize=True).clip(lower=1e-4)
psi        = float(((test_pct - train_pct) * np.log(test_pct / train_pct)).sum())
psi_status = "Stable" if psi < 0.10 else ("Monitor" if psi < 0.25 else "Unstable — retrain")


# ── 3. Threshold sweep — one MLflow run per threshold ─────────────────────────
# This is the main loop. For each threshold we compute bad-loan metrics and
# log them so the MLflow UI lets you compare all thresholds side by side.
print(f"\nRunning threshold sweep over: {THRESHOLDS}")
print(f"{'Threshold':<12} {'Recall':>8} {'Precision':>10} {'F1':>8} {'TP':>8} {'FN':>8} {'FP':>8}")
print("-" * 65)

threshold_summary = []

for threshold in THRESHOLDS:
    # Flag a loan as bad when pd_est >= (1 - threshold)
    # i.e. when the model is not confident enough that the loan is good
    y_pred_bad = (pd_est_test >= (1 - threshold)).astype(int)

    cm = confusion_matrix(y_true_bad, y_pred_bad)
    tn, fp, fn, tp = cm.ravel()

    recall_bad    = tp / (tp + fn)
    precision_bad = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_bad        = (2 * precision_bad * recall_bad) / (precision_bad + recall_bad) if (precision_bad + recall_bad) > 0 else 0.0

    print(f"{threshold:<12.2f} {recall_bad:>8.4f} {precision_bad:>10.4f} {f1_bad:>8.4f} {tp:>8,} {fn:>8,} {fp:>8,}")

    threshold_summary.append({
        "threshold":    threshold,
        "recall_bad":   round(recall_bad,    4),
        "precision_bad": round(precision_bad, 4),
        "f1_bad":       round(f1_bad,         4),
        "auc":          round(auc,            4),
        "gini":         round(gini,           4),
        "ks":           round(ks,             4),
        "true_positives":  int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives":  int(tn),
    })

    with mlflow.start_run(run_name=f"threshold={threshold}"):

        # Parameters — what we set
        mlflow.log_param("threshold",    threshold)
        mlflow.log_param("n_test",       len(X_test))

        # Metrics — what we measure
        # Threshold-independent (same across all runs, but logged for easy filtering)
        mlflow.log_metric("auc",         round(auc,          4))
        mlflow.log_metric("gini",        round(gini,         4))
        mlflow.log_metric("ks",          round(ks,           4))
        mlflow.log_metric("psi",         round(psi,          4))

        # Threshold-dependent bad loan metrics
        mlflow.log_metric("recall_bad",    round(recall_bad,    4))
        mlflow.log_metric("precision_bad", round(precision_bad, 4))
        mlflow.log_metric("f1_bad",        round(f1_bad,        4))
        mlflow.log_metric("true_positives",  int(tp))
        mlflow.log_metric("false_negatives", int(fn))
        mlflow.log_metric("false_positives", int(fp))

    # Stop optimising once recall target is reached — going further
    # collapses precision and rejects too many good borrowers.
    if recall_bad >= RECALL_TARGET:
        print(f"  -> Recall target {RECALL_TARGET} reached at threshold={threshold}. Stopping sweep.")
        break


# ── 4. Plots (saved once, not per threshold) ──────────────────────────────────
print("\nGenerating plots...")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, p_good_test)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(REPORT_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# KS chart
fig, ax = plt.subplots(figsize=(7, 5))
x_axis = ks_df.index / len(ks_df)
ax.plot(x_axis, ks_df["cum_good"], label="Cumulative Good")
ax.plot(x_axis, ks_df["cum_bad"],  label="Cumulative Bad")
ks_idx = (ks_df["cum_good"] - ks_df["cum_bad"]).abs().idxmax()
ax.axvline(ks_idx / len(ks_df), color="red", linestyle="--", label=f"KS = {ks:.4f}")
ax.set_xlabel("Population fraction (sorted by score)")
ax.set_ylabel("Cumulative rate")
ax.set_title("KS Chart")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(REPORT_DIR / "ks_chart.png", dpi=150, bbox_inches="tight")
plt.close()

# Precision / Recall for bad loans — all thresholds on one chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresh_curve, rec_curve,  color="#e74c3c", lw=2, label="Recall (Bad) — % defaults caught")
ax.plot(thresh_curve, prec_curve, color="#2980b9", lw=2, label="Precision (Bad)")
ax.plot(thresh_curve, f1_curve,   color="#27ae60", lw=1.5, linestyle="--", label="F1 (Bad)")

# Mark each threshold from our grid
colors = plt.cm.Oranges(np.linspace(0.4, 1.0, len(THRESHOLDS)))
for i, t in enumerate(THRESHOLDS):
    pd_t = 1 - t
    # Find the index in thresh_curve closest to pd_t
    idx = np.argmin(np.abs(thresh_curve - pd_t))
    ax.axvline(pd_t, color=colors[i], linestyle=":", lw=1.2, alpha=0.8)
    ax.annotate(
        f"t={t}",
        xy=(pd_t, 0.05 + i * 0.07),
        fontsize=8, color=colors[i], ha="center",
    )

ax.set_xlabel("PD Threshold  (raise = stricter, fewer approvals)")
ax.set_ylabel("Score")
ax.set_title("Precision & Recall for Bad Loans — All Thresholds")
ax.legend(loc="center right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
fig.savefig(REPORT_DIR / "precision_recall_bad.png", dpi=150, bbox_inches="tight")
plt.close()

# Confusion matrix at threshold = 0.5 for reference
y_pred_ref = (p_good_test >= 0.5).astype(int)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_ref, ax=ax, colorbar=False)
ax.set_title("Confusion Matrix (threshold=0.5)")
fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# Calibration plot
fig, ax = plt.subplots(figsize=(6, 5))
CalibrationDisplay.from_predictions(
    (y_test == 0).astype(int), pd_est_test,
    n_bins=10, ax=ax, name="Model",
)
ax.set_title("Calibration Plot (PD)")
fig.savefig(REPORT_DIR / "calibration_plot.png", dpi=150, bbox_inches="tight")
plt.close()

# Score distribution — train vs test
fig, ax = plt.subplots(figsize=(8, 5))
bins_hist = np.linspace(min(scores_train.min(), scores_test.min()),
                        max(scores_train.max(), scores_test.max()), 40)
ax.hist(scores_train, bins=bins_hist, alpha=0.5, density=True, label="Train")
ax.hist(scores_test,  bins=bins_hist, alpha=0.5, density=True, label="Test")
ax.set_xlabel("Scorecard Score")
ax.set_ylabel("Density")
ax.set_title("Score Distribution — Train vs Test")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(REPORT_DIR / "score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# Log all plots as artifacts into one final summary MLflow run
with mlflow.start_run(run_name="evaluation_summary"):
    mlflow.log_metric("auc",  round(auc,  4))
    mlflow.log_metric("gini", round(gini, 4))
    mlflow.log_metric("ks",   round(ks,   4))
    mlflow.log_metric("psi",  round(psi,  4))
    for plot_file in REPORT_DIR.glob("*.png"):
        mlflow.log_artifact(str(plot_file))


# ── 5. Save model_metrics.pkl for the Streamlit app ──────────────────────────
# Pick the first row that meets the recall target; if target was never reached
# (model isn't powerful enough), fall back to the row with the highest recall.
rows_at_target = [r for r in threshold_summary if r["recall_bad"] >= RECALL_TARGET]
best_row = rows_at_target[0] if rows_at_target else max(threshold_summary, key=lambda r: r["recall_bad"])

model_metrics = {
    "auc":           round(auc,        4),
    "gini":          round(gini,       4),
    "ks":            round(ks,         4),
    "psi":           round(psi,        4),
    "psi_status":    psi_status,
    "recall_bad":    best_row["recall_bad"],
    "precision_bad": best_row["precision_bad"],
    "f1_bad":        best_row["f1_bad"],
    "best_threshold": best_row["threshold"],
    "true_positives":  best_row["true_positives"],
    "false_negatives": best_row["false_negatives"],
    "false_positives": best_row["false_positives"],
    "threshold_sweep": threshold_summary,   # full sweep so app can render the table
    "recall_target":  RECALL_TARGET,
}
with open(MODEL_DIR / "model_metrics.pkl", "wb") as f:
    pickle.dump(model_metrics, f)
print(f"Saved model_metrics.pkl → {MODEL_DIR / 'model_metrics.pkl'}")

# ── 5b. Save CSVs ─────────────────────────────────────────────────────────────
thresh_summary_df = pd.DataFrame(threshold_summary)
thresh_summary_df.to_csv(REPORT_DIR / "threshold_sweep.csv", index=False)

pd.DataFrame(
    classification_report(y_test, y_pred_ref, output_dict=True)
).T.round(4).to_csv(REPORT_DIR / "classification_report.csv")

pd.DataFrame({
    "feature":     feature_names,
    "coefficient": model.coef_[0].round(6),
    "points_per_unit_woe": (factor * model.coef_[0]).round(2),
}).sort_values("points_per_unit_woe", ascending=False).reset_index(drop=True).to_csv(
    REPORT_DIR / "scorecard_points.csv", index=False
)

print(f"\nAll reports saved to: {REPORT_DIR.resolve()}")


# ── 6. Print summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  EVALUATION REPORT")
print("=" * 65)

print(f"\n  Discrimination (threshold-independent)")
print(f"    AUC   : {auc:.4f}")
print(f"    Gini  : {gini:.4f}  {'OK' if gini > 0.3 else 'Below 0.30 threshold'}")
print(f"    KS    : {ks:.4f}  {'OK' if ks > 0.2 else 'Below 0.20 threshold'}")

print(f"\n  Stability")
print(f"    PSI   : {psi:.4f}  — {psi_status}")

print(f"\n  Bad Loan Detection by Threshold")
print(f"  {'Threshold':<12} {'Recall':>8} {'Precision':>10} {'F1':>8} {'Defaults caught':>16} {'Defaults missed':>16}")
print("  " + "-" * 70)
for row in threshold_summary:
    print(
        f"  {row['threshold']:<12.2f}"
        f" {row['recall_bad']:>8.4f}"
        f" {row['precision_bad']:>10.4f}"
        f" {row['f1_bad']:>8.4f}"
        f" {row['true_positives']:>16,}"
        f" {row['false_negatives']:>16,}"
    )

print(f"\n  Recall target        : {RECALL_TARGET}  (sweep stops when this is reached)")
print(f"  Best threshold chosen: {best_row['threshold']}  (recall={best_row['recall_bad']})")
print(f"  -> In MLflow UI, sort runs by recall_bad to compare thresholds.")
print(f"  -> Run:  mlflow ui")
print("=" * 65)
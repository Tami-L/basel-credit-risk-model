"""
train.py
--------
Trains a logistic regression scorecard model on WoE features.
Each C value tried during cross-validation is logged as its own MLflow run.
The best model gets a final "best_model" run with all artifacts attached.

Run: python train.py
Then: mlflow ui   (visit http://localhost:5000 to browse experiments)
"""

import pickle
import warnings
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = Path("scorecard_outputs")
OUTPUT_DIR = Path("scorecard_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
C_GRID       = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
CV_FOLDS     = 5

# ── MLflow experiment ─────────────────────────────────────────────────────────
mlflow.set_experiment("credit_scorecard")


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")

X = pd.read_csv(INPUT_DIR / "X_woe.csv", index_col=0)
y = pd.read_csv(INPUT_DIR / "y.csv",     index_col=0).squeeze()

with open(INPUT_DIR / "selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

X = X[[col for col in selected_features if col in X.columns]]

print(f"  Rows: {len(X):,}  |  Features: {X.shape[1]}")
print(f"  Bad rate: {(y == 0).mean():.1%}")


# ── 2. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)
print(f"\nSplit — Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# ── 3. Cross-validation over C — one MLflow run per C value ───────────────────
print(f"\nTuning C with {CV_FOLDS}-fold CV...")

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_results = []

for c in C_GRID:
    with mlflow.start_run(run_name=f"cv_C={c}"):

        model_cv = LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        scores = cross_val_score(model_cv, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        mean_auc = round(scores.mean(), 4)
        std_auc  = round(scores.std(),  4)

        # Log the C value as a parameter and CV AUC as a metric
        mlflow.log_param("C",          c)
        mlflow.log_param("cv_folds",   CV_FOLDS)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metric("cv_mean_auc", mean_auc)
        mlflow.log_metric("cv_std_auc",  std_auc)

        cv_results.append({"C": c, "mean_auc": mean_auc, "std_auc": std_auc})
        print(f"  C={c:<8}  AUC = {mean_auc:.4f} ± {std_auc:.4f}")

cv_df  = pd.DataFrame(cv_results).sort_values("mean_auc", ascending=False).reset_index(drop=True)
best_c = float(cv_df.iloc[0]["C"])
print(f"\nBest C = {best_c}  (CV AUC = {cv_df.iloc[0]['mean_auc']:.4f})")


# ── 4. Fit final model — logged as its own "best_model" run ───────────────────
print("\nFitting final model...")

with mlflow.start_run(run_name="best_model"):

    model = LogisticRegression(
        C=best_c,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # ── Parameters
    mlflow.log_param("best_C",         best_c)
    mlflow.log_param("class_weight",   "balanced")
    mlflow.log_param("test_size",      TEST_SIZE)
    mlflow.log_param("random_state",   RANDOM_STATE)
    mlflow.log_param("n_features",     X_train.shape[1])
    mlflow.log_param("n_train",        len(X_train))
    mlflow.log_param("n_test",         len(X_test))

    # ── CV AUC of the best C
    mlflow.log_metric("best_cv_auc", cv_df.iloc[0]["mean_auc"])

    # ── Log the model itself so MLflow can reload it later
    mlflow.sklearn.log_model(model, artifact_path="model")

    # ── Coefficient table as a CSV artifact
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

    # ── CV results table
    cv_path = OUTPUT_DIR / "cv_results.csv"
    cv_df.to_csv(cv_path, index=False)
    mlflow.log_artifact(str(cv_path))

    print("\nTop 10 features by coefficient magnitude:")
    print(coef_df.head(10).to_string(index=False))


# ── 5. Save files to disk (needed by evaluate.py) ─────────────────────────────
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
print("\n=== TRAINING COMPLETE ===")
print(f"  Best C     : {best_c}")
print(f"  Features   : {X_train.shape[1]}")
print(f"  Train rows : {len(X_train):,}")
print(f"  Test rows  : {len(X_test):,}")
print("\nRun  mlflow ui  to browse experiment results.")
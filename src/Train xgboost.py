"""
train_xgboost.py
----------------
XGBoost PD challenger model for the Basel credit risk pipeline.

Input data
----------
Reads binned-but-NOT-WOE-encoded data produced by woe_etl_xgb.py.
Each feature is an integer bin code (0, 1, 2, ...) that preserves ordinal
rank but does NOT pre-linearise the relationship with log-odds.

This is intentionally different from train.py which feeds WOE values to
logistic regression. WOE encoding collapses each bin to a single number on
a linear log-odds scale — ideal for LR but it destroys the non-linear signal
that XGBoost would otherwise discover autonomously. Bin codes let XGBoost
find U-shapes, plateaus, interaction effects and threshold patterns that the
LR model cannot represent.

Run order
---------
    python src/woe_etl.py          
    python src/woe_etl_xgb.py      
    python src/train_xgboost.py    

MLflow experiment: credit_scorecard  (same experiment as the LR model)
  - One run per Optuna trial  (xgb_trial_N)
  - One best_model run        (xgb_best_model) with model artifact + metrics attached
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import optuna
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
from pathlib import Path

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "scorecard_outputs"
MLRUNS  = ROOT / "src" / "mlruns"

# ── Config ─────────────────────────────────────────────────────────────────────
N_TRIALS             = 50     
CV_FOLDS             = 5
RANDOM_STATE         = 42
SCALE_POS_WEIGHT_AUTO = True  # Derive from class imbalance; mirrors class_weight='balanced' in LR
MLFLOW_EXP           = "credit_scorecard"


def load_data():
    """
    Load raw-feature data from preprocess_xgboost.py.
    X_raw_train/test: raw loan features, categoricals integer-encoded.
    y_xgb_train/test: Bad=1, Good=0.
    Same row indices as LR model — AUC comparisons are on identical observations.
    """
    if not (OUTPUTS / "X_raw_train.csv").exists():
        raise FileNotFoundError(
            "X_raw_train.csv not found. Run preprocess_xgboost.py first:\n"
            "    python src/preprocess_xgboost.py"
        )

    X_train = pd.read_csv(OUTPUTS / "X_raw_train.csv", index_col=0)
    X_test  = pd.read_csv(OUTPUTS / "X_raw_test.csv",  index_col=0)
    y_train = pd.read_csv(OUTPUTS / "y_xgb_train.csv", index_col=0).iloc[:, -1].astype(int)
    y_test  = pd.read_csv(OUTPUTS / "y_xgb_test.csv",  index_col=0).iloc[:, -1].astype(int)

    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    features = list(X_train.columns)
    print(f"  Features : {len(features)} (raw — no WOE, no binning)")
    print(f"  Train    : {len(X_train):,} rows  |  Bad rate (Bad=1): {float(y_train.mean()):.2%}")
    print(f"  Test     : {len(X_test):,} rows   |  Bad rate (Bad=1): {float(y_test.mean()):.2%}")

    return X_train, X_test, y_train, y_test, features

def get_scale_pos_weight(y_train) -> float:
    """
    XGBoost scale_pos_weight = n_negative / n_positive.
    After flipping the target (Bad=1, Good=0):
      positive class = Bad  (minority, ~11%)
      negative class = Good (majority, ~89%)
    So weight = n_good / n_bad, which tells XGBoost to pay ~8x more
    attention to each default — mirrors class_weight='balanced' in LR.
    """
    arr = y_train if isinstance(y_train, np.ndarray) else np.array(y_train)
    n_bad  = int((arr == 1).sum())
    n_good = int((arr == 0).sum())
    return round(n_good / n_bad, 4)


def objective(trial, X_arr, y_arr, scale_pos_weight, cv):
    """Optuna objective — maximise mean stratified CV AUC.


    """
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "max_delta_step":   trial.suggest_int("max_delta_step", 0, 10),
        "scale_pos_weight": scale_pos_weight,
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
    }

    fold_aucs = []
    for tr_idx, va_idx in cv.split(X_arr, y_arr):
        X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
        y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr, verbose=False)
        prob = m.predict_proba(X_va)[:, 1]
        fold_aucs.append(roc_auc_score(y_va, prob))

    mean_auc = float(np.mean(fold_aucs))

    with mlflow.start_run(run_name=f"xgb_trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("cv_auc_mean", mean_auc)
        mlflow.log_metric("cv_auc_std",  float(np.std(fold_aucs)))

    return mean_auc


def train_best_model(best_params, X_train, X_test, y_train, y_test, features, scale_pos_weight):
    """Fit final model on full training set and evaluate on held-out test set."""
    final_params = {
        **best_params,
        "scale_pos_weight": scale_pos_weight,
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
    }

    model = xgb.XGBClassifier(**final_params)
    # Convert to numpy explicitly — passing a named DataFrame can cause XGBoost
    #  output constant probabilities if column alignment fails internally
    model.fit(
        X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
        y_train if isinstance(y_train, np.ndarray) else np.array(y_train),
        eval_set=[(
            X_test.values if isinstance(X_test, pd.DataFrame) else X_test,
            y_test if isinstance(y_test, np.ndarray) else np.array(y_test),
        )],
        verbose=False,
    )

    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test  = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_prob_train)
    test_auc  = roc_auc_score(y_test,  y_prob_test)
    gini      = round(2 * test_auc - 1, 4)

    # Bad=1 after target flip
    bad_scores  = y_prob_test[y_test == 1]
    good_scores = y_prob_test[y_test == 0]
    ks_stat, _  = ks_2samp(bad_scores, good_scores)

    importance_df = pd.DataFrame({
        "feature":    features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    metrics = {
        "model":                 "XGBoost",
        "input_encoding":        "binned_integer",  
        "train_auc":             round(train_auc, 4),
        "test_auc":              round(test_auc, 4),
        "gini":                  gini,
        "ks_stat":               round(ks_stat, 4),
        "basel_auc_threshold":   0.70,
        "basel_gini_threshold":  0.40,
        "basel_ks_threshold":    0.30,
        "passes_basel_auc":      test_auc >= 0.70,
        "passes_basel_gini":     gini     >= 0.40,
        "passes_basel_ks":       ks_stat  >= 0.30,
    }

    return model, metrics, importance_df, y_prob_test


def save_outputs(model, metrics, importance_df, y_prob_test, y_test, features):
    with open(OUTPUTS / "xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)

    xgb_metrics = {**metrics, "y_prob_test": y_prob_test, "y_test": y_test if isinstance(y_test, np.ndarray) else np.array(y_test)}
    with open(OUTPUTS / "xgb_model_metrics.pkl", "wb") as f:
        pickle.dump(xgb_metrics, f)

    importance_df.to_csv(OUTPUTS / "xgb_feature_importance.csv", index=False)

    with open(OUTPUTS / "xgb_feature_names.pkl", "wb") as f:
        pickle.dump(features, f)

    print("\n── Outputs saved to scorecard_outputs/ ─────────────────────────")
    print("  xgb_model.pkl")
    print("  xgb_model_metrics.pkl")
    print("  xgb_feature_importance.csv")
    print("  xgb_feature_names.pkl")


def print_results(metrics):
    pad = lambda b: "✓ PASS" if b else "✗ FAIL"
    print("\n── XGBoost PD Model Results (binned input, not WOE) ────────────")
    print(f"  Input encoding : {metrics['input_encoding']}")
    print(f"  Train AUC      : {metrics['train_auc']:.4f}")
    print(f"  Test AUC       : {metrics['test_auc']:.4f}   (≥ 0.70 → {pad(metrics['passes_basel_auc'])})")
    print(f"  Gini           : {metrics['gini']:.4f}   (≥ 0.40 → {pad(metrics['passes_basel_gini'])})")
    print(f"  KS             : {metrics['ks_stat']:.4f}   (≥ 0.30 → {pad(metrics['passes_basel_ks'])})")
    print("────────────────────────────────────────────────────────────────\n")


def main():
    print("Loading binned integer-encoded data (woe_etl_xgb.py outputs)...")
    X_train, X_test, y_train, y_test, features = load_data()

    scale_pos_weight = get_scale_pos_weight(y_train) if SCALE_POS_WEIGHT_AUTO else 1.0
    print(f"  scale_pos_weight: {scale_pos_weight}  (bad:good upweight)")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    mlflow.set_tracking_uri(MLRUNS.as_uri())
    mlflow.set_experiment(MLFLOW_EXP)

    # Convert to numpy once here — objective and train_best_model both need arrays
    X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
    X_test_arr  = X_test.values  if isinstance(X_test,  pd.DataFrame) else np.array(X_test)
    y_train_arr = y_train if isinstance(y_train, np.ndarray) else np.array(y_train)
    y_test_arr  = y_test  if isinstance(y_test,  np.ndarray) else np.array(y_test)

    # ── Deep diagnostic — print raw values before any modelling ─────────────
    print("\n── Data diagnostic ─────────────────────────────────────────────")
    print(f"  X_train_arr shape : {X_train_arr.shape}")
    print(f"  y_train_arr shape : {y_train_arr.shape}")
    print(f"  y_train unique    : {np.unique(y_train_arr, return_counts=True)}")
    print(f"  y_test  unique    : {np.unique(y_test_arr,  return_counts=True)}")
    print(f"  X_train first row : {X_train_arr[0]}")
    print(f"  X_train col range : min={X_train_arr.min()}, max={X_train_arr.max()}")
    print(f"  X_train unique vals per col (first 3 cols): "
          f"{[np.unique(X_train_arr[:, i]).tolist() for i in range(min(3, X_train_arr.shape[1]))]}")

    # Check whether features actually vary — zero variance = no signal
    col_stds = X_train_arr.std(axis=0)
    zero_var_cols = np.where(col_stds == 0)[0]
    if len(zero_var_cols) > 0:
        print(f"  WARNING: {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")
    else:
        print(f"  Feature variance: OK (min std={col_stds.min():.4f})")

    # Check correlation between first feature and target
    from scipy.stats import pointbiserialr
    r, p = pointbiserialr(y_train_arr, X_train_arr[:, 0])
    print(f"  Feature[0] vs target correlation: r={r:.4f}, p={p:.4g}")
    print("────────────────────────────────────────────────────────────────\n")

    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_arr, y_train_arr)
    dummy_auc = roc_auc_score(y_test_arr, dummy.predict_proba(X_test_arr)[:, 1])
    print(f"  Dummy classifier AUC (expect ~0.5): {dummy_auc:.4f}")
    quick_xgb = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=RANDOM_STATE,
                                    scale_pos_weight=scale_pos_weight, n_jobs=-1)
    quick_xgb.fit(X_train_arr, y_train_arr, verbose=False)
    quick_auc = roc_auc_score(y_test_arr, quick_xgb.predict_proba(X_test_arr)[:, 1])
    print(f"  Quick XGBoost AUC (expect > 0.6):  {quick_auc:.4f}")
    if quick_auc <= 0.51:
        raise ValueError(
            "Quick XGBoost AUC is still ~0.5 — see diagnostic above.\n"
            "Likely cause: X_binned and y are still misaligned. "
            "Check that X_binned_train.csv index matches y_train.csv row order."
        )

    print(f"\nRunning Optuna hyperparameter search ({N_TRIALS} trials)...")
    with mlflow.start_run(run_name="xgb_optuna_search"):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(
            lambda trial: objective(trial, X_train_arr, y_train_arr, scale_pos_weight, cv),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

        best_params = study.best_params
        print(f"\n  Best CV AUC : {study.best_value:.4f}")
        print(f"  Best params : {best_params}")

    print("\nFitting final model on full training set...")
    with mlflow.start_run(run_name="xgb_best_model"):
        model, metrics, importance_df, y_prob_test = train_best_model(
            best_params, X_train_arr, X_test_arr, y_train_arr, y_test_arr, features, scale_pos_weight
        )

        mlflow.log_params({**best_params, "input_encoding": "binned_integer"})
        mlflow.log_metrics({
            "train_auc": metrics["train_auc"],
            "test_auc":  metrics["test_auc"],
            "gini":      metrics["gini"],
            "ks_stat":   metrics["ks_stat"],
        })
        mlflow.xgboost.log_model(model, artifact_path="xgb_model")

        imp_path = OUTPUTS / "xgb_feature_importance.csv"
        importance_df.to_csv(imp_path, index=False)
        mlflow.log_artifact(str(imp_path))

    save_outputs(model, metrics, importance_df, y_prob_test, y_test_arr, features)
    print_results(metrics)


if __name__ == "__main__":
    main()
# Basel Credit Risk Model

## Overview

This project implements a **credit risk modelling framework** based on Basel regulatory standards, focused on **Probability of Default (PD)** modelling using machine learning. It covers the full model lifecycle — ETL, training, validation, and scorecard generation — and is designed for **quantitative finance and risk management applications**.

Key design decisions include handling of class imbalance (the dataset is ~89% good loans / 11% bad loans), statistically rigorous feature selection via p-value filtering, and a shared feature-list artefact that keeps all downstream scripts automatically in sync with the trained model.

---

## Project Structure

```
basel-credit-risk-model/
├── data/
│   ├── loan_data_2007_2014.csv              # Raw loan data (not versioned)
│   ├── loan_data_inputs_train.csv           # Preprocessed training features
│   ├── loan_data_targets_train.csv          # Training labels
│   ├── loan_data_inputs_test.csv            # Preprocessed test features
│   └── loan_data_targets_test.csv           # Test labels
├── src/
│   ├── preprocessing.py                     # ETL: feature engineering & train/test split
│   ├── train_pd_model.py                    # Model training & MLflow experiment tracking
│   ├── validate_pd_model.py                 # Model validation & performance metrics
│   ├── Scorecard.py                         # Credit scorecard generation
│   ├── pd_model.sav                         # Serialised final PD model (generated)
│   └── pd_model_features.pkl                # Significant feature list (generated)
└── README.md
```

---

## Scripts

### `preprocessing.py`
Handles all data preparation before modelling:
- Parses and engineers date, categorical, and continuous features (employment length, credit history age, interest rate bins, etc.)
- One-hot encodes categorical columns, fitting the encoder on the **training set only** to prevent data leakage
- Computes imputation statistics (e.g. mean annual income) from the training set and applies them to both splits
- Splits data into train/test sets and saves four CSV artefacts to `data/`

### `train_pd_model.py`
Trains three models and tracks experiments with MLflow:
- **Logistic Regression** (sklearn) — baseline
- **Random Forest** — tree-based comparison
- **Logistic Regression with Wald-test p-values** — final model used for scorecard

All models use `class_weight="balanced"` to address the ~8:1 class imbalance. Metrics reported include AUC, macro precision/recall/F1, and minority-class (bad loan) precision/recall/F1 separately.

After the initial fit, features with p-value > 0.05 are dropped and the model is refit on significant features only. The final model and its exact feature list are saved as `pd_model.sav` and `pd_model_features.pkl` respectively — ensuring all downstream scripts stay in sync automatically.

Includes a cross-validated threshold optimisation step using `StratifiedKFold`.

### `validate_pd_model.py`
Loads `pd_model.sav` and `pd_model_features.pkl` and evaluates the model on the held-out test set:
- Confusion matrix (raw and normalised)
- ROC curve and AUROC
- Gini coefficient
- Kolmogorov-Smirnov (KS) statistic and KS chart

Feature selection is driven entirely by the saved `pd_model_features.pkl` — no hardcoded column lists.

### `Scorecard.py`
Converts the trained model's coefficients into an interpretable credit scorecard:
- Uses the PDO (Points to Double Odds) scaling method
- Outputs scores on a **300–850 South African credit scale**
- Assigns risk tiers (Tier 1 Excellent → Tier 5 High Risk)
- Validates the inverse relationship between score and predicted PD
- Feature selection driven by `pd_model_features.pkl`

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `class_weight="balanced"` on all models | Dataset is ~89% good / 11% bad; without weighting, models trivially predict "good" for everything |
| P-value filtering (threshold = 0.05) | Removes statistically insignificant features after initial fit; model is refit on survivors |
| `pd_model_features.pkl` shared artefact | Saves the exact post-filter, post-zero-variance-drop feature list so validation and scorecard scripts never go out of sync with the model |
| OHE fitted on train only | Prevents test-set category distributions from leaking into the encoder |
| `StratifiedKFold` for threshold optimisation | Preserves class distribution across folds given the imbalance |

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tami-L/basel-credit-risk-model.git
cd basel-credit-risk-model
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

Run the scripts in order:

```bash
# 1. Preprocess raw data → writes train/test CSVs to data/
python src/preprocessing.py

# 2. Train models → writes pd_model.sav and pd_model_features.pkl to src/
python src/train_pd_model.py

# 3. Validate on test set
python src/validate_pd_model.py

# 4. Generate scorecard
python src/Scorecard.py
```

MLflow experiment results are written to `src/mlruns/`. Launch the MLflow UI with:
```bash
mlflow ui --backend-store-uri src/mlruns
```

---

## Dependencies

- Python 3.10+
- pandas, numpy, scipy
- scikit-learn
- mlflow
- matplotlib, seaborn

---

## Notes

- `pd_model.sav` and `pd_model_features.pkl` are generated artefacts — they are not versioned and must be regenerated by running `train_pd_model.py`
- Raw loan data is not included in the repository due to size constraints
- The scorecard uses South African credit bureau score ranges (300–850); these can be adjusted via constants at the top of `Scorecard.py`

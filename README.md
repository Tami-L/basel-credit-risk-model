# Basel Credit Risk Model

## Overview

This project implements a **credit risk modelling framework** based on Basel regulatory standards, focused on **Probability of Default (PD)** modelling using machine learning. It covers the full model lifecycle — ETL, training, validation, and scorecard generation — and is designed for **quantitative finance and risk management applications**.

Key design decisions include handling of class imbalance (the dataset is ~89% good loans / 11% bad loans), statistically rigorous feature selection via p-value filtering, a shared feature-list artefact that keeps all downstream scripts in sync with the trained model, and a full Basel-compliant Expected Loss (EL = PD × LGD × EAD) computation pipeline.

---

## Project Structure

```
basel-credit-risk-model/
├── data/
│   ├── loan_data_2007_2014.csv                            # Raw loan data (not versioned)
│   ├── loan_data_inputs_train.csv                         # Preprocessed training features
│   ├── loan_data_targets_train.csv                        # Training labels
│   ├── loan_data_inputs_test.csv                          # Preprocessed test features
│   ├── loan_data_targets_test.csv                         # Test labels
│   ├── ead_train.csv                                      # EAD per loan — train split (generated)
│   └── ead_test.csv                                       # EAD per loan — test split (generated)
├── src/
│   ├── __pycache__/                                       # Python bytecode cache (not versioned)
│   ├── mlruns/                                            # MLflow experiment tracking data
│   ├── app.py                                             # Application entry point
│   ├── Credit Risk Modeling - Model Fitting.ipynb         # Model training notebook (original)
│   ├── Credit Risk Modeling - Preparation.ipynb           # Data preparation notebook (original)
│   ├── ETL.py                                             # ETL: feature engineering, EAD & train/test split
│   ├── fit_model.py                                       # PD model training & MLflow experiment tracking
│   ├── fit_lgd_model.py                                   # LGD model (Basel II constant, grade-segmented)
│   ├── expected_loss.py                                   # EL = PD × LGD × EAD computation
│   ├── mlflow.db                                          # MLflow tracking database
│   ├── Model_Validation.py                                # Model validation & performance metrics
│   ├── pd_model_features.pkl                              # Significant feature list (generated)
│   ├── pd_model.sav                                       # Serialised final PD model (generated)
│   ├── lgd_model.pkl                                      # Serialised LGD model (generated)
│   ├── el_results.pkl                                     # Loan-level & portfolio EL results (generated)
│   └── Scorecard.py                                       # Credit scorecard generation
├── .gitignore
├── LICENSE
├── mlflow.db                                              # MLflow tracking database (root)
└── README.md
```

---

## Scripts

### `ETL.py`
Handles all data preparation before modelling:
- Parses and engineers date, categorical, and continuous features (employment length, credit history age, interest rate bins, etc.)
- One-hot encodes categorical columns, fitting the encoder on the **training set only** to prevent data leakage
- Computes imputation statistics (e.g. mean annual income) from the training set and applies them to both splits
- Computes **EAD = `funded_amnt` − `total_pymnt`** (floored at 0) from the raw data before splitting, then aligns to train/test indices
- Saves six CSV artefacts to `data/`: inputs, targets, and EAD for both train and test splits

### `app.py`
Application entry point for the project.

### `fit_model.py`
Trains three models and tracks experiments with MLflow:
- **Logistic Regression** (sklearn) — baseline
- **Random Forest** — tree-based comparison
- **Logistic Regression with Wald-test p-values** — final model used for scorecard

All models use `class_weight="balanced"` to address the ~8:1 class imbalance. Metrics reported include AUC, macro precision/recall/F1, and minority-class (bad loan) precision/recall/F1 separately.

After the initial fit, features with p-value > 0.05 are dropped and the model is refit on significant features only. The final model and its exact feature list are saved as `pd_model.sav` and `pd_model_features.pkl` respectively — ensuring all downstream scripts stay in sync automatically.

Includes a cross-validated threshold optimisation step using `StratifiedKFold`.

### `Model_Validation.py`
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

### `fit_lgd_model.py`
Builds a grade-segmented Loss Given Default (LGD) model:
- Applies the **Basel II regulatory constant of 45%** for unsecured retail exposures uniformly across all grades
- Saves a grade-keyed lookup table as `lgd_model.pkl`
- Structured so that per-grade values can be replaced with empirical recovery-rate estimates when recovery data becomes available — no downstream changes required

### `expected_loss.py`
Combines PD, LGD, and EAD to compute Expected Loss at loan and portfolio level:
- Loads `pd_model.sav`, `pd_model_features.pkl`, `lgd_model.pkl`, and `ead_test.csv`
- Reconstructs raw grade labels from one-hot encoded inputs to look up LGD
- Computes **EL = PD × LGD × EAD** for every loan
- Outputs a loan-level DataFrame, a grade-level summary table, and scalar portfolio metrics (total EAD, total EL, EL rate, mean PD, high-risk loan share) saved together in `el_results.pkl`

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `class_weight="balanced"` on all models | Dataset is ~89% good / 11% bad; without weighting, models trivially predict "good" for everything |
| P-value filtering (threshold = 0.05) | Removes statistically insignificant features after initial fit; model is refit on survivors |
| `pd_model_features.pkl` shared artefact | Saves the exact post-filter, post-zero-variance-drop feature list so validation and scorecard scripts never go out of sync with the model |
| OHE fitted on train only | Prevents test-set category distributions from leaking into the encoder |
| `StratifiedKFold` for threshold optimisation | Preserves class distribution across folds given the imbalance |
| EAD computed pre-split from raw data | Ensures EAD index stays aligned with train/test feature splits |
| LGD structured as a lookup table | Basel II constant now; swappable for empirical estimates later without changing `expected_loss.py` |
| `el_results.pkl` bundles loan + grade + portfolio | Single artefact gives the dashboard everything it needs in one load |

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
# 1. Run ETL → writes train/test CSVs + EAD CSVs to data/
python src/ETL.py

# 2. Train PD model → writes pd_model.sav and pd_model_features.pkl to src/
python src/fit_model.py

# 3. Build LGD model → writes lgd_model.pkl to src/
python src/fit_lgd_model.py

# 4. Compute Expected Loss → writes el_results.pkl to src/
python src/expected_loss.py

# 5. Validate PD model on test set
python src/Model_Validation.py

# 6. Generate scorecard
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

- `pd_model.sav`, `pd_model_features.pkl`, `lgd_model.pkl`, and `el_results.pkl` are generated artefacts — they are not versioned and must be regenerated by running the pipeline in order
- `ead_train.csv` and `ead_test.csv` in `data/` are also generated — re-run `ETL.py` to regenerate them
- Raw loan data is not included in the repository due to size constraints
- The scorecard uses South African credit bureau score ranges (300–850); these can be adjusted via constants at the top of `Scorecard.py`
- `Credit Risk Modeling - Model Fitting.ipynb` and `Credit Risk Modeling - Preparation.ipynb` are the original exploratory notebooks; the production logic has been refactored into the `.py` scripts
- `__pycache__/` and `mlruns/` are runtime-generated and should remain in `.gitignore`

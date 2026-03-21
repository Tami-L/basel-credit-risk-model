# Basel Credit Risk Model

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tami-l-basel-credit-risk-model-srcapp-dbjaph.streamlit.app/)

## Overview

This project implements a **Basel II/III compliant credit risk modelling framework** built around the industry-standard **Weight of Evidence (WoE) / Information Value (IV)** methodology. It covers the full model lifecycle — ETL, WoE transformation, logistic regression training, scorecard generation, and a Streamlit dashboard — and is calibrated to the **South African credit market** using TransUnion SA / FICO Score 6 score bands (300–850) and NCR-aligned default rate assumptions.

The pipeline is designed around two principles: regulatory defensibility (every transformation is auditable and monotonic) and production correctness (no post-origination data leaks into the model at any stage).

---

## Project Structure

```
basel-credit-risk-model/
├── data/
│   └── loan_data_2007_2014.csv          # Raw loan data (not versioned)
├── scorecard_outputs/                   # Generated artefacts — re-run pipeline to regenerate
│   ├── X_woe.csv                        # Full WoE-encoded dataset
│   ├── y.csv                            # Binary target aligned to X_woe
│   ├── woe_mappings.pkl                 # WoE bin definitions per feature
│   ├── selected_features.pkl            # Features selected after IV + correlation filter
│   ├── iv_summary.csv                   # IV per feature with band classification
│   ├── model.pkl                        # Fitted LogisticRegression
│   ├── feature_names.pkl                # Ordered feature list matching model.coef_
│   ├── X_train.csv / X_test.csv         # Train/test WoE splits
│   ├── y_train.csv / y_test.csv         # Train/test targets
│   ├── cv_results.csv                   # Cross-validation AUC per C value
│   ├── coef_summary.csv                 # Model coefficients sorted by magnitude
│   ├── model_metrics.pkl                # Evaluation metrics for Streamlit app
│   └── evaluation/                      # Plots and reports from evaluate.py
│       ├── roc_curve.png
│       ├── ks_chart.png
│       ├── calibration_plot.png
│       ├── confusion_matrix.png
│       ├── precision_recall_bad.png
│       ├── score_distribution.png
│       ├── discrimination_metrics.csv
│       ├── classification_report.csv
│       ├── threshold_sweep.csv
│       └── scorecard_points.csv
├── src/
│   ├── mlruns/                          # MLflow experiment tracking data
│   ├── app.py                           # Streamlit dashboard (3 pages)
│   ├── woe_etl.py                       # WoE/IV ETL pipeline
│   ├── train.py                         # Model training with MLflow tracking
│   ├── evaluate.py                      # Model evaluation with threshold sweep
│   ├── Scorecard.py                     # SA credit scorecard generation
│   ├── fit_lgd_model.py                 # LGD model (Basel II constant, grade-segmented)
│   ├── expected_loss.py                 # EL = PD × LGD × EAD computation
│   ├── lgd_model.pkl                    # Serialised LGD model (generated)
│   └── el_results.pkl                   # Loan-level & portfolio EL results (generated)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Scripts

### `woe_etl.py`
The core data preparation pipeline. Takes raw loan data and produces a clean, WoE-encoded, model-ready dataset.

- Creates binary target: 0 = Bad (Charged Off, Default, Late 31–120 days), 1 = Good
- Applies an **origination-time feature whitelist** — drops all post-origination columns (`last_pymnt_d`, `total_rec_prncp`, etc.) before any modelling begins, preventing data leakage
- Bins numerical variables using **quantile binning** then enforces **monotonic default rates** per bin, merging bins until the relationship is directionally consistent — the core WoE pre-condition
- Groups rare categorical levels (< 2% frequency) into `"Other"`; assigns `"Missing"` as its own bin for null values
- Computes **WoE and IV** for every variable with Laplace smoothing to avoid ±inf from empty cells
- Selects features using IV thresholds (drop < 0.02, keep ≥ 0.10) then removes correlated pairs (Spearman > 0.70), keeping the higher-IV feature
- Replaces all feature values with their WoE equivalents — the final dataset contains no raw values
- Saves all artefacts to `scorecard_outputs/` including `woe_mappings.pkl` for inference-time transformation

### `train.py`
Trains a regularised logistic regression on the WoE-encoded features with full MLflow tracking.

- Loads `X_woe.csv`, `y.csv`, and `selected_features.pkl` from `scorecard_outputs/`
- Performs **stratified train/test split** (80/20) to preserve the 89/11 good/bad ratio
- Uses `class_weight="balanced"` — the single most impactful change for recall on imbalanced credit data
- Sweeps `C ∈ [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]` with 5-fold stratified CV scored on ROC-AUC; each C value is logged as a separate MLflow run
- Fits the final model at the best C and logs it as a `best_model` run with the model artifact, coefficient table, and CV results attached
- Saves `model.pkl`, `feature_names.pkl`, train/test splits, and `coef_summary.csv` to `scorecard_outputs/`

### `evaluate.py`
Evaluates the trained model with a focus on bad loan detection and threshold optimisation.

- Computes threshold-independent metrics: **ROC-AUC, Gini coefficient, KS statistic, PSI**
- Runs a **threshold sweep** over `[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]` — each threshold is logged as a separate MLflow run with recall, precision, F1, TP, FN, FP for the bad class
- **Stops the sweep as soon as recall reaches 0.75** — pushing beyond this collapses precision and rejects too many good borrowers at an unacceptable rate; the target is configurable via `RECALL_TARGET`
- Selects the best threshold as the first one that meets the recall target (most conservative among qualifying thresholds)
- Saves `model_metrics.pkl` consumed by the Streamlit app, plus 6 plots and 4 CSV reports to `scorecard_outputs/evaluation/`

### `Scorecard.py`
Converts model coefficients into an interpretable credit scorecard calibrated to the South African market.

- Uses the standard **PDO (Points to Double Odds)** scaling formula
- Calibrated to SA market conditions: `PDO=20`, `base_odds=8:1` (~11% default rate, aligned with NCR data), `base_score=620`
- Outputs scores on the **TransUnion SA / FICO Score 6 300–850 scale**
- Applies official **TransUnion SA risk bands**: Excellent (767–850), Good (681–766), Favourable (614–680), Average (583–613), Below Average (527–582), Unfavourable (487–526), Poor (300–486)
- Approval threshold set at **614 (bottom of Favourable)** with a "Refer" zone for borderline cases (604–613)
- Includes NCA (National Credit Act) compliance note: score is one input only; affordability assessment required separately under Act 34 of 2005

### `app.py`
Streamlit dashboard with three pages.

**Applicant Assessment** — takes raw loan application inputs, runs them through `transform_inference` (applies saved WoE mappings), scores with the logistic regression, and displays credit score, risk band, PD, EL, RWA, and Basel capital requirement for the individual loan. Decision shown as Approve / Refer / Decline based on TransUnion SA score bands.

**Portfolio Dashboard** — loads `el_results.pkl` and displays portfolio-level EL, RWA, and capital requirement metrics with grade-level breakdowns and concentration curves.

**Model Performance** — loads `model_metrics.pkl` and displays AUC, Gini, KS, PSI metric cards alongside the full threshold sweep chart and table showing how recall, precision and F1 for bad loans change as the threshold moves.

### `fit_lgd_model.py`
Builds a grade-segmented LGD model applying the **Basel II regulatory constant of 45%** for unsecured retail exposures. Structured as a lookup table so empirical recovery-rate estimates can replace the constant later without downstream changes.

### `expected_loss.py`
Computes **EL = PD × LGD × EAD** at loan and portfolio level, outputs grade-level summaries and saves `el_results.pkl` for the dashboard.

---

## Methodology: Why WoE/IV

The pipeline uses WoE/IV logistic regression rather than a raw-feature ML approach because this is a **Basel II IRB regulatory PD model**, not a pure predictive system.

| Requirement | Raw ML approach | WoE/IV approach |
|---|---|---|
| SARB/PA regulatory acceptance | Difficult to defend | Standard — accepted methodology |
| Interpretability | Low (many encoded columns) | High — each feature has a WoE table |
| Monotonicity per feature | Not guaranteed | Enforced at binning stage |
| Missing value handling | Ad-hoc imputation | Clean — Missing gets its own WoE bin |
| PD calibration | Requires post-hoc calibration | Natively calibrated |
| Scorecard conversion | Complex | Direct — coefficients → points |
| NCA adverse action explainability | Difficult | Straightforward |

---

## South African Market Calibration

| Parameter | Value | Basis |
|---|---|---|
| Score scale | 300–850 | FICO Score 6 / TransUnion SA standard |
| Score bands | 7 bands (Poor → Excellent) | TransUnion SA official bands |
| Base odds | 8:1 | ~11% default rate — NCR SA unsecured credit data |
| PDO | 20 | Standard for high-default-rate unsecured markets |
| Base score | 620 | Anchors 8:1 borrower at mid-Favourable tier |
| Approval cut-off | 614 | Bottom of Favourable — SA lender convention |
| Regulatory framework | NCA Act 34 of 2005 | Score is one input; affordability assessed separately |

---

## Installation

```bash
git clone https://github.com/Tami-L/basel-credit-risk-model.git
cd basel-credit-risk-model
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## Usage

Run the pipeline in order:

```bash
# 1. WoE ETL → writes all artefacts to scorecard_outputs/
python src/woe_etl.py

# 2. Train model → writes model.pkl, feature_names.pkl, splits to scorecard_outputs/
python src/train.py

# 3. Evaluate → writes model_metrics.pkl and evaluation/ reports to scorecard_outputs/
python src/evaluate.py

# 4. Build LGD model → writes lgd_model.pkl to src/
python src/fit_lgd_model.py

# 5. Compute Expected Loss → writes el_results.pkl to src/
python src/expected_loss.py

# 6. Launch dashboard
streamlit run src/app.py
```

Browse MLflow experiment results:
```bash
mlflow ui --backend-store-uri src/mlruns
```

The `credit_scorecard` experiment contains:
- One run per C value from cross-validation (`cv_C=0.001` … `cv_C=10.0`)
- One `best_model` run with the fitted model and coefficient table attached
- One run per threshold from the evaluation sweep (`threshold=0.3` … up to the recall target)
- One `evaluation_summary` run with all plots attached

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Origination-time feature whitelist | Drops post-origination columns (payment history, last payment date etc.) before binning — prevents leakage that inflated old model AUC |
| WoE encoding replaces scaling | Compresses outliers, handles missing values, puts all features on a comparable log-odds scale — no separate normalisation needed |
| Monotonic bin enforcement | Required for WoE validity; non-monotonic features produce unstable, non-interpretable coefficients |
| `class_weight="balanced"` | 89/11 imbalance; without weighting the model learns to predict Good for everything and misses almost all defaults |
| IV thresholds for feature selection | Drop < 0.02 (useless), keep ≥ 0.10 (strong); principled and auditable vs arbitrary manual selection |
| Spearman correlation filter | Removes correlated pairs before fitting; correlated WoE features destabilise logistic regression coefficients |
| Recall target of 0.75 | Pushing recall beyond 0.75 collapses precision — too many good borrowers rejected, violating NCA fairness obligations |
| Stratified CV for C tuning | Preserves bad loan proportion across folds; without stratification, minority-class signal is uneven per fold |
| SA base odds 8:1 | NCR data shows ~11% default rate in unsecured credit; 50:1 (old setting) implied ~2% which is wrong for this market |
| `el_results.pkl` bundles loan + grade + portfolio | Single artefact gives the dashboard everything it needs in one load |

---

## Dependencies

- Python 3.10+
- pandas, numpy, scipy
- scikit-learn
- mlflow
- matplotlib
- streamlit
- plotly

---

## Notes

- All artefacts in `scorecard_outputs/` are generated — they are not versioned and must be rebuilt by running the pipeline in order
- `lgd_model.pkl` and `el_results.pkl` in `src/` are also generated
- Raw loan data is not included due to size — source from Lending Club / Kaggle
- `woe_etl.py` exposes `transform_inference()` and `load_woe_artifacts()` for applying saved WoE mappings to new data at inference time without re-fitting
- The recall target (0.75) and all SA scorecard parameters (PDO, base odds, bands) are defined as named constants at the top of their respective files and can be adjusted without touching the logic
# Basel Credit Risk Model

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tami-l-basel-credit-risk-model-srcapp-dbjaph.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Basel II/III](https://img.shields.io/badge/Basel-II%2FIII%20IRB-green.svg)]()
[![South African Market](https://img.shields.io/badge/Market-South%20Africa%20(NCR)-orange.svg)]()

---

## Model Performance at a Glance

> All metrics evaluated on a held-out test set. Basel II IRB minimum thresholds shown for reference.

| Metric | Logistic Regression (WOE) | XGBoost (binned features) | Basel II IRB Minimum | LR Status | XGB Status |
|---|---|---|---|---|---|
| **AUC** | 68.71 | 71.03 | ≥ 0.70 | ❌ Fail | ✅ Pass |
| **Gini** | 37.43 | 42.07 | ≥ 0.40 | ❌ Fail | ✅ Pass |
| **KS Statistic** | 27.10 | 30.28 | ≥ 0.30 | ❌ Fail | ✅ Pass |
| **PSI (stability)** | 0.0001 | — | < 0.10 (stable) | ✅ Stable | — |

> As trained, the WOE logistic regression falls just short of the Basel II IRB discrimination minimums (AUC/Gini/KS). The XGBoost challenger clears all three. This is exactly why the repo keeps both models side by side — see [Why Two Models](#why-two-models).

---

## What This Model Does

This is a **Basel II/III Internal Ratings-Based (IRB) PD model** for unsecured retail credit, calibrated to the South African market using NCR default rate assumptions and the TransUnion SA / FICO Score 6 score band structure (300–850).

It covers the full regulatory model lifecycle:

- **PD model** — two implementations: WOE logistic regression (regulatory standard) and XGBoost (challenger model for comparison)
- **LGD model** — Basel II 45% regulatory constant for unsecured retail, structured for empirical override
- **EAD** — drawn balance at origination
- **Expected Loss** — EL = PD × LGD × EAD at loan and portfolio level
- **Scorecard** — PDO-scaled points on the 300–850 TransUnion SA scale with NCA-compliant adverse action logic
- **Capital requirement** — RWA and regulatory capital per loan and portfolio

---

## Why Two Models

The **WOE logistic regression** is the primary regulatory model. It is the accepted methodology under SARB/PA guidance — every transformation is auditable, coefficients are interpretable, monotonicity is enforced at the binning stage, and it converts directly to a scorecard without post-hoc recalibration.

The **XGBoost challenger** deliberately uses a *different* feature representation from the LR model: integer bin codes (preserving ordinal rank) rather than WOE values. WOE collapses each bin to a single point on a linear log-odds scale — ideal for LR, but it destroys non-linear signal (U-shapes, plateaus, interactions) that XGBoost can otherwise find on its own. Both models are tracked in the same MLflow experiment. XGBoost serves two purposes: (1) a performance benchmark to confirm the LR model is not materially underfit, and (2) a candidate for a parallel shadow model if the regulator permits. It cannot be used as the primary IRB model without additional model risk management work (SHAP-based explainability, stability testing under stress).

---

## Basel II IRB Compliance

| Requirement | Implementation |
|---|---|
| PD estimation methodology | WOE/IV logistic regression — SARB/PA accepted standard |
| Through-the-cycle calibration | Base odds 8:1, anchored to NCR SA unsecured default rate (~11%) |
| Minimum discrimination (Gini ≥ 0.40) | Checked against Basel thresholds in `fit_model.py` and reported in `Model_Validation.py` (prints PASS/FAIL per metric — does not currently halt the pipeline on failure) |
| Monotonicity per risk driver | Enforced at WoE binning stage — non-monotonic bins are merged |
| Data leakage prevention | Origination-time feature whitelist removes all post-origination columns before any transformation |
| Missing value handling | Missing values receive their own WOE bin — no imputation required |
| Capital computation | RWA using Basel II IRB formula; LGD floored at 45% for unsecured retail |
| Adverse action explainability | Each score band maps to specific WOE features — NCA Act 34 of 2005 compliant |

---

## South African Market Calibration

| Parameter | Value | Basis |
|---|---|---|
| Score scale | 300–850 | FICO Score 6 / TransUnion SA |
| Base odds | 8:1 | ~11% default rate — NCR SA unsecured credit |
| PDO | 20 | Standard for high-default-rate unsecured markets |
| Base score | 620 | Anchors 8:1 borrower at mid-Favourable |
| Approval cut-off | 614 | Bottom of Favourable — SA lender convention |
| Regulatory framework | NCA Act 34 of 2005 | Score is one input; affordability assessed separately |

**TransUnion SA Risk Bands:**

| Band | Score Range | Typical Default Rate |
|---|---|---|
| Excellent | 767–850 | < 2% |
| Good | 681–766 | 2–5% |
| Favourable | 614–680 | 5–10% |
| Average | 583–613 | 10–15% |
| Below Average | 527–582 | 15–22% |
| Unfavourable | 487–526 | 22–35% |
| Poor | 300–486 | > 35% |

---

## Repository Structure

```
basel-credit-risk-model/
├── src/
│   ├── ETL.py                          # WOE/IV ETL — bins, encodes, selects features
│   ├── fit_model.py                    # Logistic regression PD model with MLflow tracking
│   ├── preprocess_xgboost.py           # Raw + integer-binned feature prep for the XGB challenger
│   ├── Train xgboost.py                # XGBoost challenger model with Optuna tuning
│   ├── Model_Validation.py             # Discrimination metrics, threshold sweep, plots
│   ├── Scorecard.py                    # PDO scorecard — 300–850 TransUnion SA scale
│   ├── fit_lgd_model.py                # LGD model (Basel II 45% constant, grade-segmented)
│   ├── expected_loss.py                # EL = PD × LGD × EAD at loan and portfolio level
│   ├── app.py                          # Streamlit dashboard — assessment, portfolio, monitoring
│   ├── Credit Risk Modeling - Preparation.ipynb    # Exploratory notebook — EDA / WOE prep
│   ├── Credit Risk Modeling - Model Fittting.ipynb # Exploratory notebook — model fitting
│   ├── pd_model.sav                    # Fitted PD model (pickled, used by expected_loss.py)
│   ├── pd_model_features.pkl
│   ├── pd_model_threshold.pkl
│   ├── lgd_model.pkl                   # Fitted LGD model
│   └── el_results.pkl                  # Saved Expected Loss output
├── scorecard_outputs/                  # Generated artefacts (rebuild by re-running the pipeline)
│   ├── model.pkl, xgb_model.pkl, global_model.pkl, seg_models.pkl
│   ├── woe_mappings.pkl, bin_mappings.pkl, bin_mappings_xgb.pkl
│   ├── feature_names.pkl, xgb_feature_names.pkl, selected_features.pkl
│   ├── model_metrics.pkl, xgb_model_metrics.pkl
│   └── evaluation/                     # ROC, KS, calibration, confusion matrix plots
├── mlruns/                             # Local MLflow tracking store
├── lgd_model.pkl                       # LGD model artefact (root copy)
├── requirements.txt
└── README.md
```

> Note: `ETL.py`, `fit_model.py`, and `Model_Validation.py` correspond to what their own docstrings call `woe_etl.py`, `train.py`, and `evaluate.py` respectively — the files were renamed at some point but the internal docstrings weren't updated. Referenced below by their actual on-disk filenames.

---

## Pipeline Execution Order

```bash
# 1. WOE ETL — bins continuous variables, enforces monotonicity, computes IV, encodes features
python src/ETL.py

# 2. Train logistic regression PD model
python src/fit_model.py

# 3. Prep raw + integer-binned features for the XGBoost challenger
python src/preprocess_xgboost.py

# 4. Train XGBoost challenger model (Optuna hyperparameter search)
python "src/Train xgboost.py"

# 5. Evaluate both models — writes threshold sweep, classification report, and evaluation plots
python src/Model_Validation.py

# 6. LGD model
python src/fit_lgd_model.py

# 7. Expected Loss
python src/expected_loss.py

# 8. Streamlit dashboard
streamlit run src/app.py
```

MLflow experiment tracking:
```bash
mlflow ui --backend-store-uri mlruns
# → http://localhost:5000
# Experiment: credit_scorecard_irb
#   LR runs:  cv_C sweep … best_model
#   XGB runs: xgb_trial_N … xgb_best_model
```

---

## Key Methodology Decisions

| Decision | Rationale |
|---|---|
| WOE encoding rather than raw features | Compresses outliers, handles missing values natively, puts all features on a comparable log-odds scale — no separate normalisation needed |
| Monotonic bin enforcement | Required for WOE validity and SARB model risk guidance; non-monotonic features produce unstable, non-interpretable coefficients |
| IV thresholds for feature selection | Drop < 0.02 (no predictive power), keep ≥ 0.10 (material); principled and auditable |
| `class_weight="balanced"` (LR) / `scale_pos_weight` (XGB) | 89/11 good/bad split; without upweighting, both models learn to predict Good for everything and miss almost all defaults |
| Recall target 0.75 for threshold selection | Pushing beyond 0.75 collapses precision — too many good borrowers rejected, creating NCA fairness exposure |
| Origination-time feature whitelist | Drops all post-origination columns before any transformation — prevents leakage that inflates AUC on payment history fields |
| Spearman > 0.70 correlation filter | Correlated WOE features destabilise LR coefficients; keep the higher-IV feature from each correlated pair |
| XGBoost uses integer bin codes, not WOE values | Preserves ordinal rank without pre-linearising to log-odds, letting XGBoost discover non-linear patterns WOE would otherwise flatten out |

---

## Installation

```bash
git clone https://github.com/Tami-L/basel-credit-risk-model.git
cd basel-credit-risk-model
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

`xgboost` and `optuna` are already pinned in `requirements.txt`.

Data source: Lending Club loan data 2007–2014, available on Kaggle.

⚠️ **The data path is currently hardcoded to the original author's machine** — `ETL.py`, `preprocess_xgboost.py`, and `fit_lgd_model.py` default to an absolute local path (`/Users/lindokuhletami/Desktop/Space/data/...`). Before running the pipeline, update the `DATA_PATH` constant near the top of each of those three files to point at your own CSV (e.g. `data/loan_data_2007_2014.csv`). `expected_loss.py` is more forgiving — it checks `data/` relative to the repo root first and only falls back to the hardcoded path.

---

## Notes for Model Risk / Validation

- All artefacts in `scorecard_outputs/` are generated outputs — they are not versioned and must be rebuilt by running the pipeline in order. This is intentional: the pipeline is the single source of truth.
- `ETL.py` exposes `apply_woe_transform(df, mappings)`, which applies the saved `woe_mappings.pkl` to new data at inference time without re-fitting — critical for production deployment where the ETL must not be re-run on live data. (The dashboard in `app.py` loads `woe_mappings.pkl` directly and re-implements this logic for inference rather than importing it from `ETL.py`.)
- The recall target (0.75), SA scorecard parameters (PDO, base odds, bands), and Basel thresholds are all defined as named constants at the top of their respective files and can be adjusted without touching the pipeline logic.
- XGBoost outputs (`xgb_model.pkl`, `xgb_model_metrics.pkl`) follow the same interface as the LR outputs, allowing the Streamlit app to swap between models without code changes.
- The LGD model is deliberately structured as a grade-lookup table so the Basel II 45% constant can be replaced with empirical recovery estimates from internal workout data without any downstream changes.

---

## License

MIT
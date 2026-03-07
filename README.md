# Basel Credit Risk Model

## Overview
This project implements a **credit risk modeling framework** based on Basel regulatory standards. It focuses on **Probability of Default (PD)** modeling using machine learning techniques and provides tools for **scorecard creation, model validation, and ETL processes** for loan datasets. The project is intended for **quantitative finance and risk management applications**.

---

## Project Structure
src/
├── Credit Risk Modeling - Model Fitting.ipynb
├── ETL.py
├── Model_Validation.py
├── Scorecard.py
├── pd_model.sav
data/
├── loan_data_inputs_test.csv
README.md


- **`Credit Risk Modeling - Model Fitting.ipynb`**: Notebook for training and evaluating the PD model.  
- **`ETL.py`**: Scripts for data extraction, transformation, and loading.  
- **`Model_Validation.py`**: Performs validation, performance metrics, and statistical tests.  
- **`Scorecard.py`**: Generates credit scorecards from the trained PD model.  
- **`pd_model.sav`**: Serialized PD model saved for inference.  
- **`loan_data_inputs_test.csv`**: Sample input dataset for testing and scorecard generation.  

---

## Key Features

- **PD Modeling**: Build a model to predict probability of default on loan data.  
- **Scorecard Generation**: Converts model coefficients into interpretable credit scores.  
- **Model Validation**: Includes checks for feature importance, predictive performance, and stability.  
- **ETL Automation**: Scripts standardize input data for modeling and scoring.  

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tami-L/basel-credit-risk-model.git
cd basel-credit-risk-model

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HARD-CODED SETTINGS & MACRO FACTORS
# ---------------------------------------------------------------------------
SENSITIVITY_FACTORS = {
    "recovery_loss_per_100bps_hike": -0.045, # -4.5% Recovery per 1% Repo hike
    "recovery_loss_per_1pct_gdp_drop": -0.030, # -3.0% Recovery per 1% GDP drop
}

SCENARIOS = {
    "Baseline": {"repo_hike_bps": 0, "gdp_drop_pct": 0.0},
    "Mild Stress": {"repo_hike_bps": 200, "gdp_drop_pct": 1.5},
    "Severe Stress": {"repo_hike_bps": 450, "gdp_drop_pct": 4.0},
}

SARB_LGD_FLOOR = 0.25
MAX_PHI = 150.0  # HARD CAP: Prevents the 10^31 "Bullshit" by enforcing a minimum variance
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
LGD_SAVE_PATH = Path(__file__).resolve().parent / "lgd_model.pkl"

# ===========================================================================
# REGULARIZED MLE BETA FITTING
# ===========================================================================

def beta_log_likelihood(params, y, X):
    """
    Log-likelihood with a Penalty term to prevent Phi from exploding.
    """
    k = X.shape[1]
    beta_coeffs = params[:k]
    # We constrain phi using a sigmoid-style mapping to keep it under MAX_PHI
    phi = MAX_PHI / (1 + np.exp(-params[k])) 
    
    # Logit link
    xb = np.dot(X, beta_coeffs)
    mu = 1 / (1 + np.exp(-xb))
    mu = np.clip(mu, 1e-6, 1 - 1e-6)
    
    p = mu * phi
    q = (1 - mu) * phi
    
    # Log Likelihood calculation
    ll = np.sum(gammaln(phi) - gammaln(p) - gammaln(q) + 
                (p - 1) * np.log(y) + (q - 1) * np.log(1 - y))
    
    # Regularization: Penalize extreme beta coefficients to keep them sane
    penalty = 0.5 * np.sum(beta_coeffs**2)
    
    return -(ll - penalty) if np.isfinite(ll) else 1e15

def fit_beta_mle(df):
    print("STEP 1: Sanitizing data for MLE...")
    y = df["recovery_rate"].values
    
    # If data is perfectly constant, add a tiny bit of jitter so the math doesn't break
    if np.var(y) < 1e-8:
        print("  INFO: Constant data detected. Applying Laplace-style smoothing.")
        y = np.clip(y + np.random.normal(0, 0.001, len(y)), 0.01, 0.99)

    df['grade'] = pd.Categorical(df['grade'], categories=GRADE_ORDER, ordered=True)
    X = pd.get_dummies(df['grade']).astype(float).values
    
    # Initial Guess: Use logit of the mean
    mu_start = np.mean(y)
    logit_mu = np.log(mu_start / (1 - mu_start))
    
    # params: [beta_A, beta_B, ..., logit_phi]
    initial_params = np.append(np.full(X.shape[1], logit_mu), 0.0) 
    
    print("STEP 2: Optimizing Likelihood (L-BFGS-B with Phi Constraints)...")
    res = minimize(beta_log_likelihood, initial_params, args=(y, X), 
                   method='L-BFGS-B', options={'ftol': 1e-9})
    
    if not res.success:
        print(f"  WARNING: Optimization issue: {res.message}. Using capped fallback.")
        return np.full(X.shape[1], logit_mu), 20.0
    
    final_betas = res.x[:-1]
    final_phi = MAX_PHI / (1 + np.exp(-res.x[-1]))
    
    print(f"STEP 3: MLE Converged. Phi (Precision) capped at: {final_phi:.2f}")
    return final_betas, final_phi

# ===========================================================================
# Model Logic
# ===========================================================================

def build_lgd_model(el_results_path):
    print(f"LOG: Building model from {el_results_path.name}")
    
    with open(el_results_path, "rb") as f:
        data = pickle.load(f)
    
    # Extract
    df = next(iter(data.values())) if isinstance(data, dict) and not isinstance(data, pd.DataFrame) else data
    df.columns = [c.lower().strip() for c in df.columns]
    
    if "lgd" in df.columns: df["recovery_rate"] = 1 - df["lgd"].clip(0, 1)
    if "sub_grade" in df.columns: df["grade"] = df["sub_grade"].str[0].upper()
    
    df = df[["grade", "recovery_rate"]].dropna()
    
    # MLE Fit
    coeffs, phi = fit_beta_mle(df)
    
    beta_params = {}
    for i, g in enumerate(GRADE_ORDER):
        mu_g = 1 / (1 + np.exp(-coeffs[i]))
        beta_params[g] = {
            "mu": float(mu_g),
            "lgd_mean": float(max(1 - mu_g, SARB_LGD_FLOOR)),
            "alpha": float(mu_g * phi),
            "beta": float((1 - mu_g) * phi)
        }
    
    return {
        "beta_params": beta_params,
        "phi": phi,
        "method": "Regularized MLE Beta Regression",
        "lgd_floor": SARB_LGD_FLOOR
    }

def get_stressed_lgd(grade, lgd_model, scenario_name):
    s = SCENARIOS[scenario_name]
    shift = (s["repo_hike_bps"]/100 * SENSITIVITY_FACTORS["recovery_loss_per_100bps_hike"]) + \
            (s["gdp_drop_pct"] * SENSITIVITY_FACTORS["recovery_loss_per_1pct_gdp_drop"])
    
    base_mu = lgd_model["beta_params"][grade]["mu"]
    stressed_mu = np.clip(base_mu + shift, 0.01, 0.99)
    return max(1 - stressed_mu, SARB_LGD_FLOOR)

def main():
    root = Path(__file__).resolve().parent.parent
    el_path = root / "scorecard_outputs" / "el_results.pkl"
    if not el_path.exists(): el_path = Path(__file__).parent / "el_results.pkl"

    if not el_path.exists():
        print("CRITICAL: Data file missing.")
        return

    model = build_lgd_model(el_path)
    
    print("\n--- ZAR MACRO-SENSITIVE LGD (REGULARIZED MLE) ---")
    print(f"{'GRADE':<8} | {'BASE LGD':<10} | {'MILD (+200bps)':<15} | {'SEVERE (+450bps)':<15}")
    print("-" * 60)
    for g in GRADE_ORDER:
        base = model['beta_params'][g]['lgd_mean']
        mild = get_stressed_lgd(g, model, "Mild Stress")
        severe = get_stressed_lgd(g, model, "Severe Stress")
        print(f"{g:<8} | {base:>10.2%} | {mild:>15.2%} | {severe:>15.2%}")
    
    with open(LGD_SAVE_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel finalized and saved to: {LGD_SAVE_PATH}")

if __name__ == "__main__":
    main()
import pickle
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.special import expit

warnings.filterwarnings("ignore", category=UserWarning)

# ── PATHS ──────────────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
LOCATIONS = [
    ROOT_DIR / "data",
    Path("/Users/lindokuhletami/Desktop/Space/data")
]

LGD_MODEL_PATH = SRC_DIR / "lgd_model.pkl"
PD_MODEL_PATH  = SRC_DIR / "pd_model.sav"
FEATURES_PATH  = SRC_DIR / "pd_model_features.pkl"
EL_SAVE_PATH   = SRC_DIR / "el_results.pkl"

def find_file(name):
    for loc in LOCATIONS:
        path = loc / name
        if path.exists():
            return path
    return None

def compute_lgd_with_stress(chunk, lgd_model, scenario="baseline"):
    beta_data = lgd_model["beta_regression"]
    scales    = lgd_model["feature_scales"]
    trained_cols = lgd_model["training_columns"]
    
    stress_map = {
        "baseline": 0.0,
        "severe_recession": 0.55 
    }
    buffer = stress_map.get(scenario, 0.0)

    X_scaled = pd.DataFrame(index=chunk.index)
    for col, stat in scales.items():
        if col in chunk.columns:
            val = pd.to_numeric(chunk[col], errors="coerce").fillna(stat["mean"])
            X_scaled[col] = (val - stat["mean"]) / stat["std"]
        else:
            X_scaled[col] = 0.0

    other_cols = chunk.drop(columns=[c for c in X_scaled.columns if c in chunk.columns])
    X_full = pd.concat([X_scaled, pd.get_dummies(other_cols, dtype='float32')], axis=1)
    X_aligned = X_full.reindex(columns=trained_cols, fill_value=0).values.astype('float32')
    
    X_matrix = np.column_stack([np.ones(len(X_aligned), dtype='float32'), X_aligned])
    z = (X_matrix @ beta_data["coef"].astype('float32')) + buffer
    return expit(z)

def main():
    print("--- Basel II Expected Loss & Stress Test Calculator ---")
    
    inputs_path = find_file("loan_data_inputs_test.csv")
    ead_path = find_file("ead_test.csv")
    
    if not inputs_path or not ead_path:
        print(f"CRITICAL ERROR: Data files missing.")
        return

    with open(PD_MODEL_PATH, "rb") as f: pd_model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f: sig_features = pickle.load(f)
    with open(LGD_MODEL_PATH, "rb") as f: lgd_model = pickle.load(f)

    required_cols = list(set(sig_features) | set(lgd_model["feature_scales"].keys()))
    chunk_size = 50000 
    results_collector = []
    
    print(f"Loading EAD: {ead_path.name}")
    ead_total = pd.read_csv(ead_path).iloc[:, 0].values.astype('float32')

    print(f"Streaming Inputs: {inputs_path.name}")
    inputs_iter = pd.read_csv(
        inputs_path, 
        chunksize=chunk_size, 
        usecols=required_cols,
        low_memory=False
    )

    for i, chunk in enumerate(inputs_iter):
        start_idx = i * chunk_size
        end_idx = start_idx + len(chunk)
        current_ead = ead_total[start_idx:end_idx]

        # Calculate PD
        pd_probs = 1 - pd_model.predict_proba(chunk[sig_features].values)[:, 1]
        
        # Calculate LGDs
        lgd_baseline = compute_lgd_with_stress(chunk, lgd_model, scenario="baseline")
        lgd_stressed = compute_lgd_with_stress(chunk, lgd_model, scenario="severe_recession")
        
        # Calculate ELs
        el_baseline = pd_probs * lgd_baseline * current_ead
        el_stressed = pd_probs * lgd_stressed * current_ead
        
        results_collector.append(pd.DataFrame({
            "pd": pd_probs.astype('float32'),
            "lgd_baseline": lgd_baseline.astype('float32'),
            "lgd_stressed": lgd_stressed.astype('float32'),
            "el_baseline": el_baseline.astype('float32'),
            "el_stressed": el_stressed.astype('float32')
        }))
        
        print(f"  Processed Chunk {i+1}...")

    final_df = pd.concat(results_collector)
    
    total_el_base = final_df['el_baseline'].sum()
    total_el_stress = final_df['el_stressed'].sum()
    
    print("\n" + "="*50)
    print(f"{'PORTFOLIO RESULTS':^50}")
    print("-" * 50)
    print(f"Baseline Total EL:      ${total_el_base:,.2f}")
    print(f"Stressed Total EL:      ${total_el_stress:,.2f}")
    print(f"Risk Increase:          {((total_el_stress/total_el_base)-1):.2%}")
    print("-" * 50)
    print(f"Avg LGD (Baseline):     {final_df['lgd_baseline'].mean():.2%}")
    print(f"Avg LGD (Stressed):     {final_df['lgd_stressed'].mean():.2%}")
    print("="*50)

    with open(EL_SAVE_PATH, "wb") as f:
        pickle.dump(final_df, f)
    
    print(f"\nFull comparative results saved to: {EL_SAVE_PATH}")

if __name__ == "__main__":
    main()
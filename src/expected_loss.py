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
# Potential data locations
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

def compute_lgd_chunk(chunk, lgd_model):
    beta_data = lgd_model["beta_regression"]
    scales    = lgd_model["feature_scales"]
    trained_cols = lgd_model["training_columns"]

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
    
    return expit(X_matrix @ beta_data["coef"].astype('float32'))

def main():
    print("--- Memory Optimized EL Calculator ---")
    
    # 1. Resolve Paths
    inputs_path = find_file("loan_data_inputs_test.csv")
    ead_path = find_file("ead_test.csv")
    
    if not inputs_path or not ead_path:
        print(f"CRITICAL ERROR: Files not found.\nInputs: {inputs_path}\nEAD: {ead_path}")
        return

    # 2. Load Model Artefacts
    with open(PD_MODEL_PATH, "rb") as f: pd_model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f: sig_features = pickle.load(f)
    with open(LGD_MODEL_PATH, "rb") as f: lgd_model = pickle.load(f)

    # 3. Setup Processing
    required_cols = list(set(sig_features) | set(lgd_model["feature_scales"].keys()))
    chunk_size = 50000 
    results_list = []
    
    print(f"Loading EAD from: {ead_path.name}")
    ead_total = pd.read_csv(ead_path).iloc[:, 0].values.astype('float32')

    print(f"Streaming Inputs from: {inputs_path.name}")
    inputs_iter = pd.read_csv(
        inputs_path, 
        chunksize=chunk_size, 
        usecols=required_cols,
        low_memory=False
    )

    for i, chunk in enumerate(inputs_iter):
        start_idx = i * chunk_size
        end_idx = start_idx + len(chunk)
        
        # Slice EAD for current chunk
        if end_idx > len(ead_total):
            current_ead = ead_total[start_idx:]
        else:
            current_ead = ead_total[start_idx:end_idx]

        # Calculations
        pd_probs = 1 - pd_model.predict_proba(chunk[sig_features].values)[:, 1]
        lgd_probs = compute_lgd_chunk(chunk, lgd_model)
        chunk_el = pd_probs * lgd_probs * current_ead
        
        results_list.append(pd.DataFrame({
            "pd": pd_probs.astype('float32'),
            "lgd": lgd_probs.astype('float32'),
            "el": chunk_el.astype('float32')
        }))
        
        print(f"  Chunk {i+1} complete ({len(chunk):,} rows)...")

    # 4. Finalize
    final_df = pd.concat(results_list)
    
    print("\n" + "="*40)
    print(f"Total Portfolio EL:  ${final_df['el'].sum():,.2f}")
    print(f"Average PD:           {final_df['pd'].mean():.2%}")
    print(f"Average LGD:          {final_df['lgd'].mean():.2%}")
    print("="*40)

    with open(EL_SAVE_PATH, "wb") as f:
        pickle.dump(final_df, f)
    print(f"Full results saved to {EL_SAVE_PATH}")

if __name__ == "__main__":
    main()
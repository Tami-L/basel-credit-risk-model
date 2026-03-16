"""
LGD Model Script
================
Builds a Loss Given Default (LGD) model using the Basel II regulatory
constant of 45% for unsecured loans, segmented by grade.

Since no recovery data is available, the 45% constant is applied uniformly
across all loans. The grade-level table is retained so that when recovery
data becomes available, the grade means can be replaced with empirically
estimated values without changing the downstream interface.

Output
------
src/lgd_model.pkl  — dict with keys:
    "lgd_by_grade"  : pd.Series  — LGD value per grade (A–G)
    "lgd_default"   : float      — fallback LGD for unknown grades
    "method"        : str        — description of the estimation method
"""

import pickle
from pathlib import Path

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from pathlib import Path

SRC_DIR   = Path(__file__).resolve().parent   # .../src/
ROOT_DIR  = SRC_DIR.parent                    # .../basel-credit-risk-model/
DATA_DIR  = ROOT_DIR / "data"
MODEL_DIR = SRC_DIR
LGD_SAVE_PATH  = f"{MODEL_DIR}/lgd_model.pkl"

# ---------------------------------------------------------------------------
# Basel II regulatory constant for unsecured retail exposures
# ---------------------------------------------------------------------------
BASEL_LGD_UNSECURED = 0.45


# ---------------------------------------------------------------------------
# Build LGD table
# ---------------------------------------------------------------------------
def build_lgd_model(lgd_constant: float = BASEL_LGD_UNSECURED) -> dict:
    """
    Build a grade-segmented LGD lookup table.

    All grades are assigned the Basel II regulatory constant (45%) for
    unsecured loans. When recovery data becomes available, replace the
    per-grade values with empirically estimated means from that data.

    Parameters
    ----------
    lgd_constant : LGD to assign to every grade (default 0.45)

    Returns
    -------
    dict with keys: lgd_by_grade, lgd_default, method
    """
    grades = ["A", "B", "C", "D", "E", "F", "G"]

    lgd_by_grade = pd.Series(
        {grade: lgd_constant for grade in grades},
        name="lgd",
    )

    lgd_model = {
        "lgd_by_grade": lgd_by_grade,
        "lgd_default":  lgd_constant,   # fallback for unseen grades
        "method":       (
            f"Basel II regulatory constant ({lgd_constant:.0%}) applied "
            "uniformly to all grades. Replace grade values with empirical "
            "recovery-rate estimates when recovery data is available."
        ),
    }

    return lgd_model


def get_lgd(grade_series: pd.Series, lgd_model: dict) -> pd.Series:
    """
    Look up LGD for each loan given its grade.

    Parameters
    ----------
    grade_series : pd.Series of loan grades (e.g. 'A', 'B', ...)
    lgd_model    : dict produced by build_lgd_model()

    Returns
    -------
    pd.Series of LGD values, same index as grade_series
    """
    return grade_series.map(lgd_model["lgd_by_grade"]).fillna(
        lgd_model["lgd_default"]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Building LGD model...")
    lgd_model = build_lgd_model()

    print("\nLGD by grade:")
    print(lgd_model["lgd_by_grade"].to_string())
    print(f"\nFallback LGD (unknown grade): {lgd_model['lgd_default']:.2%}")
    print(f"\nMethod: {lgd_model['method']}")

    # Save
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    with open(LGD_SAVE_PATH, "wb") as f:
        pickle.dump(lgd_model, f)
    print(f"\nLGD model saved to: {LGD_SAVE_PATH}")


if __name__ == "__main__":
    main()
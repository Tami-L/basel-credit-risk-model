import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, low_memory=False)


def _to_months_since(series: pd.Series, reference: pd.Timestamp) -> pd.Series:
    return (reference.year - series.dt.year) * 12 + (reference.month - series.dt.month)


def prepare_inputs(
    df: pd.DataFrame,
    reference_date: str = "2017-12-01",
    target_col: Optional[str] = "good_bad",
    ohe: Optional[OneHotEncoder] = None,
    fit_ohe: bool = True,
    train_params: Optional[Dict[str, Any]] = None,
    fit_train_params: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[OneHotEncoder], Dict[str, Any]]:
    """
    Prepare model inputs from raw DataFrame.

    Parameters
    ----------
    df              : raw input DataFrame (may include target_col)
    reference_date  : date used for months-since calculations
    target_col      : name of the target column (set None if absent)
    ohe             : a pre-fitted OneHotEncoder; required when fit_ohe=False
    fit_ohe         : if True, fit a new OHE on this data; if False, use supplied ohe
    train_params    : dict of statistics computed on the training set
                      (annual_inc_mean, max_mths_issue_d, max_mths_earliest_cr_line).
                      Must be supplied when fit_train_params=False.
    fit_train_params: if True, compute and store those statistics from this data;
                      if False, use values from train_params (for test/inference).

    Returns
    -------
    (X, y, ohe, train_params)
    """
    df = df.copy()
    ref = pd.to_datetime(reference_date)

    # ------------------------------------------------------------------ #
    # 1. Date / string feature engineering                                 #
    # ------------------------------------------------------------------ #

    # emp_length -> emp_length_int
    if "emp_length" in df.columns:
        emp = df["emp_length"].astype(str).fillna("0")
        emp = emp.str.replace(r"\+ years", "", regex=True)
        emp = emp.str.replace("< 1 year", "0")
        emp = emp.str.replace("n/a", "0")
        emp = emp.str.replace(r" years", "", regex=True)
        emp = emp.str.replace(r" year", "", regex=True)
        emp = emp.str.replace(r"\+", "", regex=True)
        emp_digits = emp.str.extract(r"(\d+)").fillna("0")[0]
        df["emp_length_int"] = pd.to_numeric(emp_digits).astype(int)

    # earliest_cr_line -> months since
    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line_date"] = pd.to_datetime(
            df["earliest_cr_line"].astype(str).str.strip(), format="%b-%y", errors="coerce"
        )
        today = pd.Timestamp.today()
        mask = df["earliest_cr_line_date"] > today
        df.loc[mask, "earliest_cr_line_date"] = (
            df.loc[mask, "earliest_cr_line_date"] - pd.DateOffset(years=100)
        )
        df["mths_since_earliest_cr_line"] = _to_months_since(df["earliest_cr_line_date"], ref)

    # term -> term_int
    if "term" in df.columns:
        df["term_int"] = pd.to_numeric(
            df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )

    # issue_d -> months since
    if "issue_d" in df.columns:
        df["issue_d_date"] = pd.to_datetime(
            df["issue_d"].astype(str).str.strip(), format="%b-%y", errors="coerce"
        )
        mask = df["issue_d_date"] > pd.Timestamp.today()
        df.loc[mask, "issue_d_date"] = df.loc[mask, "issue_d_date"] - pd.DateOffset(years=100)
        df["mths_since_issue_d"] = _to_months_since(df["issue_d_date"], ref)

    # ------------------------------------------------------------------ #
    # 2. Separate target                                                   #
    # ------------------------------------------------------------------ #
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col].copy()
        df = df.drop(columns=[target_col])

    # ------------------------------------------------------------------ #
    # 3. One-hot encoding                                                  #
    # ------------------------------------------------------------------ #
    to_encode = [
        "grade", "sub_grade", "home_ownership", "verification_status",
        "loan_status", "purpose", "addr_state", "initial_list_status",
    ]
    cat_cols = [col for col in to_encode if col in df.columns]

    if cat_cols:
        X_cat = df[cat_cols].astype(str)

        if fit_ohe:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            ohe.fit(X_cat)

        if ohe is not None:
            feat_names = ohe.get_feature_names_out(cat_cols)
            feat_names = [fn.replace("_", ":", 1) for fn in feat_names]
            X_enc = pd.DataFrame(ohe.transform(X_cat), index=df.index, columns=feat_names)
            df = pd.concat([df, X_enc], axis=1)

    # ------------------------------------------------------------------ #
    # 4. Compute / apply train statistics (no leakage)                     #
    # ------------------------------------------------------------------ #
    if fit_train_params:
        train_params = {
            "annual_inc_mean": (
                df["annual_inc"].mean() if "annual_inc" in df.columns else 0.0
            ),
            "max_mths_issue_d": (
                int(df["mths_since_issue_d"].dropna().max())
                if "mths_since_issue_d" in df.columns and df["mths_since_issue_d"].notna().any()
                else 0
            ),
            "max_mths_earliest_cr_line": (
                int(df["mths_since_earliest_cr_line"].dropna().max())
                if "mths_since_earliest_cr_line" in df.columns
                and df["mths_since_earliest_cr_line"].notna().any()
                else 0
            ),
        }
    else:
        # Validate that train_params was supplied
        if train_params is None:
            raise ValueError(
                "train_params must be provided when fit_train_params=False "
                "(i.e. when transforming the test set)."
            )

    annual_inc_mean = train_params["annual_inc_mean"]
    max_mths_issue_d = train_params["max_mths_issue_d"]
    max_mths_earliest_cr_line = train_params["max_mths_earliest_cr_line"]

    # ------------------------------------------------------------------ #
    # 5. Fill missing values using TRAIN statistics                        #
    # ------------------------------------------------------------------ #
    if "total_rev_hi_lim" in df.columns and "funded_amnt" in df.columns:
        df["total_rev_hi_lim"] = df["total_rev_hi_lim"].fillna(df["funded_amnt"])
    if "annual_inc" in df.columns:
        df["annual_inc"] = df["annual_inc"].fillna(annual_inc_mean)  # train mean only
    for c in [
        "mths_since_earliest_cr_line", "acc_now_delinq", "total_acc",
        "pub_rec", "open_acc", "inq_last_6mths", "delinq_2yrs", "emp_length_int",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # ------------------------------------------------------------------ #
    # 6. Combined / grouped dummy features                                 #
    # ------------------------------------------------------------------ #

    # home_ownership grouping
    ho_cols = [f"home_ownership:{v}" for v in ("RENT", "OTHER", "NONE", "ANY", "OWN", "MORTGAGE")]
    existing_ho = [c for c in ho_cols if c in df.columns]
    if existing_ho:
        df["home_ownership:RENT_OTHER_NONE_ANY"] = df[
            [f"home_ownership:{v}" for v in ("RENT", "OTHER", "NONE", "ANY") if f"home_ownership:{v}" in df.columns]
        ].sum(axis=1)
    else:
        df["home_ownership:RENT_OTHER_NONE_ANY"] = 0

    for ho in ["OWN", "MORTGAGE"]:
        col = f"home_ownership:{ho}"
        if col not in df.columns:
            df[col] = 0

    # addr_state dummies
    addr_states = [
        "ND", "NE", "IA", "NV", "FL", "HI", "AL", "NM", "VA", "OK", "TN", "MO", "LA",
        "MD", "NC", "UT", "KY", "AZ", "NJ", "AR", "MI", "PA", "OH", "MN", "RI", "MA",
        "DE", "SD", "IN", "GA", "WA", "OR", "WI", "MT", "IL", "CT", "KS", "SC", "CO",
        "VT", "AK", "MS", "WV", "NH", "WY", "DC", "ME", "ID", "NY", "CA", "TX",
    ]
    for s in addr_states:
        col = f"addr_state:{s}"
        if col not in df.columns:
            df[col] = 0

    df["addr_state:ND_NE_IA_NV_FL_HI_AL"] = df[["addr_state:ND", "addr_state:NE", "addr_state:IA", "addr_state:NV", "addr_state:FL", "addr_state:HI", "addr_state:AL"]].sum(axis=1)
    df["addr_state:NM_VA"] = df[["addr_state:NM", "addr_state:VA"]].sum(axis=1)
    df["addr_state:OK_TN_MO_LA_MD_NC"] = df[["addr_state:OK", "addr_state:TN", "addr_state:MO", "addr_state:LA", "addr_state:MD", "addr_state:NC"]].sum(axis=1)
    df["addr_state:UT_KY_AZ_NJ"] = df[["addr_state:UT", "addr_state:KY", "addr_state:AZ", "addr_state:NJ"]].sum(axis=1)
    df["addr_state:AR_MI_PA_OH_MN"] = df[["addr_state:AR", "addr_state:MI", "addr_state:PA", "addr_state:OH", "addr_state:MN"]].sum(axis=1)
    df["addr_state:RI_MA_DE_SD_IN"] = df[["addr_state:RI", "addr_state:MA", "addr_state:DE", "addr_state:SD", "addr_state:IN"]].sum(axis=1)
    df["addr_state:GA_WA_OR"] = df[["addr_state:GA", "addr_state:WA", "addr_state:OR"]].sum(axis=1)
    df["addr_state:WI_MT"] = df[["addr_state:WI", "addr_state:MT"]].sum(axis=1)
    df["addr_state:IL_CT"] = df[["addr_state:IL", "addr_state:CT"]].sum(axis=1)
    df["addr_state:KS_SC_CO_VT_AK_MS"] = df[["addr_state:KS", "addr_state:SC", "addr_state:CO", "addr_state:VT", "addr_state:AK", "addr_state:MS"]].sum(axis=1)
    df["addr_state:WV_NH_WY_DC_ME_ID"] = df[["addr_state:WV", "addr_state:NH", "addr_state:WY", "addr_state:DC", "addr_state:ME", "addr_state:ID"]].sum(axis=1)

    # verification_status dummies
    for status in ["Not Verified", "Source Verified", "Verified"]:
        col = f"verification_status:{status}"
        if col not in df.columns:
            df[col] = 0

    # initial_list_status dummies
    for status in ["f", "w"]:
        col = f"initial_list_status:{status}"
        if col not in df.columns:
            df[col] = 0

    # purpose groupings
    purpose_groups = {
        "purpose:educ__sm_b__wedd__ren_en__mov__house": [
            "educational", "small_business", "wedding", "renewable_energy", "moving", "house"
        ],
        "purpose:oth__med__vacation": ["other", "medical", "vacation"],
        "purpose:major_purch__car__home_impr": ["major_purchase", "car", "home_improvement"],
    }
    for group, cats in purpose_groups.items():
        cols = [f"purpose:{c}" for c in cats]
        existing = [c for c in cols if c in df.columns]
        df[group] = df[existing].sum(axis=1) if existing else 0

    # ------------------------------------------------------------------ #
    # 7. Binary / bucketed features                                        #
    # ------------------------------------------------------------------ #

    # term
    if "term_int" in df.columns:
        df["term:36"] = np.where(df["term_int"] == 36, 1, 0)
        df["term:60"] = np.where(df["term_int"] == 60, 1, 0)

    # emp_length bins
    if "emp_length_int" in df.columns:
        df["emp_length:0"] = np.where(df["emp_length_int"].isin([0]), 1, 0)
        df["emp_length:1"] = np.where(df["emp_length_int"].isin([1]), 1, 0)
        df["emp_length:2-4"] = np.where(df["emp_length_int"].isin(range(2, 5)), 1, 0)
        df["emp_length:5-6"] = np.where(df["emp_length_int"].isin(range(5, 7)), 1, 0)
        df["emp_length:7-9"] = np.where(df["emp_length_int"].isin(range(7, 10)), 1, 0)
        df["emp_length:10"] = np.where(df["emp_length_int"].isin([10]), 1, 0)

    # mths_since_issue_d buckets — use TRAIN max
    if "mths_since_issue_d" in df.columns:
        max_m = max_mths_issue_d
        df["mths_since_issue_d:<38"] = np.where(df["mths_since_issue_d"].isin(range(38)), 1, 0)
        df["mths_since_issue_d:38-39"] = np.where(df["mths_since_issue_d"].isin(range(38, 40)), 1, 0)
        df["mths_since_issue_d:40-41"] = np.where(df["mths_since_issue_d"].isin(range(40, 42)), 1, 0)
        df["mths_since_issue_d:42-48"] = np.where(df["mths_since_issue_d"].isin(range(42, 49)), 1, 0)
        df["mths_since_issue_d:49-52"] = np.where(df["mths_since_issue_d"].isin(range(49, 53)), 1, 0)
        df["mths_since_issue_d:53-64"] = np.where(df["mths_since_issue_d"].isin(range(53, 65)), 1, 0)
        df["mths_since_issue_d:65-84"] = np.where(df["mths_since_issue_d"].isin(range(65, 85)), 1, 0)
        df["mths_since_issue_d:>84"] = np.where(df["mths_since_issue_d"].isin(range(85, max_m + 1)), 1, 0)

    # int_rate bins
    if "int_rate" in df.columns:
        df["int_rate:<9.548"] = np.where(df["int_rate"] <= 9.548, 1, 0)
        df["int_rate:9.548-12.025"] = np.where((df["int_rate"] > 9.548) & (df["int_rate"] <= 12.025), 1, 0)
        df["int_rate:12.025-15.74"] = np.where((df["int_rate"] > 12.025) & (df["int_rate"] <= 15.74), 1, 0)
        df["int_rate:15.74-20.281"] = np.where((df["int_rate"] > 15.74) & (df["int_rate"] <= 20.281), 1, 0)
        df["int_rate:>20.281"] = np.where(df["int_rate"] > 20.281, 1, 0)

    # funded_amnt_factor
    if "funded_amnt" in df.columns:
        df["funded_amnt_factor"] = pd.cut(df["funded_amnt"], 50)

    # mths_since_earliest_cr_line buckets — use TRAIN max
    if "mths_since_earliest_cr_line" in df.columns:
        max_m = max_mths_earliest_cr_line
        df["mths_since_earliest_cr_line:<140"] = np.where(df["mths_since_earliest_cr_line"].isin(range(140)), 1, 0)
        df["mths_since_earliest_cr_line:141-164"] = np.where(df["mths_since_earliest_cr_line"].isin(range(140, 165)), 1, 0)
        df["mths_since_earliest_cr_line:165-247"] = np.where(df["mths_since_earliest_cr_line"].isin(range(165, 248)), 1, 0)
        df["mths_since_earliest_cr_line:248-270"] = np.where(df["mths_since_earliest_cr_line"].isin(range(248, 271)), 1, 0)
        df["mths_since_earliest_cr_line:271-352"] = np.where(df["mths_since_earliest_cr_line"].isin(range(271, 353)), 1, 0)
        df["mths_since_earliest_cr_line:>352"] = np.where(df["mths_since_earliest_cr_line"].isin(range(353, max_m + 1)), 1, 0)

    # delinq_2yrs bins
    if "delinq_2yrs" in df.columns:
        df["delinq_2yrs:0"] = np.where(df["delinq_2yrs"] == 0, 1, 0)
        df["delinq_2yrs:1-3"] = np.where((df["delinq_2yrs"] >= 1) & (df["delinq_2yrs"] <= 3), 1, 0)
        df["delinq_2yrs:>=4"] = np.where(df["delinq_2yrs"] >= 4, 1, 0)

    # inq_last_6mths bins
    if "inq_last_6mths" in df.columns:
        df["inq_last_6mths:0"] = np.where(df["inq_last_6mths"] == 0, 1, 0)
        df["inq_last_6mths:1-2"] = np.where((df["inq_last_6mths"] >= 1) & (df["inq_last_6mths"] <= 2), 1, 0)
        df["inq_last_6mths:3-6"] = np.where((df["inq_last_6mths"] >= 3) & (df["inq_last_6mths"] <= 6), 1, 0)
        df["inq_last_6mths:>6"] = np.where(df["inq_last_6mths"] > 6, 1, 0)

    # open_acc bins
    if "open_acc" in df.columns:
        df["open_acc:0"] = np.where(df["open_acc"] == 0, 1, 0)
        df["open_acc:1-3"] = np.where((df["open_acc"] >= 1) & (df["open_acc"] <= 3), 1, 0)
        df["open_acc:4-12"] = np.where((df["open_acc"] >= 4) & (df["open_acc"] <= 12), 1, 0)
        df["open_acc:13-17"] = np.where((df["open_acc"] >= 13) & (df["open_acc"] <= 17), 1, 0)
        df["open_acc:18-22"] = np.where((df["open_acc"] >= 18) & (df["open_acc"] <= 22), 1, 0)
        df["open_acc:23-25"] = np.where((df["open_acc"] >= 23) & (df["open_acc"] <= 25), 1, 0)
        df["open_acc:26-30"] = np.where((df["open_acc"] >= 26) & (df["open_acc"] <= 30), 1, 0)
        df["open_acc:>=31"] = np.where(df["open_acc"] >= 31, 1, 0)

    # pub_rec bins
    if "pub_rec" in df.columns:
        df["pub_rec:0-2"] = np.where((df["pub_rec"] >= 0) & (df["pub_rec"] <= 2), 1, 0)
        df["pub_rec:3-4"] = np.where((df["pub_rec"] >= 3) & (df["pub_rec"] <= 4), 1, 0)
        df["pub_rec:>=5"] = np.where(df["pub_rec"] >= 5, 1, 0)

    # total_acc bins
    if "total_acc" in df.columns:
        df["total_acc:<=27"] = np.where(df["total_acc"] <= 27, 1, 0)
        df["total_acc:28-51"] = np.where((df["total_acc"] >= 28) & (df["total_acc"] <= 51), 1, 0)
        df["total_acc:>=52"] = np.where(df["total_acc"] >= 52, 1, 0)

    # acc_now_delinq bins
    if "acc_now_delinq" in df.columns:
        df["acc_now_delinq:0"] = np.where(df["acc_now_delinq"] == 0, 1, 0)
        df["acc_now_delinq:>=1"] = np.where(df["acc_now_delinq"] >= 1, 1, 0)

    # total_rev_hi_lim bins
    if "total_rev_hi_lim" in df.columns:
        df["total_rev_hi_lim:<=5K"] = np.where(df["total_rev_hi_lim"] <= 5000, 1, 0)
        df["total_rev_hi_lim:5K-10K"] = np.where((df["total_rev_hi_lim"] > 5000) & (df["total_rev_hi_lim"] <= 10000), 1, 0)
        df["total_rev_hi_lim:10K-20K"] = np.where((df["total_rev_hi_lim"] > 10000) & (df["total_rev_hi_lim"] <= 20000), 1, 0)
        df["total_rev_hi_lim:20K-30K"] = np.where((df["total_rev_hi_lim"] > 20000) & (df["total_rev_hi_lim"] <= 30000), 1, 0)
        df["total_rev_hi_lim:30K-40K"] = np.where((df["total_rev_hi_lim"] > 30000) & (df["total_rev_hi_lim"] <= 40000), 1, 0)
        df["total_rev_hi_lim:40K-55K"] = np.where((df["total_rev_hi_lim"] > 40000) & (df["total_rev_hi_lim"] <= 55000), 1, 0)
        df["total_rev_hi_lim:55K-95K"] = np.where((df["total_rev_hi_lim"] > 55000) & (df["total_rev_hi_lim"] <= 95000), 1, 0)
        df["total_rev_hi_lim:>95K"] = np.where(df["total_rev_hi_lim"] > 95000, 1, 0)

    # installment_factor and annual_inc bins
    if "installment" in df.columns:
        df["installment_factor"] = pd.cut(df["installment"], 50)
    if "annual_inc" in df.columns:
        df["annual_inc:<20K"] = np.where(df["annual_inc"] <= 20000, 1, 0)
        df["annual_inc:20K-30K"] = np.where((df["annual_inc"] > 20000) & (df["annual_inc"] <= 30000), 1, 0)
        df["annual_inc:30K-40K"] = np.where((df["annual_inc"] > 30000) & (df["annual_inc"] <= 40000), 1, 0)
        df["annual_inc:40K-50K"] = np.where((df["annual_inc"] > 40000) & (df["annual_inc"] <= 50000), 1, 0)
        df["annual_inc:50K-60K"] = np.where((df["annual_inc"] > 50000) & (df["annual_inc"] <= 60000), 1, 0)
        df["annual_inc:60K-70K"] = np.where((df["annual_inc"] > 60000) & (df["annual_inc"] <= 70000), 1, 0)
        df["annual_inc:70K-80K"] = np.where((df["annual_inc"] > 70000) & (df["annual_inc"] <= 80000), 1, 0)
        df["annual_inc:80K-90K"] = np.where((df["annual_inc"] > 80000) & (df["annual_inc"] <= 90000), 1, 0)
        df["annual_inc:90K-100K"] = np.where((df["annual_inc"] > 90000) & (df["annual_inc"] <= 100000), 1, 0)
        df["annual_inc:100K-120K"] = np.where((df["annual_inc"] > 100000) & (df["annual_inc"] <= 120000), 1, 0)
        df["annual_inc:120K-140K"] = np.where((df["annual_inc"] > 120000) & (df["annual_inc"] <= 140000), 1, 0)
        df["annual_inc:>140K"] = np.where(df["annual_inc"] > 140000, 1, 0)

    # mths_since_last_delinq bins including Missing
    if "mths_since_last_delinq" in df.columns:
        df["mths_since_last_delinq:Missing"] = np.where(df["mths_since_last_delinq"].isnull(), 1, 0)
        df["mths_since_last_delinq:0-3"] = np.where((df["mths_since_last_delinq"] >= 0) & (df["mths_since_last_delinq"] <= 3), 1, 0)
        df["mths_since_last_delinq:4-30"] = np.where((df["mths_since_last_delinq"] >= 4) & (df["mths_since_last_delinq"] <= 30), 1, 0)
        df["mths_since_last_delinq:31-56"] = np.where((df["mths_since_last_delinq"] >= 31) & (df["mths_since_last_delinq"] <= 56), 1, 0)
        df["mths_since_last_delinq:>=57"] = np.where(df["mths_since_last_delinq"] >= 57, 1, 0)

    # dti bins
    if "dti" in df.columns:
        df["dti:<=1.4"] = np.where(df["dti"] <= 1.4, 1, 0)
        df["dti:1.4-3.5"] = np.where((df["dti"] > 1.4) & (df["dti"] <= 3.5), 1, 0)
        df["dti:3.5-7.7"] = np.where((df["dti"] > 3.5) & (df["dti"] <= 7.7), 1, 0)
        df["dti:7.7-10.5"] = np.where((df["dti"] > 7.7) & (df["dti"] <= 10.5), 1, 0)
        df["dti:10.5-16.1"] = np.where((df["dti"] > 10.5) & (df["dti"] <= 16.1), 1, 0)
        df["dti:16.1-20.3"] = np.where((df["dti"] > 16.1) & (df["dti"] <= 20.3), 1, 0)
        df["dti:20.3-21.7"] = np.where((df["dti"] > 20.3) & (df["dti"] <= 21.7), 1, 0)
        df["dti:21.7-22.4"] = np.where((df["dti"] > 21.7) & (df["dti"] <= 22.4), 1, 0)
        df["dti:22.4-35"] = np.where((df["dti"] > 22.4) & (df["dti"] <= 35), 1, 0)
        df["dti:>35"] = np.where(df["dti"] > 35, 1, 0)

    # mths_since_last_record bins
    if "mths_since_last_record" in df.columns:
        df["mths_since_last_record:Missing"] = np.where(df["mths_since_last_record"].isnull(), 1, 0)
        df["mths_since_last_record:0-2"] = np.where((df["mths_since_last_record"] >= 0) & (df["mths_since_last_record"] <= 2), 1, 0)
        df["mths_since_last_record:3-20"] = np.where((df["mths_since_last_record"] >= 3) & (df["mths_since_last_record"] <= 20), 1, 0)
        df["mths_since_last_record:21-31"] = np.where((df["mths_since_last_record"] >= 21) & (df["mths_since_last_record"] <= 31), 1, 0)
        df["mths_since_last_record:32-80"] = np.where((df["mths_since_last_record"] >= 32) & (df["mths_since_last_record"] <= 80), 1, 0)
        df["mths_since_last_record:81-86"] = np.where((df["mths_since_last_record"] >= 81) & (df["mths_since_last_record"] <= 86), 1, 0)
        df["mths_since_last_record:>=86"] = np.where(df["mths_since_last_record"] >= 86, 1, 0)

    return df, y, ohe, train_params


def split_and_prepare(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    reference_date: str = "2017-12-01",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split raw data into train/test and apply consistent preprocessing.

    - OHE is fit on train only, then applied to test.
    - annual_inc mean and max_m bucket boundaries are computed from train only.
    - No test-set information leaks into the train pipeline.
    """
    df["good_bad"] = np.where(
        df["loan_status"].isin([
            "Charged Off", "Default",
            "Does not meet the credit policy. Status:Charged Off",
            "Late (31-120 days)",
        ]),
        0, 1,
    )

    X = df.drop(columns=["good_bad"])
    y = df["good_bad"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ---- Fit ALL transformers on TRAIN only --------------------------------
    train_df = pd.concat([X_train, y_train.rename("good_bad")], axis=1)
    X_train_prep, y_train_prep, ohe, train_params = prepare_inputs(
        train_df,
        reference_date=reference_date,
        target_col="good_bad",
        fit_ohe=True,
        fit_train_params=True,
    )

    # ---- Apply (transform only) to TEST ------------------------------------
    test_df = pd.concat([X_test, y_test.rename("good_bad")], axis=1)
    X_test_prep, y_test_prep, _, _ = prepare_inputs(
        test_df,
        reference_date=reference_date,
        target_col="good_bad",
        ohe=ohe,                    # reuse fitted OHE — no refit
        fit_ohe=False,
        train_params=train_params,  # reuse train statistics — no recompute
        fit_train_params=False,
    )

    return X_train_prep, y_train_prep, X_test_prep, y_test_prep


def compute_ead(df_raw: pd.DataFrame) -> pd.Series:
    """
    Compute Exposure at Default (EAD) as outstanding balance.

    EAD = funded_amnt - total_pymnt, floored at 0.
    A negative value would mean the borrower has overpaid (edge case),
    so we clip to 0.

    Parameters
    ----------
    df_raw : raw loan DataFrame containing funded_amnt and total_pymnt

    Returns
    -------
    pd.Series of EAD values, indexed to match df_raw
    """
    if "funded_amnt" not in df_raw.columns or "total_pymnt" not in df_raw.columns:
        raise ValueError(
            "Raw data must contain 'funded_amnt' and 'total_pymnt' to compute EAD."
        )
    ead = (df_raw["funded_amnt"] - df_raw["total_pymnt"]).clip(lower=0)
    return ead.rename("ead")


def save_outputs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: str,
    ead_train: Optional[pd.Series] = None,
    ead_test: Optional[pd.Series] = None,
):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(p / "loan_data_inputs_train.csv", index=False)
    y_train.to_csv(p / "loan_data_targets_train.csv", index=False)
    X_test.to_csv(p / "loan_data_inputs_test.csv", index=False)
    y_test.to_csv(p / "loan_data_targets_test.csv", index=False)
    if ead_train is not None:
        ead_train.to_csv(p / "ead_train.csv", index=False)
        print(f"EAD train saved: {len(ead_train)} rows")
    if ead_test is not None:
        ead_test.to_csv(p / "ead_test.csv", index=False)
        print(f"EAD test  saved: {len(ead_test)} rows")


if __name__ == "__main__":
    src = '/Users/lindokuhletami/Desktop/Space/data/loan_data_2007_2014(1).csv'
    df = load_data(str(src))

    # Compute EAD from the raw data BEFORE the train/test split so that
    # the index alignment is preserved. We then split EAD in sync with
    # the feature split using the same random_state.
    from sklearn.model_selection import train_test_split as _tts
    ead_full = compute_ead(df)

    X_train, y_train, X_test, y_test = split_and_prepare(df)

    # Re-split EAD using the same indices produced by split_and_prepare
    ead_train = ead_full.loc[X_train.index].reset_index(drop=True)
    ead_test  = ead_full.loc[X_test.index].reset_index(drop=True)

    out = str(Path(__file__).parents[1] / "data")
    save_outputs(X_train, y_train, X_test, y_test, out,
                 ead_train=ead_train, ead_test=ead_test)

    print(f"\nEAD stats (train):")
    print(ead_train.describe())

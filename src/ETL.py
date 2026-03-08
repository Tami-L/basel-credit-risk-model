import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, low_memory=False)


def _to_months_since(series: pd.Series, reference: pd.Timestamp) -> pd.Series:
    return (reference.year - series.dt.year) * 12 + (reference.month - series.dt.month)


def prepare_inputs(df: pd.DataFrame, reference_date: str = "2017-12-01", target_col: Optional[str] = "good_bad", ohe: Optional[OneHotEncoder] = None, fit_ohe: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[OneHotEncoder]]:
    df = df.copy()
    ref = pd.to_datetime(reference_date)

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

    # earliest_cr_line -> date -> months
    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line_date"] = pd.to_datetime(df["earliest_cr_line"].astype(str).str.strip(), format="%b-%y", errors="coerce")
        today = pd.Timestamp.today()
        mask = df["earliest_cr_line_date"] > today
        df.loc[mask, "earliest_cr_line_date"] = df.loc[mask, "earliest_cr_line_date"] - pd.DateOffset(years=100)
        df["mths_since_earliest_cr_line"] = _to_months_since(df["earliest_cr_line_date"], ref)

    # term -> term_int
    if "term" in df.columns:
        df["term_int"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    # issue_d -> date -> months
    if "issue_d" in df.columns:
        df["issue_d_date"] = pd.to_datetime(df["issue_d"].astype(str).str.strip(), format="%b-%y", errors="coerce")
        mask = df["issue_d_date"] > pd.Timestamp.today()
        df.loc[mask, "issue_d_date"] = df.loc[mask, "issue_d_date"] - pd.DateOffset(years=100)
        df["mths_since_issue_d"] = _to_months_since(df["issue_d_date"], ref)

    # separate target before encoding
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col].copy()
        df = df.drop(columns=[target_col])

    # OneHotEncode categorical columns 
    to_encode = [
        "grade", "sub_grade", "home_ownership", "verification_status",
        "loan_status", "purpose", "addr_state", "initial_list_status"
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

    # fill missing values
    if "total_rev_hi_lim" in df.columns and "funded_amnt" in df.columns:
        df["total_rev_hi_lim"] = df["total_rev_hi_lim"].fillna(df["funded_amnt"])
    if "annual_inc" in df.columns:
        df["annual_inc"] = df["annual_inc"].fillna(df["annual_inc"].mean())
    for c in ["mths_since_earliest_cr_line", "acc_now_delinq", "total_acc", "pub_rec", "open_acc", "inq_last_6mths", "delinq_2yrs", "emp_length_int"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # create combined category
    ho_cols = [f"home_ownership:{v}" for v in ("RENT", "OTHER", "NONE", "ANY", "OWN", "MORTGAGE")]
    existing_ho = [c for c in ho_cols if c in df.columns]
    if existing_ho:
        df["home_ownership:RENT_OTHER_NONE_ANY"] = df[[f"home_ownership:{v}" for v in ("RENT", "OTHER", "NONE", "ANY")]].sum(axis=1)
    else:
        df["home_ownership:RENT_OTHER_NONE_ANY"] = 0

    # home_ownership dummies 
    for ho in ["OWN", "MORTGAGE"]:
        col = f"home_ownership:{ho}"
        if col not in df.columns:
            df[col] = 0

    #addr_state dummies 
    addr_states = ["ND","NE","IA","NV","FL","HI","AL","NM","VA","OK","TN","MO","LA","MD","NC","UT","KY","AZ","NJ","AR","MI","PA","OH","MN","RI","MA","DE","SD","IN","GA","WA","OR","WI","MT","IL","CT","KS","SC","CO","VT","AK","MS","WV","NH","WY","DC","ME","ID","NY","CA","TX"]
    for s in addr_states:
        col = f"addr_state:{s}"
        if col not in df.columns:
            df[col] = 0

    df["addr_state:ND_NE_IA_NV_FL_HI_AL"] = df[["addr_state:ND","addr_state:NE","addr_state:IA","addr_state:NV","addr_state:FL","addr_state:HI","addr_state:AL"]].sum(axis=1)
    df["addr_state:NM_VA"] = df[["addr_state:NM","addr_state:VA"]].sum(axis=1)
    df["addr_state:OK_TN_MO_LA_MD_NC"] = df[["addr_state:OK","addr_state:TN","addr_state:MO","addr_state:LA","addr_state:MD","addr_state:NC"]].sum(axis=1)
    df["addr_state:UT_KY_AZ_NJ"] = df[["addr_state:UT","addr_state:KY","addr_state:AZ","addr_state:NJ"]].sum(axis=1)
    df["addr_state:AR_MI_PA_OH_MN"] = df[["addr_state:AR","addr_state:MI","addr_state:PA","addr_state:OH","addr_state:MN"]].sum(axis=1)
    df["addr_state:RI_MA_DE_SD_IN"] = df[["addr_state:RI","addr_state:MA","addr_state:DE","addr_state:SD","addr_state:IN"]].sum(axis=1)
    df["addr_state:GA_WA_OR"] = df[["addr_state:GA","addr_state:WA","addr_state:OR"]].sum(axis=1)
    df["addr_state:WI_MT"] = df[["addr_state:WI","addr_state:MT"]].sum(axis=1)
    df["addr_state:IL_CT"] = df[["addr_state:IL","addr_state:CT"]].sum(axis=1)
    df["addr_state:KS_SC_CO_VT_AK_MS"] = df[["addr_state:KS","addr_state:SC","addr_state:CO","addr_state:VT","addr_state:AK","addr_state:MS"]].sum(axis=1)
    df["addr_state:WV_NH_WY_DC_ME_ID"] = df[["addr_state:WV","addr_state:NH","addr_state:WY","addr_state:DC","addr_state:ME","addr_state:ID"]].sum(axis=1)

    # ensure verification_status dummies exist
    for status in ["Not Verified", "Source Verified", "Verified"]:
        col = f"verification_status:{status}"
        if col not in df.columns:
            df[col] = 0

    # ensure initial_list_status dummies exist
    for status in ["f", "w"]:
        col = f"initial_list_status:{status}"
        if col not in df.columns:
            df[col] = 0

    # purpose groupings
    purpose_groups = {
        "purpose:educ__sm_b__wedd__ren_en__mov__house": ["educational", "small_business", "wedding", "renewable_energy", "moving", "house"],
        "purpose:oth__med__vacation": ["other", "medical", "vacation"],
        "purpose:major_purch__car__home_impr": ["major_purchase", "car", "home_improvement"]
    }
    for group, cats in purpose_groups.items():
        cols = [f"purpose:{c}" for c in cats]
        existing = [c for c in cols if c in df.columns]
        if existing:
            df[group] = df[existing].sum(axis=1)
        else:
            df[group] = 0

    # term binaries
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

    # mths_since_issue_d buckets
    if "mths_since_issue_d" in df.columns:
        max_m = int(df["mths_since_issue_d"].dropna().max()) if df["mths_since_issue_d"].notna().any() else 0
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

    # mths_since_earliest_cr_line buckets
    if "mths_since_earliest_cr_line" in df.columns:
        max_m = int(df["mths_since_earliest_cr_line"].dropna().max()) if df["mths_since_earliest_cr_line"].notna().any() else 0
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

    return df, y, ohe


def split_and_prepare(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, reference_date: str = "2017-12-01") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df["good_bad"] = np.where(df["loan_status"].isin([
        "Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", "Late (31-120 days)"
    ]), 0, 1)
    X = df.drop(columns=["good_bad"])
    y = df["good_bad"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Fit on combined train+test data to ensure all categories are encoded
    X_combined = pd.concat([X_train, X_test], axis=0)
    y_combined = pd.concat([y_train, y_test], axis=0)
    combined_df = pd.concat([X_combined, y_combined.rename("good_bad")], axis=1)
    
    X_prep, y_prep, ohe = prepare_inputs(combined_df, reference_date=reference_date, target_col="good_bad", fit_ohe=True)
    
    # Split back to train and test using original indices
    X_train_prep = X_prep.loc[X_train.index]
    X_test_prep = X_prep.loc[X_test.index]
    y_train_prep = y_prep.loc[y_train.index]
    y_test_prep = y_prep.loc[y_test.index]
    
    return X_train_prep, y_train_prep, X_test_prep, y_test_prep


def save_outputs(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, out_dir: str):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(p / "loan_data_inputs_train.csv", index=False)
    y_train.to_csv(p / "loan_data_targets_train.csv", index=False)
    X_test.to_csv(p / "loan_data_inputs_test.csv", index=False)
    y_test.to_csv(p / "loan_data_targets_test.csv", index=False)


if __name__ == "__main__":
    src = '/Users/lindokuhletami/Desktop/Space/data/loan_data_2007_2014(1).csv'
    df = load_data(str(src))
    X_train, y_train, X_test, y_test = split_and_prepare(df)
    save_outputs(X_train, y_train, X_test, y_test, str(Path(__file__).parents[1] / "data"))
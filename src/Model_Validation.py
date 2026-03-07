"""
Model Validation Script for Credit Risk PD Model
Loads saved model and test data, performs validation metrics
"""

# Import Libraries
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    classification_report
)
from scipy.stats import ks_2samp

# Set style for plots
sns.set()

# =============================================================================
# Load Test Data
# =============================================================================

# Load test datasets
loan_data_inputs_test = pd.read_csv('/Users/lindokuhletami/Desktop/Space/data/loan_data_inputs_test.csv')
loan_data_targets_test = pd.read_csv('/Users/lindokuhletami/Desktop/Space/data/loan_data_targets_test.csv')

print(f"Inputs test shape: {loan_data_inputs_test.shape}")
print(f"Targets test shape: {loan_data_targets_test.shape}")

# =============================================================================
# Select Features (as per the original notebook)
# =============================================================================

# Select all the feature columns (excluding reference categories)
inputs_test_with_ref_cat = loan_data_inputs_test.loc[: , [
    'grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F', 'grade:G',
    'home_ownership:RENT_OTHER_NONE_ANY', 'home_ownership:OWN', 'home_ownership:MORTGAGE',
    'addr_state:ND_NE_IA_NV_FL_HI_AL', 'addr_state:NM_VA', 'addr_state:NY',
    'addr_state:OK_TN_MO_LA_MD_NC', 'addr_state:CA', 'addr_state:UT_KY_AZ_NJ',
    'addr_state:AR_MI_PA_OH_MN', 'addr_state:RI_MA_DE_SD_IN', 'addr_state:GA_WA_OR',
    'addr_state:WI_MT', 'addr_state:TX', 'addr_state:IL_CT', 
    'addr_state:KS_SC_CO_VT_AK_MS', 'addr_state:WV_NH_WY_DC_ME_ID',
    'verification_status:Not Verified', 'verification_status:Source Verified',
    'verification_status:Verified',
    'purpose:educ__sm_b__wedd__ren_en__mov__house', 'purpose:credit_card',
    'purpose:debt_consolidation', 'purpose:oth__med__vacation',
    'purpose:major_purch__car__home_impr',
    'initial_list_status:f', 'initial_list_status:w',
    'term:36', 'term:60',
    'emp_length:0', 'emp_length:1', 'emp_length:2-4', 'emp_length:5-6', 
    'emp_length:7-9', 'emp_length:10',
    'mths_since_issue_d:<38', 'mths_since_issue_d:38-39', 'mths_since_issue_d:40-41',
    'mths_since_issue_d:42-48', 'mths_since_issue_d:49-52', 'mths_since_issue_d:53-64',
    'mths_since_issue_d:65-84', 'mths_since_issue_d:>84',
    'int_rate:<9.548', 'int_rate:9.548-12.025', 'int_rate:12.025-15.74',
    'int_rate:15.74-20.281', 'int_rate:>20.281',
    'mths_since_earliest_cr_line:<140', 'mths_since_earliest_cr_line:141-164',
    'mths_since_earliest_cr_line:165-247', 'mths_since_earliest_cr_line:248-270',
    'mths_since_earliest_cr_line:271-352', 'mths_since_earliest_cr_line:>352',
    'inq_last_6mths:0', 'inq_last_6mths:1-2', 'inq_last_6mths:3-6', 'inq_last_6mths:>6',
    'acc_now_delinq:0', 'acc_now_delinq:>=1',
    'annual_inc:<20K', 'annual_inc:20K-30K', 'annual_inc:30K-40K', 'annual_inc:40K-50K',
    'annual_inc:50K-60K', 'annual_inc:60K-70K', 'annual_inc:70K-80K', 'annual_inc:80K-90K',
    'annual_inc:90K-100K', 'annual_inc:100K-120K', 'annual_inc:120K-140K', 'annual_inc:>140K',
    'dti:<=1.4', 'dti:1.4-3.5', 'dti:3.5-7.7', 'dti:7.7-10.5', 'dti:10.5-16.1',
    'dti:16.1-20.3', 'dti:20.3-21.7', 'dti:21.7-22.4', 'dti:22.4-35', 'dti:>35',
    'mths_since_last_delinq:Missing', 'mths_since_last_delinq:0-3', 
    'mths_since_last_delinq:4-30', 'mths_since_last_delinq:31-56', 'mths_since_last_delinq:>=57',
    'mths_since_last_record:Missing', 'mths_since_last_record:0-2', 
    'mths_since_last_record:3-20', 'mths_since_last_record:21-31',
    'mths_since_last_record:32-80', 'mths_since_last_record:81-86', 'mths_since_last_record:>=86'
]]

# Define reference categories to drop
ref_categories = [
    'grade:G',
    'home_ownership:RENT_OTHER_NONE_ANY',
    'addr_state:ND_NE_IA_NV_FL_HI_AL',
    'verification_status:Verified',
    'purpose:educ__sm_b__wedd__ren_en__mov__house',
    'initial_list_status:f',
    'term:60',
    'emp_length:0',
    'mths_since_issue_d:>84',
    'int_rate:>20.281',
    'mths_since_earliest_cr_line:<140',
    'inq_last_6mths:>6',
    'acc_now_delinq:0',
    'annual_inc:<20K',
    'dti:>35',
    'mths_since_last_delinq:0-3',
    'mths_since_last_record:0-2'
]

# Drop reference categories to get final test features
inputs_test = inputs_test_with_ref_cat.drop(ref_categories, axis=1)
print(f"Final test features shape: {inputs_test.shape}")
print(f"Number of features: {inputs_test.shape[1]}")


from sklearn import linear_model
import scipy.stats as stat
import numpy as np

class LogisticRegression_with_p_values:

    def __init__(self,*args,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)

    def fit(self,X,y):
        self.model.fit(X,y)

        denom = (2.0*(1.0+np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X/denom).T,X)
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))

        z_scores = self.model.coef_[0]/sigma_estimates
        p_values = [stat.norm.sf(abs(x))*2 for x in z_scores]

        self.p_values = p_values

# =============================================================================
# Load Saved Model
# =============================================================================

# Load the model using pickle (adjust filename/path as needed)
with open('/Users/lindokuhletami/Desktop/Space/src/pd_model.sav', 'rb') as file:
    reg2 = pickle.load(file)

print("Model loaded successfully")

# =============================================================================
# Generate Predictions
# =============================================================================

# Generate class predictions
y_hat_test = reg2.model.predict(inputs_test)
print(f"Class predictions shape: {y_hat_test.shape}")

# Generate probability predictions
y_hat_test_proba = reg2.model.predict_proba(inputs_test)
print(f"Probability predictions shape: {y_hat_test_proba.shape}")

# Extract probabilities for class 1 (bad loans)
y_hat_test_proba = y_hat_test_proba[:, 1]
print(f"Probability for class 1 shape: {y_hat_test_proba.shape}")

# =============================================================================
# Create DataFrame with Actual and Predicted Values
# =============================================================================

# Prepare targets DataFrame
loan_data_targets_test_temp = loan_data_targets_test.copy()
loan_data_targets_test_temp.reset_index(drop=True, inplace=True)

# Combine actual and predicted probabilities
df_actual_predicted_probs = pd.concat([
    loan_data_targets_test_temp, 
    pd.DataFrame(y_hat_test_proba)
], axis=1)

df_actual_predicted_probs.columns = ['loan_data_targets_test', 'y_hat_test_proba']
df_actual_predicted_probs.index = loan_data_inputs_test.index

print(f"\nActual vs Predicted DataFrame shape: {df_actual_predicted_probs.shape}")
print(df_actual_predicted_probs.head())

# =============================================================================
# Accuracy Metrics with Threshold
# =============================================================================

# Apply threshold for classification
tr = 0.9
df_actual_predicted_probs['y_hat_test'] = np.where(
    df_actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0
)

# Confusion matrix - using sklearn
cm = confusion_matrix(
    df_actual_predicted_probs['loan_data_targets_test'], 
    df_actual_predicted_probs['y_hat_test']
)
print(f"\nConfusion Matrix (threshold={tr}):")
print(cm)

# Normalized confusion matrix - using sklearn with normalize parameter
cm_norm = confusion_matrix(
    df_actual_predicted_probs['loan_data_targets_test'], 
    df_actual_predicted_probs['y_hat_test'],
    normalize='all'
)
print(f"\nNormalized Confusion Matrix (threshold={tr}):")
print(cm_norm)

# Calculate accuracy - using sklearn
accuracy = accuracy_score(
    df_actual_predicted_probs['loan_data_targets_test'], 
    df_actual_predicted_probs['y_hat_test']
)
print(f"\nAccuracy (threshold={tr}): {accuracy:.4f}")

# =============================================================================
# ROC Curve and AUROC
# =============================================================================

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(
    df_actual_predicted_probs['loan_data_targets_test'],
    df_actual_predicted_probs['y_hat_test_proba']
)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, linewidth=2)
plt.plot(fpr, fpr, linestyle='--', color='k', alpha=0.7)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True, alpha=0.3)
plt.show()

# Calculate AUROC
AUROC = roc_auc_score(
    df_actual_predicted_probs['loan_data_targets_test'], 
    df_actual_predicted_probs['y_hat_test_proba']
)
print(f"\nAUROC: {AUROC:.4f}")

# =============================================================================
# Gini Coefficient
# =============================================================================

# Calculate Gini coefficient from AUROC (this is already a simple formula)
Gini = AUROC * 2 - 1
print(f"Gini Coefficient: {Gini:.4f}")

# =============================================================================
# Kolmogorov-Smirnov (KS) Statistic
# =============================================================================

# Calculate KS statistic using scipy
good_probs = df_actual_predicted_probs[df_actual_predicted_probs['loan_data_targets_test'] == 0]['y_hat_test_proba']
bad_probs = df_actual_predicted_probs[df_actual_predicted_probs['loan_data_targets_test'] == 1]['y_hat_test_proba']

ks_statistic, ks_pvalue = ks_2samp(good_probs, bad_probs)
print(f"\nKS Statistic: {ks_statistic:.4f}")

# For the KS chart, we need to recalculate cumulative percentages
df_sorted = df_actual_predicted_probs.sort_values('y_hat_test_proba').reset_index(drop=True)
total_pop = df_sorted.shape[0]
total_good = (df_sorted['loan_data_targets_test'] == 0).sum()
total_bad = (df_sorted['loan_data_targets_test'] == 1).sum()

df_sorted['Cumulative Perc Good'] = (df_sorted['loan_data_targets_test'] == 0).cumsum() / total_good
df_sorted['Cumulative Perc Bad'] = (df_sorted['loan_data_targets_test'] == 1).cumsum() / total_bad

# Plot KS chart
plt.figure(figsize=(10, 6))
plt.plot(
    df_sorted['y_hat_test_proba'], 
    df_sorted['Cumulative Perc Bad'], 
    color='r', 
    linewidth=2, 
    label='Bad'
)
plt.plot(
    df_sorted['y_hat_test_proba'], 
    df_sorted['Cumulative Perc Good'], 
    color='b', 
    linewidth=2, 
    label='Good'
)
plt.xlabel('Estimated Probability for being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov Chart')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# Summary Statistics
# =============================================================================

print("\n" + "="*50)
print("MODEL VALIDATION SUMMARY")
print("="*50)
print(f"Test Set Size: {total_pop}")
print(f"Number of Features: {inputs_test.shape[1]}")
print(f"Good Loans (0): {total_good}")
print(f"Bad Loans (1): {total_bad}")
print(f"Bad Rate: {total_bad/total_pop:.4f}")
print("-"*50)
print(f"AUROC: {AUROC:.4f}")
print(f"Gini Coefficient: {Gini:.4f}")
print(f"KS Statistic: {ks_statistic:.4f}")
print(f"Accuracy (threshold={tr}): {accuracy:.4f}")
print("="*50)
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

os.makedirs('eda_plots', exist_ok=True)
os.makedirs('processed_data', exist_ok=True)
os.makedirs('processed_data/raw_splits', exist_ok=True)

# 1. Load Data
df = pd.read_csv('bank_customers.csv')
print("Initial Dataset Shape:", df.shape)
print("\nMissing Values Count:")
missing = df.isnull().sum()
print(missing[missing > 0])

# 2. Descriptive Statistics Summary
desc_stats = df.describe().T
numeric_cols = df.select_dtypes(include=[np.number]).columns
desc_stats['skewness'] = df[numeric_cols].skew()
desc_stats['kurtosis'] = df[numeric_cols].kurt()
desc_stats.to_csv('processed_data/descriptive_statistics.csv')
print("\nSaved descriptive statistics to processed_data/descriptive_statistics.csv")

# 3. EDA Visualizations

# 3.1 Target & Feature Distributions
numerical_cols = [c for c in df.columns if c != 'CUST_ID']

fig, axes = plt.subplots(6, 3, figsize=(18, 22))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='#1f77b4', bins=30)
    axes[i].set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
for j in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig('eda_plots/01_feature_distributions.png', dpi=300)
plt.close()
print("Generated eda_plots/01_feature_distributions.png")

# 3.2 Correlation Heatmap
plt.figure(figsize=(14, 11))
corr_matrix = df[numerical_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt='.2f', annot_kws={"size": 8})
plt.title('Correlation Matrix of Customer Features', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('eda_plots/02_correlation_heatmap.png', dpi=300)
plt.close()
print("Generated eda_plots/02_correlation_heatmap.png")

# 3.3 Outlier Diagnostics with Boxplots
fig, axes = plt.subplots(6, 3, figsize=(18, 22))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    sns.boxplot(y=df[col], ax=axes[i], color='#2ca02c')
    axes[i].set_title(f'Boxplot of {col}', fontsize=11, fontweight='bold')
    axes[i].set_ylabel(col)
for j in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig('eda_plots/03_outlier_boxplots.png', dpi=300)
plt.close()
print("Generated eda_plots/03_outlier_boxplots.png")

# 3.4 Key Domain Relationships
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# Purchases vs Payments
sns.scatterplot(data=df, x='PURCHASES', y='PAYMENTS', hue='TENURE', palette='viridis', alpha=0.6, ax=axes[0, 0])
axes[0, 0].set_title('Purchases vs Payments by Tenure', fontweight='bold')

# Balance vs Credit Limit
sns.scatterplot(data=df, x='BALANCE', y='CREDIT_LIMIT', hue='PRC_FULL_PAYMENT', palette='coolwarm', alpha=0.6, ax=axes[0, 1])
axes[0, 1].set_title('Balance vs Credit Limit (Hue: Full Payment %)', fontweight='bold')

# Cash Advance vs Cash Advance Frequency
sns.scatterplot(data=df, x='CASH_ADVANCE_FREQUENCY', y='CASH_ADVANCE', color='#d62728', alpha=0.6, ax=axes[1, 0])
axes[1, 0].set_title('Cash Advance Frequency vs Amount', fontweight='bold')

# One-off Purchases vs Installments Purchases
sns.scatterplot(data=df, x='ONEOFF_PURCHASES', y='INSTALLMENTS_PURCHASES', color='#9467bd', alpha=0.6, ax=axes[1, 1])
axes[1, 1].set_title('One-Off vs Installment Purchases', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_plots/04_behavioral_relationships.png', dpi=300)
plt.close()
print("Generated eda_plots/04_behavioral_relationships.png")

# 4. Data Cleaning & Feature Engineering

df_clean = df.copy()

# Imputation (Median is robust against extreme skewness)
credit_limit_median = float(df_clean['CREDIT_LIMIT'].median())
min_payments_median = float(df_clean['MINIMUM_PAYMENTS'].median())

df_clean['CREDIT_LIMIT'] = df_clean['CREDIT_LIMIT'].fillna(credit_limit_median)
df_clean['MINIMUM_PAYMENTS'] = df_clean['MINIMUM_PAYMENTS'].fillna(min_payments_median)

# Domain Feature Engineering
# 1. Credit Utilization Ratio
df_clean['BALANCE_TO_LIMIT_RATIO'] = df_clean['BALANCE'] / (df_clean['CREDIT_LIMIT'] + 1e-5)

# 2. Payment to Minimum Payment Ratio
df_clean['PAYMENT_TO_MIN_PAYMENT_RATIO'] = df_clean['PAYMENTS'] / (df_clean['MINIMUM_PAYMENTS'] + 1e-5)

# 3. Monthly Averages based on Tenure
df_clean['MONTHLY_AVG_PURCHASES'] = df_clean['PURCHASES'] / df_clean['TENURE']
df_clean['MONTHLY_AVG_CASH_ADVANCE'] = df_clean['CASH_ADVANCE'] / df_clean['TENURE']
df_clean['MONTHLY_AVG_PAYMENTS'] = df_clean['PAYMENTS'] / df_clean['TENURE']

# 4. Cash Advance Ratio to Total Expenditure
df_clean['CASH_ADVANCE_SHARE'] = df_clean['CASH_ADVANCE'] / (df_clean['PURCHASES'] + df_clean['CASH_ADVANCE'] + 1e-5)

# 5. Purchase Segmentation Typology
def get_purchase_type(row):
    if row['ONEOFF_PURCHASES'] > 0 and row['INSTALLMENTS_PURCHASES'] > 0:
        return 'Both'
    elif row['ONEOFF_PURCHASES'] > 0:
        return 'One_Off_Only'
    elif row['INSTALLMENTS_PURCHASES'] > 0:
        return 'Installments_Only'
    else:
        return 'No_Purchases'

df_clean['PURCHASE_TYPE'] = df_clean.apply(get_purchase_type, axis=1)

# 6. Targets:
# Classification Target: Full balance payer (1 if full payment ratio >= 0.1, else 0: Revolver vs Transactor)
df_clean['IS_FULL_PAYER'] = (df_clean['PRC_FULL_PAYMENT'] >= 0.1).astype(int)

# Classification Target 2: High Credit Limit Tier (> 5000)
df_clean['HIGH_LIMIT_TIER'] = (df_clean['CREDIT_LIMIT'] > 5000).astype(int)

print("\nCleaned & Engineered Dataset Shape:", df_clean.shape)
print("Purchase Type Distribution:\n", df_clean['PURCHASE_TYPE'].value_counts())
print("Full Payer Target Ratio:\n", df_clean['IS_FULL_PAYER'].value_counts(normalize=True))

# 4.1 Visualizing Engineered Features
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
sns.countplot(data=df_clean, x='PURCHASE_TYPE', hue='IS_FULL_PAYER', palette='Set2', ax=axes[0, 0])
axes[0, 0].set_title('Purchase Type by Full Payer Status', fontweight='bold')

sns.histplot(data=df_clean, x='BALANCE_TO_LIMIT_RATIO', bins=40, kde=True, ax=axes[0, 1], color='#e377c2')
axes[0, 1].set_xlim(0, 1.5)
axes[0, 1].set_title('Balance to Credit Limit Ratio (Utilization)', fontweight='bold')

sns.boxplot(data=df_clean, x='PURCHASE_TYPE', y='MONTHLY_AVG_PURCHASES', ax=axes[1, 0], palette='Pastel1')
axes[1, 0].set_ylim(0, 1000)
axes[1, 0].set_title('Monthly Purchases by Purchase Type', fontweight='bold')

sns.scatterplot(data=df_clean, x='CASH_ADVANCE_SHARE', y='BALANCE_TO_LIMIT_RATIO', hue='IS_FULL_PAYER', palette='coolwarm', alpha=0.6, ax=axes[1, 1])
axes[1, 1].set_title('Cash Advance Share vs Credit Utilization', fontweight='bold')
axes[1, 1].set_ylim(0, 1.5)

plt.tight_layout()
plt.savefig('eda_plots/05_engineered_feature_insights.png', dpi=300)
plt.close()
print("Generated eda_plots/05_engineered_feature_insights.png")

# Save full clean dataset
df_clean.to_csv('processed_data/full_cleaned_engineered.csv', index=False)
print("Saved full cleaned dataset to processed_data/full_cleaned_engineered.csv")

# 5. Dataset Splitting & Preprocessing Pipeline (Pure Python & NumPy, Leak-Free)

def custom_train_val_test_split(df_in, target_col, test_ratio=0.15, val_ratio=0.15, stratify=False, random_state=42):
    np.random.seed(random_state)
    n = len(df_in)
    
    if stratify:
        train_idx, val_idx, test_idx = [], [], []
        for val, group in df_in.groupby(target_col):
            indices = group.index.values.copy()
            np.random.shuffle(indices)
            n_group = len(indices)
            n_test = int(n_group * test_ratio)
            n_val = int(n_group * val_ratio)
            
            test_idx.extend(indices[:n_test])
            val_idx.extend(indices[n_test:n_test + n_val])
            train_idx.extend(indices[n_test + n_val:])
    else:
        indices = np.arange(n)
        np.random.shuffle(indices)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]
        
    return df_in.iloc[train_idx].copy(), df_in.iloc[val_idx].copy(), df_in.iloc[test_idx].copy()

class DataPreprocessor:
    """Standard & Robust Scaler with One-Hot Encoder fitted strictly on training data."""
    def __init__(self, num_cols, cat_cols, method='standard'):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.method = method
        self.params = {}
        self.cat_categories = {}
        
    def fit(self, df_train):
        self.params = {}
        if self.method == 'robust':
            for col in self.num_cols:
                q25 = df_train[col].quantile(0.25)
                q50 = df_train[col].median()
                q75 = df_train[col].quantile(0.75)
                iqr = q75 - q25
                iqr = iqr if iqr > 1e-6 else 1.0
                self.params[col] = {'center': q50, 'scale': iqr}
        else: # standard
            for col in self.num_cols:
                mean = df_train[col].mean()
                std = df_train[col].std()
                std = std if std > 1e-6 else 1.0
                self.params[col] = {'center': mean, 'scale': std}
                
        for col in self.cat_cols:
            cats = sorted(df_train[col].unique().tolist())
            # drop first category to avoid multicollinearity
            self.cat_categories[col] = cats[1:]
        return self

    def transform(self, df_input):
        df_out = pd.DataFrame(index=df_input.index)
        # Transform numericals
        for col in self.num_cols:
            p = self.params[col]
            df_out[col] = (df_input[col] - p['center']) / p['scale']
            
        # One-hot encode categoricals
        for col in self.cat_cols:
            for cat in self.cat_categories[col]:
                col_name = f"{col}_{cat}"
                df_out[col_name] = (df_input[col] == cat).astype(float)
                
        return df_out

    def to_dict(self):
        return {
            'num_cols': self.num_cols,
            'cat_cols': self.cat_cols,
            'method': self.method,
            'params': self.params,
            'cat_categories': self.cat_categories
        }

# Base feature list
base_features = [
    'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
    'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
    'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
    'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
    'PAYMENTS', 'MINIMUM_PAYMENTS', 'TENURE',
    'PAYMENT_TO_MIN_PAYMENT_RATIO', 'MONTHLY_AVG_PURCHASES',
    'MONTHLY_AVG_CASH_ADVANCE', 'MONTHLY_AVG_PAYMENTS', 'CASH_ADVANCE_SHARE',
    'PURCHASE_TYPE'
]

# =========================================================================
# 5.1 REGRESSION DATASET (Target: CREDIT_LIMIT)
# =========================================================================
reg_features = base_features + ['PRC_FULL_PAYMENT']
reg_df = df_clean[reg_features + ['CREDIT_LIMIT']].copy()

train_reg, val_reg, test_reg = custom_train_val_test_split(
    reg_df, target_col='CREDIT_LIMIT', test_ratio=0.15, val_ratio=0.15, stratify=False, random_state=42
)

reg_num_cols = [c for c in reg_features if c != 'PURCHASE_TYPE']
reg_cat_cols = ['PURCHASE_TYPE']

reg_scaler = DataPreprocessor(reg_num_cols, reg_cat_cols, method='robust')
reg_scaler.fit(train_reg)

X_train_reg_scaled = reg_scaler.transform(train_reg)
X_val_reg_scaled = reg_scaler.transform(val_reg)
X_test_reg_scaled = reg_scaler.transform(test_reg)

X_train_reg_scaled['TARGET_CREDIT_LIMIT'] = train_reg['CREDIT_LIMIT'].values
X_val_reg_scaled['TARGET_CREDIT_LIMIT'] = val_reg['CREDIT_LIMIT'].values
X_test_reg_scaled['TARGET_CREDIT_LIMIT'] = test_reg['CREDIT_LIMIT'].values

X_train_reg_scaled.to_csv('processed_data/regression_train.csv', index=False)
X_val_reg_scaled.to_csv('processed_data/regression_val.csv', index=False)
X_test_reg_scaled.to_csv('processed_data/regression_test.csv', index=False)

# Save Raw splits
train_reg.to_csv('processed_data/raw_splits/reg_raw_train.csv', index=False)
val_reg.to_csv('processed_data/raw_splits/reg_raw_val.csv', index=False)
test_reg.to_csv('processed_data/raw_splits/reg_raw_test.csv', index=False)

with open('processed_data/regression_preprocessor_meta.json', 'w') as f:
    json.dump(reg_scaler.to_dict(), f, indent=2)
print(f"Regression datasets saved. Train: {X_train_reg_scaled.shape}, Val: {X_val_reg_scaled.shape}, Test: {X_test_reg_scaled.shape}")

# =========================================================================
# 5.2 CLASSIFICATION DATASET (Target: IS_FULL_PAYER)
# =========================================================================
clf_features = base_features + ['CREDIT_LIMIT', 'BALANCE_TO_LIMIT_RATIO']
clf_df = df_clean[clf_features + ['IS_FULL_PAYER']].copy()

train_clf, val_clf, test_clf = custom_train_val_test_split(
    clf_df, target_col='IS_FULL_PAYER', test_ratio=0.15, val_ratio=0.15, stratify=True, random_state=42
)

clf_num_cols = [c for c in clf_features if c != 'PURCHASE_TYPE']
clf_cat_cols = ['PURCHASE_TYPE']

clf_scaler = DataPreprocessor(clf_num_cols, clf_cat_cols, method='standard')
clf_scaler.fit(train_clf)

X_train_clf_scaled = clf_scaler.transform(train_clf)
X_val_clf_scaled = clf_scaler.transform(val_clf)
X_test_clf_scaled = clf_scaler.transform(test_clf)

X_train_clf_scaled['TARGET_IS_FULL_PAYER'] = train_clf['IS_FULL_PAYER'].values
X_val_clf_scaled['TARGET_IS_FULL_PAYER'] = val_clf['IS_FULL_PAYER'].values
X_test_clf_scaled['TARGET_IS_FULL_PAYER'] = test_clf['IS_FULL_PAYER'].values

X_train_clf_scaled.to_csv('processed_data/classification_train.csv', index=False)
X_val_clf_scaled.to_csv('processed_data/classification_val.csv', index=False)
X_test_clf_scaled.to_csv('processed_data/classification_test.csv', index=False)

# Save Raw splits
train_clf.to_csv('processed_data/raw_splits/clf_raw_train.csv', index=False)
val_clf.to_csv('processed_data/raw_splits/clf_raw_val.csv', index=False)
test_clf.to_csv('processed_data/raw_splits/clf_raw_test.csv', index=False)

with open('processed_data/classification_preprocessor_meta.json', 'w') as f:
    json.dump(clf_scaler.to_dict(), f, indent=2)
print(f"Classification datasets saved. Train: {X_train_clf_scaled.shape}, Val: {X_val_clf_scaled.shape}, Test: {X_test_clf_scaled.shape}")

print("\n--- Summary of Prepared Datasets ---")
print("1. Processed Data Folder: ./processed_data/")
print("2. EDA Plots Folder: ./eda_plots/")
print("3. Preprocessed Scaled Datasets: Ready for Linear/Ridge/Lasso, Logistic Regression, Neural Nets, SVMs.")
print("4. Raw Split Datasets: Ready for Decision Trees, Random Forests, XGBoost, LightGBM.")

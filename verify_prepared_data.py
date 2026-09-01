import os
import pandas as pd
import numpy as np

def run_verifications():
    print("--- Starting Automated Data & Artifact Verification ---")
    
    # 1. Check EDA Plots
    expected_plots = [
        'eda_plots/01_feature_distributions.png',
        'eda_plots/02_correlation_heatmap.png',
        'eda_plots/03_outlier_boxplots.png',
        'eda_plots/04_behavioral_relationships.png',
        'eda_plots/05_engineered_feature_insights.png'
    ]
    for p in expected_plots:
        assert os.path.exists(p), f"Plot {p} does not exist!"
        assert os.path.getsize(p) > 50000, f"Plot {p} is too small / corrupted!"
    print("[OK] All 5 EDA visualization plots verified.")
    
    # 2. Check Full Cleaned Dataset
    full_df = pd.read_csv('processed_data/full_cleaned_engineered.csv')
    assert full_df.isnull().sum().sum() == 0, "Full cleaned dataset contains NaNs!"
    assert len(full_df) == 8950, f"Expected 8950 rows, got {len(full_df)}"
    print(f"[OK] Full cleaned dataset verified ({full_df.shape[0]} rows, {full_df.shape[1]} columns, 0 NaNs).")
    
    # 3. Check Regression Splits
    reg_train = pd.read_csv('processed_data/regression_train.csv')
    reg_val = pd.read_csv('processed_data/regression_val.csv')
    reg_test = pd.read_csv('processed_data/regression_test.csv')
    
    for name, df_split in [('Train', reg_train), ('Val', reg_val), ('Test', reg_test)]:
        assert df_split.isnull().sum().sum() == 0, f"Regression {name} contains NaNs!"
        assert 'TARGET_CREDIT_LIMIT' in df_split.columns, f"Regression {name} missing target column!"
    assert len(reg_train) + len(reg_val) + len(reg_test) == 8950, "Row count mismatch in regression splits!"
    print(f"[OK] Regression splits verified (Train: {reg_train.shape}, Val: {reg_val.shape}, Test: {reg_test.shape}).")
    
    # 4. Check Classification Splits
    clf_train = pd.read_csv('processed_data/classification_train.csv')
    clf_val = pd.read_csv('processed_data/classification_val.csv')
    clf_test = pd.read_csv('processed_data/classification_test.csv')
    
    for name, df_split in [('Train', clf_train), ('Val', clf_val), ('Test', clf_test)]:
        assert df_split.isnull().sum().sum() == 0, f"Classification {name} contains NaNs!"
        assert 'TARGET_IS_FULL_PAYER' in df_split.columns, f"Classification {name} missing target column!"
    assert len(clf_train) + len(clf_val) + len(clf_test) == 8950, "Row count mismatch in classification splits!"
    
    # Check Stratification
    r_train = clf_train['TARGET_IS_FULL_PAYER'].mean()
    r_val = clf_val['TARGET_IS_FULL_PAYER'].mean()
    r_test = clf_test['TARGET_IS_FULL_PAYER'].mean()
    print(f"[OK] Stratified positive class ratios: Train={r_train:.4f}, Val={r_val:.4f}, Test={r_test:.4f}")
    assert abs(r_train - r_test) < 0.02, "Stratification ratio mismatch!"
    
    # 5. Check Raw Splits
    for prefix in ['reg_raw', 'clf_raw']:
        for split in ['train', 'val', 'test']:
            raw_path = f'processed_data/raw_splits/{prefix}_{split}.csv'
            assert os.path.exists(raw_path), f"Missing {raw_path}"
    print("[OK] Raw unscaled splits for tree models verified.")
    
    # 6. Check Notebook
    assert os.path.exists('eda_and_data_preparation.ipynb'), "Missing notebook!"
    print("[OK] Jupyter Notebook verified.")
    print("\n--- ALL VERIFICATIONS PASSED SUCCESSFULLY! ---")

if __name__ == '__main__':
    run_verifications()

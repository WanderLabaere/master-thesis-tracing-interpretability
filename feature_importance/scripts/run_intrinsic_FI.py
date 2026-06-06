"""
run_intrinsic.py
----------------------------
Exectution script to calculate and visualize intrinsic (model-native) feature importance metrics, 
supplemented by validation-set permutation importance, for spatial regression models.

Input data:
1. Preprocessed multi-element CSV datasets (ICPMS or XRF) for Soy and Timber (including genus-specific subsets).
2. Precomputed optimal hyperparameter logs (CSV) for configuring Random Forest and XGBoost algorithms.

Generates and saves:
1. Intrinsic feature importance logs (CSV) aggregating model-native metrics—such as Mean Decrease 
    Impurity (MDI) for Random Forests and Gain/Weight/Cover for XGBoost—alongside validation-set Permutation Importance scores.
2. Comparative visualization plots (PNG) for Random Forest (MDI vs. MDA rank scatter plots) 
    and XGBoost (normalized horizontal bar charts) to identify and rank the most influential geochemical 
    features across spatial cross-validation folds.
"""


import os
import pandas as pd
import time
import datetime

from feature_importance.functions.intrinsic_FI import run_native_feature_importance, load_data

# Directories
tX_imputedNA_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceNA\tX_imputed_NA_0_Ba_Br.csv"
sI_imputedLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\sI_imputed_mice.csv"

cv_params_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\machine_learning\output\CV\tables"

# NEW Output Directory for Native FI
out_fi_base_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\intrinsic_FI"

N_SPLITS = 5
N_CLUSTERS = 10

def execute_native_pipeline(data_name: str, csv_path: str, data_type: str, params_base: str, subfolder_path: str):
    optimal_params_csv = os.path.join(params_base, f"{data_name}_CVtable_optimal_parameters.csv")
    
    run_native_feature_importance(
        data_name=data_name, 
        csv_path=csv_path, 
        data_type=data_type, 
        params_file=optimal_params_csv,
        base_out_dir=out_fi_base_dir,
        subfolder_path=subfolder_path,
        n_splits=N_SPLITS,
        n_clusters=N_CLUSTERS
    )

if __name__ == "__main__":

    print(f"\n{'#'*70}")
    print(f"STARTING PIPELINE FOR NATIVE/INTRINSIC FEATURE IMPORTANCE")
    print(f"{'#'*70}")
    
    #  Process Soy (ICPMS)
    soy_time = time.time()
    execute_native_pipeline(
        data_name="sI", 
        csv_path=sI_imputedLOD_path, 
        data_type="ICPMS", 
        params_base=os.path.join(cv_params_dir, "sI"), 
        subfolder_path="sI"
    )
    print(f"\n[========== SOY PROCESSING COMPLETE ==========]")

    #  Process Timber (XRF) Setup
    df_tX = pd.read_csv(tX_imputedNA_path)
    tx_genera_dir = os.path.join(os.path.dirname(tX_imputedNA_path), "tX_genera_splits")
    tx_params_base = os.path.join(cv_params_dir, "tX")
    os.makedirs(tx_genera_dir, exist_ok=True)

    print("\nPreparing Native Feature Importance for Full Timber Dataset...")
    categorical_col = 'Genus'
    species_dummies = pd.get_dummies(df_tX[categorical_col], prefix=categorical_col).astype(int)
    df_tX_encoded = pd.concat([df_tX, species_dummies], axis=1)
    
    full_encoded_csv_path = os.path.join(tx_genera_dir, "tX_Full_Encoded.csv")
    df_tX_encoded.to_csv(full_encoded_csv_path, index=False)

    try:
        execute_native_pipeline(
            data_name="tX_Full_All_Genera", 
            csv_path=full_encoded_csv_path, 
            data_type="XRF", 
            params_base=os.path.join(tx_params_base, "tX_Full_All_Genera"), 
            subfolder_path=os.path.join("tX", "tX_Full_All_Genera") 
        )
    except Exception as e:
        print(f"  [!] Skipping Full Timber Dataset due to error: {e}")

    # ---> INDIVIDUAL GENERA <---
    for genus, group in df_tX.groupby("Genus"):
        genus_clean = str(genus).strip().replace(" ", "_")
        if not genus_clean or genus_clean.lower() == "nan":
            continue

        if len(group) < 2:
            continue
            
        genus_csv_path = os.path.join(tx_genera_dir, f"tX_{genus_clean}.csv")
        group.to_csv(genus_csv_path, index=False)
        
        try:
            execute_native_pipeline(
                data_name=genus_clean, 
                csv_path=genus_csv_path, 
                data_type="XRF", 
                params_base=os.path.join(tx_params_base, genus_clean), 
                subfolder_path=os.path.join("tX", genus_clean)
            )
        except Exception as e:
            print(f"  [!] Skipping {genus_clean} due to error: {e}")
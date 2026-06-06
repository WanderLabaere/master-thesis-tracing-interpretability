"""
run_feature_extraction_permutation.py
----------------------------
Execution script to perform spatially-stratified cross-validated feature importance analysis 
using Permutation and Conditional Permutation Importance (CPI) methods. This pipeline assesses the predictive 
contribution of individual trace elements and species-level categorical features by measuring the drop in spatial
predictive skill (Haversine Skill Score) upon feature perturbation.

Input data:
1. Preprocessed multi-element CSV datasets (ICPMS or XRF) containing numerical trace features, 
    metadata, and spatial coordinate targets.
2. Model-specific optimal hyperparameter logs (CSV) to configure the evaluated machine learning algorithms.

Generates and saves:
1. Feature importance logs (CSV) detailing the mean predictive impact and cross-fold variance for each 
    feature across the evaluated models.
2. Visual importance rank plots (PNG) featuring error bars to quantify the stability and reliability of 
    feature rankings across spatial folds.
"""

import os
import pandas as pd
import time
import datetime

from feature_importance.functions.feature_extraction_permutation import run_feature_importance, load_data

# Define directories
tX_imputedNA_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceNA\tX_imputed_NA_0_Ba_Br.csv"
sI_imputedLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\sI_imputed_mice.csv"
cI_imputedLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\cI_imputed_mice.csv"

# Define Parameter Read Directory 
# USING CONTINUOUS RANDOM GRID SEARCH RESULTS!!!
cv_params_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\machine_learning\output\CV\tables"

# Define Output Directory
out_fi_base_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\PI"

# ---------------------------------------------------------
# CV and clustering parameters
# ---------------------------------------------------------
N_SPLITS = 5    # Number of cross-validation folds
N_CLUSTERS = 10 # Number of spatial clusters for stratification (independent of n_splits)

# ---------------------------------------------------------
# Define Feature Extraction Methods
# ---------------------------------------------------------
METHODS_TO_RUN = [
    "permutation",
    "conditional_permutation"
]

def execute_fi_pipeline(data_name: str, csv_path: str, data_type: str, params_base: str, out_base: str, method: str, subfolder: str = ""):
    if subfolder:
        method_out_dir = os.path.join(out_base, method, subfolder)
    else:
        method_out_dir = os.path.join(out_base, method)
        
    os.makedirs(method_out_dir, exist_ok=True)
    
    optimal_params_csv = os.path.join(params_base, f"{data_name}_CVtable_optimal_parameters.csv")
    
    run_feature_importance(
        data_name=data_name, 
        csv_path=csv_path, 
        data_type=data_type, 
        params_file=optimal_params_csv,
        output_dir=method_out_dir,
        method=method,
        n_splits=N_SPLITS,
        n_clusters=N_CLUSTERS
    )

if __name__ == "__main__":

    X, y = load_data(sI_imputedLOD_path, "ICPMS")
    print("Min value in X:", X.min().min())
    print("Negative cells:", (X < 0).sum().sum())
    print("Columns with negatives:", X.columns[(X < 0).any()].tolist())
    
    for current_method in METHODS_TO_RUN:
        print(f"\n{'#'*70}")
        print(f"STARTING PIPELINE FOR METHOD: {current_method.upper()}")
        print(f"{'#'*70}")
        
        #  Process Soy (ICPMS)
        soy_time = time.time()

        execute_fi_pipeline("sI", sI_imputedLOD_path, "ICPMS", os.path.join(cv_params_dir, "sI"), out_fi_base_dir, current_method, subfolder = "sI")
        
        soy_elapsed = time.time() - soy_time
        formatted_soy = str(datetime.timedelta(seconds=int(soy_elapsed)))
        print(f"\n[========== ALL PROCESSING COMPLETE IN {formatted_soy} ==========]")

        #  Process Cocoa (ICPMS)
        # execute_fi_pipeline("cI", cI_imputedLOD_path, "ICPMS", os.path.join(cv_params_dir, "cI"), out_fi_base_dir, current_method, subfolder = "cI")

        # ---------------------------------------------------------
        #  Process Timber (XRF) Setup
        # ---------------------------------------------------------
        df_tX = pd.read_csv(tX_imputedNA_path)
        tx_genera_dir = os.path.join(os.path.dirname(tX_imputedNA_path), "tX_genera_splits")
        tx_params_base = os.path.join(cv_params_dir, "tX")
        
        os.makedirs(tx_genera_dir, exist_ok=True)

        # ---> FULL TIMBER DATASET WITH SPECIES INFO <---
        print("\nPreparing Feature Importance for Full Timber Dataset with Species...")
        categorical_col = 'Genus'
        species_dummies = pd.get_dummies(df_tX[categorical_col], prefix=categorical_col).astype(int)
        df_tX_encoded = pd.concat([df_tX, species_dummies], axis=1)
        
        full_encoded_csv_path = os.path.join(tx_genera_dir, "tX_Full_Encoded.csv")
        df_tX_encoded.to_csv(full_encoded_csv_path, index=False)

        try:
            timber_full_time = time.time()

            execute_fi_pipeline(
                data_name="tX_Full_All_Genera", 
                csv_path=full_encoded_csv_path, 
                data_type="XRF", 
                # --- FIX IS HERE: Add the specific subfolder ---
                params_base=os.path.join(tx_params_base, "tX_Full_All_Genera"), 
                out_base=out_fi_base_dir,
                method=current_method,
                subfolder=os.path.join("tX", "tX_Full_All_Genera") 
            )
            
            timber_full_elapsed = time.time() - timber_full_time
            formatted_timber_full = str(datetime.timedelta(seconds=int(timber_full_elapsed)))
            print(f"\n[========== ALL PROCESSING COMPLETE IN {formatted_timber_full} ==========]")
        except Exception as e:
            print(f"  [!] Skipping Full Timber Dataset due to error: {e}")

        # ---> INDIVIDUAL GENERA <---
        for genus, group in df_tX.groupby("Genus"):
            genus_clean = str(genus).strip().replace(" ", "_")
            if not genus_clean or genus_clean.lower() == "nan":
                continue

            n_genus_samples = len(group)
            if n_genus_samples < 2:
                print(f"  [!] Skipping {genus_clean}: only {n_genus_samples} sample(s), cannot run CV.")
                continue
                
            genus_csv_path = os.path.join(tx_genera_dir, f"tX_{genus_clean}.csv")
            group.to_csv(genus_csv_path, index=False)
            
            try:
                execute_fi_pipeline(
                    data_name=genus_clean, 
                    csv_path=genus_csv_path, 
                    data_type="XRF", 
                    # --- FIX IS HERE: Add genus_clean to the params_base path ---
                    params_base=os.path.join(tx_params_base, genus_clean), 
                    out_base=out_fi_base_dir, 
                    method=current_method,
                    subfolder=os.path.join("tX", genus_clean)
                )
            except Exception as e:
                print(f"  [!] Skipping {genus_clean} due to error: {e}")
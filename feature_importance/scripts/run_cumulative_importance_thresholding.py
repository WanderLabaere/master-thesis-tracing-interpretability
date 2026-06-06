"""
run_cumulative_importance_thresholding.py
----------------------------
Execution script for assessing reduced feature set performances by iteratively building model feature sets based on the order of 
precomputed importance rankings. The pipeline performs cumulative cross-validation to identify the minimal 
feature subset required to meet a performance threshold.

Input data:
1. Preprocessed multi-element CSV datasets for Soy (ICPMS) and Timber (XRF).
2. Precomputed feature importance logs (CSV) for both standard Permutation and Conditional 
    Permutation Importance (CPI) methods.
3. Model-specific optimal hyperparameter logs (CSV).

Generates and saves:
1. Cumulative importance assessment logs (CSV) tracking predictive skill (Haversine Skill Score) 
    and cross-fold variance at each iterative feature addition step.
2. Trajectory visualization plots (PNG) illustrating the cumulative performance gains against the baseline HSS
"""

import os
import pandas as pd
import time
import datetime

# Import the newly updated function
from feature_importance.functions.cumulative_importance_thresholding import run_cumulative_thresholding

# =========================================================
# EXECUTION TOGGLES
# =========================================================
RUN_95_THRESHOLD = False
RUN_FULL_FEATURE_SET = True

# Define directories
tX_imputed_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceNA\tX_imputed_NA_0_Ba_Br.csv"
sI_imputedLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\sI_imputed_mice.csv"
cI_imputedLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\cI_imputed_mice.csv"

# Define Parameter Read Directory
cv_params_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\machine_learning\output\CV_v2\tables"

# Precomputed bases mapping
PRECOMPUTED_BASES = {
    "permutation": r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\PI",
    "conditional_permutation": r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\PI"
}

# Output Directories
out_fi_cumulative_base_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\cumulative_importance_TH"
out_fi_full_set_base_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\cumulative_importance_TH\full_feature_set"

# ---------------------------------------------------------
# Define Methods & Spatial Stratification Parameters
# ---------------------------------------------------------
METHODS_TO_RUN = [
    "permutation",
    "conditional_permutation"
]

N_SPLITS = 5
N_CLUSTERS = 10

def execute_pipeline_modes(data_name: str, csv_path: str, data_type: str, params_base: str, precomputed_base: str, method: str, subfolder: str = ""):
    """Wrapper that fires off the pipeline based on the toggles activated."""
    
    #  Run 95% Threshold
    if RUN_95_THRESHOLD:
        out_base = os.path.join(out_fi_cumulative_base_dir, method, subfolder) if subfolder else os.path.join(out_fi_cumulative_base_dir, method)
        precomputed_imp_dir = os.path.join(precomputed_base, method, subfolder) if subfolder else os.path.join(precomputed_base, method)
        os.makedirs(out_base, exist_ok=True)
        
        run_cumulative_thresholding(
            data_name=data_name, 
            csv_path=csv_path, 
            data_type=data_type, 
            params_file=os.path.join(params_base, f"{data_name}_CVtable_optimal_parameters.csv"),
            precomputed_imp_dir=precomputed_imp_dir,
            output_dir=out_base,
            method=method,
            n_splits=N_SPLITS,
            n_clusters=N_CLUSTERS,
            stop_at_threshold=True
        )

    #  Run Full Feature Set
    if RUN_FULL_FEATURE_SET:
        out_base = os.path.join(out_fi_full_set_base_dir, method, subfolder) if subfolder else os.path.join(out_fi_full_set_base_dir, method)
        precomputed_imp_dir = os.path.join(precomputed_base, method, subfolder) if subfolder else os.path.join(precomputed_base, method)
        os.makedirs(out_base, exist_ok=True)
        
        run_cumulative_thresholding(
            data_name=data_name, 
            csv_path=csv_path, 
            data_type=data_type, 
            params_file=os.path.join(params_base, f"{data_name}_CVtable_optimal_parameters.csv"),
            precomputed_imp_dir=precomputed_imp_dir,
            output_dir=out_base,
            method=method,
            n_splits=N_SPLITS,
            n_clusters=N_CLUSTERS,
            stop_at_threshold=False
        )

if __name__ == "__main__":

    for current_method in METHODS_TO_RUN:
        print(f"\n{'#'*70}")
        print(f"STARTING CUMULATIVE PIPELINE: {current_method.upper()}")
        print(f"{'#'*70}")
        
        current_precomputed_base = PRECOMPUTED_BASES[current_method]
        
        #  Process Soy (ICPMS)
        soy_time = time.time()
        execute_pipeline_modes("sI", sI_imputedLOD_path, "ICPMS", os.path.join(cv_params_dir, "sI"), current_precomputed_base, current_method, subfolder="sI")
        
        soy_elapsed = time.time() - soy_time
        formatted_soy = str(datetime.timedelta(seconds=int(soy_elapsed)))
        print(f"\n[========== Soy PROCESSING COMPLETE IN {formatted_soy} ==========]")

        #  Process Cocoa (ICPMS)
        # execute_pipeline_modes("cI", cI_imputedLOD_path, "ICPMS", os.path.join(cv_params_dir, "cI"), current_precomputed_base, current_method, subfolder="cI")

        # ---------------------------------------------------------
        #  Process Timber (XRF)
        # ---------------------------------------------------------
        df_tX = pd.read_csv(tX_imputed_path)
        tx_genera_dir = os.path.join(os.path.dirname(tX_imputed_path), "tX_genera_splits")
        tx_params_base = os.path.join(cv_params_dir, "tX")
        os.makedirs(tx_genera_dir, exist_ok=True)

        # ---> FULL TIMBER DATASET WITH SPECIES INFO <---
        # print("\nPreparing Cumulative Importance for Full Timber Dataset...")
        # categorical_col = 'Genus'
        # species_dummies = pd.get_dummies(df_tX[categorical_col], prefix=categorical_col).astype(int)
        # df_tX_encoded = pd.concat([df_tX, species_dummies], axis=1)
        
        # full_encoded_csv_path = os.path.join(tx_genera_dir, "tX_Full_Encoded.csv")
        # df_tX_encoded.to_csv(full_encoded_csv_path, index=False)

        # try:
        #     timber_full_time = time.time()
        #     execute_pipeline_modes(
        #         data_name="tX_Full_All_Genera", 
        #         csv_path=full_encoded_csv_path, 
        #         data_type="XRF", 
        #         params_base=os.path.join(tx_params_base, "tX_Full_All_Genera"), 
        #         precomputed_base=current_precomputed_base,
        #         method=current_method,
        #         subfolder=os.path.join("tX", "tX_Full_All_Genera") 
        #     )
        #     timber_full_elapsed = time.time() - timber_full_time
        #     print(f"\n[========== TIMBER FULL COMPLETE IN {str(datetime.timedelta(seconds=int(timber_full_elapsed)))} ==========]")
        # except Exception as e:
        #     print(f"  [!] Skipping Full Timber Dataset due to error: {e}")

        # ---> INDIVIDUAL GENERA <---
        for genus, group in df_tX.groupby("Genus"):
            genus_clean = str(genus).strip().replace(" ", "_")
            if not genus_clean or genus_clean.lower() == "nan": continue
            if len(group) < 2: continue
                
            genus_csv_path = os.path.join(tx_genera_dir, f"tX_{genus_clean}.csv")
            group.to_csv(genus_csv_path, index=False)
            
            try:
                execute_pipeline_modes(
                    data_name=genus_clean, 
                    csv_path=genus_csv_path, 
                    data_type="XRF", 
                    params_base=os.path.join(tx_params_base, genus_clean), 
                    precomputed_base=current_precomputed_base,
                    method=current_method,
                    subfolder=os.path.join("tX", genus_clean)
                )
            except Exception as e:
                print(f"  [!] Skipping {genus_clean} due to error: {e}")
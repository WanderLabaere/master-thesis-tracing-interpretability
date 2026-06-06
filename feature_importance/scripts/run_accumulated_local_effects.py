"""
run_accumulated_local_effects.py
----------------------------
Exectution script to execute the Accumulated Local Effects (ALE) computation and visualization pipeline 
across diverse geochemical datasets and machine learning models (Random Forest, XGBoost, SVM).

Input data:
1. Imputed multi-element CSV datasets for Soy (ICPMS) and Timber (XRF).
2. Model-specific optimal hyperparameters (CSV) generated during tuning.

Generates and saves:
1. Standardized model-wise ALE grid plots (PNG) displaying marginal feature effects.
2. Custom synchronized multi-model ALE subplots (PNG) for target features, featuring variable line thickness 
    (based on data density) and rug plots to assess feature importance reliability in regions of data scarcity.
"""

import os
import pandas as pd
import logging

# Silence PyALE's chatty INFO messages
logging.getLogger('PyALE._ALE_generic').setLevel(logging.WARNING)

# Import the pipeline functions (Updated)
from feature_importance.functions.accumulated_local_effects import run_ale_pipeline, create_custom_ale_subplots

# ==========================================
# EXECUTION TOGGLES
# ==========================================
ALL_ALE_PLOTS = True  # Set True to recalculate all standard ALE plots & CSVs
ALE_SUBPLOTS = False    # Set True to read CSVs and generate custom 4-variable subplots

# TARGET FEATURES FOR SUBPLOTS (Rows)
CUSTOM_FEATURES = ["Ba", "Co", "Cu", "Ni"]
# ==========================================

# Input paths
tX_imputed_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceNA\tX_imputed_NA_0_Ba_Br.csv"
sI_imputedLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\sI_imputed_mice.csv"
cI_imputedLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\cI_imputed_mice.csv"

cv_params_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\machine_learning\output\CV\tables"

# TARGET ALE DIRECTORIES
out_ale_base_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\accumulated_local_effects\logAx_fixed"
out_subplots_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\accumulated_local_effects\ALE_subplots"

def execute_wrapper(data_name: str, csv_path: str, data_type: str, params_base_dir: str, file_prefix: str):
    """Small wrapper to locate the exact parameters file and run the pipeline depending on toggles."""
    optimal_params_csv = os.path.join(params_base_dir, f"{file_prefix}_CVtable_optimal_parameters.csv")
    
    if ALL_ALE_PLOTS:
        print(f"\n[ALL_ALE_PLOTS] Running full pipeline for {data_name}...")
        try:
            run_ale_pipeline(
                data_name=data_name,
                csv_path=csv_path,
                data_type=data_type,
                params_file=optimal_params_csv,
                base_output_dir=out_ale_base_dir
            )
        except Exception as e:
            print(f"  [!] Failed processing ALE for {data_name}: {e}")

    if ALE_SUBPLOTS:
        print(f"\n[ALE_SUBPLOTS] Generating custom subplots for {data_name}...")
        try:
            create_custom_ale_subplots(
                data_name=data_name,
                base_ale_dir=out_ale_base_dir,
                output_dir=out_subplots_dir,
                features=CUSTOM_FEATURES,
                csv_path=csv_path,       # <- Added to pass data for density calculation
                data_type=data_type      # <- Added to pass data for density calculation
            )
        except Exception as e:
            print(f"  [!] Failed creating subplots for {data_name}: {e}")


if __name__ == "__main__":
    
    #  Process Soy (ICPMS)
    execute_wrapper("sI", sI_imputedLOD_path, "ICPMS", os.path.join(cv_params_dir, "sI"), "sI")

    #  Process Cocoa (ICPMS)
    # execute_wrapper("cI", cI_imputedLOD_path, "ICPMS", os.path.join(cv_params_dir, "cI"), "cI")

    # ---------------------------------------------------------
    #  Process Timber (XRF)
    # ---------------------------------------------------------
    df_tX = pd.read_csv(tX_imputed_path)
    tx_genera_dir = os.path.join(os.path.dirname(tX_imputed_path), "tX_genera_splits")
    tx_params_base = os.path.join(cv_params_dir, "tX")
    
    # -> FULL TIMBER DATASET <-
    full_encoded_csv_path = os.path.join(tx_genera_dir, "tX_Full_Encoded.csv")
    execute_wrapper(
        data_name=os.path.join("tX", "tX_All_Genera"), 
        csv_path=full_encoded_csv_path, 
        data_type="XRF", 
        params_base_dir=os.path.join(tx_params_base, "tX_Full_All_Genera"),
        file_prefix="tX_Full_All_Genera"
    )

    # -> INDIVIDUAL GENERA <-
    for genus, group in df_tX.groupby("Genus"):
        genus_clean = str(genus).strip().replace(" ", "_")
        if not genus_clean or genus_clean.lower() == "nan":
            continue

        genus_csv_path = os.path.join(tx_genera_dir, f"tX_{genus_clean}.csv")
        
        if len(group) >= 5: 
            execute_wrapper(
                data_name=os.path.join("tX", genus_clean),
                csv_path=genus_csv_path,
                data_type="XRF",
                params_base_dir=os.path.join(tx_params_base, genus_clean),
                file_prefix=genus_clean
            )
        else:
            print(f"  [!] Skipping {genus_clean} ALEs: only {len(group)} sample(s).")
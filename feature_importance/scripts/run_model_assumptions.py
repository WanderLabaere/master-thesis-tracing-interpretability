
"""
run_model_assumptions.py
----------------------------
Execution script to empirically check the assumptions for potential Conditional models. This pipeline tests how accurately the regression models 
(Ridge, ElasticNet, Random Forest) can reconstruct individual features from all remaining features in the dataset.

Input data:
1. Preprocessed multi-element CSV datasets for Soy, Cocoa, and Timber (including genus-specific subsets).

Generates and saves:
1. Feature-level assumption logs (CSV) detailing the mean cross-validated R-squared and absolute residual
    correlation for each feature group.
2. Global dataset assumption logs (CSV) aggregating performance metrics across all features to quantify 
    the overall reliability of CPI-based importance estimates for each specific dataset and model configuration.
"""

import pandas as pd
import os

# Updated import to reflect the generalized function
from feature_importance.functions.model_assumptions import (
    evaluate_cpi_model_r2
)

# ----------------------------------------------
#  Main Orchestration
# ----------------------------------------------

if __name__ == "__main__":
    
    # Input Directories
    tX_imputedNA_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceNA\tX_imputed_NA_0_Ba_Br.csv"
    sI_imputedLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\sI_imputed_mice.csv"
    cI_imputedLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\cI_imputed_mice.csv"
    
    # Base Output Directory
    out_base_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\PI\CPI_model_assumptions"
    
    # Parameters
    N_SPLITS = 5    
    N_CLUSTERS = 10 
    
    # Models to evaluate for the conditional distribution
    MODELS_TO_TEST = ["Ridge", "ElasticNet", "RandomForest"]
    
    # -----------------
    #  Process Soy (ICPMS)
    # -----------------
    print("\n" + "="*50 + "\nProcessing Soy (ICPMS)\n" + "="*50)
    for model_name in MODELS_TO_TEST:
        sI_out = os.path.join(out_base_dir, "sI", model_name)
        os.makedirs(sI_out, exist_ok=True)
        evaluate_cpi_model_r2("sI", sI_imputedLOD_path, "ICPMS", sI_out, model_type=model_name, n_splits=N_SPLITS, n_clusters=N_CLUSTERS)
    
    # -----------------
    #  Process Cocoa (ICPMS)
    # -----------------
    # print("\n" + "="*50 + "\nProcessing Cocoa (ICPMS)\n" + "="*50)
    # for model_name in MODELS_TO_TEST:
    #     cI_out = os.path.join(out_base_dir, "cI", model_name)
    #     os.makedirs(cI_out, exist_ok=True)
    #     evaluate_cpi_model_r2("cI", cI_imputedLOD_path, "ICPMS", cI_out, model_type=model_name, n_splits=N_SPLITS, n_clusters=N_CLUSTERS)

    # -----------------
    #  Process Timber (XRF) Setup
    # -----------------
    print("\n" + "="*50 + "\nPreparing Timber Dataset with Species (XRF)\n" + "="*50)
    df_tX = pd.read_csv(tX_imputedNA_path)
    
    categorical_col = 'Genus'
    species_dummies = pd.get_dummies(df_tX[categorical_col], prefix=categorical_col).astype(int)
    df_tX_encoded = pd.concat([df_tX, species_dummies], axis=1)
    
    # Temp file for the encoded timber
    temp_tx_dir = os.path.join(os.path.dirname(tX_imputedNA_path), "tX_genera_splits")
    os.makedirs(temp_tx_dir, exist_ok=True)
    full_encoded_csv_path = os.path.join(temp_tx_dir, "tX_Full_Encoded.csv")
    df_tX_encoded.to_csv(full_encoded_csv_path, index=False)

    # --- Evaluate Full Timber Dataset across all models ---
    for model_name in MODELS_TO_TEST:
        tx_full_out = os.path.join(out_base_dir, "tX", model_name)
        os.makedirs(tx_full_out, exist_ok=True)
        try:
            evaluate_cpi_model_r2(
                data_name="tX_Full_All_Genera", 
                csv_path=full_encoded_csv_path, 
                data_type="XRF", 
                output_dir=tx_full_out,
                model_type=model_name,
                n_splits=N_SPLITS,
                n_clusters=N_CLUSTERS
            )
        except Exception as e:
            print(f"  [!] Skipping Full Timber Dataset ({model_name}) due to error: {e}")

    # --- Evaluate Individual Genera across all models ---
    for genus, group in df_tX.groupby("Genus"):
        genus_clean = str(genus).strip().replace(" ", "_")
        if not genus_clean or genus_clean.lower() == "nan":
            continue

        n_genus_samples = len(group)
        if n_genus_samples < 2:
            print(f"  [!] Skipping {genus_clean}: only {n_genus_samples} sample(s), cannot run CV.")
            continue
            
        genus_csv_path = os.path.join(temp_tx_dir, f"tX_{genus_clean}.csv")
        group.to_csv(genus_csv_path, index=False)
        
        for model_name in MODELS_TO_TEST:
            tx_genus_out = os.path.join(out_base_dir, "tX", model_name)
            os.makedirs(tx_genus_out, exist_ok=True)
            try:
                evaluate_cpi_model_r2(
                    data_name=genus_clean, 
                    csv_path=genus_csv_path, 
                    data_type="XRF", 
                    output_dir=tx_genus_out,
                    model_type=model_name,
                    n_splits=N_SPLITS,
                    n_clusters=N_CLUSTERS
                )
            except Exception as e:
                print(f"  [!] Skipping {genus_clean} ({model_name}) due to error: {e}")
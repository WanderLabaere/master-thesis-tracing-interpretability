"""
run_extract_SHAP.py
----------------------------
Execution script to perform SHAP analysis across geochemical datasets, 
focusing on model interpretability rather than cross-validated predictive performance.

Input data:
1. Preprocessed multi-element CSV datasets (ICPMS or XRF) for Soy, Cocoa, and Timber.
2. Precomputed optimal hyperparameter logs (CSV) for configuring the evaluated machine learning algorithms 
    (Random Forest, XGBoost, SVM).

Generates and saves:
1. Per-sample SHAP interaction logs (CSV) mapping true spatial coordinates to the local attribution of each trace element feature.
2. Global SHAP beeswarm summary plots (PNG) for both Longitude and Latitude.
3. Tabular feature importance logs (CSV) aggregating the mean absolute impact of all features across both spatial axes.
"""

import os
import pandas as pd
import time
import datetime

from feature_importance.functions.extract_SHAP import run_shap_pipeline

# ---------------------------------------------------------------------------
# Raw Data Paths
# ---------------------------------------------------------------------------
tX_imputedNA_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceNA\tX_imputed_NA_0_Ba_Br.csv"
sI_imputedLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\sI_imputed_mice.csv"
cI_imputedLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\cI_imputed_mice.csv"

# ---------------------------------------------------------------------------
# Input / Output Directories
# ---------------------------------------------------------------------------
cv_params_dir   = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\machine_learning\output\CV\tables"
out_fi_base_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output"

shap_out_base = os.path.join(out_fi_base_dir, "SHAP")
os.makedirs(shap_out_base, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Soy (ICPMS)
    soy_time = time.time()
    # run_shap_pipeline(
    #     data_name  = "sI",
    #     csv_path   = sI_imputedLOD_path,
    #     data_type  = "ICPMS",
    #     params_file = os.path.join(cv_params_dir, "sI", "sI_CVtable_optimal_parameters.csv"),
    #     output_dir = shap_out_base,
    # )

    soy_elapsed = time.time() - soy_time
    formatted_soy = str(datetime.timedelta(seconds=int(soy_elapsed)))
    print(f"\n[========== ALL PROCESSING COMPLETE IN {formatted_soy} ==========]")

    #  Cocoa (ICPMS)
    # run_shap_pipelin_simple(
    #     data_name  = "cI",
    #     csv_path   = cI_imputedLOD_path,
    #     data_type  = "ICPMS",
    #     params_file = os.path.join(cv_params_dir, "cI", "cI_CVtable_optimal_parameters.csv"),
    #     output_dir = shap_out_base,
    # )

    #  Timber (XRF) — per-genus splits
    df_tX = pd.read_csv(tX_imputedNA_path)
    tx_genera_dir = os.path.join(os.path.dirname(tX_imputedNA_path), "tX_genera_splits")
    tx_shap_out   = os.path.join(shap_out_base, "tX")
    os.makedirs(tx_genera_dir, exist_ok=True)
    os.makedirs(tx_shap_out,   exist_ok=True)

    for genus, group in df_tX.groupby("Genus"):
        genus_time = time.time()

        genus_clean = str(genus).strip().replace(" ", "_")
        if not genus_clean or genus_clean.lower() == "nan":
            continue

        genus_csv_path = os.path.join(tx_genera_dir, f"tX_{genus_clean}.csv")
        group.to_csv(genus_csv_path, index=False)

        try:
            run_shap_pipeline(
                data_name   = genus_clean,
                csv_path    = genus_csv_path,
                data_type   = "XRF",
                params_file = os.path.join(cv_params_dir, "tX", genus_clean, f"{genus_clean}_CVtable_optimal_parameters.csv"),
                output_dir  = tx_shap_out,
            )
        except Exception as e:
            print(f"  [!] Skipping {genus_clean} due to error: {e}")


        genus_elapsed = time.time() - genus_time
        formatted_genus = str(datetime.timedelta(seconds=int(genus_elapsed)))
        print(f"\n[========== ALL PROCESSING COMPLETE IN {formatted_genus} ==========]")



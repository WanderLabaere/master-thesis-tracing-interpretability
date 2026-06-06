"""
run_plot_SHAP_beeswarm.py
----------------------------
Execution script for regenerating SHAP visualizations from previously 
saved per-sample data logs. This pipeline enables creating beeswarm plots without 
the high computational cost of running the SHAP analysis.

Input data:
1. Per-sample SHAP data logs (CSV) containing original feature values and calculated directional SHAP 
    contributions for spatial model targets.

Generates and saves:
1. Reconstructed global SHAP beeswarm summary plots (PNG) for both Longitude and Latitude.
"""

import os
import pandas as pd
import time
import datetime

from feature_importance.functions.plot_SHAP_beeswarm import reconstruct_shap_plots

# ---------------------------------------------------------------------------
# Raw Data Paths (used to extract timber genera)
# ---------------------------------------------------------------------------
tX_imputedNA_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceNA\tX_imputed_NA_0_Ba_Br.csv"

# ---------------------------------------------------------------------------
# Input / Output Directories
# ---------------------------------------------------------------------------
# Where the CSVs were generated
shap_in_base = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\SHAP"

# Where the new reconstructed plots will go
shap_out_base = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\SHAP\SHAP_beeswarm"

models_to_plot = ["RandomForest", "XGBoost", "SVM"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Soy (ICPMS)
    soy_time = time.time()
    for model in models_to_plot:
        reconstruct_shap_plots(
            csv_path=os.path.join(shap_in_base, "sI", f"sI_{model}_SHAP_per_sample.csv"),
            data_name="sI",
            model_name=model,
            output_dir=os.path.join(shap_out_base, "sI"),
        )
    soy_elapsed = time.time() - soy_time
    formatted_soy = str(datetime.timedelta(seconds=int(soy_elapsed)))
    print(f"\n[========== Soy PROCESSING COMPLETE IN {formatted_soy} ==========]")

    #  Cocoa (ICPMS)
    # cocoa_time = time.time()
    # for model in models_to_plot:
    #     reconstruct_shap_plots(
    #         csv_path=os.path.join(shap_in_base, "cI", f"cI_{model}_SHAP_per_sample.csv"),
    #         data_name="cI",
    #         model_name=model,
    #         output_dir=os.path.join(shap_out_base, "cI"),
    #     )
    # cocoa_elapsed = time.time() - cocoa_time
    # formatted_cocoa = str(datetime.timedelta(seconds=int(cocoa_elapsed)))
    # print(f"\n[========== COCOA PROCESSING COMPLETE IN {formatted_cocoa} ==========]")

    #  Timber (XRF) — per-genus splits
    print("\n[========== STARTING TIMBER (tX) ==========]")
    df_tX = pd.read_csv(tX_imputedNA_path)
    
    tx_shap_in = os.path.join(shap_in_base, "tX")
    tx_shap_out = os.path.join(shap_out_base, "tX")

    for genus, group in df_tX.groupby("Genus"):
        genus_time = time.time()

        genus_clean = str(genus).strip().replace(" ", "_")
        if not genus_clean or genus_clean.lower() == "nan":
            continue

        for model in models_to_plot:
            csv_path = os.path.join(tx_shap_in, genus_clean, f"{genus_clean}_{model}_SHAP_per_sample.csv")
            out_dir = os.path.join(tx_shap_out, genus_clean)
            
            try:
                reconstruct_shap_plots(
                    csv_path=csv_path,
                    data_name=genus_clean,
                    model_name=model,
                    output_dir=out_dir,
                )
            except Exception as e:
                print(f"  [!] Skipping {genus_clean} ({model}) due to error: {e}")

        genus_elapsed = time.time() - genus_time
        formatted_genus = str(datetime.timedelta(seconds=int(genus_elapsed)))
        print(f"[{genus_clean} complete in {formatted_genus}]")
        
    print("\n[========== ALL PROCESSING COMPLETE ==========]")
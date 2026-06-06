"""
run_feature_correlations.py
----------------------------
Execution script to run the feature correlation pipeline on preprocessed trace element datasets (Soy, Cocoa, and Timber).

Input data:
1. Preprocessed (filtered or imputed) multi-element CSV datasets for Timber (XRF), Soy (ICPMS), and Cocoa (ICPMS).

Generates and saves:
1. Pearson correlation matrices (CSV) and corresponding upper-triangle masked heatmaps (PNG).
2. Variance Inflation Factor (VIF) tables to assess feature multicollinearity (CSV).
3. Pairwise Mutual Information (MI) matrices (CSV) and corresponding upper-triangle masked heatmaps (PNG).
4. Genus-specific dataset splits for Timber (CSV) alongside subsequent individual correlation outputs for each isolated genus.
"""

import pandas as pd
import os

# Import the run_correlation_pipeline function
from data_exploration.functions.feature_correlations import run_correlation_pipeline 

# Define input directories
# Assuming these point to the original RAW data since the pipeline handles log+std.
# If these already point to logStd data, simply adjust the pipeline function to skip the transform step.
# tX_raw_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_transformation\output\dataframes\raw\tX_raw.csv"
# sI_raw_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_transformation\output\dataframes\raw\sI_raw.csv"
# cI_raw_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_transformation\output\dataframes\raw\cI_raw.csv"

tX_stripped_path    = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\filtered\NA_0_pct\tX_filtered_NA_0_pct.csv"
sI_imputedLOD_path  = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\sI_imputed_mice.csv"
cI_imputedLOD_path  = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\cI_imputed_mice.csv"


# NEW CORRELATION OUTPUT DIRECTORY
out_base_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_exploration\output\feature_correlation"

def execute_correlation(data_name: str, csv_path: str, data_type: str, base_out_dir: str):
    """
    Runs the correlation pipeline for a specific dataset and outputs 
    the files into dedicated subdirectories.
    """
    display_name = data_name.replace(os.sep, " - ")
    
    print(f"\n{'='*60}")
    print(f"PROCESSING CORRELATION FOR DATASET: {display_name.upper()} ({data_type})")
    print(f"{'='*60}")
    
    try:
        run_correlation_pipeline(
            input_path=csv_path,
            base_output_path=base_out_dir,
            data_name=data_name,
            data_type=data_type
        )
    except Exception as e:
        print(f"  [!] Error running Correlation pipeline for {display_name}: {e}")

if __name__ == "__main__": 
    
    #  Process Soy (ICPMS)
    execute_correlation(os.path.join("sI", "sI_All"), sI_imputedLOD_path, "ICPMS", out_base_dir)

    #  Process Cocoa (ICPMS)
    execute_correlation(os.path.join("cI", "cI_All"), cI_imputedLOD_path, "ICPMS", out_base_dir)

    # ---------------------------------------------------------
    #  Process Timber (XRF) Setup
    # ---------------------------------------------------------
    df_tX = pd.read_csv(tX_stripped_path)
    
    tx_genera_dir = os.path.join(os.path.dirname(tX_stripped_path), "tX_genera_splits_for_corr")
    os.makedirs(tx_genera_dir, exist_ok=True)

    # ---> FULL TIMBER DATASET <---
    # Note: Placed in tX/tX_All_Genera as requested
    execute_correlation(
        data_name=os.path.join("tX", "tX_All_Genera"), 
        csv_path=tX_stripped_path, 
        data_type="XRF", 
        base_out_dir=out_base_dir
    )
    
    # ---------------------------------------------------------
    #  Process Timber (XRF) - Splitting by Genus
    # ---------------------------------------------------------
    for genus, group in df_tX.groupby("Genus"):
        genus_clean = str(genus).strip().replace(" ", "_")
        if not genus_clean or genus_clean.lower() == "nan":
            continue
            
        genus_csv_path = os.path.join(tx_genera_dir, f"tX_{genus_clean}.csv")
        
        # Require enough samples to compute meaningful statistics
        if len(group) >= 5: 
            group.to_csv(genus_csv_path, index=False)
            
            # Note the os.path.join to nest the sub-groups inside the tX folder
            execute_correlation(
                data_name=os.path.join("tX", genus_clean), 
                csv_path=genus_csv_path, 
                data_type="XRF", 
                base_out_dir=out_base_dir
            )
        else:
            print(f"  [!] Skipping {genus_clean} due to insufficient samples for correlation matrices (n={len(group)}).")
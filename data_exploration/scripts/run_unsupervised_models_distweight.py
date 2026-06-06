"""
run_unsupervised_models_distweight.py
----------------------------
Execution script to run the unsupervised dimensionality reduction pipeline (PCA, sPCA, and MAF) 
on preprocessed trace element datasets (Soy, Cocoa, and Timber).

Input data:
1. Preprocessed (imputed) multi-element CSV datasets for Timber (XRF), Soy (ICPMS), and Cocoa (ICPMS).
2. Local world map shapefiles for geospatial visualization.

Generates and saves:
1. Genus-specific dataset splits for Timber (CSV) alongside subsequent individual analyses for 
    each isolated genus with sufficient sample size.
2. Automated execution of PCA, sPCA, and MAF pipelines for all defined datasets and subgroups, generating their respective outputs 
   (ranked feature loadings, scores matrices, eigenvalue summaries, biplots, scree plots, and geographical spatial maps) organized into method-specific subdirectories.
"""

import os
import pandas as pd
from data_exploration.functions.unsupervised_models_distweight import run_analysis

# -----------------------------------------------------------------------
# Input paths
# -----------------------------------------------------------------------

tX_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceNA\tX_imputed_NA_0_Ba_Br.csv"
sI_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\sI_imputed_mice.csv"
cI_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\cI_imputed_mice.csv"

# -----------------------------------------------------------------------
# Geographical Map Data directory
# -----------------------------------------------------------------------

MAP_DATA_DIR = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_input\world_map_data\World_countries"

# -----------------------------------------------------------------------
# Output base directory
# -----------------------------------------------------------------------

OUT_BASE = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_exploration\output\unsupervised_models_distweight"

# -----------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------

def execute_models(
    data_name: str,
    csv_path:  str,
    data_type: str,
    base_out_dir: str,
    map_data_dir: str,
    n_clusters: int = 3,
    pc_index: int = 2,
):
    """
    Runs PCA, sPCA, and MAF for a given dataset.
    """
    display_name = data_name.replace(os.sep, " > ")
    print(f"\n{'='*60}")
    print(f"DATASET : {display_name.upper()}  ({data_type})")
    print(f"{'='*60}")

    for method in ["PCA", "sPCA", "MAF"]:
        try:
            run_analysis(
                input_path=csv_path,
                base_output_path=base_out_dir,
                data_name=data_name,
                data_type=data_type,
                method=method,
                map_data_dir=map_data_dir,
                pc_index=pc_index,
                n_clusters=n_clusters,
            )
        except Exception as e:
            print(f"Error — {method} for {display_name}: {e}")


if __name__ == "__main__":

    # change pc_index here. 0 = PC1, 1 = PC2, etc.
    # for map score plots
    target_component_index = 0

    # Soy (ICP-MS)
    execute_models("sI", sI_path, "ICPMS", OUT_BASE, MAP_DATA_DIR, n_clusters=4, pc_index=target_component_index)

    # Cocoa (ICP-MS)
    execute_models("cI", cI_path, "ICPMS", OUT_BASE, MAP_DATA_DIR, n_clusters=3, pc_index=target_component_index)

    # Timber (XRF) — split by Genus
    print("\n" + "="*60)
    print("TIMBER (XRF) — GENUS SPLITS")
    print("="*60)

    df_tX = pd.read_csv(tX_path)

    tx_genera_dir = os.path.join(os.path.dirname(tX_path), "tX_genera_splits")
    os.makedirs(tx_genera_dir, exist_ok=True)

    for genus, group in df_tX.groupby("Genus"):
        genus_clean = str(genus).strip().replace(" ", "_")
        if not genus_clean or genus_clean.lower() == "nan":
            continue

        if len(group) < 4:
            print(f"Skipping {genus_clean} — insufficient samples (n={len(group)}).")
            continue

        genus_csv_path = os.path.join(tx_genera_dir, f"tX_{genus_clean}.csv")
        group.to_csv(genus_csv_path, index=False)

        execute_models(
            data_name=os.path.join("tX", genus_clean),
            csv_path=genus_csv_path,
            data_type="XRF",
            base_out_dir=OUT_BASE,
            map_data_dir=MAP_DATA_DIR,
            n_clusters=4,
            pc_index=target_component_index
        )

    print("\nAll analyses complete.")
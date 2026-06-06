"""
run_basic_exploration.py
----------------------------
Execution script to run the basic data exploration pipeline on cleaned trace element datasets (Soy, Cocoa, and Timber).

Input data:
1. Cleaned multi-element CSV datasets for Timber (XRF), Soy (ICPMS), and Cocoa (ICPMS).
2. Local world map shapefiles for geospatial visualization.

Generates and saves:
1. Cross-dataset general overviews and category distribution tables (CSV).
2. Categorical frequency tables detailing sample occurrences per category (CSV).
3. Individual feature lists detailing metadata, numerical features, and spatial coordinate ranges (CSV).
4. NA and LOD (Limit of Detection) percentage tables for numerical features (CSV).
5. Single-dataset geographic distribution plots overlaid on a world map (PNG).
6. Multi-panel geographic subplots comparing specific timber genera (PNG).
7. Genus-specific dataset splits for Timber (CSV) alongside individual exploration outputs for each isolated genus.
"""

import pandas as pd
import os
import geopandas as gpd 

# Import the required functions from your exploration pipeline script
from data_exploration.functions.basic_DE import (
    run_all, 
    create_general_overview, 
    create_individual_feature_lists, 
    create_geoplot, 
    create_category_distribution_table, 
    create_genera_subplots,
    create_na_lod_table
)

if __name__ == "__main__":
    
    # Define input directories
    tX_OG_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\cleaned\tX_cleaned.csv"
    sI_OG_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\cleaned\sI_cleaned.csv"
    cI_OG_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\cleaned\cI_cleaned.csv"

    # EXPLORATION OUTPUT DIRECTORIES
    freq_base_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_exploration\output\basic_DE\frequencies"
    overview_base_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_exploration\output\basic_DE\datasets_overview"
    geoplots_base_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_exploration\output\basic_DE\samples_geoplots"
    na_lod_base_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_exploration\output\basic_DE\NA_LOD_tables"

    def execute_exploration(data_name: str, csv_path: str, data_type: str, base_out_dir: str):
        """
        Runs the categorical frequency pipeline for a specific dataset.
        """
        display_name = data_name.replace(os.sep, " - ")
        
        print(f"\n{'='*60}")
        print(f"PROCESSING FREQUENCIES FOR DATASET: {display_name.upper()} ({data_type})")
        print(f"{'='*60}")
        
        try:
            run_all(
                input_path=csv_path,
                base_output_path=base_out_dir,
                data_name=data_name,
                data_type=data_type
            )
        except Exception as e:
            print(f"Error running exploration for {display_name}: {e}")

    # ---------------------------------------------------------
    # INITIAL SETUP & LOAD WORLD MAP
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("LOADING DATASETS & BASE MAP")
    print("="*60)
    
    df_tX = pd.read_csv(tX_OG_path)
    df_sI = pd.read_csv(sI_OG_path)
    df_cI = pd.read_csv(cI_OG_path)

    map_data_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_input\world_map_data\World_countries"
    try:
        world_df = gpd.read_file(map_data_dir)
        print("Successfully loaded world map shapefile.")
    except Exception as e:
        raise FileNotFoundError(f"Could not load world map from:\n  {map_data_dir}\nError: {e}")

    # ---------------------------------------------------------
    #  GENERAL OVERVIEW (Side-by-side comparison of all 3)
    # ---------------------------------------------------------
    all_overview_dir = os.path.join(overview_base_dir, "all")
    overview_datasets = {
        "timber": {"df": df_tX, "type": "XRF"},
        "soy": {"df": df_sI, "type": "ICPMS"},
        "cocoa": {"df": df_cI, "type": "ICPMS"}
    }
    create_general_overview(overview_datasets, all_overview_dir)

    print("\nGenerating Timber Genera Overview...")
    create_category_distribution_table(
        df=df_tX,
        category_col="Genus",
        out_dir=all_overview_dir,
        file_name="tX_genera_nsamples_overview.csv"
    )

    # ---------------------------------------------------------
    #  INDIVIDUAL DATASET PROCESSING
    # ---------------------------------------------------------
    
    # --- Process Soy (ICPMS) ---
    execute_exploration("sI", sI_OG_path, "ICPMS", freq_base_dir)
    create_individual_feature_lists(df_sI, "sI", "ICPMS", overview_base_dir)
    create_geoplot(df_sI, "sI", geoplots_base_dir, world_df)
    create_na_lod_table(df_sI, "sI", na_lod_base_dir, "ICPMS")

    # --- Process Cocoa (ICPMS) ---
    execute_exploration("cI", cI_OG_path, "ICPMS", freq_base_dir)
    create_individual_feature_lists(df_cI, "cI", "ICPMS", overview_base_dir)
    create_geoplot(df_cI, "cI", geoplots_base_dir, world_df)
    create_na_lod_table(df_cI, "cI", na_lod_base_dir, "ICPMS")

    # --- Process Timber (XRF) ---
    tx_genera_dir = os.path.join(os.path.dirname(tX_OG_path), "tX_genera_splits")
    os.makedirs(tx_genera_dir, exist_ok=True)
    
    # -> Full Timber Dataset
    print("\nPreparing Full Timber Dataset...")
    tx_full_name = os.path.join("tX", "tX_All_Genera")
    execute_exploration(tx_full_name, tX_OG_path, "XRF", freq_base_dir)
    create_individual_feature_lists(df_tX, tx_full_name, "XRF", overview_base_dir)
    create_geoplot(df_tX, tx_full_name, geoplots_base_dir, world_df)
    create_na_lod_table(df_tX, tx_full_name, na_lod_base_dir, "XRF")
    
    # -> Timber Genera Subplots (NEW)
    print("\nGenerating Genera Comparison Subplots...")
    
    # EDIT THIS LIST to choose which genera you want in your subplot grid
    genera_to_plot = ["Betula", "Fagus", "Quercus", "Pinus"]
    
    create_genera_subplots(
        df=df_tX, 
        selected_genera=genera_to_plot, 
        base_out_dir=geoplots_base_dir, 
        world_map=world_df
    )

    # -> Splitting by Genus
    print("\nSplitting Timber Dataset by Genus...")
    for genus, group in df_tX.groupby("Genus"):
        genus_clean = str(genus).strip().replace(" ", "_")
        if not genus_clean or genus_clean.lower() == "nan":
            continue
            
        # Save split CSV
        genus_csv_path = os.path.join(tx_genera_dir, f"tX_{genus_clean}.csv")
        group.to_csv(genus_csv_path, index=False)
        
        # Apply pipeline logic
        genus_data_name = os.path.join("tX", genus_clean)
        
        execute_exploration(genus_data_name, genus_csv_path, "XRF", freq_base_dir)
        create_individual_feature_lists(group, genus_data_name, "XRF", overview_base_dir)
        create_geoplot(group, genus_data_name, geoplots_base_dir, world_df)
        create_na_lod_table(group, genus_data_name, na_lod_base_dir, "XRF")
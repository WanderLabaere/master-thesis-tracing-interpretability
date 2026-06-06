"""
run_imputation_exploration.py
----------------------------
Execution script to run the missing data and Limit of Detection (LOD) exploration pipeline on cleaned trace element datasets (Timber, Soy, and Cocoa).

Input data:
1. Cleaned multi-element CSV datasets for Timber (XRF), Soy (ICPMS), and Cocoa (ICPMS).

Generates and saves:
1. Binned missingno nullity matrices (PNG) visually mapping missing data for specified target variables 
    (e.g., Ba, Br) in the Timber dataset, sorted by country.
2. Classification reports (CSV) evaluating a Random Forest model's ability to predict country based purely 
    on target variable missingness in the Timber dataset.
3. LOD-specific nullity matrices (PNG) visualizing the spatial distribution of Limit of Detection ('<') 
    values across the Soy and Cocoa datasets.
4. Classification reports (CSV) evaluating the spatial predictive power of LOD presence across all applicable 
    features in the Soy and Cocoa datasets.
5. Classification reports (CSV) evaluating spatial predictive power using strictly high-frequency LOD variables 
    (default >40% threshold) in the Soy and Cocoa datasets.
"""

import os
from data_exploration.functions.imputation_exploration import (
    create_sorted_nullity_matrix, 
    predict_country_from_missingness,
    create_lod_nullity_matrix,
    predict_country_from_lods,
    predict_country_from_high_lods
)

### Explicit Path Definitions
tX_cleaned = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\cleaned\tX_cleaned.csv"
sI_cleaned = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\cleaned\sI_cleaned.csv"
cI_cleaned = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\cleaned\cI_cleaned.csv"

# Mapping to dictionary for the loop
datasets = {
    "tX": tX_cleaned,
    "sI": sI_cleaned,
    "cI": cI_cleaned
}

# Output Base
base_exploration_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_exploration\output\imputation_exploration"

### Define parameters
spatial_structure_var = "Country"
variables_to_check = ["Ba", "Br"]
high_lod_threshold = 0.40 # 40% threshold


### Run visualization pipeline
print("RUNNING MISSINGNESS AND LOD EXPLORATION PIPELINE\n")
print("=========================================")
print(f" CHECKING STRUCTURE FOR: {spatial_structure_var}")
print(f" TARGET VARIABLES: {variables_to_check}")
print("=========================================\n")

# Loop through each dataset
for ds_name, ds_path in datasets.items():
    print(f"--- Processing Dataset: {ds_name} ---")
    ds_out_path = os.path.join(base_exploration_path, ds_name)
    os.makedirs(ds_out_path, exist_ok=True)
    
    if ds_name == "tX":
        # Standard Analysis for tX (Fixed Variables)
        create_sorted_nullity_matrix(ds_path, ds_out_path, ds_name, spatial_structure_var, variables_to_check)
        predict_country_from_missingness(ds_path, ds_out_path, ds_name, spatial_structure_var, variables_to_check)

    elif ds_name in ["sI", "cI"]:
        # Automated LOD Analysis for sI/cI
        create_lod_nullity_matrix(ds_path, ds_out_path, ds_name, spatial_structure_var)
        
        # Predict country using ALL LOD variables
        predict_country_from_lods(ds_path, ds_out_path, ds_name, spatial_structure_var)
        
        # Predict country using ONLY >40% LOD variables
        predict_country_from_high_lods(ds_path, ds_out_path, ds_name, spatial_structure_var, threshold=high_lod_threshold)
    
    print("\n")
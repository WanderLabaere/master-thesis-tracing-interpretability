"""
run_data_filtering.py
----------------------------
Execution script to apply multiple missing value (NA) filtering thresholds across cleaned trace element datasets (Timber, Soy, and Cocoa).
The different NA percentages were to compare the results based on how much NA we allowed. 
-> Did not matter that much and we ended up only using Ba and Br for timber. 

Input data:
1. Cleaned multi-element CSV datasets for Timber (XRF), Soy and Cocoa (ICPMS).

Generates and saves:
1. Multiple filtered datasets (CSV) corresponding to specific missing value thresholds (0%, 0.5%, and 50%), organized into threshold-specific subdirectories.
2. Detailed removal logs (CSV) for each filtering run, tracking exactly which samples and variables were excluded and why.
3. Specific threshold-exception datasets (e.g., Timber datasets explicitly retaining target features like Ba and Br during the strictest 0% NA runs).
"""

import os
from data_preparation.functions.data_filtering import filter_clean_df # Commented out if running directly

### Define directories
# Input
tX_clean_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\cleaned\tX_cleaned.csv"
sI_clean_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\cleaned\sI_cleaned.csv"
cI_clean_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\cleaned\cI_cleaned.csv"

# Output Base
base_df_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\filtered"
base_log_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\logs\filtering"

# Define the runs and thresholds
runs = {
    "NA_0_pct": 0.0,
    "NA_0.5_pct": 0.005,
    "NA_50_pct": 0.5
}

### Run data cleaning
print("RUNNING DATA FILTERING\n")

for folder_name, threshold in runs.items():
    print(f"=========================================")
    print(f"  RUNNING THRESHOLD: {folder_name} (> {threshold*100}% NA)")
    print(f"=========================================\n")
    
    # Generate specific sub-paths for this iteration
    out_dataFiltering_df_path = os.path.join(base_df_path, folder_name)
    out_dataFiltering_log_path = os.path.join(base_log_path, folder_name)
    
    # Exception handling for tX naming on NA_0_pct
    if folder_name == "NA_0_pct":
        tX_csv_name = f"tX_filtered_{folder_name}_Ba_Br"
    else:
        tX_csv_name = f"tX_filtered_{folder_name}"

    # Process the three files and append the NA percentage suffix to the file name
    filter_clean_df(tX_clean_path, out_dataFiltering_df_path, out_dataFiltering_log_path, csv_name=tX_csv_name, data_type="XRF", na_threshold=threshold)
    filter_clean_df(sI_clean_path, out_dataFiltering_df_path, out_dataFiltering_log_path, csv_name=f"sI_filtered_{folder_name}", data_type="ICPMS", na_threshold=threshold)
    filter_clean_df(cI_clean_path, out_dataFiltering_df_path, out_dataFiltering_log_path, csv_name=f"cI_filtered_{folder_name}", data_type="ICPMS", na_threshold=threshold)
"""
run_data_stripping.py
----------------------------
Execution script to run the data stripping pipeline on filtered trace element datasets (Timber, Soy, and Cocoa).

Input data:
1. Filtered multi-element CSV datasets for Timber (XRF), Soy (ICPMS), and Cocoa (ICPMS).

Generates and saves:
1. Stripped datasets (CSV) that are entirely free of LOD and NA values.
2. removal logs (CSV) tracking which variables and samples were dropped during the stripping process.
"""

### import the required functions
from data_preparation.functions.data_stripping import strip_filtered_df


### Define directories
# Input
tX_filtered_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\filtered\tX_filtered.csv"
sI_filtered_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\filtered\sI_filtered.csv"
cI_filtered_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\filtered\cI_filtered.csv"
# Output
out_dataStripping_df_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\stripped"
out_dataStripping_log_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\logs\stripping"

### Run data cleaning
# No WFID duplicates are found.
print("RUNNING DATA FILTERING\n")
strip_filtered_df(tX_filtered_path, out_dataStripping_df_path, out_dataStripping_log_path, csv_name = "tX_stripped", data_type = "XRF")
strip_filtered_df(sI_filtered_path, out_dataStripping_df_path, out_dataStripping_log_path, csv_name = "sI_stripped", data_type = "ICPMS")
strip_filtered_df(cI_filtered_path, out_dataStripping_df_path, out_dataStripping_log_path, csv_name = "cI_stripped", data_type = "ICPMS")



"""
run_data_cleaning.py
----------------------------
Execution script to run the data cleaning functions on raw trace element Excel datasets (Timber, Soy, and Cocoa).

Input data:
1. Raw multi-element Excel datasets for Timber (XRF), Soy (ICPMS), and Cocoa (ICPMS).

Generates and saves:
1. Fully cleaned and formatted complete datasets (CSV).
2. Isolated metadata feature datasets (CSV).
3. Isolated numerical feature datasets (CSV) with standardized decimal formatting and missing value representations.
"""

### import the required functions
from data_preparation.functions.data_cleaning import excel_to_clean_csv


### Define directories
# Input
tX_excel_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\data\OG_data\CONFIDENTIAL_WorldForestID_CAT3_Timber_XRF.xlsx"
sI_excel_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\data\OG_data\CONFIDENTIAL_WorldForestID_CAT4_Soy_ICPMS.xlsx"
cI_excel_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\data\OG_data\CONFIDENTIAL_WorldForestID_CAT4_Cacao_ICPMS.xlsx"
# Output
out_dataCleaning_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\cleaned"

### Run data cleaning
print("RUNNING DATA CLEANING...")
excel_to_clean_csv(tX_excel_path, out_dataCleaning_path, csv_name = "tX_cleaned", data_type = "XRF")
excel_to_clean_csv(sI_excel_path, out_dataCleaning_path, csv_name = "sI_cleaned", data_type = "ICPMS")
excel_to_clean_csv(cI_excel_path, out_dataCleaning_path, csv_name = "cI_cleaned", data_type = "ICPMS")
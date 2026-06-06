"""
run_imputation.py
----------------------------
Execution script for running imputation processes (via R integration) and subsequently evaluating the structural impacts on trace element distributions using visualization tools.

Input data:
1. Filtered multi-element CSV datasets (specifically NA 0% filtered Timber, Soy, and Cocoa).
2. Cleaned original Timber dataset for baseline comparisons.
3. R-generated MICE-imputed multi-element CSV datasets.

Generates and saves:
1. Pre-imputation count tables mapping baseline NA, LOD, and zero value frequencies across all filtered datasets (CSV).
2. Post-imputation count tables verifying the successful resolution of LOD/missing values across imputed datasets (CSV).
3. Histogram plots overlaying pre- and post-imputation feature distributions on both linear and logarithmic scales (PNG), providing a visual evaluation of imputation efficacy and outlier mitigation.
"""

import pandas as pd
import os

### Load in needed functions
from data_preparation.functions.data_imputation import count_NA_LOD_zero
from data_preparation.functions.data_imputation import plot_dist_imputation

### Define directories
# Input
tX_filtered_0_Ba_Br_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\filtered\NA_0_pct\tX_filtered_NA_0_pct_Ba_Br.csv"
sI_filtered_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\filtered\NA_0_pct\sI_filtered_NA_0_pct.csv"
cI_filtered_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\filtered\NA_0_pct\cI_filtered_NA_0_pct.csv"
# Output
out_NaLodZero_filtered_log_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\logs\NA_LOD_zeros\filtered"
out_NaLodZero_imputed_miceLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\logs\NA_LOD_zeros\imputed\miceLOD"
out_impDistribution_miceLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\evaluations\imputation\miceLOD"
out_impDistribution_miceNA_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\evaluations\imputation\miceNA"


### Run NA, LOD and zero counting ###
print("RUNNING IMPUTATION COUNTING\n")
count_NA_LOD_zero(tX_filtered_0_Ba_Br_path, out_NaLodZero_filtered_log_path, csv_name = "tX_NaLodZero_filt", data_type = "XRF")
count_NA_LOD_zero(sI_filtered_path, out_NaLodZero_filtered_log_path, csv_name = "sI_NaLodZero_filt", data_type = "ICPMS")
count_NA_LOD_zero(cI_filtered_path, out_NaLodZero_filtered_log_path, csv_name = "cI_NaLodZero_filt", data_type = "ICPMS")
# RESULTS:
# Timber: 5 missing values for the Cd variable.
# Cocoa: Only LOD values. 0-40% range per variable.
# Soy: Only LOD values. 0-40% range per variable. 


### Run tobit Random Forest imputation for LOD values in cI and sI
# This is done in R, and saved in the imputed dataframes folder.





# ----------------
### MICE LOD
# ----------------
sI_imputed_miceLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\sI_imputed_mice.csv"
cI_imputed_miceLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\cI_imputed_mice.csv"

# Distribution plotting
plot_dist_imputation(sI_filtered_path, sI_imputed_miceLOD_path, out_impDistribution_miceLOD_path, "sI_ImpDist_miceLOD", "ICPMS", log_plot = False)
# plot_dist_imputation(cI_filtered_path, cI_imputed_miceLOD_path, out_impDistribution_miceLOD_path, "cI_ImpDist_miceLOD", "ICPMS", log_plot = False)
plot_dist_imputation(sI_filtered_path, sI_imputed_miceLOD_path, out_impDistribution_miceLOD_path, "sI_ImpDist_miceLOD_log", "ICPMS", log_plot = True)
# plot_dist_imputation(cI_filtered_path, cI_imputed_miceLOD_path, out_impDistribution_miceLOD_path, "cI_ImpDist_miceLOD_log", "ICPMS", log_plot = True)

# RESULTS:
# The distributions are more bell shaped and outliers have way less influence. The log-transformed data looks okay. 

# Checking NA and LOD amounts
count_NA_LOD_zero(sI_imputed_miceLOD_path, out_NaLodZero_imputed_miceLOD_path, csv_name = "sI_NaLodZero_miceLOD", data_type = "ICPMS")
count_NA_LOD_zero(cI_imputed_miceLOD_path, out_NaLodZero_imputed_miceLOD_path, csv_name = "cI_NaLodZero_miceLOD", data_type = "ICPMS")



# ---------------
### MICE NA
# ---------------

tX_imputed_miceNA_0_Ba_Br_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceNA\tX_imputed_NA_0_Ba_Br.csv"

tX_OG_cleaned_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\cleaned\tX_cleaned.csv"

plot_dist_imputation(tX_OG_cleaned_path, tX_imputed_miceNA_0_Ba_Br_path, out_impDistribution_miceNA_path, "tX_ImpDist_miceNA_0_Ba_Br_log", "XRF", log_plot = True)
plot_dist_imputation(tX_OG_cleaned_path, tX_imputed_miceNA_0_Ba_Br_path, out_impDistribution_miceNA_path, "tX_ImpDist_miceNA_0_Ba_Br_raw", "XRF", log_plot = False)

"""
run_summarize_PI_global.py
----------------------------
Exectution script to synthesize and aggregate feature importance rankings across the different models and interpretation 
methods (PI, CPI), creating a global summary over all metrics and models.

Input data:
1. Aggregated model-level summary logs (CSV) generated from preceding feature importance pipeline stages (summarize_PI_modelLevel).

Generates and saves:
1. Consolidated Consensus Datasets (CSV) mapping the median, minimum, and maximum ranks for each feature across all method-model combinations.
2. Grand Consensus Visualization Plots (PNG) including:
   - MedRed and AvgRed Spread Plots: Statistical summaries of rank variance.
   - Model Interval Cleveland Dot Plots (MAIN PLOT IN THESIS): Detailed interval views of rank shifts between interpretation methods, 
   annotated with model-specific shorthand labels (e.g., 'R', 'X', 'S') for clarity.
"""

import os

# Import our new function
from feature_importance.functions.summarize_PI_global import calculate_grand_consensus

# ---------------------------------------------------------
# Define Paths
# ---------------------------------------------------------
# Input is the output of the previous summarization script
INPUT_BASE_DIR = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\PI\PI_ModelLevel_summaries"

# Output goes exactly where you requested
OUTPUT_BASE_DIR = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\PI\PI_Global_summaries"

# ---------------------------------------------------------
# Define Scope
# ---------------------------------------------------------
METHODS_USED = [
    "permutation", 
    "conditional_permutation"
]

MODELS_USED = [
    "RandomForest", 
    "XGBoost", 
    "SVM"
]

if __name__ == "__main__":
    
    calculate_grand_consensus(
        input_base=INPUT_BASE_DIR,
        output_base=OUTPUT_BASE_DIR,
        methods=METHODS_USED,
        models=MODELS_USED
    )
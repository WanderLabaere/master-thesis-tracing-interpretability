"""
imputation_evaluation.py
----------------------------
Functions script to explore missing values, limit of detection (LOD) occurrences, 
and evaluate the impact of imputation strategies on element distributions.

Input data:
1. Filtered multi-element CSV datasets (ICPMS or XRF).
2. Imputed multi-element CSV datasets for comparative distribution plotting.

Generates and saves:
1. Comprehensive count tables detailing the frequency and percentage of NA, LOD, and zero values per individual sample (CSV).
2. Comprehensive count tables detailing the frequency and percentage of NA, LOD, and zero values per individual feature/variable (CSV).
3. Multi-panel histogram grid plots comparing original filtered distributions against post-imputation distributions, including mapped LOD threshold markers (PNG).
"""

import pandas as pd
import os

### EXPLORATION FOR IMPUTATION

def count_NA_LOD_zero(df_filt_path, out_csv_path, csv_name, data_type):
    """
    Creates csv file that count the NA, LOD and zero values in a dataframe.
    In: dataframe path (filtered)
    Out: CSV table with counts of NA, LOD and zero values per sample and per variable.
    """
    
    print(f"Counting NA, LOD, and zero values for {csv_name[:2]}:")
    
    # Load the filtered dataframe
    df = pd.read_csv(df_filt_path)
    
    # Determine numerical columns based on data type
    if data_type == "ICPMS":
        trace_cols = df.columns[16:]
    elif data_type == "XRF":
        trace_cols = df.columns[13:]
    else:
        raise ValueError("Data type not supported. Must be ICPMS or XRF.")
    
    # Initialize results lists
    per_sample_results = []
    per_variable_results = []
    
    
    ### COUNT PER SAMPLE ###
    for idx, row in df.iterrows():
        na_count = 0
        lod_count = 0
        zero_count = 0
        
        for col in trace_cols:
            value = row[col]
            
            # Check if NA
            if pd.isna(value):
                na_count += 1
            else:
                # Convert to string to check for LOD
                value_str = str(value)
                if value_str.startswith("<"):
                    lod_count += 1
                # Check for zero
                elif float(value) == 0:
                    zero_count += 1
        
        total_vars = len(trace_cols)
        per_sample_results.append({
            "Sample": row["WFID Identifier"] if "WFID Identifier" in df.columns else idx,
            "Country": row["Country"] if "Country" in df.columns else idx,
            "NA_count": na_count,
            "NA_percentage": (na_count / total_vars) * 100,
            "LOD_count": lod_count,
            "LOD_percentage": (lod_count / total_vars) * 100,
            "Zero_count": zero_count,
            "Zero_percentage": (zero_count / total_vars) * 100,
            "Total_variables": total_vars
        })
    
    
    ### COUNT PER VARIABLE ###
    for col in trace_cols:
        na_count = 0
        lod_count = 0
        zero_count = 0
        
        for value in df[col]:
            # Check if NA
            if pd.isna(value):
                na_count += 1
            else:
                # Convert to string to check for LOD
                value_str = str(value)
                if value_str.startswith("<"):
                    lod_count += 1
                # Check for zero
                elif float(value) == 0:
                    zero_count += 1
        
        total_samples = len(df)
        per_variable_results.append({
            "Variable": col,
            "NA_count": na_count,
            "NA_percentage": (na_count / total_samples) * 100,
            "LOD_count": lod_count,
            "LOD_percentage": (lod_count / total_samples) * 100,
            "Zero_count": zero_count,
            "Zero_percentage": (zero_count / total_samples) * 100,
            "Total_samples": total_samples
        })
    
    
    ### SAVE PER SAMPLE RESULTS ###
    per_sample_df = pd.DataFrame(per_sample_results)
    # Keep only rows with at least one non-zero count
    per_sample_df = per_sample_df[
        (per_sample_df["NA_count"] > 0) | 
        (per_sample_df["LOD_count"] > 0) | 
        (per_sample_df["Zero_count"] > 0)
    ]
    per_sample_file = os.path.join(out_csv_path, csv_name + "_per_sample_counts.csv")
    per_sample_df.to_csv(per_sample_file, index=False)
    print(f"Per-sample counts saved to {per_sample_file}")
    
    
    ### SAVE PER VARIABLE RESULTS ###
    per_variable_df = pd.DataFrame(per_variable_results)
    # Keep only rows with at least one non-zero count
    per_variable_df = per_variable_df[
        (per_variable_df["NA_count"] > 0) | 
        (per_variable_df["LOD_count"] > 0) | 
        (per_variable_df["Zero_count"] > 0)
    ]
    per_variable_file = os.path.join(out_csv_path, csv_name + "_per_variable_counts.csv")
    per_variable_df.to_csv(per_variable_file, index=False)
    print(f"Per-variable counts saved to {per_variable_file}\n")
    
    return per_sample_df, per_variable_df


# --------------------------------------------------------------------
### IMPUTATION FUNCTIONS
# --------------------------------------------------------------------





# --------------------------------------------------------------------
### EVALUATION OF IMPUTATION
# --------------------------------------------------------------------

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_dist_imputation(df_filtered_path, df_imputed_path, out_plot_path, plot_name, data_type, log_plot = False):
    """
    Plots the distribution of the filtered data vs the imputed data, to see how the imputation went. 
    In: paths of filtered and imputed datasets.
    Out: Plot of distributions.
    """

    # Read in the dataframes
    df_filt = pd.read_csv(df_filtered_path)
    df_imp  = pd.read_csv(df_imputed_path)

    # Select numerical columns based on datatype
    if data_type == "ICPMS":
        trace_cols_filt = df_filt.columns[16:]
        trace_cols_imp  = df_imp.columns[16:]
    elif data_type == "XRF":
        trace_cols_filt = df_filt.columns[13:]
        trace_cols_imp  = df_imp.columns[13:]
    else:
        raise ValueError("Data type not supported. Must be ICPMS or XRF.")

    # --- Find common columns (omits variables dropped during imputation) ---
    common_cols = [col for col in trace_cols_filt if col in trace_cols_imp]

    # --- For LOD columns in filtered data: extract threshold and replace "<x" with NA ---
    lod_thresholds = {}
    df_filt_clean  = df_filt.copy()

    for col in common_cols:
        col_vals = df_filt[col].astype(str)
        lod_mask = col_vals.str.startswith("<")
        if lod_mask.any():
            threshold = float(col_vals[lod_mask].iloc[0].replace("<", ""))
            lod_thresholds[col] = threshold
            df_filt_clean.loc[lod_mask, col] = np.nan

    # Convert trace cols to numeric after cleaning
    df_filt_clean[common_cols] = df_filt_clean[common_cols].apply(pd.to_numeric, errors="coerce")

    # --- Grid layout ---
    n_cols = 4
    n_vars = len(common_cols)
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = np.atleast_1d(axes).flatten() # Ensure it's iterable even if 1x1

    for i, col in enumerate(common_cols):
        ax = axes[i]

        # Get values and drop NAs
        filt_vals = df_filt_clean[col].dropna()
        imp_vals  = df_imp[col].dropna()

        # optional: log-transform before plotting
        if log_plot:
            filt_vals = np.log10(filt_vals).replace(-np.inf, np.nan)
            imp_vals  = np.log10(imp_vals).replace(-np.inf, np.nan)

        x_min = np.nanmin([filt_vals.min(), imp_vals.min()])
        x_max = np.nanmax([filt_vals.max(), imp_vals.max()])
        bins  = np.linspace(x_min, x_max, 200)

        ax.hist(filt_vals, bins=bins, color="red",  alpha=0.5, label="Filtered", density=False)
        ax.hist(imp_vals,  bins=bins, color="blue", alpha=0.5, label="Imputed",  density=False)

        if col in lod_thresholds:
            threshold = lod_thresholds[col]
            threshold_plot = np.log10(threshold) if log_plot else threshold
            ax.axvline(x=threshold_plot, color="black", linestyle="--", linewidth=1.2)
            ax.set_xlim(x_min, x_max)             # Full range


        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("Value", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=7)

    # --- Legend + hide unused subplots ---
    handles = [
        mpatches.Patch(color="red",  alpha=0.5, label="Filtered"),
        mpatches.Patch(color="blue", alpha=0.5, label="Imputed"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="LOD threshold")
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=10, frameon=True)

    for j in range(len(common_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Distribution: Filtered vs Imputed ({data_type})", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    # Save plot to output path
    os.makedirs(out_plot_path, exist_ok=True)
    full_path = os.path.join(out_plot_path, plot_name + ".png") # Added .png to be safe if not provided
    plt.savefig(full_path, dpi=150, bbox_inches="tight")
    plt.close()
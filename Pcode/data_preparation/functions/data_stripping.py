"""
data_stripping.py
----------------------------
Functions script to apply strict data stripping on filtered geochemical datasets by dropping all variables containing Limit of Detection (LOD) values and subsequently removing all samples containing missing (NA) values.
Main use was to compare this analysis with the imputed dataset analysis. There were no major differences.

Input data:
1. Filtered multi-element CSV datasets (ICPMS or XRF).

Generates and saves:
1. Stripped datasets (CSV) that are entirely free of LOD and missing values.
2. Removal logs (CSV) detailing exactly which variables and samples were dropped during the 
    stripping process and the reason for their exclusion.
"""

import pandas as pd
import os

def strip_filtered_df(input_df_path, out_df_stripped_path, out_strip_log_path, csv_name, data_type):
    """
    Applies very strict filtering on TE dataframe:
    - Remove all variables with LOD values
    - Then, remove all samples with NA values.
    
    In: Filtered dataframe (csv).
    Out: Stripped dataset with no LOD or NA values.
    """
    
    print(f"Stripping {csv_name[:2]}...")
    
    # Load in the dataframe
    df = pd.read_csv(input_df_path)
    
    # Identify trace columns (numerical columns)
    if data_type == "ICPMS":
        trace_cols = df.columns[16:]
    elif data_type == "XRF":
        trace_cols = df.columns[13:]
    else:
        raise ValueError("Data type not supported. Must be ICPMS or XRF.")
    
    # Initialize tracking lists for removals
    samples_removed_log = []
    variables_removed_log = []
    
    
    ### REMOVE ALL VARIABLES (COLUMNS) WITH ANY LOD VALUES ###
    lod_cols_to_drop = []
    for c_name in trace_cols:
        col = df[c_name].astype(str)  # create str for safety
        # Check if ANY values contain "<" (LOD indicator)
        has_lod = (col.str.startswith("<")).any()
        if has_lod:
            lod_cols_to_drop.append(c_name)
            # Add LOD variables to log
            variables_removed_log.append({
                "Type": "Variable",
                "Name": c_name,
                "Reason": "Contains LOD values"
            })
    
    # Drop the columns with LOD values
    df = df.drop(columns=lod_cols_to_drop)
    print(f"Removed {len(lod_cols_to_drop)} columns with LOD values:")
    for col in lod_cols_to_drop:
        print(f"  {col}")
    
    # Update trace_cols to only include remaining columns
    trace_cols = [col for col in trace_cols if col in df.columns]
    print()
    
    
    ### REMOVE ALL SAMPLES (ROWS) WITH ANY NA VALUES ###
    df_before_na = df.copy()
    df = df.dropna(subset=trace_cols, how='any')
    removed_na_samples = df_before_na.loc[~df_before_na.index.isin(df.index)]
    
    # Save NA samples to log
    for idx, row in removed_na_samples.iterrows():
        samples_removed_log.append({
            "Type": "Sample",
            "Name": row["WFID Identifier"] if "WFID Identifier" in row else "Unknown",
            "Reason": "Contains missing values in numerical columns"
        })
    
    print(f"Removed {len(removed_na_samples)} samples with NA values\n")
    
    
    ### SAVE STRIPPED DATAFRAME ###
    output_df_file = os.path.join(out_df_stripped_path, csv_name + ".csv")
    df.to_csv(output_df_file, index=False, na_rep="NA")
    print(f"Stripped data saved to {output_df_file}\n")
    
    ### SAVE REMOVAL LOG ###
    removals_log = samples_removed_log + variables_removed_log
    if removals_log:
        removals_df = pd.DataFrame(removals_log)
        removals_file = os.path.join(out_strip_log_path, csv_name + "_removals.csv")
        removals_df.to_csv(removals_file, index=False)
        print(f"Stripping removal log saved to {removals_file}\n")
    
    return df
"""
data_filtering.py
----------------------------
Functions script containing functions to filter and clean structured multi element datasets based on specific data quality thresholds and domain rules.

Input data:
1. Cleaned multi-element CSV datasets (ICPMS or XRF).

Generates and saves:
1. Filtered datasets (CSV) where samples and variables failing to meet criteria have been removed.
2. Removal logs (CSV) detailing exactly which samples or variables were dropped and the reason for their exclusion (e.g., duplicate WFID, high NA/LOD percentage, geographic bounds, or manual outliers).
"""

import pandas as pd
import os

def find_duplicates(df, data_type):
    """
    Identifies all WFID duplicates in a dataframe that should be removed.
    In: Dataframe of TE data.
    Out: dataframe with duplicate IDs.
    """

    # Select numerical columns
    if data_type == "ICPMS":
        df_md = df.iloc[:, 0:16]
        df_num = df.iloc[:,16:]
    elif data_type == "XRF":
        df_md = df.iloc[:, 0:13]
        df_num = df.iloc[:,13:]
    else:
        raise ValueError("Data type not supported. Must be ICPMS or XRF.")
        
    # Group by WFID ID and remove duplicates with identical numerical columns, if their WFID matches. 
    trace_columns = df_num.columns
    grouped = df.groupby("WFID Identifier")

    true_duplicates = []
    for wfid, group in grouped:
        if len(group) > 1:
            # Check if the values are identical
            same_ratios_mask = all(
                group[col].fillna(0).round(5).nunique() == 1
                for col in trace_columns
            )

            if same_ratios_mask:
                true_duplicates.append((wfid, len(group)))
    
    return true_duplicates

    

def filter_clean_df(input_cleanDf_path, out_df_path, out_log_path, csv_name, data_type, na_threshold):
    """
    Filters the cleaned up ICPMS dataframes, based on:
    - Duplicate WFID
    - NOT based on lab (Every sample has the same lab for ICPMS and XRF).
    - Custom constraints: Remove Timber from Spain/France, Fagus outlier.
    - Coordinates. Samples outside of europe are removed.
    - Numerical columns, remove:
        - Specific manual exclusions for XRF (Ti, V, Cr, etc.)
        - samples with all NA values 
        - variables with >40% LOD values
        - variables with > user-defined NA threshold
    - NOT on special cases based on the comments (this is the case for SIRA though!).
        

    In: cleaned up excel dataframe (csv).
    Out: Filtered dataset.
    """

    print(f"Filtering {csv_name[:2]}...")

    # Load in the dataframe
    df = pd.read_csv(input_cleanDf_path)
    
    # Ensure output directories exist
    os.makedirs(out_df_path, exist_ok=True)
    os.makedirs(out_log_path, exist_ok=True)

    if data_type == "ICPMS":
        trace_cols = df.columns[16:]
    elif data_type == "XRF":
        trace_cols = df.columns[13:]
    else:
        raise ValueError("Data type not supported. Must be ICPMS or XRF.")

    # Initialize tracking lists for removals
    samples_removed_log = []
    variables_removed_log = []


    ### CUSTOM FILTERING: TIMBER & FAGUS OUTLIER ###
    #  Filter Timber: remove Spain and France (Assuming "tX" designates timber)
    if "tX" in csv_name and "Country" in df.columns:
        spain_france_mask = df["Country"].isin(["Spain", "France"])
        removed_timber = df[spain_france_mask]
        
        for idx, row in removed_timber.iterrows():
            samples_removed_log.append({
                "Type": "Sample",
                "Name": row["WFID Identifier"] if "WFID Identifier" in row else "Unknown",
                "Reason": f"Timber sample manually excluded (Country: {row['Country']})"
            })
            
        df = df[~spain_france_mask]
        print(f"Removed {len(removed_timber)} timber samples from Spain and France.")

    #  Filter Fagus Outlier
    if "Genus" in df.columns:
        fagus_mask = df["Genus"] == "Fagus"
        fagus_df = df[fagus_mask]
        
        if not fagus_df.empty:
            cols_to_check = ["P", "S", "K", "Ba", "Ca"]
            avail_cols = [c for c in cols_to_check if c in df.columns]
            
            if avail_cols:
                # Find the maximum multi-dimensional extreme via standardized z-scores
                temp_num = fagus_df[avail_cols].apply(pd.to_numeric, errors='coerce')
                z_scores = (temp_num - temp_num.mean()) / temp_num.std()
                outlier_idx = z_scores.sum(axis=1).idxmax()
                
                if pd.notna(outlier_idx):
                    outlier_row = df.loc[outlier_idx]
                    samples_removed_log.append({
                        "Type": "Sample",
                        "Name": outlier_row["WFID Identifier"] if "WFID Identifier" in outlier_row else "Unknown",
                        "Reason": f"Fagus outlier with extreme values in {avail_cols}"
                    })
                    df = df.drop(index=outlier_idx)
                    print(f"Removed 1 Fagus outlier (Index: {outlier_idx}).")


    ### FILTERING BASED ON WFID DUPLICATES ###
    # Identify the duplicates
    duplicates = find_duplicates(df, data_type = data_type) # Ensure find_duplicates is defined/imported

    # remove duplicates
    df_filtered = df.copy()
    rows_removed = 0
    
    for wfid, count in duplicates:
        # Get all indices for this WFID
        wfid_indices = df_filtered[df_filtered["WFID Identifier"] == wfid].index.tolist()
        # Remove all but the first occurrence
        indices_to_remove = wfid_indices[1:]
        # save to log
        for idx in indices_to_remove:
            samples_removed_log.append({
                "Type": "Sample",
                "Name": df_filtered.loc[idx, "WFID Identifier"] if "WFID Identifier" in df_filtered.columns else "Unknown",
                "Reason": "Duplicate WFID with identical numerical values"
            })
        df_filtered = df_filtered.drop(indices_to_remove) # adapt dataframe
        rows_removed += len(indices_to_remove)

    # Print the amount of samples removed
    print(f"Removed {rows_removed} duplicate rows")
    print(f"Duplicate WFIDs found: {len(duplicates)}\n")

    # update df 
    df = df_filtered.copy()


    ### FILTERING BASED ON COORDINATES
    if data_type == "XRF":
        df_before_eu = df.copy()

        # Europe bounding box filter 
        # update the dataframe
        df = df[
            (df["Longitude"].between(-31, 90)) &
            (df["Latitude"].between(27, 72))
        ]

        # samples removed because outside europe
        removed_eu_samples = df_before_eu.loc[~df_before_eu.index.isin(df.index)]

        # add samples outside of europe to log
        for idx, row in removed_eu_samples.iterrows():
            samples_removed_log.append({
                "Type": "Sample",
                "Name": row["WFID Identifier"] if "WFID Identifier" in row else "Unknown",
                "Reason": f"Outside Europe (Country: {row['Country']})"
            })

        print(f"Removed {len(removed_eu_samples)} samples outside Europe:")
        for country in removed_eu_samples["Country"].unique():
            count = len(removed_eu_samples[removed_eu_samples["Country"] == country])
            print(f"  {country}: {count} samples\n")
     
    

    ### FILTERING BASED ON NUMERICAL COLUMNS ### 

    ### MANUAL EXCLUSIONS FOR XRF ###
    # if data_type == "XRF":
    #     xrf_exclude = ["Ti", "V", "Cr", "Ga", "Ge", "As", "Se", "Y", "Zr", "Nb", "Mo", "Cd", "I", "Cs", "Bi", "Th", "U"]
    #     xrf_cols_to_drop = [col for col in xrf_exclude if col in df.columns]
        
    #     if xrf_cols_to_drop:
    #         for col in xrf_cols_to_drop:
    #             variables_removed_log.append({
    #                 "Type": "Variable",
    #                 "Name": col,
    #                 "Reason": "Manually excluded XRF element"
    #             })
    #         df = df.drop(columns=xrf_cols_to_drop)
    #         print(f"Removed {len(xrf_cols_to_drop)} manually excluded XRF columns:")
    #         for col in xrf_cols_to_drop:
    #             print(f"  {col}")
                
    #         # Update trace_cols to only include remaining columns
    #         trace_cols = [col for col in trace_cols if col in df.columns]


    ### Remove samples that have missing values for all numerical columns ------------
    df_before_allna = df.copy()
    df = df.dropna(subset=trace_cols, how='all')
    removed_allna = df_before_allna.loc[~df_before_allna.index.isin(df.index)]
    # Save NA samples to log
    for idx, row in removed_allna.iterrows():
        samples_removed_log.append({
            "Type": "Sample",
            "Name": row["WFID Identifier"] if "WFID Identifier" in row else "Unknown",
            "Reason": "All numerical columns have missing values"
        })


    ### Remove columns where >40% of values are "<" values
    lod_cols_to_drop = []
    for c_name in trace_cols:
        col = df[c_name].astype(str) #create str for safety
        # percentage of values with "<"
        pct_lod = (col.str.startswith("<")).sum() / len(col)
        if pct_lod > 0.40:
            lod_cols_to_drop.append(c_name)
            # Add LOD variables to log
            variables_removed_log.append({
                "Type": "Variable",
                "Name": c_name,
                "Reason": f">40% LOD values ({pct_lod*100:.2f}%)"
            })
    # Drop the columns
    df = df.drop(columns=lod_cols_to_drop)
    print(f"Removed {len(lod_cols_to_drop)} columns with >40% '<' values:")
    for col in lod_cols_to_drop:
        print(f"  {col}")

    # Update trace_cols to only include remaining columns
    trace_cols = [col for col in trace_cols if col in df.columns]

    ### Filter columns based on inputted threshold missing values
    missing_pct = df[trace_cols].isna().sum() / len(df)
    missing_to_drop = missing_pct[missing_pct > na_threshold].index.tolist()
    
    # EXCEPTION: Keep "Ba" and "Br" for the specific tX NA 0 pct run
    if csv_name == "tX_filtered_NA_0_pct_Ba_Br":
        missing_to_drop = [col for col in missing_to_drop if col not in ["Ba", "Br"]]

    # Drop the columns 
    for col in missing_to_drop:
        variables_removed_log.append({
            "Type": "Variable",
            "Name": col,
            "Reason": f">{na_threshold*100}% missing values ({missing_pct[col]*100:.2f}%)"
        })
    df = df.drop(columns=missing_to_drop)
    print(f"Removed {len(missing_to_drop)} columns with >{na_threshold*100}% missing values:")
    for col in missing_to_drop:
        print(f"  {col}")


    ### SAVE FILTERED DATAFRAME ###
    output_df_file = os.path.join(out_df_path, csv_name + ".csv")
    df.to_csv(output_df_file, index=False, na_rep="NA")
    print(f"Filtered data saved to {output_df_file}\n")

    ### SAVE REMOVAL LOG ###
    removals_log = samples_removed_log + variables_removed_log
    if removals_log:
        removals_df = pd.DataFrame(removals_log)
        removals_file = os.path.join(out_log_path, csv_name + "_removals.csv")
        removals_df.to_csv(removals_file, index=False)
        print(f"Removal log saved to {removals_file}\n")

    return df
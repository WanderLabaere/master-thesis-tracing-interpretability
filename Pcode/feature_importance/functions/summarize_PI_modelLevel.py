"""
summarize_PI_modelLevel.py
----------------------------
Functions script to aggregate feature importance ranks from multiple machine 
learning models and interpretability methods, determining a robust "consensus" ranking.

Input data:
1. Individual feature importance logs (CSV) for each model-method pair, containing ranked trace element importance metrics.

Generates and saves:
1. Merged consensus datasets (CSV) aggregating ranks across models and 
    providing median consensus ranks with min/max intervals.
2. Stability-focused error bar plots (PNG) displaying the median consensus rank for top-30 features, 
    supplemented with asymmetric error bars (min/max range) and model-specific shorthand annotations ('R', 'X', 'S').
"""


import os
import glob
import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.use('Agg') # Forces background rendering to prevent Tkinter threading crashes
import matplotlib.pyplot as plt

def calculate_rank_consensus(input_base: str, output_base: str, methods: list, models: list):
    """
    Scans the input directory for feature importance CSVs, groups them by 
    dataset and method, calculates a median consensus rank, and generates summary plots 
    with min/max ranges and specific model labels (R, X, S).
    """
    print(f"\n{'='*70}")
    print("STARTING CROSS-MODEL RANK AGGREGATION & PLOTTING")
    print(f"{'='*70}\n")
    
    # Find all importance CSVs
    search_pattern = os.path.join(input_base, "**", "*_importance.csv")
    all_csvs = glob.glob(search_pattern, recursive=True)
    
    # Dictionary to hold groupings: Key -> (Output_Dir, Dataset_Name, Method)
    groupings = {}
    
    for file_path in all_csvs:
        if "permutation_summarized" in file_path or "cumulative" in file_path:
            continue
            
        rel_path = os.path.relpath(file_path, input_base)
        parts = rel_path.split(os.sep)
        method = parts[0]
        
        if method not in methods:
            continue
            
        sub_dirs = parts[1:-1]
        filename = parts[-1]
        
        model_in_file = next((m for m in models if m in filename), None)
        if not model_in_file:
            continue
            
        dataset_name = filename.split(f"_{model_in_file}")[0]
        target_out_dir = os.path.join(output_base, *sub_dirs)
        
        group_key = (target_out_dir, dataset_name, method)
        if group_key not in groupings:
            groupings[group_key] = []
            
        groupings[group_key].append((model_in_file, file_path))

    if not groupings:
        print("[!] No files found to summarize. Please check your input_base path.")
        return

    # Process each grouping, calculate consensus, and plot
    for (target_out_dir, dataset_name, method), files in groupings.items():
        os.makedirs(target_out_dir, exist_ok=True)
        print(f" -> Processing: {dataset_name} ({method}) | Models found: {len(files)}")
        
        merged_df = None
        
        # Track which models actually made it into this specific plot
        found_models = []
        
        for model_name, file_path in files:
            df = pd.read_csv(file_path)
            
            if "Feature" not in df.columns or "Importance_Mean" not in df.columns:
                print(f"    [!] Skipping {file_path} due to missing columns.")
                continue
                
            found_models.append(model_name)
            df[f"Rank_{model_name}"] = df["Importance_Mean"].rank(ascending=False, method="min")
            df_subset = df[["Feature", f"Rank_{model_name}", "Importance_Mean"]].rename(
                columns={"Importance_Mean": f"Score_{model_name}"}
            )
            
            if merged_df is None:
                merged_df = df_subset
            else:
                merged_df = pd.merge(merged_df, df_subset, on="Feature", how="outer")
                
        if merged_df is not None:
            rank_cols = [c for c in merged_df.columns if c.startswith("Rank_")]
            
            # Calculate Median, Min, and Max
            merged_df["Consensus_Rank"] = merged_df[rank_cols].median(axis=1)
            merged_df["Rank_Min"] = merged_df[rank_cols].min(axis=1)
            merged_df["Rank_Max"] = merged_df[rank_cols].max(axis=1)
            
            # Sort so rank #1 is at the top of the dataframe
            merged_df = merged_df.sort_values(by="Consensus_Rank", ascending=True).reset_index(drop=True)
            
            # --- SAVE CSV ---
            out_filename = f"{dataset_name}_{method}_model_consensus.csv"
            out_path = os.path.join(target_out_dir, out_filename)
            merged_df.to_csv(out_path, index=False)
            
            # --- GENERATE PLOT ---
            plot_df = merged_df.copy()
            
            # Cap at top 30 features to prevent unreadable, overly tall plots
            if len(plot_df) > 30:
                plot_df = plot_df.head(30)
                title_suffix = " (Top 30)"
            else:
                title_suffix = ""
                
            # Matplotlib plots from bottom to top, so reverse dataframe
            plot_df = plot_df.iloc[::-1]

            fig_height = max(5, min(12, len(plot_df) * 0.4))
            fig, ax = plt.subplots(figsize=(8, fig_height))
            
            # Calculate asymmetric errors for matplotlib
            lower_error = plot_df["Consensus_Rank"] - plot_df["Rank_Min"]
            upper_error = plot_df["Rank_Max"] - plot_df["Consensus_Rank"]
            asymmetric_error = [lower_error, upper_error]
            
            # Plot the main error bars and median dot
            ax.errorbar(
                x=plot_df["Consensus_Rank"],
                y=plot_df["Feature"],
                xerr=asymmetric_error,
                fmt='o',                  
                color='black',            
                ecolor='royalblue',       
                elinewidth=3,             
                capsize=4,                
                markersize=6,
                zorder=2
            )
            
            # ==========================================================
            # NEW: PLOT MODEL LETTER LABELS (R, X, S) OVER POINTS
            # ==========================================================
            for _, row in plot_df.iterrows():
                y_pos = row["Feature"]
                
                # Group models by their exact rank to prevent overlapping text
                rank_dict = defaultdict(list)
                for m_name in found_models:
                    col_name = f"Rank_{m_name}"
                    if pd.notna(row[col_name]):
                        rank_val = row[col_name]
                        # Take the first letter of the model name 
                        initial = m_name[0].upper() 
                        rank_dict[rank_val].append(initial)
                
                # Plot the grouped text labels
                for rank_val, initials in rank_dict.items():
                    label_text = ",".join(initials) # Creates "R,X" if they share the same rank
                    ax.text(
                        x=rank_val, 
                        y=y_pos, 
                        s=label_text, 
                        color='darkred', 
                        fontsize=8, 
                        fontweight='bold',
                        ha='center',   # Center horizontally on  point
                        va='bottom',    # Place bottom of text just above the line
                        zorder=3
                    )
            # ==========================================================
            
            ax.set_xlabel("Consensus Rank (Median) with [Min, Max] Range", fontsize=10)
            ax.set_ylabel("Feature", fontsize=10)
            ax.set_title(
                f"Feature Ranking Consensus Across Models\n"
                f"{dataset_name} - {method.replace('_', ' ').title()}{title_suffix}", 
                fontsize=12, fontweight='bold'
            )
            
            ax.grid(axis='x', linestyle='--', alpha=0.7, zorder=1)
            ax.set_xlim(left=0.5)
            
            plt.tight_layout()
            png_filename = f"{dataset_name}_{method}_model_consensus.png"
            plt.savefig(os.path.join(target_out_dir, png_filename), dpi=150)
            plt.close()
            
    print(f"\nSuccess! All summaries and plots written to:\n    {output_base}")
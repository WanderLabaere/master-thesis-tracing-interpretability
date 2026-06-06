"""
summarize_PI_global.py
----------------------------
Functions script to aggregate feature importance rankings across multiple 
spatial regression models and interpretability methods, calculating a "grand consensus" to evaluate feature stability.

Input data:
1. Precomputed individual model consensus logs (CSV) (from summarize_PI_modelLevel.py) detailing feature ranks across different ML models 
    (RF, XGB, SVM) and permutation methods (Permutation Importance, Conditional Permutation Importance).

Generates and saves:
1. Consensus Spread Plots (PNG) illustrating the median (MedRed) and average (AvgRed) rank variance 
    for top features across both models and permutation methods.
2. Cleveland Dot Plots / Model Intervals (PNG) visually tracking the specific rank shifts 
    (e.g., the range from PI to CPI) for individual features within each specific machine learning algorithm.
    THIS IS THE MAIN PLOT USED IN THE THESIS.
"""


import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from functools import reduce
from matplotlib.lines import Line2D

def calculate_grand_consensus(input_base: str, output_base: str, methods: list, models: list):
    """
    Aggregates model consensus CSVs and generates three distinct visualizations:
    1. MedRed (Median Method Spread)
    2. AvgRed (Average Method Spread)
    3. Model Intervals (Cleveland Dot Plot with Method tips P/C) -> USED IN THESIS
    """
    print(f"\n{'='*70}")
    print("STARTING GRAND CONSENSUS (ALL VISUALIZATIONS)")
    print(f"{'='*70}\n")
    
    search_pattern = os.path.join(input_base, "**", "*_model_consensus.csv")
    all_csvs = glob.glob(search_pattern, recursive=True)
    
    groupings = {}
    sorted_methods = sorted(methods, key=len, reverse=True)
    
    for file_path in all_csvs:
        if "general_summ_model_method" in file_path:
            continue
            
        rel_path = os.path.relpath(file_path, input_base)
        parts = rel_path.split(os.sep)
        sub_dirs = parts[:-1]
        filename = parts[-1]
        
        method_in_file = next((m for m in sorted_methods if f"_{m}_model_consensus" in filename), None)
        if not method_in_file:
            continue
            
        dataset_name = filename.split(f"_{method_in_file}")[0]
        target_out_dir = os.path.join(output_base, *sub_dirs)
        
        group_key = (target_out_dir, dataset_name)
        if group_key not in groupings:
            groupings[group_key] = []
        groupings[group_key].append((method_in_file, file_path))

    if not groupings:
        print("[!] No files found. Please ensure previous summaries exist.")
        return

    method_initials = {"permutation": "P", "conditional_permutation": "C"}
    model_colors = {"RF": "forestgreen", "XGB": "darkorange", "SVM": "purple"}
    default_colors = ["forestgreen", "darkorange", "purple", "brown", "teal"]

    for (target_out_dir, dataset_name), files in groupings.items():
        os.makedirs(target_out_dir, exist_ok=True)
        print(f" -> Processing Dataset: {dataset_name} | Methods: {len(files)}")
        
        dataframes = []
        found_methods = []
        
        for method_name, file_path in files:
            df = pd.read_csv(file_path)
            rank_cols = [c for c in df.columns if c.startswith("Rank_") and "Std_Dev" not in c and "Min" not in c and "Max" not in c]
            keep_cols = ["Feature"] + rank_cols
            if not all(col in df.columns for col in keep_cols): continue
                
            found_methods.append(method_name)
            df_subset = df[keep_cols].copy()
            rename_map = {col: f"{col}_{method_name}" for col in rank_cols}
            df_subset.rename(columns=rename_map, inplace=True)
            dataframes.append(df_subset)
            
        if not dataframes: continue
            
        merged_df = reduce(lambda left, right: pd.merge(left, right, on="Feature", how="outer"), dataframes)
        all_rank_cols = [c for c in merged_df.columns if c.startswith("Rank_")]
        
        merged_df["Grand_Consensus_Rank"] = merged_df[all_rank_cols].median(axis=1)
        merged_df["Overall_Min"] = merged_df[all_rank_cols].min(axis=1)
        merged_df["Overall_Max"] = merged_df[all_rank_cols].max(axis=1)
        
        #  Method Spread (Median)
        method_cols = []
        for method in methods:
            m_cols = [f"Rank_{model}_{method}" for model in models if f"Rank_{model}_{method}" in all_rank_cols]
            if m_cols:
                merged_df[f"Median_{method}"] = merged_df[m_cols].median(axis=1)
                method_cols.append(f"Median_{method}")
        merged_df["Method_Median"] = merged_df[method_cols].median(axis=1)
        merged_df["Method_Min"] = merged_df[method_cols].min(axis=1)
        merged_df["Method_Max"] = merged_df[method_cols].max(axis=1)
        
        #  Method Spread (Average)
        method_mean_cols = []
        for method in methods:
            m_cols = [f"Rank_{model}_{method}" for model in models if f"Rank_{model}_{method}" in all_rank_cols]
            if m_cols:
                merged_df[f"Mean_{method}"] = merged_df[m_cols].mean(axis=1)
                method_mean_cols.append(f"Mean_{method}")
        merged_df["Method_Mean_Min"] = merged_df[method_mean_cols].min(axis=1)
        merged_df["Method_Mean_Max"] = merged_df[method_mean_cols].max(axis=1)

        #  Model Spread (Medians) & Model Intervals
        model_cols = []
        for model in models:
            m_cols = [f"Rank_{model}_{method}" for method in methods if f"Rank_{model}_{method}" in all_rank_cols]
            if m_cols:
                merged_df[f"Median_{model}"] = merged_df[m_cols].median(axis=1)
                model_cols.append(f"Median_{model}")
                merged_df[f"{model}_Min"] = merged_df[m_cols].min(axis=1)
                merged_df[f"{model}_Max"] = merged_df[m_cols].max(axis=1)
                merged_df[f"{model}_Mid"] = merged_df[m_cols].mean(axis=1)
                
        merged_df["Model_Median"] = merged_df[model_cols].median(axis=1)
        merged_df["Model_Min"] = merged_df[model_cols].min(axis=1)
        merged_df["Model_Max"] = merged_df[model_cols].max(axis=1)
        
        merged_df = merged_df.sort_values(by="Model_Median", ascending=True).reset_index(drop=True)
        
        # --- GENERATE PLOTS (Top 20 Compact, Top 10 Expanded) ---
        for view_limit in [20, 10]:
            plot_df = merged_df.copy()
            if len(plot_df) > view_limit: 
                plot_df = plot_df.head(view_limit)
                suffix = f"_top{view_limit}" if view_limit == 10 else ""
            else:
                if view_limit == 10 and len(merged_df) <= 10: continue 
                suffix = ""
                
            plot_df = plot_df.iloc[::-1].reset_index(drop=True) 
            
            # --- DYNAMIC SPACING LOGIC ---
            if view_limit == 10:
                # Expanded view for Top 10
                fig_height = max(8, min(24, len(plot_df) * 0.90)) 
                y_base = np.arange(len(plot_df), dtype=float) * 1.5 
                
                spread_offset = 0.25
                text_offset = 0.40
                interval_jitter = 0.35
                interval_text_offset = 0.12
            else:
                # Compact view for Top 20
                fig_height = max(6, min(16, len(plot_df) * 0.70)) 
                y_base = np.arange(len(plot_df), dtype=float)
                
                spread_offset = 0.15
                text_offset = 0.27
                interval_jitter = 0.25
                interval_text_offset = 0.08
            
            # ------------------------------------------------------
            # PLOTS 1 & 2: MedRed and AvgRed Spread Plots
            # ------------------------------------------------------
            for red_mode in ["MedRed", "AvgRed"]:
                col_min = "Method_Min" if red_mode == "MedRed" else "Method_Mean_Min"
                col_max = "Method_Max" if red_mode == "MedRed" else "Method_Mean_Max"
                col_prefix = "Median_" if red_mode == "MedRed" else "Mean_"

                fig, ax = plt.subplots(figsize=(10, fig_height))
                
                model_err_lower = plot_df["Model_Median"] - plot_df["Model_Min"]
                model_err_upper = plot_df["Model_Max"] - plot_df["Model_Median"]
                
                ax.errorbar(
                    x=plot_df["Model_Median"], y=y_base + spread_offset, 
                    xerr=[model_err_lower, model_err_upper], 
                    fmt='o', color='royalblue', markersize=5, ecolor='royalblue', elinewidth=3, capsize=4, alpha=0.8, label="Spread across Models"
                )
                
                method_center = (plot_df[col_min] + plot_df[col_max]) / 2
                method_spread = (plot_df[col_max] - plot_df[col_min]) / 2
                ax.errorbar(
                    x=method_center, y=y_base - spread_offset, 
                    xerr=method_spread, 
                    fmt='none', ecolor='crimson', elinewidth=3, capsize=4, alpha=0.8, label="Spread across Methods"
                )

                for i, row in plot_df.iterrows():
                    rank_dict = defaultdict(list)
                    for m_name in models:
                        col_name = f"Median_{m_name}"
                        if col_name in row and pd.notna(row[col_name]):
                            rank_dict[row[col_name]].append(m_name[0].upper())
                    for rank_val, initials in rank_dict.items():
                        ax.text(x=rank_val, y=y_base[i] + text_offset, s=",".join(initials), color='darkblue', fontsize=9, fontweight='bold', ha='center', va='bottom', zorder=3)

                for i, row in plot_df.iterrows():
                    rank_dict = defaultdict(list)
                    for m_name in found_methods:
                        col_name = f"{col_prefix}{m_name}"
                        if col_name in row and pd.notna(row[col_name]):
                            rank_dict[row[col_name]].append(method_initials.get(m_name, m_name[0].upper()))
                    for rank_val, initials in rank_dict.items():
                        ax.text(x=rank_val, y=y_base[i] - text_offset, s=",".join(initials), color='darkred', fontsize=9, fontweight='bold', ha='center', va='top', zorder=3)

                ax.plot(plot_df["Model_Median"], y_base, 'ko', markersize=7, label="Model Consensus Rank (Median)", zorder=4)
                
                ax.set_yticks(y_base)
                ax.set_yticklabels(plot_df["Feature"], fontsize=10)
                ax.set_xlabel(f"Consensus Rank with Spreads ({red_mode})", fontsize=11)
                # ax.set_title(f"Feature Importance Stability Analysis\n{dataset_name}", fontsize=14, fontweight='bold', pad=20)
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                ax.set_xlim(left=0.5)
                ax.legend(loc='upper right')
                
                plt.tight_layout()
                plt.savefig(os.path.join(target_out_dir, f"{dataset_name}_Grand_Consensus_{red_mode}{suffix}.png"), dpi=150)
                plt.close()

            # ------------------------------------------------------
            # PLOT 3: Interval Plot (Cleveland Dot Plot with Methods) -- MAIN PLOT
            # ------------------------------------------------------
            fig, ax = plt.subplots(figsize=(10, fig_height))
            
            ax.hlines(y=y_base, xmin=plot_df["Overall_Min"], xmax=plot_df["Overall_Max"], color='lightgrey', linewidth=4, alpha=0.5, zorder=1)

            for idx, model in enumerate(models):
                if f"{model}_Min" not in plot_df.columns: continue
                m_color = model_colors.get(model, default_colors[idx % len(default_colors)])
                
                offset_multiplier = (idx - (len(models)-1)/2)
                y_offset = y_base + (offset_multiplier * interval_jitter)
                
                ax.hlines(y=y_offset, xmin=plot_df[f"{model}_Min"], xmax=plot_df[f"{model}_Max"], color=m_color, linewidth=3, zorder=3, alpha=0.8)
                ax.plot(plot_df[f"{model}_Min"], y_offset, marker='o', color=m_color, markersize=5, linestyle='None', zorder=4)
                ax.plot(plot_df[f"{model}_Max"], y_offset, marker='o', color=m_color, markersize=5, linestyle='None', zorder=4)
                
                for i, row in plot_df.iterrows():
                    mid_val = row[f"{model}_Mid"]
                    if pd.notna(mid_val):
                        # Text above
                        ax.text(x=mid_val, y=y_offset[i] + interval_text_offset, s=model[0].upper(), color=m_color, fontsize=9, fontweight='bold', ha='center', va='bottom', zorder=5)

                    if len(found_methods) >= 2:
                        m1, m2 = found_methods[0], found_methods[1]
                        c1, c2 = f"Rank_{model}_{m1}", f"Rank_{model}_{m2}"
                        char1, char2 = method_initials.get(m1, m1[0].upper()), method_initials.get(m2, m2[0].upper())

                        if c1 in plot_df.columns and c2 in plot_df.columns:
                            val1, val2 = row[c1], row[c2]
                            if pd.notna(val1) and pd.notna(val2):
                                # Text below
                                if val1 < val2:
                                    ax.text(x=val1, y=y_offset[i] - interval_text_offset, s=char1, color=m_color, fontsize=8, fontweight='bold', ha='center', va='top', zorder=5)
                                    ax.text(x=val2, y=y_offset[i] - interval_text_offset, s=char2, color=m_color, fontsize=8, fontweight='bold', ha='center', va='top', zorder=5)
                                elif val1 > val2:
                                    ax.text(x=val2, y=y_offset[i] - interval_text_offset, s=char2, color=m_color, fontsize=8, fontweight='bold', ha='center', va='top', zorder=5)
                                    ax.text(x=val1, y=y_offset[i] - interval_text_offset, s=char1, color=m_color, fontsize=8, fontweight='bold', ha='center', va='top', zorder=5)
                                else:
                                    ax.text(x=val1, y=y_offset[i] - interval_text_offset, s=f"{char1},{char2}", color=m_color, fontsize=8, fontweight='bold', ha='center', va='top', zorder=5)

            ax.set_yticks(y_base)
            ax.set_yticklabels(plot_df["Feature"], fontsize=17)
            ax.set_xlabel("Feature Rank Range (PI to CPI)", fontsize=17)
            # ax.set_title(f"Feature Importance Method Sensitivity per Model\n{dataset_name}", fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.set_xlim(left=0.5)
            
            handles, labels = ax.get_legend_handles_labels()
            for idx, model in enumerate(models):
                m_color = model_colors.get(model, default_colors[idx % len(default_colors)])
                handles.append(Line2D([0], [0], color=m_color, lw=3, marker='o'))
                labels.append(f"{model} Interval (PI ↔ CPI)")
            ax.legend(handles, labels, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(target_out_dir, f"{dataset_name}_Model_Intervals{suffix}.png"), dpi=150)
            plt.close()
            
    print(f"\nSuccess! All visualizations written to:\n    {output_base}")
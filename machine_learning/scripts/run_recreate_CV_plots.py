"""
reconstruct_fold_plots.py
----------------------------
Utility script to batch reconstruct geographic fold visualization plots using previously saved CSV data.

Input data:
1. Previously generated raw data logs (CSV) capturing the fold visualization point coordinates and error metrics.
2. Local world map shapefiles for geogarphic visualization.

Generates and saves:
1. Reconstructed multi-panel geographic fold plots (PNG), allowing global visual styling adjustments 
(e.g., jitter alpha, marker sizes) without requiring complete model retraining.
"""

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==========================================
#  Configuration 
# ==========================================
# base directory
base_search_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\machine_learning\output\CV\fold_images"

# output
output_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\machine_learning\output\CV\fold_images\reconstructed_plots"

# geo data
map_data_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_input\world_map_data\World_countries"

# Visual Settings (for all plots!)

# JITTER HIDDEN

# ==========================================
#  Setup & Map Loading
# ==========================================
os.makedirs(output_dir, exist_ok=True)

try:
    print("Loading world map...")
    world = gpd.read_file(map_data_dir)
except Exception as e:
    raise FileNotFoundError(f"Could not load world map: {e}")

# Find all CSV files recursively inside the base directory
search_pattern = os.path.join(base_search_dir, "**", "*_kfold_cv_folds_data.csv")
csv_files = glob.glob(search_pattern, recursive=True)

if not csv_files:
    print(f"No CSV files found in {base_search_dir}")
    exit()

print(f"Found {len(csv_files)} datasets to process. Starting batch reconstruction...\n")

# ==========================================
#  Batch Processing Loop
# ==========================================
for csv_path in csv_files:
    # Extract the base name (e.g., 'sI_RandomForest') from the file
    base_name = os.path.basename(csv_path).replace("_kfold_cv_folds_data.csv", "")
    output_filename = f"Reconstructed_{base_name}.png"
    save_target = os.path.join(output_dir, output_filename)
    
    print(f"  -> Reconstructing: {base_name}...")
    
    df = pd.read_csv(csv_path)

    # Setup Plot Grid
    n_folds = df["Fold"].nunique()
    grid_rows = int(np.sqrt(n_folds))
    grid_cols = int(np.ceil(n_folds / grid_rows))
    if grid_rows * grid_cols < n_folds:
        grid_rows += 1

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(5 * grid_cols, 4.5 * grid_rows))
    if grid_rows == 1 and grid_cols == 1: axes = np.array([[axes]])
    elif grid_rows == 1 or grid_cols == 1: axes = axes.reshape(grid_rows, grid_cols)

    # Calculate global map bounds with padding
    lon_min, lon_max = df["Actual_Lon"].min(), df["Actual_Lon"].max()
    lat_min, lat_max = df["Actual_Lat"].min(), df["Actual_Lat"].max()
    lon_pad, lat_pad = (lon_max - lon_min) * 0.05, (lat_max - lat_min) * 0.05

    all_errors = []
    all_skills = []

    # Draw the Plots for this CSV
    fold_num = 1
    for row in range(grid_rows):
        for col in range(grid_cols):
            ax = axes[row, col]
            
            if fold_num > n_folds:
                ax.set_visible(False)
                continue
                
            # Isolate data for this specific fold
            fold_data = df[df["Fold"] == fold_num]
            train_data = fold_data[fold_data["Point_Type"] == "Train"]
            test_data = fold_data[fold_data["Point_Type"] == "Test"].copy()
            
            # Get metrics
            fold_error = fold_data["Fold_Error_km"].iloc[0]
            fold_skill = fold_data["Fold_Skill_Score"].iloc[0]
            all_errors.append(fold_error)
            all_skills.append(fold_skill)

            # Draw Map
            world.plot(ax=ax, alpha=0.2, edgecolor='k', linewidth=0.2, color='lightgray')
            ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
            ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)

            # Draw Train Points 
            ax.scatter(train_data["Jittered_Lon"], train_data["Jittered_Lat"], 
                       c='dodgerblue', alpha=JITTER_TRAIN_ALPHA, s=JITTER_TRAIN_SIZE, 
                       edgecolors='navy', linewidth=0.4, zorder=5)

            # Draw Error Lines
            test_clean = test_data.dropna(subset=['Jittered_Lon', 'Jittered_Predicted_Lon'])
            for _, point in test_clean.iterrows():
                ax.plot([point["Jittered_Lon"], point["Jittered_Predicted_Lon"]], 
                        [point["Jittered_Lat"], point["Jittered_Predicted_Lat"]],
                        color='dimgray', linestyle='-', linewidth=1.0, alpha=LINE_ALPHA, zorder=4)

            # Draw Test Actual
            ax.scatter(test_data["Jittered_Lon"], test_data["Jittered_Lat"], 
                       c='red', alpha=JITTER_TEST_ALPHA, s=JITTER_TEST_SIZE, 
                       edgecolors='darkred', linewidth=0.6, marker='^', zorder=6)
            
            # Draw Test Predicted
            ax.scatter(test_data["Jittered_Predicted_Lon"], test_data["Jittered_Predicted_Lat"], 
                       c='orange', alpha=JITTER_PRED_ALPHA, s=JITTER_PRED_SIZE, 
                       edgecolors='black', linewidth=0.6, marker='X', zorder=6)

            # Styling
            ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.3, color='gray')
            ax.set_axisbelow(True)
            ax.set_title(f"Fold {fold_num} | Error: {fold_error:.0f} km (Skill: {fold_skill:.2f})\n"
                         f"Train: {len(train_data)} | Test: {len(test_data)}", 
                         fontsize=10, fontweight='bold')
            
            if col == 0: ax.set_ylabel("Latitude (°)", fontsize=8)
            if row == grid_rows - 1: ax.set_xlabel("Longitude (°)", fontsize=8)
            ax.tick_params(labelsize=7)
            
            fold_num += 1

    # Final Polish & Save for this specific CSV
    mean_err = np.mean(all_errors)
    mean_skill = np.mean(all_skills)

    fig.suptitle(f"{base_name.upper()} — K-Fold CV\nOverall Mean Error: {mean_err:.0f} km (Skill: {mean_skill:.2f})", 
                 fontsize=16, fontweight='bold', y=0.995)

    legend_elements = [
        Patch(facecolor='dodgerblue', edgecolor='navy', label='Train'),
        Patch(facecolor='red', edgecolor='darkred', label='Test Actual'),
        Patch(facecolor='orange', edgecolor='black', label='Test Predicted')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.02), framealpha=0.95)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(save_target, dpi=300, bbox_inches='tight')
    plt.close()

print(f"\n[========== ALL {len(csv_files)} PLOTS SUCCESSFULLY RECONSTRUCTED ==========]")
print(f"Check the output folder: {output_dir}")
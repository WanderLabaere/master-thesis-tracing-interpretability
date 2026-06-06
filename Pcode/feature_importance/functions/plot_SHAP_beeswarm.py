
"""
plot_SHAP_beeswarm.py
----------------------------
Functions script to reconstruct global SHAP beeswarm summary plots directly from previously saved 
per-sample SHAP data logs, so no rerunning of the SHAP extraction is needed.

Input data:
1. Generated wide SHAP data logs (CSV) (from extract_SHAP) containing the original trace feature values alongside 
    their corresponding per-sample SHAP "push" values for both Longitude and Latitude.

Generates and saves:
1. Reconstructed global SHAP beeswarm plots (PNG) for both Longitude and Latitude.
"""

import os
import pandas as pd
import shap
import matplotlib.pyplot as plt

def reconstruct_shap_plots(csv_path: str, data_name: str, model_name: str, output_dir: str):
    """
    Reads the saved wide SHAP CSV and reconstructs the Longitude and Latitude 
    summary (beeswarm) plots, saving them to the specified output directory.
    """
    if not os.path.exists(csv_path):
        print(f"SHAP data not found at: {csv_path}")
        return

    print(f"  -> Reconstructing {model_name} plots for {data_name}...")
    
    # Load the saved per-sample data
    df = pd.read_csv(csv_path)
    
    # Identify the columns dynamically
    shap_lon_cols = [c for c in df.columns if c.startswith("shap_lon_")]
    shap_lat_cols = [c for c in df.columns if c.startswith("shap_lat_")]
    
    # Extract the pure feature names
    feature_names = [c.replace("shap_lon_", "") for c in shap_lon_cols]
    
    # Isolate the matrices required for plotting
    X_features = df[feature_names]
    shap_lon_matrix = df[shap_lon_cols].values
    shap_lat_matrix = df[shap_lat_cols].values

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Reconstruct Longitude Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_lon_matrix, X_features, show=False)

    # Increase axis ticks and label fonts
    ax_lon = plt.gca()
    ax_lon.tick_params(axis='both', which='major', labelsize=15) 
    ax_lon.set_xlabel("SHAP Value (Degrees Longitude)", fontsize=14)    

    plt.title(
        f"Longitude SHAP\n(Positive = Pushes East  |  Negative = Pushes West)",
        y=1.05,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{data_name}_{model_name}_Reconstructed_SHAP_Longitude.png"), 
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Reconstruct Latitude Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_lat_matrix, X_features, show=False)

    # Increase axis ticks and label fonts 
    ax_lat = plt.gca()
    ax_lat.tick_params(axis='both', which='major', labelsize=15) 
    ax_lat.set_xlabel("SHAP Value (Degrees Latitude)", fontsize=14)

    plt.title(
        f"Latitude SHAP\n(Positive = Pushes North |  Negative = Pushes South)",
        y=1.05,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{data_name}_{model_name}_Reconstructed_SHAP_Latitude.png"), 
        dpi=300, bbox_inches="tight"
    )
    plt.close()
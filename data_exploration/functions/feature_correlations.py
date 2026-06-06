"""
correlation_pipeline.py
----------------------------
Functions script to perform feature correlation analysis on trace element concentrations.

Input data:
1. Preprocessed multi-element CSV datasets (ICPMS or XRF).

Generates and saves:
1. Pearson correlation matrices (CSV) and corresponding upper-triangle masked heatmaps (PNG).
2. Variance Inflation Factor (VIF) tables to assess multicollinearity (CSV).
3. Pairwise Mutual Information (MI) matrices (CSV) and corresponding upper-triangle masked heatmaps (PNG).
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Literal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# ----------------------------------------------
# Data Loading and Preprocessing
# ----------------------------------------------

def load_and_preprocess_data(
    path: str,
    data_type: Literal["ICPMS", "XRF"],
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Loads data, extracts trace elements, applies log1p transformation, 
    and standardizes the features.
    """
    df = pd.read_csv(path)

    if data_type == "ICPMS":
        trace_cols = df.columns[16:]
    elif data_type == "XRF":
        trace_cols = df.columns[13:]
    else:
        raise ValueError("data_type must be 'ICPMS' or 'XRF'.")

    # Extract strictly numeric trace element columns
    X_raw = df[trace_cols].select_dtypes(include="number")
    features = X_raw.columns

    # Log1p transform (handles zeros by calculating log(1 + x))
    X_log = np.log1p(X_raw)

    # Standardize
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_log)
    X_scaled = pd.DataFrame(X_scaled_array, columns=features, index=df.index)

    return X_scaled, features

# ----------------------------------------------
# Statistical Calculations
# ----------------------------------------------

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Variance Inflation Factor (VIF) for all features.
    """
    # Adding a constant is required for VIF calculations in statsmodels
    X_with_const = add_constant(X)
    
    vif_data = []
    for i in range(X_with_const.shape[1]):
        feature_name = X_with_const.columns[i]
        if feature_name == "const":
            continue
            
        try:
            vif_val = variance_inflation_factor(X_with_const.values, i)
        except RuntimeWarning:
            vif_val = np.inf # Handle perfect multicollinearity 
            
        vif_data.append({
            "Feature": feature_name,
            "VIF": vif_val
        })
        
    df_vif = pd.DataFrame(vif_data).sort_values(by="VIF", ascending=False)
    return df_vif

def calculate_mutual_information_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a pairwise Mutual Information matrix.
    Computes the upper triangle to save computation time and mirrors it.
    """
    n_features = X.shape[1]
    mi_matrix = np.zeros((n_features, n_features))
    features = X.columns

    for i in range(n_features):
        mi_matrix[i, i] = 1.0 # Self MI is max/normalized to 1 or left raw; we'll leave as raw max, but normally diagonal isn't strictly defined for continuous self-MI. We'll set to 0 and fill below.
        for j in range(i + 1, n_features):
            # Calculate MI between feature i and feature j
            mi_val = mutual_info_regression(X.iloc[:, [i]], X.iloc[:, j], random_state=42)[0]
            mi_matrix[i, j] = mi_val
            mi_matrix[j, i] = mi_val # Mirror

    # Set diagonal to the maximum MI found in the dataset for visual scaling purposes
    np.fill_diagonal(mi_matrix, mi_matrix.max())

    return pd.DataFrame(mi_matrix, index=features, columns=features)

# ----------------------------------------------
# Plotting Functions
# ----------------------------------------------

def plot_heatmap(
    matrix: pd.DataFrame, 
    title: str, 
    base_output_path: str, 
    data_name: str,
    filename: str,
    cmap: str = "coolwarm",
    vmin: float = None,
    vmax: float = None
) -> None:
    """
    Generates and saves a heatmap for a given symmetric matrix (Correlation or MI).
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Generate a mask for the upper triangle for cleaner visualization
    mask = np.triu(np.ones_like(matrix, dtype=bool))

    sns.heatmap(
        matrix, 
        mask=mask, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax, 
        center=0 if vmin == -1 else None,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .75},
        ax=ax
    )

    ax.set_title(f"{title} ({data_name})", fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    out_dir = os.path.join(base_output_path, data_name)
    os.makedirs(out_dir, exist_ok=True)
    
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to: {save_path}")

# ----------------------------------------------
# Main Dispatcher
# ----------------------------------------------

def run_correlation_pipeline(
    input_path: str, 
    base_output_path: str, 
    data_name: str,
    data_type: Literal["ICPMS", "XRF"]
) -> None:
    """
    Executes the full correlation and mutual information pipeline.
    """
    print(f"Starting Correlation Pipeline for {data_name}...")
    
    # Create a safe version of the data name for filenames (replaces slashes with underscores)
    safe_data_name = data_name.replace('\\', '_').replace('/', '_')
    
    # Load and Preprocess
    X_scaled, features = load_and_preprocess_data(input_path, data_type)
    print(f"  [-] Processed data: Log-transformed and Standardized ({X_scaled.shape[0]} samples, {X_scaled.shape[1]} features).")

    out_dir = os.path.join(base_output_path, data_name)
    os.makedirs(out_dir, exist_ok=True)

    # Pearson Correlation Matrix
    corr_matrix = X_scaled.corr(method='pearson')
    corr_csv_path = os.path.join(out_dir, "correlation_matrix.csv")
    corr_matrix.to_csv(corr_csv_path)
    
    plot_heatmap(
        matrix=corr_matrix, 
        title="Feature Correlation Matrix", 
        base_output_path=base_output_path, 
        data_name=data_name, 
        filename=f"{safe_data_name}_heatmap_correlation.png", # <- Updated filename
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0
    )

    # Variance Inflation Factor (VIF)
    vif_df = calculate_vif(X_scaled)
    vif_csv_path = os.path.join(out_dir, "vif_table.csv")
    vif_df.to_csv(vif_csv_path, index=False)
    print(f"Saved VIF table to: {vif_csv_path}")

    # Mutual Information Matrix
    mi_matrix = calculate_mutual_information_matrix(X_scaled)
    mi_csv_path = os.path.join(out_dir, "mutual_information_matrix.csv")
    mi_matrix.to_csv(mi_csv_path)
    
    plot_heatmap(
        matrix=mi_matrix, 
        title="Mutual Information Matrix", 
        base_output_path=base_output_path, 
        data_name=data_name, 
        filename=f"{safe_data_name}_heatmap_mutual_information.png", # <- Updated filename
        cmap="mako", 
        vmin=0.0
    )
    
    print("Pipeline execution complete.")
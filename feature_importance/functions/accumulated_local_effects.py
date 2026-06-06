"""
accumulated_local_effects.py
----------------------------
Functions script to calculate and visualize Accumulated Local Effects (ALE) for multi-output spatial regression models.

Input data:
1. Preprocessed multi-element CSV datasets (ICPMS or XRF) containing numerical trace features and spatial targets 
    (Longitude/Latitude).
2. Optimal hyperparameter logs (CSV) to configure the evaluated machine learning algorithms (Random Forest, XGBoost, SVM).

Generates and saves:
1. Raw numerical ALE tracking data (CSV) mapping the marginal spatial effects 
    (grid values vs. predicted Longitude/Latitude) for every feature across all models.
2. Single-model ALE grid plots (PNG) displaying the longitudinal and latitudinal impact of every feature on dual axes.
3. Custom multi-model comparative ALE subplots (PNG) enhanced with visual data density 
    representations (variable line thicknesses and rug plots) for accurate interpretation of localized feature impacts.
"""

import os
import math
import ast
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.base import clone
from PyALE import ale

from typing import Tuple, Dict, Literal

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

import logging
from matplotlib.collections import LineCollection

# Silence PyALE's chatty INFO messages
logging.getLogger('PyALE._ALE_generic').setLevel(logging.WARNING)

# ----------------------------------------------
# Data & Model Setup
# ----------------------------------------------

def load_data(path: str, data_type: Literal["ICPMS", "XRF"]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    if data_type == "ICPMS":
        trace_cols = df.columns[16:]
    elif data_type == "XRF":
        trace_cols = df.columns[13:]
    else:
        raise ValueError("data_type must be 'ICPMS' or 'XRF'.")

    X = df[trace_cols].select_dtypes(include="number")
    y = df[["Longitude", "Latitude"]]

    # Added this safety check
    overlap = set(X.columns) & set(y.columns)
    if overlap:
        raise ValueError(f"X and y share columns: {overlap}. Check column indexing in load_data.")

    print(f"  -> Loaded {len(X)} samples, {X.shape[1]} features. "
          f"Coord range: Lon [{y['Longitude'].min():.2f}, {y['Longitude'].max():.2f}], "
          f"Lat [{y['Latitude'].min():.2f}, {y['Latitude'].max():.2f}]")
    
    return X, y

def get_models(random_state: int = 42) -> Dict[str, object]:
    def log_with_epsilon(X):
        epsilon = 1e-6
        X_arr = np.array(X, dtype=float)  # strips all pandas metadata including column names
        X_arr = np.clip(X_arr, 0.0, None)
        return np.log(X_arr + epsilon)
    
    log_transformer = FunctionTransformer(log_with_epsilon, validate=False)
    
    return {
        "RandomForest": Pipeline([
            ('log', log_transformer),
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state))
        ]),
        "XGBoost": Pipeline([
            ('log', log_transformer),
            ('scaler', StandardScaler()),
            ('model', XGBRegressor(n_estimators=200, learning_rate=0.05, n_jobs=-1, random_state=random_state, verbosity=0)) 
        ]),
        "SVM": Pipeline([
            ('log', log_transformer),
            ('scaler', StandardScaler()),
            ('model', MultiOutputRegressor(SVR(kernel="rbf", C=10, epsilon=0.1)))
        ]),
    }

def get_best_params(csv_path: str, model_name: str) -> dict:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Params file not found: {csv_path}\n"
            f"Run hyperparameter tuning first, or pass params_file=None to use defaults."
        )
    
    df = pd.read_csv(csv_path)
    row = df[df['model'] == model_name]
    
    if not row.empty:
        param_str = row['optimal_parameters'].values[0]
        
        # FIX: Strip numpy wrapper functions before evaluating
        param_str = re.sub(r'np\.[a-zA-Z0-9_]+\((.*?)\)', r'\1', param_str)
        
        try:
            return ast.literal_eval(param_str)
        except Exception as e:
            print(f"  [!] Error parsing parameters for {model_name}: {e}")
            return {}
    else:
        print(f"  [!] No tuned parameters found for {model_name}. Using defaults.")
        return {}

class SingleOutputWrapper:
    """
    Wrapper to isolate a specific target from a MultiOutputRegressor.
    PyALE strictly requires models that return a 1D array of predictions.
    """
    def __init__(self, model, target_idx: int):
        self.model = model
        self.target_idx = target_idx
        
    def predict(self, X):
        return self.model.predict(X)[:, self.target_idx]

def compute_custom_ale(model, X: pd.DataFrame, feature: str, grid_resolution: int = 50):
    """
    Computes 1D Accumulated Local Effects for a multi-output (Lon, Lat) regressor.
    Shifts the centered ALE effect by the global mean to match true coordinates.
    """
    # Handle edge case: features with zero variance
    if X[feature].nunique() <= 1:
        dummy_preds = model.predict(X)
        return np.array([X[feature].min(), X[feature].max()]), np.array([0.0, 0.0]), np.array([0.0, 0.0])

    # Split the multi-output model into two distinct 1D models for PyALE
    model_lon = SingleOutputWrapper(model, target_idx=0)
    model_lat = SingleOutputWrapper(model, target_idx=1)

    # PyALE can be noisy with warnings when generating quantiles for highly skewed elements
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Calculate ALE
            ale_lon_df = ale(X=X, model=model_lon, feature=[feature], grid_size=grid_resolution, include_CI=False, plot=False)
            ale_lat_df = ale(X=X, model=model_lat, feature=[feature], grid_size=grid_resolution, include_CI=False, plot=False)
            
            grid = ale_lon_df.index.values
            
            # ALE naturally centers effects around 0. We add the global mean back.
            # CHANGE: don't add the global mean back. Is more intuitive to just see how much degrees 
            # were changed based onto the average bc of the feature.
            # mean_lon = model_lon.predict(X).mean()
            # mean_lat = model_lat.predict(X).mean()
            
            ale_lon = ale_lon_df['eff'].values #+ mean_lon
            ale_lat = ale_lat_df['eff'].values #+ mean_lat
            
            return grid, np.array(ale_lon), np.array(ale_lat)
            
        except Exception as e:
            print(f"      [!] PyALE failed for {feature}: {e}. Returning flatline.")
            # Fallback for heavily zero-inflated dummy variables
            dummy_preds = model.predict(X)
            return np.array([X[feature].min(), X[feature].max()]), np.array([0.0, 0.0]), np.array([0.0, 0.0])

def run_ale_pipeline(
    data_name: str, 
    csv_path: str, 
    data_type: str, 
    params_file: str, 
    base_output_dir: str
) -> None:
    """
    Executes the full ALE pipeline, saving data and plots.
    """
    print(f"\n{'='*60}")
    print(f"GENERATING ALE PLOTS: {data_name.upper()} ({data_type})")
    print(f"{'='*60}")

    data_dir = os.path.join(base_output_dir, data_name, "ale_data")
    plots_dir = os.path.join(base_output_dir, data_name, "ale_plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    X, y = load_data(csv_path, data_type)
    models = get_models() 
    features = X.columns.tolist()

    for model_name, model_obj in models.items():
        print(f"  -> Processing {model_name}...")
    
        tuned_params = get_best_params(params_file, model_name)

        if tuned_params:
            print(f"     [+] Tuned hyperparameters found and applied for {model_name}:")
            for param_name, param_val in tuned_params.items():
                print(f"         - {param_name}: {param_val}")
        else:
            print(f"     [-] Running {model_name} with default parameters.")

        model = clone(model_obj)
        if tuned_params:
            model.set_params(**tuned_params)
        
        model.fit(X, y)

        # ---------------------------------------------------------
        # PHASE 1: Compute all ALEs and find global Y-axis bounds
        # ---------------------------------------------------------
        all_ale_data = []
        feature_results = {}
        
        g_lon_min, g_lon_max = float('inf'), float('-inf')
        g_lat_min, g_lat_max = float('inf'), float('-inf')

        for feature in features:
            grid, ale_lon, ale_lat = compute_custom_ale(model, X, feature, grid_resolution=50)
            
            # Update global minimums and maximums across Longitude and Latitude separately
            g_lon_min = min(g_lon_min, ale_lon.min())
            g_lon_max = max(g_lon_max, ale_lon.max())
            g_lat_min = min(g_lat_min, ale_lat.min())
            g_lat_max = max(g_lat_max, ale_lat.max())
            
            # Store array data so we don't have to compute it again in Phase 2
            feature_results[feature] = (grid, ale_lon, ale_lat)
            
            for g_val, lon_val, lat_val in zip(grid, ale_lon, ale_lat):
                all_ale_data.append({
                    "Feature": feature,
                    "Grid_Value": g_val,
                    "ALE_Longitude": lon_val,
                    "ALE_Latitude": lat_val
                })

        # Add a 5% visual padding so lines don't hit the exact ceiling/floor
        lon_pad = (g_lon_max - g_lon_min) * 0.05 if (g_lon_max - g_lon_min) != 0 else 0.1
        lat_pad = (g_lat_max - g_lat_min) * 0.05 if (g_lat_max - g_lat_min) != 0 else 0.1
        
        g_lon_min -= lon_pad
        g_lon_max += lon_pad
        g_lat_min -= lat_pad
        g_lat_max += lat_pad

        # ---------------------------------------------------------
        # PHASE 2: Plot the Main Grid
        # ---------------------------------------------------------
        n_features = len(features)
        n_cols = 5
        n_rows = math.ceil(n_features / n_cols)
        
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 4, n_rows * 3.5))
        axes = axes.flatten()

        for i, feature in enumerate(features):
            grid, ale_lon, ale_lat = feature_results[feature]

            # ---------------------------------------------------------
            # Decoupled Data Density Calculation
            # ---------------------------------------------------------
            safe_min = X[feature].min() if X[feature].min() > 0 else 1e-6
            safe_max = X[feature].max() if X[feature].max() > 0 else 1e-5
            if safe_min >= safe_max: safe_max = safe_min + 1e-5
            
            density_bins = np.geomspace(safe_min, safe_max, 50)
            hist_counts, _ = np.histogram(X[feature], bins=density_bins)
            
            midpoints = (grid[:-1] + grid[1:]) / 2
            bin_indices = np.digitize(midpoints, density_bins) - 1
            bin_indices = np.clip(bin_indices, 0, len(hist_counts) - 1)
            segment_densities = hist_counts[bin_indices]
            
            min_lw, max_lw = 0.5, 5.0
            if segment_densities.max() > 0:
                widths = min_lw + (segment_densities / segment_densities.max()) * (max_lw - min_lw)
            else:
                widths = np.full(len(grid)-1, min_lw)

            ax = axes[i]
            
            # Longitude on Left Y-Axis 
            color_lon = 'dodgerblue'
            pts_lon = np.array([grid, ale_lon]).T.reshape(-1, 1, 2)
            segs_lon = np.concatenate([pts_lon[:-1], pts_lon[1:]], axis=1)
            
            lc_lon = LineCollection(segs_lon, linewidths=widths, color=color_lon)
            ax.add_collection(lc_lon)
            
            ax.set_ylabel('Longitude', color=color_lon, fontsize=9)
            ax.tick_params(axis='y', labelcolor=color_lon, labelsize=8)
            
            ax.set_xlim(grid.min() if grid.min() > 0 else 1e-6, grid.max())
            ax.set_ylim(g_lon_min, g_lon_max) # <-- GLOBALLY LOCKED
            ax.set_xscale('log')

            # Force rotation and right-alignment on BOTH major and minor ticks
            ax.tick_params(axis='x', which='both', labelsize=8, labelrotation=45)
            plt.setp(ax.get_xticklabels(which='both'), ha='right', rotation_mode='anchor')

            # Rug plot
            ax.plot(X[feature], np.full_like(X[feature], g_lon_min), 
                    '|', color='black', alpha=0.15, markersize=6, zorder=3)
            
            # Latitude on Right Y-Axis
            ax2 = ax.twinx()
            color_lat = 'darkred'
            
            pts_lat = np.array([grid, ale_lat]).T.reshape(-1, 1, 2)
            segs_lat = np.concatenate([pts_lat[:-1], pts_lat[1:]], axis=1)
            
            lc_lat = LineCollection(segs_lat, linewidths=widths, color=color_lat)
            ax2.add_collection(lc_lat)
            
            ax2.set_ylabel('Latitude', color=color_lat, fontsize=9)
            ax2.tick_params(axis='y', labelcolor=color_lat, labelsize=8)
            
            ax2.set_ylim(g_lat_min, g_lat_max) # <-- GLOBALLY LOCKED

            ax.set_title(f"{feature}", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')

        # Clean up empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        fig.suptitle(f"Accumulated Local Effects (ALE) - {model_name} ({data_name})", fontsize=16, fontweight='bold', y=1.02)

        # Added data_name to the output filename
        safe_data_name = str(data_name).replace("\\", "_").replace("/", "_")
        plot_path = os.path.join(plots_dir, f"{safe_data_name}_{model_name}_ALE_Grid.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()

        df_ale = pd.DataFrame(all_ale_data)
        csv_path_out = os.path.join(data_dir, f"{model_name}_ALE_Data.csv")
        df_ale.to_csv(csv_path_out, index=False)

        print(f"    [+] Saved plots and data for {model_name}.")

    print("  Done!")


def create_custom_ale_subplots(data_name: str, base_ale_dir: str, output_dir: str, features: list, csv_path: str, data_type: str):
    """
    Reads pre-calculated ALE CSV data and plots a custom subplot grid:
    Columns = Models (RandomForest, XGBoost, SVM), Rows = Features.
    Scales all subplots using the absolute global min/max of Lat and Lon across the selected features.
    Now includes variable line thickness based on data density, and x-axis rug plots.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    models = ["RandomForest", "XGBoost", "SVM"]
    data_dir = os.path.join(base_ale_dir, data_name, "ale_data")

    if not os.path.exists(data_dir):
        print(f"  [!] Data directory not found: {data_dir}. Cannot create subplots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw data to calculate density
    X, _ = load_data(csv_path, data_type)

    # ---------------------------------------------------------
    # PHASE 1: Find Global Y-axis Bounds Across All Features & Models
    # ---------------------------------------------------------
    g_lon_min, g_lon_max = float('inf'), float('-inf')
    g_lat_min, g_lat_max = float('inf'), float('-inf')
    
    valid_data_found = False

    for model in models:
        csv_m_path = os.path.join(data_dir, f"{model}_ALE_Data.csv")
        if os.path.exists(csv_m_path):
            df = pd.read_csv(csv_m_path)
            for feature in features:
                if feature in df['Feature'].values:
                    feat_df = df[df['Feature'] == feature]
                    # Compute min/max for longitude and latitude separately
                    g_lon_min = min(g_lon_min, feat_df['ALE_Longitude'].min())
                    g_lon_max = max(g_lon_max, feat_df['ALE_Longitude'].max())
                    g_lat_min = min(g_lat_min, feat_df['ALE_Latitude'].min())
                    g_lat_max = max(g_lat_max, feat_df['ALE_Latitude'].max())
                    valid_data_found = True

    if not valid_data_found:
        print(f"  [!] No valid features found in CSVs for {data_name}. Skipping.")
        return

    lon_pad = (g_lon_max - g_lon_min) * 0.05 if (g_lon_max - g_lon_min) != 0 else 0.1
    lat_pad = (g_lat_max - g_lat_min) * 0.05 if (g_lat_max - g_lat_min) != 0 else 0.1
    
    g_lon_min -= lon_pad
    g_lon_max += lon_pad
    g_lat_min -= lat_pad
    g_lat_max += lat_pad

    # ---------------------------------------------------------
    # PHASE 2: Plot the Subplots
    # ---------------------------------------------------------
    n_rows = len(features)
    n_cols = len(models)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for col_idx, model in enumerate(models):
        csv_m_path = os.path.join(data_dir, f"{model}_ALE_Data.csv")
        
        if not os.path.exists(csv_m_path):
            print(f"  [!] Missing CSV for {model} in {data_name}. Skipping column.")
            continue
            
        df = pd.read_csv(csv_m_path)

        for row_idx, feature in enumerate(features):
            ax = axes[row_idx, col_idx]
            
            if feature not in df['Feature'].values or feature not in X.columns:
                ax.text(0.5, 0.5, f"{feature}\nnot in dataset", ha='center', va='center', color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(f"{model}", fontsize=12, fontweight='bold')
                continue

            feat_df = df[df['Feature'] == feature]
            grid = feat_df['Grid_Value'].values
            ale_lon = feat_df['ALE_Longitude'].values
            ale_lat = feat_df['ALE_Latitude'].values

            # ---------------------------------------------------------
            # Decoupled Data Density Calculation
            # ---------------------------------------------------------
            safe_min = X[feature].min() if X[feature].min() > 0 else 1e-6
            safe_max = X[feature].max() if X[feature].max() > 0 else 1e-5
            if safe_min >= safe_max: safe_max = safe_min + 1e-5
            
            density_bins = np.geomspace(safe_min, safe_max, 50)
            hist_counts, _ = np.histogram(X[feature], bins=density_bins)
            
            midpoints = (grid[:-1] + grid[1:]) / 2
            bin_indices = np.digitize(midpoints, density_bins) - 1
            bin_indices = np.clip(bin_indices, 0, len(hist_counts) - 1)
            segment_densities = hist_counts[bin_indices]
            
            min_lw, max_lw = 0.5, 5.0
            if segment_densities.max() > 0:
                widths = min_lw + (segment_densities / segment_densities.max()) * (max_lw - min_lw)
            else:
                widths = np.full(len(grid)-1, min_lw)

            # Longitude (Left Y-axis)
            color_lon = 'dodgerblue'
            pts_lon = np.array([grid, ale_lon]).T.reshape(-1, 1, 2)
            segs_lon = np.concatenate([pts_lon[:-1], pts_lon[1:]], axis=1)
            
            lc_lon = LineCollection(segs_lon, linewidths=widths, color=color_lon)
            ax.add_collection(lc_lon)
            
            ax.set_ylabel('Longitude', color=color_lon, fontsize=9)
            ax.tick_params(axis='y', labelcolor=color_lon, labelsize=8)
            ax.set_xscale('log')
            
            ax.set_xlim(grid.min() if grid.min() > 0 else 1e-6, grid.max())
            ax.set_ylim(g_lon_min, g_lon_max)

            # Force rotation and right-alignment on BOTH major and minor ticks
            ax.tick_params(axis='x', which='both', labelsize=8, labelrotation=45)
            plt.setp(ax.get_xticklabels(which='both'), ha='right', rotation_mode='anchor')

            # Rug plot for density
            ax.plot(X[feature], np.full_like(X[feature], g_lon_min), 
                    '|', color='black', alpha=0.15, markersize=6, zorder=3)

            # Latitude (Right Y-axis)
            ax2 = ax.twinx()
            color_lat = 'darkred'
            pts_lat = np.array([grid, ale_lat]).T.reshape(-1, 1, 2)
            segs_lat = np.concatenate([pts_lat[:-1], pts_lat[1:]], axis=1)
            
            lc_lat = LineCollection(segs_lat, linewidths=widths, color=color_lat)
            ax2.add_collection(lc_lat)
            
            ax2.set_ylabel('Latitude', color=color_lat, fontsize=9)
            ax2.tick_params(axis='y', labelcolor=color_lat, labelsize=8)
            ax2.set_ylim(g_lat_min, g_lat_max)

            # Titles
            if row_idx == 0:
                ax.set_title(f"{model}\n{feature}", fontsize=11, fontweight='bold')
            else:
                ax.set_title(f"{feature}", fontsize=11, fontweight='bold')

            ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.suptitle(f"Custom ALE Subplots - {data_name}", fontsize=16, fontweight='bold', y=1.02)

    safe_name = str(data_name).replace("\\", "_").replace("/", "_")
    plot_path = os.path.join(output_dir, f"{safe_name}_Custom_Subplots.png")
    
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"    [+] Saved custom subplots for {data_name}")
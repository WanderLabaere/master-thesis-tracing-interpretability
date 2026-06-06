"""
CV.py
----------------------------
Pipeline functions script to perform spatially stratified nested cross-validation (CV) for predictive 
spatial modeling using multi-element data.

Input data:
1. Preprocessed multi-element CSV datasets (ICPMS or XRF) containing numerical trace features, 
categorical metadata, and spatial coordinates.

Generates and saves:
1. CSV tables detailing nested CV evaluation metrics (R2, RMSE, mean Haversine distance, 
    and Haversine Skill Score) across tested models (Random Forest, XGBoost, SVM).
2. CSV logs of the optimal hyperparameters found during the inner randomized search for each algorithm.
3. Multi-panel geographical plots mapping predicted vs. actual test sample coordinates per fold, overlaid on a world map (PNG).
4. Raw data logs (CSV) for the fold visualization plots, capturing individual jittered and actual coordinates for 
    train/test splits.
"""


import os
import ast
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import Counter
from typing import Dict, List, Literal, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.base import clone
from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer

import warnings
# to ignore warnings about CPU usage:
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

# KILL-SWITCH for the specific 'delayed' joblib spam
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

# ----------------------------------------------
# Helper functions
# ----------------------------------------------

def get_best_params(csv_path: str, model_name: str) -> dict:
    if not os.path.exists(csv_path):
        print(f"Warning: Params file not found at {csv_path}. Using default parameters.")
        return {}
    df = pd.read_csv(csv_path)
    row = df[df['model'] == model_name]
    if not row.empty:
        param_str = row['optimal_parameters'].values[0]
        try:
            return ast.literal_eval(param_str)
        except Exception as e:
            print(f"Error parsing parameters for {model_name}: {e}")
            return {}
    else:
        print(f"No tuned parameters found for {model_name}. Using defaults.")
        return {}

# ----------------------------------------------
# Data loading
# ----------------------------------------------

def load_data(path: str, data_type: Literal["ICPMS", "XRF"]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads a geochemical dataset from a CSV file and separates it into numerical feature columns
    and spatial coordinate targets based on  specific column structure of the 
    ICPMS or XRF data type.
    """
    df = pd.read_csv(path)
    if data_type == "ICPMS":
        trace_cols = df.columns[16:]
    elif data_type == "XRF":
        trace_cols = df.columns[13:]
    else:
        raise ValueError("data_type must be 'ICPMS' or 'XRF'.")
    X = df[trace_cols].select_dtypes(include="number")
    y = df[["Longitude", "Latitude"]]
    return X, y

# ----------------------------------------------
# Model definitions
# ----------------------------------------------

from sklearn.compose import ColumnTransformer

# Pass X.columns to the function so it knows which columns are which
# adapted for one hot encoding of genera
def get_models(X_columns, random_state: int = 42) -> Dict[str, object]:
    """
    Initializes and returns a dictionary of machine learning regression pipelines (Random Forest, XGBoost, and SVM).
    Constructs a ColumnTransformer preprocessing step that applies a log transformation (with epsilon) 
        and standard scaling to continuous trace element features, 
        while passing one-hot encoded binary features through without modification.
    """
    
    def log_with_epsilon(X):
        epsilon = 1e-6
        return np.log(X + epsilon)
        
    log_transformer = FunctionTransformer(log_with_epsilon, validate=False)
    
    # Isolate continuous trace elements vs binary Genus columns
    binary_cols = [col for col in X_columns if col.startswith('Genus_')]
    continuous_cols = [col for col in X_columns if col not in binary_cols]
    
    # Create a sub-pipeline just for the continuous trace elements
    continuous_pipeline = Pipeline([
        ('log', log_transformer),
        ('scaler', StandardScaler())
    ])
    
    # Combine them: process continuous, pass binary (OHE) through untouched
    preprocessor = ColumnTransformer(
        transformers=[
            ('cont', continuous_pipeline, continuous_cols),
            ('bin', 'passthrough', binary_cols)
        ]
    )

    return {
        "RandomForest": Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state))
        ]),
        "XGBoost": Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(n_estimators=200, learning_rate=0.05, n_jobs=-1, random_state=random_state, verbosity=0))
        ]),
        "SVM": Pipeline([
            ('preprocessor', preprocessor),
            ('model', MultiOutputRegressor(SVR(kernel="rbf", C=10, epsilon=0.1)))
        ]),
    }

# ----------------------------------------------
# Scoring functions
# ----------------------------------------------

def _r2_lon(y_true, y_pred):
    return r2_score(np.asarray(y_true)[:, 0], np.asarray(y_pred)[:, 0])

def _r2_lat(y_true, y_pred):
    return r2_score(np.asarray(y_true)[:, 1], np.asarray(y_pred)[:, 1])

def _rmse_lon(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.asarray(y_true)[:, 0], np.asarray(y_pred)[:, 0]))

def _rmse_lat(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.asarray(y_true)[:, 1], np.asarray(y_pred)[:, 1]))

def _r2_combined(y_true, y_pred):
    y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)
    return np.mean([r2_score(y_true_arr[:, i], y_pred_arr[:, i]) for i in range(y_true_arr.shape[1])])

def _rmse_combined(y_true, y_pred):
    y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)
    return np.mean([np.sqrt(mean_squared_error(y_true_arr[:, i], y_pred_arr[:, i])) for i in range(y_true_arr.shape[1])])

def _calculate_haversine_array(y_true_arr, y_pred_arr):
    lon1, lat1 = np.radians(y_true_arr[:, 0]), np.radians(y_true_arr[:, 1])
    lon2, lat2 = np.radians(y_pred_arr[:, 0]), np.radians(y_pred_arr[:, 1])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

def _haversine_distance_mean(y_true, y_pred):
    return np.mean(_calculate_haversine_array(np.asarray(y_true), np.asarray(y_pred)))

### FIX: EXPLICIT CENTROID ARGUMENTS 
# This makes sure we can use the training data only to make the naive prediction
def _haversine_skill_score(y_true, y_pred, center_lon, center_lat):
    """
    Calculates a custom spatial skill score (analogous to R-squared) using Haversine distances. 
    It evaluates the model's predictive performance by comparing its total distance error against a naive baseline
    that always predicts the safely provided central coordinates.
    """
    y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)
    model_distances = _calculate_haversine_array(y_true_arr, y_pred_arr)
    
    # The naive guess is now forced to use safely provided coordinates
    y_naive_arr = np.full_like(y_true_arr, [center_lon, center_lat])
    naive_distances = _calculate_haversine_array(y_true_arr, y_naive_arr)
    
    sum_naive = np.sum(naive_distances)
    if sum_naive == 0:
        return 0.0
    return 1 - (np.sum(model_distances) / sum_naive)

# ----------------------------------------------
# CV summary helper
# ----------------------------------------------

def _summarise_nested_cv(
    outer_fold_metrics: dict,
    best_params: dict,
    model_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compiles raw evaluation metrics tracked across all outer folds during nested cross-validation into 
    a structured statistical summary DataFrame (calculating mean, standard deviation, min, and max for each metric), 
    while also formatting the optimal hyperparameters for export.
    """

    rows = []
    for metric_name, values in outer_fold_metrics.items():
        values = np.array(values)
        if metric_name.endswith("_0"):
            target, clean_metric = "Longitude", metric_name[:-2]
        elif metric_name.endswith("_1"):
            target, clean_metric = "Latitude", metric_name[:-2]
        elif metric_name.endswith("_combined"):
            target, clean_metric = "2D_combined", metric_name[:-9]
        elif metric_name == "haversine_distance_within_fold_std":
            target, clean_metric = "2D_combined", "haversine_distance_within_fold_std"
        else:
            target, clean_metric = "2D_combined", metric_name

        rows.append({
            "model": model_name,
            "target": target,
            "metric": clean_metric,
            "mean": round(values.mean(), 4),
            "std": round(values.std(), 4),
            "min": round(values.min(), 4),
            "max": round(values.max(), 4),
        })

    summary_df = pd.DataFrame(rows)
    optimal_params_df = pd.DataFrame([{
        "model": model_name,
        "optimal_parameters": str(best_params)
    }])
    return summary_df, optimal_params_df

# ----------------------------------------------
# Core CV function — nested CV with spatial stratification
# ----------------------------------------------

def _get_safe_spatial_clusters(coords_df: pd.DataFrame, n_splits: int, target_clusters: int = 5, random_state: int = 42, loop_name: str = "Outer"):
    """
    Dynamically reduces the number of K-Means clusters until every cluster 
    contains at least 'n_splits' members, preventing StratifiedKFold crashes.
    """
    coords = coords_df[["Latitude", "Longitude"]]
    
    # Mathematical ceiling: can't have more clusters than (Total Samples / n_splits)
    max_possible_clusters = max(1, len(coords) // n_splits)
    current_target = min(target_clusters, max_possible_clusters)
    
    for k in range(current_target, 0, -1):
        if k == 1:
            print(f"Warning ({loop_name}): Extreme spatial imbalance. Falling back to k=1 (standard random split).")
            return np.zeros(len(coords), dtype=int)
            
        clusters = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(coords)
        
        # Check the size of the smallest cluster
        smallest_cluster_size = np.min(np.bincount(clusters))
        
        # If the smallest cluster is mathematically safe for the folds, we found our k!
        if smallest_cluster_size >= n_splits:
            
            ### UPDATED PRINT LOGIC 
            if k == target_clusters:
                print(f"  [i] {loop_name} CV: Optimal k={k} clusters used for {n_splits} splits (Smallest region: {smallest_cluster_size} samples).")
            else:
                print(f"  [!] {loop_name} CV: Adjusted to k={k} clusters for {n_splits} splits (Smallest region: {smallest_cluster_size} samples).")

            
            return clusters
            
    return np.zeros(len(coords), dtype=int)

def kfold_cv(
    model,
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_name: str,
    param_grid: dict,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List, dict]:
    """
    Executes a comprehensive, spatially-stratified nested K-Fold cross-validation routine. 
    - It clusters spatial coordinates to ensure geographic balance across folds
    - uses an inner loop randomized search for optimal hyperparameter tuning guided by a custom Haversine skill score
    - evaluates the final generalized performance on the outer folds, ultimately returning performance metrics,
        optimal parameters, and the internally fitted models.
    """

    ### INNER LOOP WRAPPER
    # Enables centroid on training data only
    # Calculate the static global center for the tuning engine to use
    global_center_lon = np.mean(y["Longitude"])
    global_center_lat = np.mean(y["Latitude"])

    # Create a wrapper function so scikit-learn can pass just (y_true, y_pred)
    def inner_skill_score(y_true, y_pred):
        return _haversine_skill_score(y_true, y_pred, global_center_lon, global_center_lat)

    scoring = {
        "r2_0": make_scorer(_r2_lon),
        "r2_1": make_scorer(_r2_lat),
        "rmse_0": make_scorer(_rmse_lon, greater_is_better=False),
        "rmse_1": make_scorer(_rmse_lat, greater_is_better=False),
        "r2_combined": make_scorer(_r2_combined),
        "rmse_combined": make_scorer(_rmse_combined, greater_is_better=False),
        "haversine_distance": make_scorer(_haversine_distance_mean, greater_is_better=False),
        # Use the wrapper function
        "haversine_skill_score": make_scorer(inner_skill_score) 
    }


    # OUTER SPLIT
    spatial_clusters = _get_safe_spatial_clusters(
        coords_df=y, 
        n_splits=n_splits, 
        target_clusters=n_splits, 
        random_state=random_state, 
        loop_name="Outer"
    )

    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    outer_fold_metrics = {k: [] for k in scoring}
    outer_fold_metrics["haversine_distance_within_fold_std"] = [] 

    best_params_per_fold = []
    fold_models = []

    for train_idx, val_idx in outer_cv.split(X, spatial_clusters):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # INNER SPLIT
        inner_clusters = _get_safe_spatial_clusters(
            coords_df=y_train, 
            n_splits=n_splits, 
            target_clusters=n_splits, 
            random_state=random_state, 
            loop_name="Inner"
        )

        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + 1)
        inner_splits = list(inner_cv.split(X_train, inner_clusters))

        search = RandomizedSearchCV(
            clone(model), 
            param_distributions=param_grid,  
            n_iter=100,                      
            cv=inner_splits,
            scoring=scoring,
            refit="haversine_skill_score",
            n_jobs=-1,
            random_state=random_state,       
            return_train_score=False
        )
        search.fit(X_train, y_train)

        best_params_per_fold.append(search.best_params_)

        y_pred = search.best_estimator_.predict(X_val)

        ### FIX: OUTER LOOP PERFECT BASELINE 
        # Calculate the theoretically perfect Train-Only Centroid for this fold
        fold_train_lon = np.mean(y_train["Longitude"])
        fold_train_lat = np.mean(y_train["Latitude"])

        outer_fold_metrics["r2_0"].append(_r2_lon(y_val, y_pred))
        outer_fold_metrics["r2_1"].append(_r2_lat(y_val, y_pred))
        outer_fold_metrics["rmse_0"].append(_rmse_lon(y_val, y_pred))
        outer_fold_metrics["rmse_1"].append(_rmse_lat(y_val, y_pred))
        outer_fold_metrics["r2_combined"].append(_r2_combined(y_val, y_pred))
        outer_fold_metrics["rmse_combined"].append(_rmse_combined(y_val, y_pred))
        outer_fold_metrics["haversine_distance"].append(_haversine_distance_mean(y_val, y_pred))

        # add to calc within fold spread of haversine distances. 
        fold_distances = _calculate_haversine_array(np.asarray(y_val), y_pred)
        outer_fold_metrics["haversine_distance_within_fold_std"].append(np.std(fold_distances))

        
        # Pass the leak-proof train center to the final evaluation metric
        outer_fold_metrics["haversine_skill_score"].append(
            _haversine_skill_score(y_val, y_pred, fold_train_lon, fold_train_lat)
        )

        fold_models.append((search.best_estimator_, train_idx, val_idx))

    param_counts = Counter([str(p) for p in best_params_per_fold])
    most_common_params = best_params_per_fold[
        [str(p) for p in best_params_per_fold].index(param_counts.most_common(1)[0][0])
    ]

    summary_df, optimal_params_df = _summarise_nested_cv(
        outer_fold_metrics, most_common_params, model_name
    )

    return summary_df, optimal_params_df, fold_models, most_common_params

# ----------------------------------------------
# Fold visualisation — uses pre-fitted fold models
# ----------------------------------------------

def plot_kfold_cv_folds(
    fold_models: List,     
    model_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    coords: pd.DataFrame,
    data_name: str,
    output_path: str,
    verbose: bool = True,
) -> None:

    map_data_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_input\world_map_data\World_countries"
    try:
        world = gpd.read_file(map_data_dir)
    except Exception as e:
        raise FileNotFoundError(f"Could not load world map from:\n  {map_data_dir}\nError: {e}")

    lon = coords["Longitude"].values
    lat = coords["Latitude"].values
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    lon_padding = (lon_max - lon_min) * 0.05
    lat_padding = (lat_max - lat_min) * 0.05

    if verbose:
        print(f" Generating fold plots from pre-fitted models...")

    n_folds = len(fold_models)
    grid_rows, grid_cols = _calculate_grid_dims(n_folds)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(5 * grid_cols, 4.5 * grid_rows))

    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([[axes]])
    elif grid_rows == 1 or grid_cols == 1:
        axes = axes.reshape(grid_rows, grid_cols)

    all_skill, all_haversine = [], []
    
    # NEW: List to store dataframe rows for CSV export
    all_fold_data = []

    fold_num = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            ax = axes[row, col]
            fold_num += 1

            if fold_num > n_folds:
                ax.set_visible(False)
                continue

            # FIX :unpack pre-fitted model and indices
            fitted_model, train_idx, val_idx = fold_models[fold_num - 1]

            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            y_pred = fitted_model.predict(X_val)
            
            ### FIX: Calculate train centroid for the plot
            y_train_plot = y.iloc[train_idx]
            plot_train_lon = np.mean(y_train_plot["Longitude"])
            plot_train_lat = np.mean(y_train_plot["Latitude"])

            # Pass the 4 arguments
            fold_skill = _haversine_skill_score(y_val, y_pred, plot_train_lon, plot_train_lat)
            
            fold_haversine = _haversine_distance_mean(y_val, y_pred)
            all_skill.append(fold_skill)
            all_haversine.append(fold_haversine)

            world.plot(ax=ax, alpha=0.2, edgecolor='k', linewidth=0.2, color='lightgray')
            ax.set_xlim(lon_min - lon_padding, lon_max + lon_padding)
            ax.set_ylim(lat_min - lat_padding, lat_max + lat_padding)

            # ------------------------------------------------
            # --- START OF MODIFIED JITTER & STYLING BLOCK ---
            # ------------------------------------------------
            # j_range = # HIDDEN
            
            #  Train Samples
            train_jitter_lon = np.random.uniform(-j_range, j_range, size=len(train_idx))
            train_jitter_lat = np.random.uniform(-j_range, j_range, size=len(train_idx))
            
            ax.scatter(lon[train_idx] + train_jitter_lon, lat[train_idx] + train_jitter_lat, 
                       c='dodgerblue', alpha=0.3, s=120,
                       edgecolors='navy', linewidth=0.4, zorder=5)

            #  Test/Predicted Setup
            test_lon = coords["Longitude"].values[val_idx]
            test_lat = coords["Latitude"].values[val_idx]
            pred_lon, pred_lat = y_pred[:, 0], y_pred[:, 1]
            
            test_j_lon = np.random.uniform(-j_range, j_range, size=len(test_lon))
            test_j_lat = np.random.uniform(-j_range, j_range, size=len(test_lat))
            pred_j_lon = np.random.uniform(-j_range, j_range, size=len(pred_lon))
            pred_j_lat = np.random.uniform(-j_range, j_range, size=len(pred_lat))

            #  Connective Lines
            for i in range(len(test_lon)):
                ax.plot([test_lon[i] + test_j_lon[i], pred_lon[i] + pred_j_lon[i]], 
                        [test_lat[i] + test_j_lat[i], pred_lat[i] + pred_j_lat[i]],
                        color='dimgray', linestyle='-', linewidth=1.0, alpha=0.4, zorder=4)

            #  Test Actual
            ax.scatter(test_lon + test_j_lon, test_lat + test_j_lat, 
                       c='red', alpha=0.4, s=160,
                       edgecolors='darkred', linewidth=0.6, marker='^', zorder=6)
            
            #  Test Predicted
            ax.scatter(pred_lon + pred_j_lon, pred_lat + pred_j_lat, 
                       c='orange', alpha=0.5, s=120,
                       edgecolors='black', linewidth=0.6, marker='X', zorder=6)
            # ------------------------------------------------
            # --- END OF MODIFIED JITTER & STYLING BLOCK ---
            # ------------------------------------------------

            #  PACKAGE DATA FOR THIS FOLD 
            train_df = pd.DataFrame({
                "Fold": fold_num,
                "Point_Type": "Train",
                "Actual_Lon": lon[train_idx],
                "Actual_Lat": lat[train_idx],
                "Jittered_Lon": lon[train_idx] + train_jitter_lon,
                "Jittered_Lat": lat[train_idx] + train_jitter_lat,
                "Predicted_Lon": np.nan,
                "Predicted_Lat": np.nan,
                "Jittered_Predicted_Lon": np.nan,
                "Jittered_Predicted_Lat": np.nan,
                "Fold_Error_km": fold_haversine,
                "Fold_Skill_Score": fold_skill
            })
            
            test_df = pd.DataFrame({
                "Fold": fold_num,
                "Point_Type": "Test",
                "Actual_Lon": test_lon,
                "Actual_Lat": test_lat,
                "Jittered_Lon": test_lon + test_j_lon,
                "Jittered_Lat": test_lat + test_j_lat,
                "Predicted_Lon": pred_lon,
                "Predicted_Lat": pred_lat,
                "Jittered_Predicted_Lon": pred_lon + pred_j_lon,
                "Jittered_Predicted_Lat": pred_lat + pred_j_lat,
                "Fold_Error_km": fold_haversine,
                "Fold_Skill_Score": fold_skill
            })
            
            all_fold_data.extend([train_df, test_df])

            ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.3, color='gray')
            ax.set_axisbelow(True)
            ax.set_title(
                f"Fold {fold_num} | Error: {fold_haversine:.0f} km (Skill Score: {fold_skill:.2f})\n"
                f"Train: {len(train_idx)} | Test: {len(val_idx)}",
                fontsize=10, fontweight='bold'
            )
            if col > 0: ax.set_ylabel('')
            else: ax.set_ylabel("Latitude (°)", fontsize=8)
            if row < grid_rows - 1: ax.set_xlabel('')
            else: ax.set_xlabel("Longitude (°)", fontsize=8)
            ax.tick_params(labelsize=7)

    mean_haversine = np.mean(all_haversine)
    mean_skill = np.mean(all_skill)

    fig.suptitle(
        f"{data_name.upper()} — All {n_folds} Folds: Nested K-Fold CV ({model_name})\n"
        f"Overall Mean Error: {mean_haversine:.0f} km (Skill Score: {mean_skill:.2f})",
        fontsize=16, fontweight='bold', y=0.995
    )

    legend_elements = [
        Patch(facecolor='dodgerblue', edgecolor='navy', label='Train'),
        Patch(facecolor='red', edgecolor='darkred', label='Test Actual'),
        Patch(facecolor='orange', edgecolor='black', label='Test Predicted')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.95)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, f"{data_name}_{model_name}_kfold_cv_folds_withSkill.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # EXPORT CSV FILE 
    if all_fold_data:
        csv_subfolder = os.path.join(output_path, "fold_image_data")
        os.makedirs(csv_subfolder, exist_ok=True)

        final_df = pd.concat(all_fold_data, ignore_index=True)
        csv_save_path = os.path.join(csv_subfolder, f"{data_name}_{model_name}_kfold_cv_folds_data.csv")
        final_df.to_csv(csv_save_path, index=False)
        if verbose:
            print(f"  [i] Saved plot coordinates and metrics to: {csv_save_path}")

# ----------------------------------------------
# Grid dims helper
# ----------------------------------------------

def _calculate_grid_dims(n_folds: int) -> Tuple[int, int]:
    """
    Calculate fitting subplot grid so we have a nice plot. 
    """
    sqrt_n = int(np.sqrt(n_folds))
    for rows in range(sqrt_n, n_folds + 1):
        cols = int(np.ceil(n_folds / rows))
        if rows <= cols:
            return rows, cols
    return sqrt_n, sqrt_n

# ----------------------------------------------
# Dispatcher and run_all
# ----------------------------------------------

def evaluate(
    model,
    model_name: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    param_grid: dict,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List, dict]:
    """
    Execute the kfold evaluation
    """

    if verbose:
        print(f"  [nested_kfold_cv]  model={model_name}  tuning parameters...")

    return kfold_cv(model, X, y, model_name, param_grid, n_splits=n_splits, random_state=random_state)


def run_all(
    input_path: str,
    output_path: str,
    table_name: str,
    data_type: Literal["ICPMS", "XRF"],
    param_grids: Dict[str, dict],
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List]]:
    """
    Main dispatcher function that orchestrates the entire predictive modeling pipeline. 
    - Sequentially loads the data, initializes the machine learning models (Random Forest, XGBoost, SVM)
    - Passes each model through the nested spatially-stratified K-Fold cross-validation engine alongside its respective hyperparameter grid 
    - concatenates and exports the performance metrics and optimal parameters to CSV files,
        while retaining the fitted fold models in memory for subsequent visualization.
    """
    

    X, y = load_data(input_path, data_type)
    models = get_models(X.columns.tolist(), random_state)

    all_summaries = []
    all_optimal_params = []
    fold_models_by_model = {} 

    for model_name, model in models.items():
        current_grid = param_grids.get(model_name, {})

        summary, optimal_params, fold_models, best_params = evaluate(
            model, model_name,
            X=X, y=y,
            param_grid=current_grid,
            n_splits=n_splits,
            random_state=random_state,
            verbose=verbose,
        )
        all_summaries.append(summary)
        all_optimal_params.append(optimal_params)
        fold_models_by_model[model_name] = fold_models

    results = pd.concat(all_summaries, ignore_index=True)
    optimal_params_results = pd.concat(all_optimal_params, ignore_index=True)

    filtered_results = results[
        (results["target"] == "2D_combined") &
        (results["metric"].isin(["r2", "rmse", "haversine_distance", "haversine_skill_score"]))
    ].reset_index(drop=True)

    os.makedirs(output_path, exist_ok=True)

    filtered_results.to_csv(os.path.join(output_path, table_name + ".csv"), index=False)
    filtered_results[filtered_results["metric"] == "haversine_skill_score"].reset_index(drop=True).to_csv(
        os.path.join(output_path, table_name + "_summary.csv"), index=False
    )
    optimal_params_results.to_csv(
        os.path.join(output_path, table_name + "_optimal_parameters.csv"), index=False
    )

    return filtered_results, fold_models_by_model



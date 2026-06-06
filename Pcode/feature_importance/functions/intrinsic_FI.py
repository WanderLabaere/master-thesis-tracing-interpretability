"""
intrinsic_FI.py
----------------------------
Functions script to calculate, compare, and visualize native tree-based feature importance metrics 
alongside model-agnostic Permutation Importance for Random Forest and XGBoost spatial regression models.

Input data:
1. Preprocessed multi-element CSV datasets (ICPMS or XRF) containing numerical trace features, 
     metadata, and (Longitude/Latitude).
2. Existing optimal hyperparameter logs (CSV) to put into the models.

Generates and saves:
1. Tabular intrinsic importance logs (CSV) for Random Forest (MDI and MDA ranks) 
    and XGBoost (Gain, Weight, Cover, and MDA scores).
2. Random Forest comparative scatter plots (PNG) mapping Mean Decrease Impurity (MDI) 
    rank against Mean Decrease Accuracy (MDA) rank to identify metric alignment.
3. XGBoost normalized horizontal bar charts (PNG) visually comparing relative feature 
    impact across internal Gain, Weight, and Cover metrics.
"""

import os
import ast
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Forces background rendering to prevent Tkinter threading crashes
import matplotlib.pyplot as plt
from typing import Literal, Tuple, Dict

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

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

    overlap = set(X.columns) & set(y.columns)
    if overlap:
        raise ValueError(f"X and y share columns: {overlap}.")
    
    return X, y

def get_models(random_state: int = 42) -> Dict[str, object]:
    def log_with_epsilon(X):
        epsilon = 1e-6
        X_arr = np.array(X, dtype=float)
        X_arr = np.clip(X_arr, 0.0, None)
        return np.log(X_arr + epsilon)
    
    log_transformer = FunctionTransformer(log_with_epsilon, validate=False)
    
    # Only RF and XGB natively support these tree-based metrics
    return {
        "RF": Pipeline([
            ('log', log_transformer),
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state))
        ]),
        "XGB": Pipeline([
            ('log', log_transformer),
            ('scaler', StandardScaler()),
            ('model', XGBRegressor(n_estimators=200, learning_rate=0.05, n_jobs=-1, random_state=random_state, verbosity=0)) 
        ])
    }

def get_best_params(csv_path: str, model_name: str) -> dict:
    if not os.path.exists(csv_path):
        return {}
    
    # Map back to pipeline names for params check
    search_name = "RandomForest" if model_name == "RF" else "XGBoost"
    df = pd.read_csv(csv_path)
    row = df[df['model'] == search_name]
    
    if not row.empty:
        param_str = row['optimal_parameters'].values[0]
        try:
            clean_param_str = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', param_str)
            return ast.literal_eval(clean_param_str)
        except Exception:
            return {}
    return {}

# ----------------------------------------------
# Custom Scorer & Kmeans (From previous script)
# ----------------------------------------------

def _calculate_haversine_array(y_true_arr, y_pred_arr):
    lon1, lat1 = np.radians(y_true_arr[:, 0]), np.radians(y_true_arr[:, 1])
    lon2, lat2 = np.radians(y_pred_arr[:, 0]), np.radians(y_pred_arr[:, 1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0 
    return c * r

def _haversine_skill_score(y_true, y_pred, train_centroid: np.ndarray):
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    model_distances = _calculate_haversine_array(y_true_arr, y_pred_arr)
    
    y_naive_arr = np.full_like(y_true_arr, train_centroid)
    naive_distances = _calculate_haversine_array(y_true_arr, y_naive_arr)
    
    sum_naive = np.sum(naive_distances)
    if sum_naive == 0: return 0.0 
    return 1 - (np.sum(model_distances) / sum_naive)

def _get_safe_spatial_clusters(coords_df: pd.DataFrame, n_splits: int, target_clusters: int = 5, random_state: int = 42):
    coords = coords_df[["Latitude", "Longitude"]]
    max_possible_clusters = max(1, len(coords) // n_splits)
    current_target = min(target_clusters, max_possible_clusters)
    
    for k in range(current_target, 0, -1):
        if k == 1: return np.zeros(len(coords), dtype=int)
        clusters = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(coords)
        if np.min(np.bincount(clusters)) >= n_splits:
            return clusters
    return np.zeros(len(coords), dtype=int)

# ----------------------------------------------
# Native Importance Extraction
# ----------------------------------------------

def _extract_rf_importance(model, X_val, y_val, train_centroid, random_state) -> pd.DataFrame:
    """ 
    Extracts MDI (from model) and MDA (via Permutation) for Random Forest 
    """
    rf_estimator = model.named_steps['model']
    mdi = rf_estimator.feature_importances_
    
    # Calculate MDA using custom scorer
    baseline_score = _haversine_skill_score(y_val, model.predict(X_val), train_centroid)
    rng = np.random.default_rng(random_state)
    n_repeats = 10  # Set to 5 for speed, can adjust
    
    mda_scores = []
    features = X_val.columns
    
    for col in features:
        col_scores = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[col] = X_perm[col].values[perm_idx]
            score = _haversine_skill_score(y_val, model.predict(X_perm), train_centroid)
            col_scores.append(baseline_score - score)
        mda_scores.append(np.mean(col_scores))
        
    return pd.DataFrame({"Feature": features, "MDI": mdi, "MDA": mda_scores})

# AFTER
def _extract_xgb_importance(model, X_val, y_val, train_centroid, random_state) -> pd.DataFrame:
    """
    Extracts native Gain, Weight, Cover from XGBoost booster (training-based),
    AND permutation-based MDA on validation data using the Haversine skill scorer.
    This mirrors the RF approach so both models are evaluated on held-out data.
    """
    features = X_val.columns
    xgb_estimator = model.named_steps['model']
    booster = xgb_estimator.get_booster()
    booster.feature_names = list(features)

    gain = booster.get_score(importance_type='gain')
    weight = booster.get_score(importance_type='weight')
    cover = booster.get_score(importance_type='cover')

    df = pd.DataFrame({"Feature": features})
    df["Gain"] = df["Feature"].map(gain).fillna(0)
    df["Weight"] = df["Feature"].map(weight).fillna(0)
    df["Cover"] = df["Feature"].map(cover).fillna(0)

    # Permutation MDA on validation set (same approach as RF)
    baseline_score = _haversine_skill_score(y_val, model.predict(X_val), train_centroid)
    rng = np.random.default_rng(random_state)
    n_repeats = 10

    mda_scores = []
    for col in features:
        col_scores = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[col] = X_perm[col].values[perm_idx]
            score = _haversine_skill_score(y_val, model.predict(X_perm), train_centroid)
            col_scores.append(baseline_score - score)
        mda_scores.append(np.mean(col_scores))

    df["MDA"] = mda_scores
    return df

# ----------------------------------------------
# Plotting Functions
# ----------------------------------------------

def plot_rf_scatter(df: pd.DataFrame, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(df['MDA_Rank'], df['MDI_Rank'], color='darkorange', s=100, edgecolors='k', zorder=3)
    
    for i, row in df.iterrows():
        ax.annotate(row['Feature'], 
                    (row['MDA_Rank'], row['MDI_Rank']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', fontsize=9)

    # Invert axes so Rank 1 (Best) is at the Top-Right
    ax.set_xlim(df['MDA_Rank'].max() + 1, 0)
    ax.set_ylim(df['MDI_Rank'].max() + 1, 0)
    
    # Diagonal line for perfect correlation
    lims = [0, max(df['MDA_Rank'].max(), df['MDI_Rank'].max()) + 1]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=1)

    ax.set_xlabel("Mean Decrease Accuracy (MDA) Rank", fontsize=12)
    ax.set_ylabel("Mean Decrease Impurity (MDI/MDG) Rank", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_xgb_bars(df: pd.DataFrame, out_path: str, title: str):
    df_sorted = df.sort_values('Gain', ascending=True).copy()
    
    # Normalize to 0-1 range relative to the max of each metric for visual comparison
    for col in ['Gain', 'Weight', 'Cover']:
        max_val = df_sorted[col].max()
        if max_val > 0:
            df_sorted[f'{col}_Norm'] = df_sorted[col] / max_val
        else:
            df_sorted[f'{col}_Norm'] = 0

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
    
    y = np.arange(len(df_sorted))
    height = 0.25
    
    ax.barh(y - height, df_sorted['Gain_Norm'], height, label='Gain (Normalized)', color='#1f77b4')
    ax.barh(y, df_sorted['Weight_Norm'], height, label='Weight (Normalized)', color='#ff7f0e')
    ax.barh(y + height, df_sorted['Cover_Norm'], height, label='Cover (Normalized)', color='#2ca02c')
    
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted['Feature'])
    ax.set_xlabel("Normalized Relative Importance (0 to 1 scale)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

# ----------------------------------------------
# Main Execution
# ----------------------------------------------

def run_native_feature_importance(
    data_name: str, 
    csv_path: str, 
    data_type: str, 
    params_file: str, 
    base_out_dir: str, 
    subfolder_path: str, # expected to be like 'sI' or 'tX/tX_Full_All_Genera'
    random_state: int = 42,
    n_splits: int = 5,
    n_clusters: int = 10
):
    X, y = load_data(csv_path, data_type)
    models = get_models(random_state)
    
    n_samples = len(y)
    effective_n_clusters = min(n_clusters, n_samples)
    effective_n_splits = min(n_splits, n_samples)

    if effective_n_splits < 2:
        print(f"  [!] Skipping {data_name}: not enough samples.")
        return

    spatial_clusters = _get_safe_spatial_clusters(y, effective_n_splits, effective_n_clusters, random_state)
    outer_cv = StratifiedKFold(n_splits=effective_n_splits, shuffle=True, random_state=random_state)

    for model_name, model_obj in models.items():
        print(f"  -> Extracting Native FI for {model_name}...")
        
        # Build specific output directories
        # Target: native_FI / {RF, XGB} / {sI, tX...} / intrinsic_FI_plots & intrinsic_FI_data
        model_out_dir = os.path.join(base_out_dir, model_name, subfolder_path)
        plot_dir = os.path.join(model_out_dir, "intrinsic_FI_plots")
        data_dir = os.path.join(model_out_dir, "intrinsic_FI_data")
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        tuned_params = get_best_params(params_file, model_name)

        if tuned_params:
            print(f"    [+] Tuned hyperparameters found and applied for {model_name}:")
            for param_name, param_val in tuned_params.items():
                print(f"        - {param_name}: {param_val}")
        else:
            print(f"    [-] Running {model_name} with default parameters.")

        fold_importances = []

        for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, spatial_clusters)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_centroid = np.array([y_train["Longitude"].mean(), y_train["Latitude"].mean()])
            
            model = clone(model_obj)
            if tuned_params:
                model.set_params(**tuned_params)
                
            model.fit(X_train, y_train)
            
            if model_name == "RF":
                imp_df = _extract_rf_importance(model, X_val, y_val, train_centroid, random_state + fold)
            elif model_name == "XGB":
                imp_df = _extract_xgb_importance(model, X_val, y_val, train_centroid, random_state + fold)

            
            fold_importances.append(imp_df)

        # Aggregate Data
        all_folds_df = pd.concat(fold_importances)
        
        if model_name == "RF":
            final_df = all_folds_df.groupby("Feature")[["MDI", "MDA"]].mean().reset_index()
            # Rank descending (1 = highest value)
            final_df['MDI_Rank'] = final_df['MDI'].rank(ascending=False, method='min')
            final_df['MDA_Rank'] = final_df['MDA'].rank(ascending=False, method='min')
            
            csv_out = os.path.join(data_dir, f"{data_name}_RF_native_FI.csv")
            final_df.to_csv(csv_out, index=False)
            
            plot_out = os.path.join(plot_dir, f"{data_name}_RF_native_FI.png")
            plot_rf_scatter(final_df, plot_out, f"Random Forest: MDI vs MDA Ranks ({data_name})")
            
        elif model_name == "XGB":
            final_df = all_folds_df.groupby("Feature")[["Gain", "Weight", "Cover", "MDA"]].mean().reset_index()

            csv_out = os.path.join(data_dir, f"{data_name}_XGB_native_FI.csv")
            final_df.to_csv(csv_out, index=False)

            plot_out = os.path.join(plot_dir, f"{data_name}_XGB_native_FI.png")
            plot_xgb_bars(final_df, plot_out, f"XGBoost Native Metrics Ranked by Gain ({data_name})")

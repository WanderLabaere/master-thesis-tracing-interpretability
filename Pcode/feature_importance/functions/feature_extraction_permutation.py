"""
feature_extraction_permutation.py
----------------------------
function script to calculate and rank feature importance for multi-output spatial regression models
using spatially-stratified K-Fold cross-validation. The pipeline supports both standard PI 
and CPI to account for feature collinearity, evaluating predictive 
loss via  custom Haversine Skill Score.

Input data:
1. Preprocessed multi-element CSV datasets (ICPMS or XRF) containing numerical trace features, 
    categorical metadata (e.g., One-Hot Encoded Genus/Species), and spatial coordinate targets (Longitude/Latitude).
2. Precomputed optimal hyperparameter logs (CSV) for configuring the evaluated machine learning 
    algorithms (Random Forest, XGBoost, SVM).

Generates and saves:
1. Tabular feature importance logs (CSV) aggregating the mean predictive importance (Haversine Skill Score drop) 
    and cross-fold standard deviation for every feature or feature group.
2. Horizontal bar charts (PNG) visually ranking the features by importance, with error bars 
    to illustrate variance across the spatial cross-validation folds.
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Forces background rendering to prevent Tkinter threading crashes
import matplotlib.pyplot as plt

from typing import Literal, Tuple, Dict
from sklearn.model_selection import KFold # Swapped train_test_split for KFold
from sklearn.base import clone
from sklearn.linear_model import Ridge

# Import models from your existing pipeline logic
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# to transform ridge regression data
from sklearn.pipeline import make_pipeline

# Stratified Kfold
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# To load continuous hyperparameters
import re

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

    # safety check
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

# updated to handle continuous grid results:
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
        try:
            # FIX: Remove np.float64(), np.int64(), etc. using regex
            # This turns "np.float64(0.96)" into "0.96" so ast.literal_eval can read it.
            clean_param_str = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', param_str)
            
            return ast.literal_eval(clean_param_str)
        except Exception as e:
            print(f"  [!] Error parsing parameters for {model_name}: {e}")
            print(f"  [!] Problematic string: {param_str}") # Helpful for debugging future errors
            return {}
    else:
        print(f"  [!] No tuned parameters found for {model_name}. Using defaults.")
        return {}

# ----------------------------------------------
# Custom Scorer (Haversine Skill Score)
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

def _mean_squared_haversine_loss(y_true, y_pred):
    # Convert pandas DataFrames/Series to numpy arrays to allow [:, 0] slicing
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    distances = _calculate_haversine_array(y_true_arr, y_pred_arr) 
    return np.mean(distances ** 2)

def _haversine_skill_score(y_true, y_pred, train_centroid: np.ndarray):
    """
    Compute the Haversine skill score.

    Parameters
    ----------
    y_true : array-like, shape (n, 2)
        True [Longitude, Latitude] values for the test fold.
    y_pred : array-like, shape (n, 2)
        Predicted [Longitude, Latitude] values.
    train_centroid : np.ndarray, shape (2,)
        Fixed [mean_Longitude, mean_Latitude] computed from the *training* fold.
        Using the training centroid keeps the naive baseline stable and
        comparable across folds.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    model_distances = _calculate_haversine_array(y_true_arr, y_pred_arr)
    
    # Use the training centroid as the naive baseline — NOT the test centroid.
    y_naive_arr = np.full_like(y_true_arr, train_centroid)
    naive_distances = _calculate_haversine_array(y_true_arr, y_naive_arr)
    
    sum_naive = np.sum(naive_distances)
    if sum_naive == 0: return 0.0 
        
    return 1 - (np.sum(model_distances) / sum_naive)

# ----------------------------------------------
# Preprocessing helper (mirrors the main pipeline)
# ----------------------------------------------

def _apply_selective_log(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies log(x + epsilon) strictly to trace element columns.
    Leaves one hot encoded columns (Species_, Genus_) untouched.
    Returns a pandas DataFrame to maintain column indexing before extracting .values.
    """
    X_out = X_df.copy()
    ohe_cols = [c for c in X_out.columns if c.startswith(("Species_", "Genus_"))]
    trace_cols = [c for c in X_out.columns if c not in ohe_cols]
    
    epsilon = 1e-6
    if trace_cols:
        X_trace = np.clip(X_out[trace_cols].values.astype(float), 0.0, None)
        X_out[trace_cols] = np.log(X_trace + epsilon)
        
    return X_out

# ----------------------------------------------
# Group Detection Helper
# ----------------------------------------------
def _get_feature_groups(columns):
    groups = {}
    for col in columns:
        if col.startswith("Species_"):
            groups.setdefault("Species", []).append(col)
        elif col.startswith("Genus_"):
            groups.setdefault("Genus", []).append(col)
        else:
            groups[col] = [col]
    return groups

# ----------------------------------------------
# Dynamic Kmeans Helper
# ----------------------------------------------
def _get_safe_spatial_clusters(coords_df: pd.DataFrame, n_splits: int, target_clusters: int = 5, random_state: int = 42, loop_name: str = "Outer"):
    """
    Dynamically reduces the number of K-Means clusters until every cluster 
    contains at least 'n_splits' members, preventing StratifiedKFold crashes.
    """
    coords = coords_df[["Latitude", "Longitude"]]
    
    # Mathematical ceiling: You can't have more clusters than (Total Samples / n_splits)
    max_possible_clusters = max(1, len(coords) // n_splits)
    current_target = min(target_clusters, max_possible_clusters)
    
    for k in range(current_target, 0, -1):
        if k == 1:
            print(f"  [!] Warning ({loop_name}): Extreme spatial imbalance. Falling back to k=1 (standard random split).")
            return np.zeros(len(coords), dtype=int)
            
        clusters = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(coords)
        
        # Check the size of the smallest cluster
        smallest_cluster_size = np.min(np.bincount(clusters))
        
        # If the smallest cluster is mathematically safe for the folds, we found our k!
        if smallest_cluster_size >= n_splits:
            if k == target_clusters:
                print(f"  [i] {loop_name} CV: Optimal k={k} clusters used for {n_splits} splits (Smallest region: {smallest_cluster_size} samples).")
            else:
                print(f"  [!] {loop_name} CV: Adjusted to k={k} clusters for {n_splits} splits (Smallest region: {smallest_cluster_size} samples).")
            return clusters
            
    return np.zeros(len(coords), dtype=int)

# ----------------------------------------------
# Feature Importance Methods
# These are the main function that calculate the PI and CPI
# ----------------------------------------------

def _permutation_importance_method(model, X_test, y_test, train_centroid, random_state) -> pd.DataFrame:
    baseline_score = _haversine_skill_score(y_test, model.predict(X_test), train_centroid)
    rng = np.random.default_rng(random_state)
    n_repeats = 10
    
    # here we look at the genus importance
    groups = _get_feature_groups(X_test.columns)
    importances_mean, importances_std, feature_names = [], [], []
    
    for group_name, cols in groups.items():
        scores = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[cols] = X_perm[cols].values[perm_idx, :]
            
            score = _haversine_skill_score(y_test, model.predict(X_perm), train_centroid)
            scores.append(baseline_score - score)
            
        feature_names.append(group_name)
        importances_mean.append(np.mean(scores))
        importances_std.append(np.std(scores))
        
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance_Mean": importances_mean, "Importance_Std": importances_std})
    return df_imp

def _conditional_permutation_importance_method(model, X_train, X_test, y_test, train_centroid, random_state) -> pd.DataFrame:
    baseline_score = _haversine_skill_score(y_test, model.predict(X_test), train_centroid)
    rng = np.random.default_rng(random_state)
    n_repeats = 10
    
    # Leave out the Genus variables, is meaningless for CPI. 
    groups = {k: v for k, v in _get_feature_groups(X_test.columns).items() 
          if k not in ["Species", "Genus"]}
    importances_mean, importances_std, feature_names = [], [], []

    # Use the selective log transform. OHE columns remain binary (0/1).
    X_train_trans = _apply_selective_log(X_train).values
    X_test_trans  = _apply_selective_log(X_test).values
    col_index   = {col: i for i, col in enumerate(X_train.columns)}
    
    for group_name, cols in groups.items():
        # treat variable subclasses as 1 big variable:
        is_ohe_group = group_name in ["Species", "Genus"]
        
        # Create index for currently handled feature & all the other features
        col_idx      = [col_index[c] for c in cols]
        other_idx    = [i for i in range(X_train_trans.shape[1]) if i not in col_idx]

        # split data to train ridge model, selecting only the features that we're not shuffling
        X_train_other_trans  = X_train_trans[:, other_idx]
        X_test_other_trans   = X_test_trans[:, other_idx]
        target_feats_train   = X_train_trans[:, col_idx]

        # Standardize the data, and fit the ridge model
        cond_model = make_pipeline(StandardScaler(), Ridge())
        cond_model.fit(X_train_other_trans, target_feats_train)

        # Predict the selected element based on all other variables 
        # We do this based on the test data since feature importance 
        #   is also calculated based on the test data
        pred_feats_trans = cond_model.predict(X_test_other_trans)
        # reshape so we can extract
        if pred_feats_trans.ndim == 1:
            pred_feats_trans = pred_feats_trans.reshape(-1, 1)

        # calculate the differenc between the actual feature value and the 
        # predicted value based on the other features
        residuals_trans = X_test_trans[:, col_idx] - pred_feats_trans
        
        scores = []
        # 10 REPEATS OF DIFFERENT SHUFFLING
        for _ in range(n_repeats):
            perm_idx = rng.permutation(len(residuals_trans))
            permuted_residuals_trans = residuals_trans[perm_idx, :]

            permuted_trans = pred_feats_trans + permuted_residuals_trans
            
            # No need to split between numerical and OHE variables.
            permuted_raw = np.exp(permuted_trans) - 1e-6
            permuted_raw = np.clip(permuted_raw, 0.0, None)

            X_perm = X_test.copy()
            X_perm[cols] = permuted_raw
            
            score = _haversine_skill_score(y_test, model.predict(X_perm), train_centroid)
            scores.append(baseline_score - score)
            
        feature_names.append(group_name)
        importances_mean.append(np.mean(scores))
        importances_std.append(np.std(scores))
        
    return pd.DataFrame({"Feature": feature_names, "Importance_Mean": importances_mean, "Importance_Std": importances_std})




def extract_feature_importance(
    method: Literal["permutation", "conditional_permutation", "sobol_cpi"],
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    train_centroid: np.ndarray,
    random_state: int = 42
) -> pd.DataFrame:
    if method == "permutation":
        return _permutation_importance_method(model, X_test, y_test, train_centroid, random_state)
    elif method == "conditional_permutation":
        return _conditional_permutation_importance_method(model, X_train, X_test, y_test, train_centroid, random_state)
    else:
        raise ValueError(f"Method '{method}' not supported.")

# ----------------------------------------------
# Main Pipeline Execution
# ----------------------------------------------

def run_feature_importance(
    data_name: str, 
    csv_path: str, 
    data_type: str, 
    params_file: str, 
    output_dir: str, 
    method: Literal["permutation", "conditional_permutation", "sobol_cpi"] = "permutation",
    random_state: int = 42,
    n_splits: int = 5,
    n_clusters: int = 10
):
    """
    This function bundles all above functions. 
    Cross validation is performed in this function, wrapping around all other functions. 
    """
    print(f"\n{'='*60}")
    print(f"EXTRACTING FEATURES (CV): {data_name.upper()} ({data_type}) - Method: {method.upper()}")
    print(f"{'='*60}")

    X, y = load_data(csv_path, data_type)
    models = get_models(random_state)
    os.makedirs(output_dir, exist_ok=True)

    # Clamp n_clusters and n_splits to the number of available samples
    n_samples = len(y)
    effective_n_clusters = min(n_clusters, n_samples)
    effective_n_splits = min(n_splits, n_samples)

    if effective_n_splits < 2:
        print(f"  [!] Skipping {data_name}: not enough samples ({n_samples}) for CV.")
        return

    # Not needed anymore
    # if effective_n_clusters != n_clusters:
    #     print(f"  [!] n_clusters reduced from {n_clusters} to {effective_n_clusters} (only {n_samples} samples).")
    # if effective_n_splits != n_splits:
    #     print(f"  [!] n_splits reduced from {n_splits} to {effective_n_splits} (only {n_samples} samples).")

    # Unnecessary since we have safe Kmeans cluster function
    # print(f"  -> Clustering coordinates for spatial stratification "
    #       f"(target n_clusters={effective_n_clusters}, n_splits={effective_n_splits})...")
    
    spatial_clusters = _get_safe_spatial_clusters(
        coords_df=y,
        n_splits=effective_n_splits,
        target_clusters=effective_n_clusters,
        random_state=random_state,
        loop_name="Outer"
    )

    # Create stratifiedKFold object to use to split the data in for loop
    outer_cv = StratifiedKFold(n_splits=effective_n_splits, shuffle=True, random_state=random_state)

    # Loop through models
    for model_name, model_obj in models.items():
        print(f"  -> Processing {model_name} with {effective_n_splits}-Fold Spatial CV...")
        
        tuned_params = get_best_params(params_file, model_name)

        # Safetiy check if hyperparameters are loaded in right
        if tuned_params:
            print(f"     [+] Tuned hyperparameters found and applied for {model_name}:")
            for param_name, param_val in tuned_params.items():
                print(f"         - {param_name}: {param_val}")
        else:
            print(f"     [-] Running {model_name} with default parameters.")

        fold_importances = []

        # Loop trough spatially balanced folds
        for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, spatial_clusters)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Compute centroid from the training fold — used as the naive baseline
            train_centroid = np.array([
                y_train["Longitude"].mean(),
                y_train["Latitude"].mean()
            ])
            
            model = clone(model_obj)
            if tuned_params:
                model.set_params(**tuned_params)
                
            model.fit(X_train, y_train)
            
            imp_df = extract_feature_importance(
                method=method, 
                model=model, 
                X_train=X_train, 
                X_test=X_val, 
                y_test=y_val, 
                train_centroid=train_centroid, 
                random_state=random_state + fold
            )
            fold_importances.append(imp_df)

        # Aggregate across folds: mean and cross-fold std of the per-fold importance means
        all_folds_df = pd.concat(fold_importances)
        final_imp_df = (
            all_folds_df
            .groupby("Feature")["Importance_Mean"]
            .agg(['mean', 'std'])
            .reset_index()
            .rename(columns={'mean': 'Importance_Mean', 'std': 'Importance_Std_CrossFold'})
        )
        final_imp_df['Importance_Std_CrossFold'] = final_imp_df['Importance_Std_CrossFold'].fillna(0.0)

        # --- SAVE NUMERICAL RESULTS ---
        base_filename = f"{data_name}_{model_name}_{method}_importance"
        csv_out_path = os.path.join(output_dir, f"{base_filename}.csv")
        final_imp_df.to_csv(csv_out_path, index=False)

        # --- PLOTTING LOGIC ---
        plot_df = final_imp_df.sort_values(by="Importance_Mean", ascending=True)
        
        fig_height = max(4, min(12, len(plot_df) * 0.3))
        fig, ax = plt.subplots(figsize=(8, fig_height))
        
        ax.barh(
            plot_df["Feature"], 
            plot_df["Importance_Mean"], 
            xerr=plot_df["Importance_Std_CrossFold"], 
            color="skyblue", 
            edgecolor="black",
            capsize=3
        )
        
        # FIXED: Dynamic X-axis label
        if False: #method == "sobol_cpi":
            x_label = "Importance (Expected Increase in Squared Haversine Loss)"
        else:
            x_label = "Importance (Haversine Skill Score Drop)"
            
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel("Feature", fontsize=10)

        ax.set_title(
            f"Feature Importance ({method})\n"
            f"{data_name} - {model_name} ({effective_n_splits}-Fold Spatial CV)",
            fontsize=12, fontweight="bold"
        )
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        
        plt.tight_layout()
        
        png_out_path = os.path.join(output_dir, f"{base_filename}.png")
        plt.savefig(png_out_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Plots and CSVs saved to {output_dir}")
"""
cumulative_importance_thresholding.py
----------------------------
Functions script to evaluate cumulative feature performance using spatially-stratified K-Fold cross-validation. Features are iteratively added based on precomputed importance rankings to track marginal spatial predictive gains and establish minimal optimal feature subsets.

Input data:
1. Preprocessed multi-element CSV datasets (ICPMS or XRF) containing numerical trace features 
    and spatial targets (Longitude/Latitude).
2. Precomputed feature importance ranking logs (CSV).
3. Optimal hyperparameter logs (CSV) to configure the evaluated machine learning algorithms (Random Forest, XGBoost, SVM).

Generates and saves:
1. Cumulative evaluation logs (CSV) tracking subset predictive scores (Haversine Skill Score) 
    and cross-fold variance at each feature addition step.
2. Performance trajectory curve plots (PNG) illustrating the model's spatial skill progression against defined baselines.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# New imports for Spatial CV
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
        raise ValueError(f"X and y share columns: {overlap}. Check column indexing in load_data.")

    print(f"  -> Loaded {len(X)} samples, {X.shape[1]} features.")
    return X, y

def get_models(random_state: int = 42) -> Dict[str, object]:
    def log_with_epsilon(X):
        epsilon = 1e-6
        X_arr = np.array(X, dtype=float) 
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
        return {}
    
    df = pd.read_csv(csv_path)
    row = df[df['model'] == model_name]
    
    if not row.empty:
        param_str = row['optimal_parameters'].values[0]
        
        # FIX: Strip numpy wrapper functions before evaluating to prevent ast crashes
        param_str = re.sub(r'np\.[a-zA-Z0-9_]+\((.*?)\)', r'\1', param_str)
        
        try:
            return ast.literal_eval(param_str)
        except Exception as e:
            print(f"  [!] Error parsing parameters for {model_name}: {e}")
            return {}
    return {}

# ----------------------------------------------
# Scoring & Preprocessing Helpers
# ----------------------------------------------

def _calculate_haversine_array(y_true_arr, y_pred_arr):
    lon1, lat1 = np.radians(y_true_arr[:, 0]), np.radians(y_true_arr[:, 1])
    lon2, lat2 = np.radians(y_pred_arr[:, 0]), np.radians(y_pred_arr[:, 1])
    a = np.sin((lat2 - lat1) / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371.0 

def _haversine_skill_score(y_true, y_pred, train_centroid: np.ndarray):
    y_true_arr = np.asarray(y_true)
    model_distances = _calculate_haversine_array(y_true_arr, np.asarray(y_pred))
    naive_distances = _calculate_haversine_array(y_true_arr, np.full_like(y_true_arr, train_centroid))
    sum_naive = np.sum(naive_distances)
    if sum_naive == 0: return 0.0 
    return 1 - (np.sum(model_distances) / sum_naive)

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
# Cumulative Thresholding Pipeline (Spatial CV)
# ----------------------------------------------

def run_cumulative_thresholding(
    data_name: str, 
    csv_path: str, 
    data_type: str, 
    params_file: str, 
    precomputed_imp_dir: str,
    output_dir: str, 
    method: Literal["permutation", "conditional_permutation"] = "permutation",
    random_state: int = 42,
    n_splits: int = 5,
    n_clusters: int = 10,
    stop_at_threshold: bool = True # <-- NEW TOGGLE
):
    """
    Function that iteratively adds features to the total feature set for the given model.
    Model performance evaluated via Kfold stratification on validation set.
    Generates a plot that shows how the HSS evolves when adding features.
    """
    mode_str = "95_Threshold" if stop_at_threshold else "Full_Feature_Set"
    print(f"\n{'='*70}")
    print(f"CUMULATIVE IMPORTANCE ({mode_str.upper()}): {data_name.upper()} | Method: {method}")
    print(f"{'='*70}")

    X, y = load_data(csv_path, data_type)
    os.makedirs(output_dir, exist_ok=True)
    
    n_samples = len(y)
    effective_n_clusters = min(n_clusters, n_samples)
    effective_n_splits = min(n_splits, n_samples)

    if effective_n_splits < 2:
        print(f"  [!] Skipping {data_name}: not enough samples ({n_samples}) for CV.")
        return

    print(f"  -> Clustering coordinates for spatial stratification (n_clusters={effective_n_clusters}, n_splits={effective_n_splits})...")
    spatial_clusters = KMeans(n_clusters=effective_n_clusters, random_state=random_state, n_init=10).fit_predict(y[["Latitude", "Longitude"]])
    outer_cv = StratifiedKFold(n_splits=effective_n_splits, shuffle=True, random_state=random_state)
    
    groups = _get_feature_groups(X.columns)
    models = get_models(random_state)
    
    for model_name, model_obj in models.items():
        print(f"\n -> Processing {model_name}...")
        
        imp_csv_name = f"{data_name}_{model_name}_{method}_importance.csv"
        imp_csv_path = os.path.join(precomputed_imp_dir, imp_csv_name)
        
        if not os.path.exists(imp_csv_path):
            print(f"    [!] Precomputed importance file NOT FOUND: {imp_csv_path}. Skipping.")
            continue
        
        imp_df = pd.read_csv(imp_csv_path)
        
        if "Feature" not in imp_df.columns or "Importance_Mean" not in imp_df.columns:
            print(f"    [!] Precomputed CSV is missing required columns. Skipping.")
            continue
        if imp_df.empty:
            print(f"    [!] Precomputed CSV is empty. Skipping.")
            continue
        
        imp_df = imp_df.sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)
        
        print("\n    [+] Precomputed Feature Ranking:")
        for rank, row in imp_df.iterrows():
            print(f"        {rank + 1:02d}. {row['Feature']:<15} | Importance: {row['Importance_Mean']:.4f}")
        print("-" * 50)
        
        tuned_params = get_best_params(params_file, model_name)

        if tuned_params:
            print(f"     [+] Tuned hyperparameters found and applied for {model_name}:")
            for param_name, param_val in tuned_params.items():
                print(f"         - {param_name}: {param_val}")
        else:
            print(f"     [-] Running {model_name} with default parameters.")
        
        #  Train baseline model with ALL features across CV to establish threshold
        base_scores = []
        for train_idx, val_idx in outer_cv.split(X, spatial_clusters):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_centroid = np.array([y_train["Longitude"].mean(), y_train["Latitude"].mean()])
            
            base_model = clone(model_obj)
            if tuned_params: base_model.set_params(**tuned_params)
            
            base_model.fit(X_train, y_train)
            score = _haversine_skill_score(y_val, base_model.predict(X_val), train_centroid)
            base_scores.append(score)
            
        P_base_mean = np.mean(base_scores)
        
        if P_base_mean <= 0:
            print(f"Baseline model failed to beat naive baseline (Mean CV Score: {P_base_mean:.4f}). Skipping cumulative thresholding.")
            continue
        
        T = 1 * P_base_mean
        print(f"CV Baseline Score (All Features): {P_base_mean:.4f} | Target Threshold (95%): {T:.4f}")

        #  Iteratively build feature set based on precomputed ranking
        selected_groups = []
        selected_cols = []
        history = []
        reached_threshold = False
        
        print("Iteratively adding features (Evaluating via CV):")
        for idx, row in imp_df.iterrows():
            grp = row["Feature"]
            selected_groups.append(grp)
            selected_cols.extend(groups[grp])
            
            subset_scores = []
            for train_idx, val_idx in outer_cv.split(X, spatial_clusters):
                X_train_sub, X_val_sub = X.iloc[train_idx][selected_cols], X.iloc[val_idx][selected_cols]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                train_centroid = np.array([y_train["Longitude"].mean(), y_train["Latitude"].mean()])
                
                subset_model = clone(model_obj)
                if tuned_params: subset_model.set_params(**tuned_params)
                
                subset_model.fit(X_train_sub, y_train)
                score = _haversine_skill_score(y_val, subset_model.predict(X_val_sub), train_centroid)
                subset_scores.append(score)
                
            mean_subset_score = np.mean(subset_scores)
            std_subset_score = np.std(subset_scores)
            
            history.append({
                "Iteration": idx + 1,
                "Added_Feature_Group": grp,
                "Total_Columns": len(selected_cols),
                "Subset_Score_Mean": mean_subset_score,
                "Subset_Score_Std": std_subset_score
            })
            
            print(f"        Step {idx+1:02d} | Added: {grp: <15} | Mean CV Score: {mean_subset_score:.4f} (+/- {std_subset_score:.4f})")
            
            # Only break if stop_at_threshold is True
            if stop_at_threshold and mean_subset_score >= T:
                print(f"    [!] THRESHOLD REACHED at {len(selected_groups)} feature groups ({len(selected_cols)} total columns).")
                reached_threshold = True
                break
                
        if stop_at_threshold and not reached_threshold:
            print("    [-] Iterated through all features but did not hit the 95% threshold exactly.")
        elif not stop_at_threshold:
            print("    [+] Finished evaluating full feature set.")

        #  Export Results & Plot curve
        history_df = pd.DataFrame(history)
        base_filename = f"{data_name}_{model_name}_{method}_cumulative_CV_{mode_str}"
        
        history_df.to_csv(os.path.join(output_dir, f"{base_filename}.csv"), index=False)
        
        # Plotting the threshold journey with standard deviation shading
        plt.figure(figsize=(10, 6)) # Slightly wider to accommodate text
        plt.plot(history_df["Total_Columns"], history_df["Subset_Score_Mean"], marker='o', label="Mean CV Score", color='blue')
        
        # <-- NEW LOGIC: Annotate the added feature name above every dot
        for i, row_data in history_df.iterrows():
            plt.annotate(
                row_data["Added_Feature_Group"],
                (row_data["Total_Columns"], row_data["Subset_Score_Mean"]),
                textcoords="offset points",
                xytext=(0, 15),  # 10 points vertical offset
                ha='center',
                va='bottom',
                rotation=0,     # Rotate vertically to prevent overlaps
                fontsize=9
            )

        # Add shading for variance across folds
        plt.fill_between(
            history_df["Total_Columns"], 
            history_df["Subset_Score_Mean"] - history_df["Subset_Score_Std"], 
            history_df["Subset_Score_Mean"] + history_df["Subset_Score_Std"], 
            color='blue', alpha=0.2, label="+/- 1 Std Dev"
        )
        
        plt.axhline(T, color='red', linestyle='--', label=f"100% of features Threshold ({T:.4f})")
        # plt.axhline(P_base_mean, color='green', linestyle=':', label=f"Mean Baseline Score ({P_base_mean:.4f})")
        
        # Dynamically scale Y-axis to make room for vertical text
        y_min, y_max = plt.ylim()
        plt.ylim(y_min, y_max + (y_max - y_min) * 0.25) # Add 25% padding to the top for the text
        
        plt.xlabel("Total Feature Columns Used", fontsize=10)
        plt.ylabel("Haversine Skill Score", fontsize=10)
        plt.title(f"Cumulative Feature Addition (Spatial CV - {method})\n{data_name} - {model_name} ({mode_str.replace('_', ' ')})")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=150)
        plt.close()
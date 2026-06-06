"""
model_assumptions.py
----------------------------
Utility Functions script to empirically evaluate the assumptions of the CPI conditional model. 
It measures how accurately a regression model (e.g., Ridge, ElasticNet, Random Forest) can 
predict a target feature based on all other features (R-squared), and saves residual correlation to see if these are independent. 

Input data:
1. Preprocessed multi-element CSV datasets (ICPMS or XRF) containing numerical trace features 
    and spatial coordinates for stratified cross-validation mapping.

Generates and saves:
1. Feature-level evaluation logs (CSV) detailing the mean cross-validated R-squared and absolute 
    residual correlation for every individual feature or group.
2. dataset assumption logs (CSV) aggregating the overall predictability and residual 
    independence across the entire dataset.
"""
import os
import numpy as np
import pandas as pd

from typing import Literal, Tuple
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

# ----------------------------------------------
#  Base Setup & Data Loading
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
    return X, y

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
#  R2 & Residual Evaluation Core Logic
# ----------------------------------------------

def evaluate_cpi_model_r2(
    data_name, 
    csv_path, 
    data_type, 
    output_dir, 
    model_type="Ridge", 
    n_splits=5, 
    n_clusters=10, 
    random_state=42
):
    X, y = load_data(csv_path, data_type)
    os.makedirs(output_dir, exist_ok=True)

    # Clamp n_clusters and n_splits to the number of available samples
    n_samples = len(y)
    effective_n_clusters = min(n_clusters, n_samples)
    effective_n_splits = min(n_splits, n_samples)

    if effective_n_splits < 2:
        print(f"  [!] Skipping {data_name}: not enough samples ({n_samples}) for CV.")
        return

    print(f"  -> Clustering coordinates for spatial stratification "
          f"(n_clusters={effective_n_clusters}, n_splits={effective_n_splits})...")
    
    spatial_clusters = KMeans(
        n_clusters=effective_n_clusters, random_state=random_state, n_init=10
    ).fit_predict(y[["Latitude", "Longitude"]])

    outer_cv = StratifiedKFold(n_splits=effective_n_splits, shuffle=True, random_state=random_state)

    metrics_records = []

    # Iterate through folds exactly like the CPI pipeline
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, spatial_clusters)):
        X_val = X.iloc[val_idx]
        groups = _get_feature_groups(X_val.columns)
        
        # For each feature group, fit the dynamic model and calculate R2 & Residuals
        for group_name, cols in groups.items():
            X_other = X_val.drop(columns=cols)
            target_feats = X_val[cols]
            
            # --- DYNAMIC MODEL SELECTION ---
            if model_type == "Ridge":
                cond_model = make_pipeline(StandardScaler(), Ridge(random_state=random_state + fold))
            elif model_type == "ElasticNet":
                cond_model = make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state + fold))
            elif model_type == "RandomForest":
                cond_model = RandomForestRegressor(max_depth=30, min_samples_leaf=5, n_estimators=300, random_state=random_state + fold, n_jobs=-1)
            else:
                raise ValueError(f"Model type '{model_type}' not supported.")
            
            # Fit and predict
            # Use np.ravel when its a single column to satisfy sklearn's 1D array expectation
            y_train_fit = target_feats.values.ravel() if target_feats.shape[1] == 1 else target_feats
            cond_model.fit(X_other, y_train_fit)
            pred_feats = cond_model.predict(X_other)
            
            # Calculate R2 score
            fold_r2 = r2_score(target_feats, pred_feats)
            
            # -----------------------------------------------------
            # RESIDUAL INDEPENDENCE CHECK (PER FEATURE)
            # -----------------------------------------------------
            # Calculate residuals
            pred_feats_df = pd.DataFrame(pred_feats, columns=cols, index=target_feats.index)
            residuals = target_feats - pred_feats_df
            
            group_corrs = []
            for i, col in enumerate(cols):
                # Handle 1D vs 2D prediction arrays
                preds_i = pred_feats if pred_feats.ndim == 1 else pred_feats[:, i]
                
                # Check Pearson correlation between residuals and predictions
                # Avoid division by zero if predictions or residuals are constant
                if np.std(preds_i) > 0 and np.std(residuals[col]) > 0:
                    corr = np.corrcoef(residuals[col], preds_i)[0, 1]
                    group_corrs.append(np.abs(corr))
                else:
                    group_corrs.append(0.0)
            
            # Average the absolute correlation if it's a group (e.g., Species_...)
            mean_res_corr = np.mean(group_corrs)

            metrics_records.append({
                'Feature': group_name,
                'Fold': fold,
                'R2': fold_r2,
                'Residual_Corr': mean_res_corr
            })

    # Aggregate R2 and Residual metrics across folds
    df_metrics = pd.DataFrame(metrics_records)
    
    final_feature_metrics = (
        df_metrics.groupby('Feature')
        .agg(
            R2_Mean=('R2', 'mean'),
            R2_Std=('R2', 'std'),
            Res_Corr_Mean=('Residual_Corr', 'mean'),
            Res_Corr_Std=('Residual_Corr', 'std')
        )
        .reset_index()
        .sort_values(by='R2_Mean', ascending=False)
    )
    final_feature_metrics['R2_Std'] = final_feature_metrics['R2_Std'].fillna(0.0)
    final_feature_metrics['Res_Corr_Std'] = final_feature_metrics['Res_Corr_Std'].fillna(0.0)
    
    # Calculate dataset-level global metrics
    dataset_mean_r2 = final_feature_metrics['R2_Mean'].mean()
    dataset_std_r2 = final_feature_metrics['R2_Mean'].std()
    dataset_mean_res_corr = final_feature_metrics['Res_Corr_Mean'].mean()
    
    df_global = pd.DataFrame({
        'Dataset': [data_name],
        'Global_R2_Mean': [dataset_mean_r2],
        'Global_R2_Std': [dataset_std_r2],
        'Global_Res_Corr_Mean': [dataset_mean_res_corr]
    })
    
    # Save Outputs
    feature_csv_path = os.path.join(output_dir, f"{data_name}_Feature_Assumptions.csv")
    global_csv_path = os.path.join(output_dir, f"{data_name}_Dataset_Global_Assumptions.csv")
    
    final_feature_metrics.to_csv(feature_csv_path, index=False)
    df_global.to_csv(global_csv_path, index=False)
    
    print(f"  -> Saved Feature Metrics to: {feature_csv_path}")
    print(f"  -> Saved Global Metrics to: {global_csv_path}")
    print(f"  -> Global Dataset R2: {dataset_mean_r2:.4f} (±{dataset_std_r2:.4f})")
    print(f"  -> Global Residual Correlation: {dataset_mean_res_corr:.4f}")
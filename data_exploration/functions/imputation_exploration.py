"""
imputation_exploration.py
----------------------------
Functions script to visually and statistically explore missing data and Limit of Detection (LOD) patterns, 
and to evaluate their predictive power for spatial metadata.

Input data:
1. Preprocessed multi-element CSV datasets.

Generates and saves:
1. Missingno nullity matrices (PNG) visually mapping missing data, sorted and binned by a specified spatial variable.
2. Classification reports (CSV) evaluating a Random Forest model's ability to predict spatial 
    categories based purely on missingness patterns.
3. LOD-specific nullity matrices (PNG) visualizing the spatial distribution of Limit of Detection ('<') 
    values across the dataset. 
4. Classification reports (CSV) evaluating the spatial predictive power of LOD presence across all applicable features.
5. Classification reports (CSV) evaluating spatial predictive power using strictly high-frequency LOD
     variables (default >40% LOD presence).
"""

import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def create_sorted_nullity_matrix(input_df_path, out_dir_path, dataset_name, spatial_var, target_vars):
    """
    Generates and saves a missingno nullity matrix sorted by a spatial variable,
    with visual bins and labels centered inside the spatial variable column.
    """
    print(f"Generating binned nullity matrix for {dataset_name}...")
    df = pd.read_csv(input_df_path)
    os.makedirs(out_dir_path, exist_ok=True)

    cols_to_plot = [spatial_var] + target_vars
    missing_cols = [col for col in cols_to_plot if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns were not found in the dataframe: {missing_cols}")

    for col in target_vars:
        if df[col].dtype == object:
            df.loc[df[col].astype(str).str.startswith("<"), col] = np.nan

    df_sorted = df.sort_values(spatial_var)
    df_plot = df_sorted[cols_to_plot]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    msno.matrix(df_plot, ax=ax, sparkline=False)
    
    categories = df_plot[spatial_var].unique()
    current_y = 0
    total_rows = len(df_plot)
    
    for category in categories:
        count = (df_plot[spatial_var] == category).sum()
        mid_y = current_y + (count / 2)
        
        ax.text(0.0, mid_y, str(category), ha='center', va='center', 
                color='white', fontsize=10, weight='bold',
                bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=1))
        
        current_y += count
        if current_y < total_rows:
            ax.axhline(current_y, color='black', linewidth=2)

    plt.title(f"Nullity Matrix Sorted by {spatial_var}", fontsize=16)
    output_filename = f"{dataset_name}_nullity_matrix_{spatial_var}.png"
    output_file_path = os.path.join(out_dir_path, output_filename)
    
    plt.savefig(output_file_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Successfully saved binned nullity matrix to {output_file_path}\n")


def predict_country_from_missingness(input_df_path, out_dir_path, dataset_name, spatial_var, target_vars):
    """
    Trains a Random Forest Classifier to predict the spatial variable 
    based purely on the presence/absence of target variables.
    """
    print(f"Evaluating predictive power of missingness for {spatial_var}...")
    os.makedirs(out_dir_path, exist_ok=True)
    df = pd.read_csv(input_df_path)

    for col in target_vars:
        if df[col].dtype == object:
            df.loc[df[col].astype(str).str.startswith("<"), col] = np.nan

    X = df[target_vars].isna().astype(int)
    y = df[spatial_var]

    valid_indices = y.dropna().index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, zero_division=0)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    print("=========================================")
    print(f" MISSINGNESS PREDICTION RESULTS")
    print("=========================================")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(report_str)
    print("=========================================\n")
    
    report_df = pd.DataFrame(report_dict).transpose()
    output_filename = f"{dataset_name}_prediction_report_{spatial_var}.csv"
    output_file_path = os.path.join(out_dir_path, output_filename)
    report_df.to_csv(output_file_path, index=True, index_label="Class/Metric")
    print(f"Successfully saved prediction report to {output_file_path}\n")
    
    return model


def create_lod_nullity_matrix(input_df_path, out_dir_path, dataset_name, spatial_var, target_vars=None):
    """
    Scans the dataset for ANY column containing '<' and generates a binned nullity matrix.
    """
    print(f"--- Scanning {dataset_name} for LOD values ---")
    df = pd.read_csv(input_df_path, low_memory=False)
    os.makedirs(out_dir_path, exist_ok=True)

    lod_cols = []
    for col in df.columns:
        if col == spatial_var:
            continue
        if df[col].astype(str).str.contains("<", na=False).any():
            lod_cols.append(col)

    if not lod_cols:
        print(f"-> Skipping matrix: No columns with '<' found in {dataset_name}.\n")
        return

    print(f"-> Found {len(lod_cols)} columns with LOD values: {lod_cols}")

    df_sorted = df.sort_values(spatial_var)
    df_plot = df_sorted[[spatial_var] + lod_cols].copy()

    for col in lod_cols:
        is_lod = df_plot[col].astype(str).str.contains("<", na=False)
        df_plot[col] = np.where(is_lod, np.nan, 1)

    fig = plt.figure(figsize=(max(10, len(lod_cols)*1.5), 8)) 
    ax = fig.add_subplot(111)
    msno.matrix(df_plot, ax=ax, sparkline=False)
    
    categories = df_plot[spatial_var].unique()
    current_y = 0
    for category in categories:
        count = (df_plot[spatial_var] == category).sum()
        mid_y = current_y + (count / 2)
        ax.text(0.0, mid_y, str(category), ha='center', va='center', 
                color='white', fontsize=10, weight='bold',
                bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=1))
        current_y += count
        if current_y < len(df_plot):
            ax.axhline(current_y, color='black', linewidth=2)

    plt.title(f"LOD Structure Matrix: {dataset_name}", fontsize=16)
    output_path = os.path.join(out_dir_path, f"{dataset_name}_LOD_matrix.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Successfully saved LOD matrix to {output_path}\n")


def predict_country_from_lods(input_df_path, out_dir_path, dataset_name, spatial_var, target_vars=None):
    """
    Scans for ANY column with '<' and uses them as features to predict country.
    """
    print(f"Evaluating LOD predictive power for {dataset_name}...")
    df = pd.read_csv(input_df_path, low_memory=False)
    
    lod_cols = [col for col in df.columns if col != spatial_var and df[col].astype(str).str.contains("<", na=False).any()]
    
    if not lod_cols:
        return

    X = pd.DataFrame(index=df.index)
    for col in lod_cols:
        X[col] = df[col].astype(str).str.contains("<", na=False).astype(int)
    
    y = df[spatial_var]
    valid_indices = y.dropna().index
    X, y = X.loc[valid_indices], y.loc[valid_indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    os.makedirs(out_dir_path, exist_ok=True)
    out_file = os.path.join(out_dir_path, f"{dataset_name}_LOD_prediction_report.csv")
    report_df.to_csv(out_file, index=True)
    
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Report saved to {out_file}\n")


def predict_country_from_high_lods(input_df_path, out_dir_path, dataset_name, spatial_var, threshold=0.40):
    """
    Evaluates predictive power using ONLY variables where more than a specified 
    threshold (default 40%) of their values are LODs.
    """
    print(f"Evaluating predictive power of High LOD (> {threshold*100}%) variables for {dataset_name}...")
    df = pd.read_csv(input_df_path, low_memory=False)
    
    high_lod_cols = []
    for col in df.columns:
        if col == spatial_var:
            continue
        
        # Calculate the proportion of LOD values in the column
        is_lod = df[col].astype(str).str.contains("<", na=False)
        lod_proportion = is_lod.mean()
        
        if lod_proportion > threshold:
            high_lod_cols.append(col)
            
    if not high_lod_cols:
        print(f"  -> No variables found with > {threshold*100}% LOD values in {dataset_name}.\n")
        return

    print(f"  -> Found {len(high_lod_cols)} variables with > {threshold*100}% LODs: {high_lod_cols}")

    # Create Feature Matrix X (1 if LOD, 0 otherwise)
    X = pd.DataFrame(index=df.index)
    for col in high_lod_cols:
        X[col] = df[col].astype(str).str.contains("<", na=False).astype(int)
    
    y = df[spatial_var]
    valid_indices = y.dropna().index
    X, y = X.loc[valid_indices], y.loc[valid_indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    os.makedirs(out_dir_path, exist_ok=True)
    out_file = os.path.join(out_dir_path, f"{dataset_name}_High_LOD_prediction_report.csv")
    report_df.to_csv(out_file, index=True)
    
    print(f"High LOD Variables Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"High LOD Report saved to {out_file}\n")
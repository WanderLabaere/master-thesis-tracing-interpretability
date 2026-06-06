"""
run_CV.py
----------------------------
Execution script to run spatially stratified nested K-Fold cross-validation on imputed trace element datasets (Soy and Timber).

Input data:
1. Preprocessed (imputed) multi-element CSV datasets for Timber (XRF) and Soy (ICPMS).
2. Hyperparameter tuning continuous grid.

Generates and saves:
1. Cross-validation evaluation tables (CSV) with predictive spatial metrics 
    (R2, RMSE, mean Haversine distance, and Haversine Skill Score).
2. Optimal hyperparameter logs (CSV) tracking best parameters found during the inner randomized search for 
    evaluated machine learning models (Random Forest, XGBoost, SVM).
3. Geographical fold plots (PNG) mapping train and test distributions alongside predicted coordinates and error vectors.
4. Data logs (CSV) capturing the fold visualization point coordinates and specific error metrics per fold.
5. One-hot encoded full dataset variations for Timber to integrate categorical species metadata into 
    the continuous feature space.
6. Genus-specific dataset splits and corresponding nested CV analyses for isolated Timber subsets.
"""

import pandas as pd
import os
import time
import datetime
from scipy.stats import randint, uniform, loguniform # for hyperparameter grid

# Import updated functions from your new K-Fold specific cv_functions script
from machine_learning.functions.CV import (
    run_all, 
    load_data, 
    get_models, 
    plot_kfold_cv_folds, 
    get_best_params
)

### Define directories
# After comparing imputed vs stripped datasets, continue working with these datasets:
# tX_stripped_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\filtered\NA_0_pct\tX_filtered_NA_0_pct.csv"
tX_imputedNA_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceNA\tX_imputed_NA_0_Ba_Br.csv"
sI_imputedLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\sI_imputed_mice.csv"
cI_imputedLOD_path = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_preparation\output\dataframes\imputed\miceLOD\cI_imputed_mice.csv"

# OUTPUT DIRECTORIES
out_base_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\machine_learning\output\CV"
out_CVtable_path = os.path.join(out_base_dir, "tables")
out_foldPlot_path = os.path.join(out_base_dir, "fold_images")
out_CV_imputation_comparison = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\machine_learning\output\CV_imputation_comparison"

n_splits_input = 5

# Master toggle for the pipeline
CALCULATE_TABLE = True
MAKE_FOLD_PLOTS = True


def execute_pipeline(data_name, csv_path, data_type, base_table_dir, base_plot_dir, tuning_grids):


    print(f"\n{'='*60}")
    print(f"PROCESSING DATASET: {data_name.upper()} ({data_type})")
    print(f"{'='*60}")

    table_dir = os.path.join(base_table_dir, data_name)
    plot_dir = os.path.join(base_plot_dir, data_name)
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    table_name = f"{data_name}_CVtable"
    fold_models_by_model = {}

    if CALCULATE_TABLE:
        # run_all now returns fold_models_by_model as second value
        _, fold_models_by_model = run_all(
            input_path=csv_path,
            output_path=table_dir,
            table_name=table_name,
            data_type=data_type,
            param_grids=tuning_grids,
            n_splits=n_splits_input,
            random_state=42,
            verbose=True,
        )

    if MAKE_FOLD_PLOTS:
        print(f"  Generating K-Fold Plots for {data_name}...")
        X, y = load_data(csv_path, data_type=data_type)
        coords = y.copy()

        # ---> CHANGED: Loop through all available models instead of just checking for SVM
        if not fold_models_by_model:
            print(f"  [!] No fold models available for plotting — run CALCULATE_TABLE first.")
        else:
            for model_name, fold_models in fold_models_by_model.items():
                print(f"    -> Plotting {model_name}...")
                plot_kfold_cv_folds(
                    fold_models=fold_models,
                    model_name=model_name,  
                    X=X, y=y, coords=coords,
                    data_name=data_name,
                    output_path=plot_dir,
                    verbose=True
                )

if __name__ == "__main__": 
    # Start the dataset timer
    script_start = time.time()

    # Updated to target the 'model' step inside the scikit-learn Pipeline
    tuning_grids = {
    "RandomForest": {
        "model__n_estimators": randint(100, 1000),
        "model__max_depth": [None, 10, 20, 30, 40],
        "model__min_samples_split": randint(2, 20),
        "model__min_samples_leaf": randint(1, 20),
        "model__max_features": ["sqrt", "log2", None, 0.33, 0.5, 0.7] 
    },
    
    "XGBoost": {
        "model__n_estimators": randint(100, 1000),
        "model__learning_rate": loguniform(0.001, 0.3),
        "model__max_depth": randint(3, 12),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__subsample": uniform(0.5, 0.5),
        "model__gamma": uniform(0, 5),
        "model__reg_lambda": loguniform(1e-3, 100),
        "model__reg_alpha": loguniform(1e-3, 100),
        "model__min_child_weight": randint(1, 10)
    },
    
    "SVM": [
        # Search space 1: Continuous Gamma
        {
            "model__estimator__C": loguniform(1e-3, 1e3),
            "model__estimator__gamma": loguniform(1e-4, 1e1),
            "model__estimator__epsilon": loguniform(1e-3, 1)
        },
        # Search space 2: Categorical Gamma
        {
            "model__estimator__C": loguniform(1e-3, 1e3),
            "model__estimator__gamma": ['scale', 'auto'],
            "model__estimator__epsilon": loguniform(1e-3, 1)
        }
    ]
    }

    #  Process Soy (ICPMS) - Now strictly using imputed paths
    soy_start = time.time()

    execute_pipeline("sI", sI_imputedLOD_path, "ICPMS", out_CVtable_path, out_foldPlot_path, tuning_grids)

    soy_elapsed = time.time() - soy_start
    formatted_soy = str(datetime.timedelta(seconds=int(soy_elapsed)))
    print(f"\n[========== ALL PROCESSING COMPLETE IN {formatted_soy} ==========]")
    
    #  Process Cocoa (ICPMS) - Now strictly using imputed paths
    # SKIP THIS FOR NOW, WE'RE NOT USING THE cI DATA ANYWAY
    # execute_pipeline("cI", cI_imputedLOD_path, "ICPMS", out_CVtable_path, out_foldPlot_path, tuning_grids)


    # ---------------------------------------------------------
    #  Process Timber (XRF) Setup
    # ---------------------------------------------------------
    # Replaced undefined tX_logStd_path with tX_imputed_path
    df_tX = pd.read_csv(tX_imputedNA_path)
    
    tx_genera_dir = os.path.join(os.path.dirname(tX_imputedNA_path), "tX_genera_splits")
    os.makedirs(tx_genera_dir, exist_ok=True)
    
    tx_table_base = os.path.join(out_CVtable_path, "tX")
    tx_plot_base = os.path.join(out_foldPlot_path, "tX")


    # ---> ADDED: FULL TIMBER DATASET WITH SPECIES INFO <---
    print("\nPreparing Full Timber Dataset with One-Hot Encoded Species...")
    
    # Check if "Genus" exists
    categorical_col = 'Genus'
    
    # One-hot encode the text column into numeric 0/1 columns
    species_dummies = pd.get_dummies(df_tX[categorical_col], prefix=categorical_col).astype(int)
    
    # Tack the new numeric columns onto the end of the dataframe
    df_tX_encoded = pd.concat([df_tX, species_dummies], axis=1)
    
    # Save the temporary file
    full_encoded_csv_path = os.path.join(tx_genera_dir, "tX_Full_Encoded.csv")
    df_tX_encoded.to_csv(full_encoded_csv_path, index=False)
    
    # Run the pipeline for the full dataset
    timber_full_start = time.time()

    execute_pipeline(
        data_name="tX_Full_All_Genera", 
        csv_path=full_encoded_csv_path, 
        data_type="XRF", 
        base_table_dir=tx_table_base, 
        base_plot_dir=tx_plot_base,
        tuning_grids=tuning_grids
    )

    
    timber_elapsed = time.time() - timber_full_start
    formatted_timber = str(datetime.timedelta(seconds=int(timber_elapsed)))
    print(f"\n[========== ALL PROCESSING COMPLETE IN {formatted_timber} ==========]")
    # ---------------------------------------------------------

    # # ---> ADDED: COUNTRY FILTERED TIMBER DATASET <---
    # print("\nPreparing Country-Filtered Timber Dataset...")
    
    # countries_to_remove = ['France', 'Solomon Islands', 'Spain', 'Turkey', 'Russia', 'Kazakhstan']
    # # Change 'Country' if your column is named differently (e.g., 'Origin')
    # df_tX_filtered = df_tX[~df_tX['Country'].isin(countries_to_remove)].copy()

    # # One-hot encode the text column into numeric 0/1 columns for the filtered dataframe
    # species_dummies_filtered = pd.get_dummies(df_tX_filtered[categorical_col], prefix=categorical_col).astype(int)
    # df_tX_encoded_filtered = pd.concat([df_tX_filtered, species_dummies_filtered], axis=1)

    # # Setup directories for filtered data
    # tx_filtered_csv_dir = os.path.join(tx_genera_dir, "country_filtered")
    # os.makedirs(tx_filtered_csv_dir, exist_ok=True)
    
    # filtered_encoded_csv_path = os.path.join(tx_filtered_csv_dir, "tX_Filtered_Encoded.csv")
    # df_tX_encoded_filtered.to_csv(filtered_encoded_csv_path, index=False)

    # # Setup subdirectories for the output (tables and plots)
    # tx_table_filtered_base = os.path.join(tx_table_base, "country_filtered")
    # tx_plot_filtered_base = os.path.join(tx_plot_base, "country_filtered")

    # execute_pipeline(
    #     data_name="tX_Country_Filtered", 
    #     csv_path=filtered_encoded_csv_path, 
    #     data_type="XRF", 
    #     base_table_dir=tx_table_filtered_base, 
    #     base_plot_dir=tx_plot_filtered_base,
    #     tuning_grids=tuning_grids
    # )
    # ---------------------------------------------------------
    
    #  Process Timber (XRF) - Splitting by Genus
    for genus, group in df_tX.groupby("Genus"):
            genus_clean = str(genus).strip().replace(" ", "_")
            if not genus_clean or genus_clean.lower() == "nan":
                continue
                
            genus_csv_path = os.path.join(tx_genera_dir, f"tX_{genus_clean}.csv")
            group.to_csv(genus_csv_path, index=False)
            
            try:
                execute_pipeline(
                    data_name=genus_clean, 
                    csv_path=genus_csv_path, 
                    data_type="XRF", 
                    base_table_dir=tx_table_base, 
                    base_plot_dir=tx_plot_base,
                    tuning_grids=tuning_grids
                )
            except ValueError as e:
                print(f"  [!] Skipping {genus_clean} due to CV splitting error: {e}")
                continue

    # ---------------------------------------------------------
    #  Process Country-Filtered Timber (XRF) - Splitting by Genus
    # ---------------------------------------------------------
    # print("\nPreparing Country-Filtered Dataset Splitting by Genus...")
    
    # for genus, group in df_tX_filtered.groupby("Genus"):
    #     genus_clean = str(genus).strip().replace(" ", "_")
    #     if not genus_clean or genus_clean.lower() == "nan":
    #         continue
            
    #     # Save these specifically in the filtered CSV folder
    #     genus_csv_path = os.path.join(tx_filtered_csv_dir, f"tX_Filtered_{genus_clean}.csv")
        
    #     # Only process if there are still samples left after filtering out the countries
    #     if len(group) > 0:
    #         group.to_csv(genus_csv_path, index=False)
            
    #         try:
    #             execute_pipeline(
    #                 data_name=f"Filtered_{genus_clean}", 
    #                 csv_path=genus_csv_path, 
    #                 data_type="XRF", 
    #                 base_table_dir=tx_table_filtered_base, 
    #                 base_plot_dir=tx_plot_filtered_base,
    #                 tuning_grids=tuning_grids
    #             )
    #         except ValueError as e:
    #             print(f"  [!] Skipping {genus_clean} (Filtered) due to CV splitting error: {e}")
    #             continue

    # Calculate and print total script time
    script_elapsed = time.time() - script_start
    formatted_total = str(datetime.timedelta(seconds=int(script_elapsed)))
    print(f"\n[========== ALL PROCESSING COMPLETE IN {formatted_total} ==========]")
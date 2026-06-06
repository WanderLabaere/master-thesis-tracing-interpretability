"""
run_plot_SHAP_push.py
----------------------------
Exectution script to generate SHAP vector maps visualizing SHAP-derived directional 
model "pushes." The pipeline maps how specific geochemical trace features influence longitudinal and 
latitudinal coordinate predictions relative to a global baseline centroid.

Input data:
1. Per-sample SHAP interaction logs (CSV) mapping true coordinates and feature-specific SHAP contributions.
2. Tabular feature importance logs (CSV) for identifying and ranking the most influential features.
3. World map shapefiles (GeoPandas) for providing geographic spatial context.

Generates and saves:
1. Directional SHAP Arrow Maps (PNG): Geographic visualizations where arrows originate from sample sites, 
    with direction and magnitude representing the model's SHAP-derived coordinate push, colored by raw feature intensity.
2. Comparative Summary Grids (PNG): 3x3 grids or multi-model comparison panels that facilitate the evaluation 
    of feature-model consistency and spatial decision-making patterns across algorithms and genus-specific taxonomic subsets.
"""

import os

# Import the plotting functions
from feature_importance.functions.plot_SHAP_push import plot_top_n_features, plot_model_comparison_grid

# ---------------------------------------------------------------------------
# Plotting Toggles
# ---------------------------------------------------------------------------
INDIVIDUAL_FEATURE_PLOTS = False
FEATURE_SUBPLOTS         = False
THREE_MODEL_PLOTS        = False     # Compares Ba, Co, Cu, Ni across models
EXTRA_SPECIFIC_PLOTS     = True      # Extra single-element model comparisons (Quercus Fe, Betula Pb)

FEATURES_TO_COMPARE = ['Ba', 'Co', 'Cu', 'Ni']

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
extracted_shap_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\SHAP"
push_maps_dir      = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\SHAP\SHAP_push"
summary_dir        = os.path.join(push_maps_dir, "model_summaries")
models             = ["RandomForest", "XGBoost", "SVM"]

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("STARTING SHAP MAP GENERATION")
    print(f"{'='*60}")

    # ---------------------------------------------------------
    # Soy (sI)
    # ---------------------------------------------------------
    print("\n--- Processing Soy (sI) ---")
    si_in_dir  = os.path.join(extracted_shap_dir, "sI")
    si_out_dir = os.path.join(push_maps_dir, "sI")
    
    shap_paths_sI = {}
    imp_paths_sI  = {}

    for model in models:
        shap_csv = os.path.join(si_in_dir, f"sI_{model}_SHAP_per_sample.csv")
        imp_csv  = os.path.join(si_in_dir, f"sI_{model}_SHAP_importance.csv")
        
        shap_paths_sI[model] = shap_csv
        imp_paths_sI[model]  = imp_csv
        
        if INDIVIDUAL_FEATURE_PLOTS or FEATURE_SUBPLOTS:
            model_out_dir = os.path.join(si_out_dir, model)
            if os.path.exists(shap_csv) and os.path.exists(imp_csv):
                plot_top_n_features(
                    shap_csv_path       = shap_csv,
                    importance_csv_path = imp_csv,
                    output_dir          = model_out_dir,
                    top_n               = 25,
                    arrow_scale         = 1.0,
                    max_arrow_len       = 10.0,
                    cmap                = "coolwarm",
                    title_prefix        = f"sI — {model}",
                    dataset_prefix      = "sI",
                    do_individual       = INDIVIDUAL_FEATURE_PLOTS,
                    do_subplot          = FEATURE_SUBPLOTS
                )
            else:
                print(f"  [!] Missing data for sI ({model}). Run extraction first.")

    # Generate 3 Model Comparison for sI
    if THREE_MODEL_PLOTS:
        comp_out_path = os.path.join(summary_dir, "sI_3_Model_Comparison_Grid.png")
        plot_model_comparison_grid(
            shap_csv_paths       = shap_paths_sI,
            importance_csv_paths = imp_paths_sI,
            models               = models,
            features             = FEATURES_TO_COMPARE,
            output_path          = comp_out_path,
            arrow_scale          = 1.0,
            max_arrow_len        = 100.0,
            title_prefix         = "sI"
        )


    # ---------------------------------------------------------
    # Timber (tX)
    # ---------------------------------------------------------
    print("\n--- Processing Timber (tX) ---")
    tx_in_base  = os.path.join(extracted_shap_dir, "tX")
    tx_out_base = os.path.join(push_maps_dir, "tX")

    if os.path.exists(tx_in_base):
        genera = [d for d in os.listdir(tx_in_base) if os.path.isdir(os.path.join(tx_in_base, d))]
        print(f"Found {len(genera)} genera to process.")

        for genus in genera:
            genus_in_dir  = os.path.join(tx_in_base, genus)
            genus_out_dir = os.path.join(tx_out_base, genus)
            
            shap_paths_tx = {}
            imp_paths_tx  = {}

            for model in models:
                shap_csv = os.path.join(genus_in_dir, f"{genus}_{model}_SHAP_per_sample.csv")
                imp_csv  = os.path.join(genus_in_dir, f"{genus}_{model}_SHAP_importance.csv")
                
                shap_paths_tx[model] = shap_csv
                imp_paths_tx[model]  = imp_csv

                if INDIVIDUAL_FEATURE_PLOTS or FEATURE_SUBPLOTS:
                    model_out_dir = os.path.join(genus_out_dir, model)
                    if os.path.exists(shap_csv) and os.path.exists(imp_csv):
                        plot_top_n_features(
                            shap_csv_path       = shap_csv,
                            importance_csv_path = imp_csv,
                            output_dir          = model_out_dir,
                            top_n               = 20,
                            arrow_scale         = 1.0,
                            max_arrow_len       = 100.0,
                            cmap                = "coolwarm",
                            title_prefix        = f"{genus} — {model}",
                            dataset_prefix      = f"tX_{genus}",
                            do_individual       = INDIVIDUAL_FEATURE_PLOTS,
                            do_subplot          = FEATURE_SUBPLOTS
                        )
            
            # Generate 3 Model Comparison for the specific Genus
            if THREE_MODEL_PLOTS:
                comp_out_path = os.path.join(summary_dir, f"{genus}_3_Model_Comparison_Grid.png")
                plot_model_comparison_grid(
                    shap_csv_paths       = shap_paths_tx,
                    importance_csv_paths = imp_paths_tx,
                    models               = models,
                    features             = FEATURES_TO_COMPARE,
                    output_path          = comp_out_path,
                    arrow_scale          = 1.0,
                    max_arrow_len        = 100.0,
                    title_prefix         = genus
                )

            # Extra single-element row plots for Quercus and Betula
            if EXTRA_SPECIFIC_PLOTS:
                if genus == "Quercus":
                    fe_out_path = os.path.join(summary_dir, f"{genus}_Fe_Comparison_Grid.png")
                    plot_model_comparison_grid(
                        shap_csv_paths       = shap_paths_tx,
                        importance_csv_paths = imp_paths_tx,
                        models               = models,
                        features             = ['Fe'],
                        output_path          = fe_out_path,
                        arrow_scale          = 1.0,
                        max_arrow_len        = 100.0,
                        title_prefix         = genus
                    )
                
                elif genus == "Betula":
                    pb_out_path = os.path.join(summary_dir, f"{genus}_Pb_Comparison_Grid.png")
                    plot_model_comparison_grid(
                        shap_csv_paths       = shap_paths_tx,
                        importance_csv_paths = imp_paths_tx,
                        models               = models,
                        features             = ['Pb'],
                        output_path          = pb_out_path,
                        arrow_scale          = 1.0,
                        max_arrow_len        = 100.0,
                        title_prefix         = genus,
                        vertical_layout      = True
                    )

    else:
        print(f"  [!] Timber extraction folder not found: {tx_in_base}")

    print(f"\n{'='*60}")
    print("[========== ALL MAPS GENERATED SUCCESSFULLY ==========]")
    print(f"{'='*60}")
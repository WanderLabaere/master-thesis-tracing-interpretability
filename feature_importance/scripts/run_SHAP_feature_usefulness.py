"""
run_shap_usefulness.py
----------------------------
Execution script for computing and plotting SHAP usefulness scores
for all datasets (sI, tX) and all models.

Usefulness score per sample per feature is based purely on:
  1. Directional alignment  : does the feature push toward the correct region?

Output structure mirrors the existing SHAP pipeline:
  SHAP_usefulness/
    sI/
      RandomForest/
        usefulness_scores.csv
        rank01_Co_usefulness.png
        ...
        Summary_3x3_Usefulness_Grid.png
      XGBoost/  ...
      SVM/      ...
    tX/
      Acacia/
        RandomForest/  ...
"""

import os
import geopandas as gpd

# ---------------------------------------------------------------------------
# Import the pipeline function
# ---------------------------------------------------------------------------
# Adjust this import path to wherever you placed compute_shap_usefulness.py
from feature_importance.functions.SHAP_feature_usefulness import run_usefulness_pipeline

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
# Where extract_SHAP.py saved its per-sample CSVs
EXTRACTED_SHAP_DIR = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\SHAP"

# Where we want the usefulness outputs saved
USEFULNESS_OUT_DIR = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\SHAP\SHAP_usefulness_V2"

# World map shapefile (same path used in plot_SHAP_push.py)
WORLD_MAP_DIR = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_input\world_map_data\World_countries"

MODELS = ["RandomForest", "XGBoost", "SVM"]

# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------
COMMON_KWARGS = dict(
    top_n               = 9,     # Process and plot top-N features
    map_padding         = 2.0,   # Degrees of padding around data on maps
    figsize             = (12, 8),
    min_dist_threshold  = 1.0,   # Samples < 1° from baseline are masked (unstable direction)
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _run_model(dataset_label: str, shap_in_dir: str, usefulness_out_dir: str, world, **kwargs):
    """Run the usefulness pipeline for all three models for a given dataset."""
    for model in MODELS:
        shap_csv = os.path.join(shap_in_dir, f"{dataset_label}_{model}_SHAP_per_sample.csv")
        imp_csv  = os.path.join(shap_in_dir, f"{dataset_label}_{model}_SHAP_importance.csv")
        out_dir  = os.path.join(usefulness_out_dir, model)

        if not os.path.exists(shap_csv):
            print(f"  [!] Missing: {shap_csv}")
            continue
        if not os.path.exists(imp_csv):
            print(f"  [!] Missing: {imp_csv}")
            continue

        print(f"\n  -> {dataset_label} / {model}")
        run_usefulness_pipeline(
            shap_csv_path       = shap_csv,
            importance_csv_path = imp_csv,
            output_dir          = out_dir,
            title_prefix        = f"{dataset_label} — {model}",
            world               = world,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("SHAP USEFULNESS PIPELINE")
    print(f"{'='*60}")

    # Load world map once and reuse across all plots
    print(f"\nLoading world map from:\n  {WORLD_MAP_DIR}")
    try:
        world = gpd.read_file(WORLD_MAP_DIR)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load world map from:\n  {WORLD_MAP_DIR}\nError: {e}"
        )
    print("  World map loaded OK.")

    # ===========================================================
    # 1. Soil (sI)
    # ===========================================================
    print(f"\n{'─'*40}")
    print("DATASET: Soil (sI)")
    print(f"{'─'*40}")

    _run_model(
        dataset_label     = "sI",
        shap_in_dir       = os.path.join(EXTRACTED_SHAP_DIR, "sI"),
        usefulness_out_dir= os.path.join(USEFULNESS_OUT_DIR, "sI"),
        world             = world,
        **COMMON_KWARGS,
    )

    # ===========================================================
    # 2. Timber (tX) — one subfolder per genus
    # ===========================================================
    print(f"\n{'─'*40}")
    print("DATASET: Timber (tX)")
    print(f"{'─'*40}")

    tx_in_base  = os.path.join(EXTRACTED_SHAP_DIR, "tX")
    tx_out_base = os.path.join(USEFULNESS_OUT_DIR,  "tX")

    if not os.path.exists(tx_in_base):
        print(f"  [!] Timber extraction folder not found: {tx_in_base}")
    else:
        genera = sorted([
            d for d in os.listdir(tx_in_base)
            if os.path.isdir(os.path.join(tx_in_base, d))
        ])
        print(f"  Found {len(genera)} genera: {genera}")

        for genus in genera:
            print(f"\n  GENUS: {genus}")
            _run_model(
                dataset_label     = genus,
                shap_in_dir       = os.path.join(tx_in_base, genus),
                usefulness_out_dir= os.path.join(tx_out_base, genus),
                world             = world,
                **COMMON_KWARGS,
            )

    print(f"\n{'='*60}")
    print("ALL USEFULNESS MAPS GENERATED")
    print(f"{'='*60}")
    print(f"Output root: {USEFULNESS_OUT_DIR}")



# """
# run_shap_usefulness.py
# ----------------------------
# Execution script for computing and plotting SHAP usefulness scores
# for all datasets (sI, tX) and all models.

# Usefulness score per sample per feature combines:
#   1. Directional alignment  : does the feature push toward the correct region?
#   2. Magnitude appropriateness : does the feature push harder for samples that
#                                   are further from the baseline (and softer for
#                                   samples already close to it)?

# Output structure mirrors the existing SHAP pipeline:
#   SHAP_usefulness/
#     sI/
#       RandomForest/
#         usefulness_scores.csv
#         rank01_Co_usefulness.png
#         rank02_Ba_usefulness.png
#         ...
#         Summary_3x3_Usefulness_Grid.png
#       XGBoost/  ...
#       SVM/      ...
#     tX/
#       Acacia/
#         RandomForest/  ...
# """

# import os
# import geopandas as gpd

# # ---------------------------------------------------------------------------
# # Import the pipeline function
# # ---------------------------------------------------------------------------
# # Adjust this import path to wherever you placed compute_shap_usefulness.py
# from feature_importance.functions.SHAP_feature_usefulness import run_usefulness_pipeline

# # ---------------------------------------------------------------------------
# # Directories
# # ---------------------------------------------------------------------------
# # Where extract_SHAP.py saved its per-sample CSVs
# EXTRACTED_SHAP_DIR = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\SHAP"

# # Where we want the usefulness outputs saved
# USEFULNESS_OUT_DIR = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\feature_importance\output\SHAP\SHAP_usefulness"

# # World map shapefile (same path used in plot_SHAP_push.py)
# WORLD_MAP_DIR = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_input\world_map_data\World_countries"

# MODELS = ["RandomForest", "XGBoost", "SVM"]

# # ---------------------------------------------------------------------------
# # Shared settings
# # ---------------------------------------------------------------------------
# COMMON_KWARGS = dict(
#     top_n               = 9,     # Process and plot top-N features
#     map_padding         = 2.0,   # Degrees of padding around data on maps
#     figsize             = (12, 8),
#     min_dist_threshold  = 1.0,   # Samples < 1° from baseline are masked (unstable direction)
# )


# # ---------------------------------------------------------------------------
# # Helper
# # ---------------------------------------------------------------------------
# def _run_model(dataset_label: str, shap_in_dir: str, usefulness_out_dir: str, world, **kwargs):
#     """Run the usefulness pipeline for all three models for a given dataset."""
#     for model in MODELS:
#         shap_csv = os.path.join(shap_in_dir, f"{dataset_label}_{model}_SHAP_per_sample.csv")
#         imp_csv  = os.path.join(shap_in_dir, f"{dataset_label}_{model}_SHAP_importance.csv")
#         out_dir  = os.path.join(usefulness_out_dir, model)

#         if not os.path.exists(shap_csv):
#             print(f"  [!] Missing: {shap_csv}")
#             continue
#         if not os.path.exists(imp_csv):
#             print(f"  [!] Missing: {imp_csv}")
#             continue

#         print(f"\n  -> {dataset_label} / {model}")
#         run_usefulness_pipeline(
#             shap_csv_path       = shap_csv,
#             importance_csv_path = imp_csv,
#             output_dir          = out_dir,
#             title_prefix        = f"{dataset_label} — {model}",
#             world               = world,
#             **kwargs,
#         )


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     print(f"\n{'='*60}")
#     print("SHAP USEFULNESS PIPELINE")
#     print(f"{'='*60}")

#     # Load world map once and reuse across all plots
#     print(f"\nLoading world map from:\n  {WORLD_MAP_DIR}")
#     try:
#         world = gpd.read_file(WORLD_MAP_DIR)
#     except Exception as e:
#         raise FileNotFoundError(
#             f"Could not load world map from:\n  {WORLD_MAP_DIR}\nError: {e}"
#         )
#     print("  World map loaded OK.")

#     # ===========================================================
#     # 1. Soil (sI)
#     # ===========================================================
#     print(f"\n{'─'*40}")
#     print("DATASET: Soil (sI)")
#     print(f"{'─'*40}")

#     _run_model(
#         dataset_label     = "sI",
#         shap_in_dir       = os.path.join(EXTRACTED_SHAP_DIR, "sI"),
#         usefulness_out_dir= os.path.join(USEFULNESS_OUT_DIR, "sI"),
#         world             = world,
#         **COMMON_KWARGS,
#     )

#     # ===========================================================
#     # 2. Timber (tX) — one subfolder per genus
#     # ===========================================================
#     print(f"\n{'─'*40}")
#     print("DATASET: Timber (tX)")
#     print(f"{'─'*40}")

#     tx_in_base  = os.path.join(EXTRACTED_SHAP_DIR, "tX")
#     tx_out_base = os.path.join(USEFULNESS_OUT_DIR,  "tX")

#     if not os.path.exists(tx_in_base):
#         print(f"  [!] Timber extraction folder not found: {tx_in_base}")
#     else:
#         genera = sorted([
#             d for d in os.listdir(tx_in_base)
#             if os.path.isdir(os.path.join(tx_in_base, d))
#         ])
#         print(f"  Found {len(genera)} genera: {genera}")

#         for genus in genera:
#             print(f"\n  GENUS: {genus}")
#             _run_model(
#                 dataset_label     = genus,
#                 shap_in_dir       = os.path.join(tx_in_base, genus),
#                 usefulness_out_dir= os.path.join(tx_out_base, genus),
#                 world             = world,
#                 **COMMON_KWARGS,
#             )

#     print(f"\n{'='*60}")
#     print("ALL USEFULNESS MAPS GENERATED")
#     print(f"{'='*60}")
#     print(f"Output root: {USEFULNESS_OUT_DIR}")
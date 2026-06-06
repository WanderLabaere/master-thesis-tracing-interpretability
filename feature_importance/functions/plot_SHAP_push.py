"""
plot_SHAP_push.py
----------------------------
Functions script to generate geographic directional maps mapping the 
spatial impact of individual features using SHAP vectors. The pipeline illustrates how specific trace 
elements or categorical features "push" geographic predictions (Longitude/Latitude) relative to a naive baseline centroid.

Input data:
1. Precomputed wide SHAP data logs (CSV) containing original feature values and their 
    corresponding per-sample longitudinal and latitudinal SHAP values.
2. Precomputed tabular feature importance logs (CSV) to identify and sequentially rank the 
    top-N most influential features for plotting.
3. Local world map shapefiles for geospatial boundaries and context.

Generates and saves:
1. Individual high-resolution geographic maps (PNG) for the top-N features, displaying spatial SHAP vectors 
    (arrows) colored by the raw underlying feature value.
2. A 3x3 summary grid (PNG) consolidating the directional SHAP maps of the top 9 most important features 
    for rapid spatial comparison.
3. Custom multi-model comparative grids (PNG) aligning the spatial SHAP behaviors of specific target 
    features across the different machine learning algorithms.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_data(shap_csv_path: str, importance_csv_path: str):
    df   = pd.read_csv(shap_csv_path)
    imp  = pd.read_csv(importance_csv_path)
    required = {"Longitude", "Latitude"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"SHAP CSV is missing columns: {missing}")
    return df, imp

def _get_feature_names(imp_df: pd.DataFrame, top_n: int) -> list:
    return imp_df["Feature"].head(top_n).tolist()

def _baseline(df: pd.DataFrame) -> tuple:
    return df["Longitude"].mean(), df["Latitude"].mean()

def _cap_magnitude(dx: np.ndarray, dy: np.ndarray, max_len: float):
    mag = np.sqrt(dx**2 + dy**2)
    scale = np.where(mag > max_len, max_len / np.clip(mag, 1e-12, None), 1.0)
    return dx * scale, dy * scale

# ---------------------------------------------------------------------------
# Core plotting function
# ---------------------------------------------------------------------------

def plot_shap_arrows_for_feature(
    df: pd.DataFrame,
    world: gpd.GeoDataFrame,
    feature: str,
    baseline_lon: float,
    baseline_lat: float,
    output_path: str,
    arrow_scale: float = 10.0,
    max_arrow_len: float = 2.0,
    map_padding: float = 2.0,
    cmap: str = "coolwarm",
    figsize: tuple = (12, 8),
    title_prefix: str = "",
    ax = None,
    show_colorbar: bool = True,
    show_legend: bool = True,
    is_subplot: bool = False,
    rank_num: int = None,
    xlim: tuple = None,
    ylim: tuple = None,
    tick_labelsize: int = 11,
    legend_fontsize: int = 11,
    title_fontsize: int = 12,
    cbar_label_fontsize: int = 8,
):
    """
    Main function that creates SHAP vector maps.
    """
    shap_lon_col = f"shap_lon_{feature}"
    shap_lat_col = f"shap_lat_{feature}"

    for col in [feature, shap_lon_col, shap_lat_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found. Available: {list(df.columns)}")

    lons       = df["Longitude"].values
    lats       = df["Latitude"].values

    np.random.seed(42)
    lons = lons # + RANDOM JITTER
    lats = lats # + RANDOM JITTER
    np.random.seed(None)

    feat_vals  = df[feature].values
    dx_raw     = df[shap_lon_col].values / arrow_scale
    dy_raw     = df[shap_lat_col].values / arrow_scale

    dx, dy = _cap_magnitude(dx_raw, dy_raw, max_arrow_len / arrow_scale)

    vmin, vmax = np.nanpercentile(feat_vals, 2), np.nanpercentile(feat_vals, 98)
    norm       = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj   = cm.get_cmap(cmap)
    colors     = cmap_obj(norm(feat_vals))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        created_fig = False
        fig = ax.figure

    world.plot(ax=ax, color="whitesmoke", edgecolor="lightgray", zorder=0)
    ax.scatter(lons, lats, c="lightgrey", s=100, alpha=0.75, zorder=1, label="True locations" if show_legend else "")

    for i in range(len(lons)):
        ax.annotate(
            "",
            xy       = (lons[i] + dx[i], lats[i] + dy[i]),
            xytext   = (lons[i],         lats[i]),
            arrowprops=dict(arrowstyle="-|>", color=colors[i], lw=1.2, mutation_scale=8),
            zorder=2,
        )

    ax.plot(
        baseline_lon, baseline_lat,
        marker="o", markersize=14 if not is_subplot else 8,
        markerfacecolor="yellow", markeredgecolor="black", linestyle="None", zorder=5,
        label=f"Baseline ({baseline_lon:.2f}°, {baseline_lat:.2f}°)" if show_legend else "",
    )

    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        min_lon = min(np.min(lons), np.min(lons + dx), baseline_lon) - map_padding
        max_lon = max(np.max(lons), np.max(lons + dx), baseline_lon) + map_padding
        min_lat = min(np.min(lats), np.min(lats + dy), baseline_lat) - map_padding
        max_lat = max(np.max(lats), np.max(lats + dy), baseline_lat) + map_padding
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)

    ax.set_aspect("equal")

    if show_colorbar:
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(sm, cax=cax)

        cbar.set_label(f"{feature} (raw values)", fontsize=cbar_label_fontsize)

    if not is_subplot:
        ax.set_xlabel("Longitude (°)", fontsize=13)
        ax.set_ylabel("Latitude (°)",  fontsize=13)
        ax.tick_params(axis='both', which='major', labelsize=14)
        title_main = f"{title_prefix} — SHAP directional map: {feature}" if title_prefix else f"SHAP directional map: {feature}"
        ax.set_title(
            f"{title_main}\n"
            "Arrow origin = true location  |  Arrow direction = SHAP push (lon, lat)\n"
            "Arrow away from baseline -> correct direction  |  Arrow toward baseline -> wrong direction",
            fontsize=10,
        )
    else:
        rank_str = f"{rank_num}. " if rank_num is not None else ""
        ax.set_title(f"{rank_str}{feature}", fontsize=title_fontsize, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)

    if show_legend:
        ax.legend(loc="lower right", fontsize=legend_fontsize)

    if created_fig and not is_subplot:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

def plot_top_n_features(
    shap_csv_path: str,
    importance_csv_path: str,
    output_dir: str,
    top_n: int = 5,
    arrow_scale: float = 10.0,
    max_arrow_len: float = 2.0,
    map_padding: float = 2.0,
    cmap: str = "coolwarm",
    figsize: tuple = (12, 8),
    title_prefix: str = "",
    dataset_prefix: str = "",
    do_individual: bool = True,
    do_subplot: bool = True
):
    """
    Individual plots for all model-feature-dataset combinations.
    """
    os.makedirs(output_dir, exist_ok=True)

    map_data_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_input\world_map_data\World_countries"
    try:
        world = gpd.read_file(map_data_dir)
    except Exception as e:
        raise FileNotFoundError(f"Could not load world map from:\n  {map_data_dir}\nError: {e}")

    df, imp = _load_data(shap_csv_path, importance_csv_path)
    
    fetch_count = max(top_n, 9) if do_subplot else top_n
    features = _get_feature_names(imp, fetch_count)
    baseline_lon, baseline_lat = _baseline(df)

    # ==========================================
    #  INDIVIDUAL PLOTS LOOP
    # ==========================================
    if do_individual:
        print(f"\nPlotting individual SHAP arrow maps for top-{top_n} features...")
        features_to_plot_individually = features[:top_n]
        for rank_idx, feature in enumerate(features_to_plot_individually):
            current_rank = rank_idx + 1
            out_path = os.path.join(output_dir, f"rank{current_rank:02d}_{feature}_SHAP_arrows.png")
            print(f"  [{current_rank}/{top_n}] {feature}")
            try:
                plot_shap_arrows_for_feature(
                    df=df, world=world, feature=feature,
                    baseline_lon=baseline_lon, baseline_lat=baseline_lat,
                    output_path=out_path, arrow_scale=arrow_scale,
                    max_arrow_len=max_arrow_len, map_padding=map_padding,
                    cmap=cmap, figsize=figsize, title_prefix=title_prefix
                )
            except KeyError as e:
                print(f"  [!] Skipping {feature}: {e}")

    # ==========================================
    #  3x3 SUMMARY GRID
    # ==========================================
    if do_subplot:
        grid_features = features[:9]
        g_min_lon, g_max_lon = float('inf'), float('-inf')
        g_min_lat, g_max_lat = float('inf'), float('-inf')
        
        np.random.seed(42)
        lons_j = df["Longitude"].values # + RANDOM JITTER
        lats_j = df["Latitude"].values # + RANDOM JITTER
        np.random.seed(None)

        for feature in grid_features:
            shap_lon_col = f"shap_lon_{feature}"
            shap_lat_col = f"shap_lat_{feature}"
            if shap_lon_col in df.columns and shap_lat_col in df.columns:
                dx_raw = df[shap_lon_col].values / arrow_scale
                dy_raw = df[shap_lat_col].values / arrow_scale
                dx, dy = _cap_magnitude(dx_raw, dy_raw, max_arrow_len / arrow_scale)

                g_min_lon = min(g_min_lon, np.min(lons_j), np.min(lons_j + dx), baseline_lon)
                g_max_lon = max(g_max_lon, np.max(lons_j), np.max(lons_j + dx), baseline_lon)
                g_min_lat = min(g_min_lat, np.min(lats_j), np.min(lats_j + dy), baseline_lat)
                g_max_lat = max(g_max_lat, np.max(lats_j), np.max(lats_j + dy), baseline_lat)

        if g_min_lon != float('inf'):
            global_xlim = (g_min_lon - map_padding, g_max_lon + map_padding)
            global_ylim = (g_min_lat - map_padding, g_max_lat + map_padding)
        else:
            global_xlim, global_ylim = None, None

        print(f"\nGenerating 3x3 summary grid for top {len(grid_features)} features...")
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()
        
        for idx, feature in enumerate(grid_features):
            ax = axes[idx]
            current_rank = idx + 1
            try:
                plot_shap_arrows_for_feature(
                    df=df, world=world, feature=feature,
                    baseline_lon=baseline_lon, baseline_lat=baseline_lat,
                    output_path="", arrow_scale=arrow_scale,
                    max_arrow_len=max_arrow_len, map_padding=map_padding,
                    cmap=cmap, ax=ax, show_colorbar=True, 
                    show_legend=(idx == 8), 
                    is_subplot=True, rank_num=current_rank,
                    xlim=global_xlim, ylim=global_ylim
                )
            except KeyError as e:
                ax.set_title(f"{current_rank}. {feature} (Data Missing)", fontsize=12)
                ax.axis('off')
                print(f"  [!] Skipping {feature} in grid: {e}")

        for idx in range(len(grid_features), 9):
            axes[idx].axis('off')

        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        plt.suptitle(f"{title_prefix} — Top 9 Features Summary", fontsize=20, y=0.92)
        grid_filename = f"{dataset_prefix}_Summary_3x3_SHAP_Grid.png" if dataset_prefix else "Summary_3x3_SHAP_Grid.png"
        grid_out_path = os.path.join(output_dir, grid_filename)
        plt.savefig(grid_out_path, dpi=300, bbox_inches="tight", facecolor='white')
        plt.close()
        print(f"  Saved Grid: {grid_out_path}")

# ---------------------------------------------------------------------------
# 3 Models Comparison Plotting Function
# ---------------------------------------------------------------------------

def plot_model_comparison_grid(
    shap_csv_paths: dict,
    importance_csv_paths: dict,
    models: list,
    features: list,
    output_path: str,
    arrow_scale: float = 1.0,
    max_arrow_len: float = 100.0,
    map_padding: float = 2.0,
    cmap: str = "coolwarm",
    title_prefix: str = "",
    vertical_layout: bool = False
):
    """
    Creates a grid comparing specific features across multiple models.
    Rows = Features, Columns = Models (or inverted if vertical_layout=True).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    map_data_dir = r"C:\Users\wande\Documents\Bioinformatics_2mas_2025-2026\Master_Thesis\Pcode\data_input\world_map_data\World_countries"
    try:
        world = gpd.read_file(map_data_dir)
    except Exception as e:
        raise FileNotFoundError(f"Could not load world map from:\n  {map_data_dir}\nError: {e}")

    # Load data for all models
    data = {}
    for m in models:
        if m in shap_csv_paths and m in importance_csv_paths:
            df, imp = _load_data(shap_csv_paths[m], importance_csv_paths[m])
            baseline_lon, baseline_lat = _baseline(df)
            data[m] = {"df": df, "imp": imp, "base_lon": baseline_lon, "base_lat": baseline_lat}

    if not data:
        print(f"  [!] No data loaded for model comparison.")
        return

    # Calculate global map limits
    g_min_lon, g_max_lon = float('inf'), float('-inf')
    g_min_lat, g_max_lat = float('inf'), float('-inf')

    np.random.seed(42)
    for m in data:
        df = data[m]["df"]
        lons_j = df["Longitude"].values # + RANDOM JITTER
        lats_j = df["Latitude"].values # + RANDOM JITTER
        
        for feature in features:
            shap_lon_col = f"shap_lon_{feature}"
            shap_lat_col = f"shap_lat_{feature}"
            if shap_lon_col in df.columns and shap_lat_col in df.columns:
                dx_raw = df[shap_lon_col].values / arrow_scale
                dy_raw = df[shap_lat_col].values / arrow_scale
                dx, dy = _cap_magnitude(dx_raw, dy_raw, max_arrow_len / arrow_scale)

                g_min_lon = min(g_min_lon, np.min(lons_j), np.min(lons_j + dx), data[m]["base_lon"])
                g_max_lon = max(g_max_lon, np.max(lons_j), np.max(lons_j + dx), data[m]["base_lon"])
                g_min_lat = min(g_min_lat, np.min(lats_j), np.min(lats_j + dy), data[m]["base_lat"])
                g_max_lat = max(g_max_lat, np.max(lats_j), np.max(lats_j + dy), data[m]["base_lat"])
    np.random.seed(None)

    if g_min_lon != float('inf'):
        global_xlim = (g_min_lon - map_padding, g_max_lon + map_padding)
        global_ylim = (g_min_lat - map_padding, g_max_lat + map_padding)
    else:
        global_xlim, global_ylim = None, None

    # Plot grid setup is now initialized ONCE before the loop
    if vertical_layout:
        nrows = len(models)
        ncols = len(features)
        fig, axes = plt.subplots(nrows, ncols, figsize=(8, 3.5 * nrows), constrained_layout=True)
    else:
        nrows = len(features)
        ncols = len(models)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    
    # Standardize to 2D array if 1D
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    LARGE_FEATURES = {"Pb", "Fe"}

    for i, feature in enumerate(features):
        for j, m in enumerate(models):
            if vertical_layout:
                row_idx, col_idx = j, i
            else:
                row_idx, col_idx = i, j
                
            ax = axes[row_idx, col_idx]
            if m not in data:
                ax.axis('off')
                continue
            
            df = data[m]["df"]
            imp = data[m]["imp"]
            
            # Find the feature rank in this specific model
            try:
                rank = imp.index[imp['Feature'] == feature][0] + 1
            except IndexError:
                rank = None

            is_large = feature in LARGE_FEATURES
            tick_sz  = 16 if is_large else 11
            legend_sz = 14 if is_large else 9
            title_sz  = 15 if is_large else 12
            cbar_sz   = 13 if is_large else 8
            
            try:
                plot_shap_arrows_for_feature(
                    df=df, world=world, feature=feature,
                    baseline_lon=data[m]["base_lon"], baseline_lat=data[m]["base_lat"],
                    output_path="", arrow_scale=arrow_scale,
                    max_arrow_len=max_arrow_len, map_padding=map_padding,
                    cmap=cmap, ax=ax, show_colorbar=True,
                    show_legend=(row_idx == nrows - 1 and col_idx == ncols - 1),
                    is_subplot=True, rank_num=rank,
                    xlim=global_xlim, ylim=global_ylim,
                    tick_labelsize=tick_sz,
                    legend_fontsize=legend_sz,
                    title_fontsize=title_sz,
                    cbar_label_fontsize=cbar_sz,
                )
                
                # Remove individual feature titles for the grid to clean it up
                ax.set_title("")
                
                # Add Model name as column header on the top row and push it up to avoid the value legend
                if vertical_layout:
                    y_offset = 1.05 
                    ax.annotate(m, xy=(0.5, y_offset), xycoords='axes fraction', ha='center', fontsize=16, fontweight='bold')
                    
                    # Hide X-axis labels for all but the bottom plot to pack them flush
                    if row_idx < nrows - 1:
                        ax.set_xlabel("")
                        ax.tick_params(labelbottom=False)

                elif i == 0:
                    y_offset = 1.12
                    ax.annotate(m, xy=(0.5, y_offset), xycoords='axes fraction', ha='center', fontsize=18, fontweight='bold')
                    
            except KeyError:
                ax.set_title(f"Data Missing for {feature}", fontsize=12)
                ax.axis('off')

    # constrained_layout handles vertical packing automatically, so we only use subplots_adjust if NOT vertical
    if not vertical_layout:
        plt.subplots_adjust(wspace=0.3, hspace=0.1)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"  Saved Comparison Grid: {output_path}")
"""
basic_DE.py
----------------------------
Functions script to perform basic data exploration on geochemical metadata.

Input data:
1. Preprocessed multi element datasets.

Generates and saves:
1. Frequency tables for metadata variables, detailing sample occurrences per category.
2. General dataset overviews comparing samples and feature counts.
3. Individual feature lists detailing categorical/metadata and numerical feature names, and coordinate ranges.
4. Geographic distribution plots.
5. Multi-panel subplots comparing selected genera.
6. NA and LOD percentage tables for numerical features.
"""

import os
import math
import pandas as pd
from typing import Literal

# geographical plotting
import geopandas as gpd
import matplotlib.pyplot as plt

import numpy as np

# ----------------------------------------------
# Data loading
# ----------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """
    Loads the raw CSV dataset into a Pandas DataFrame.
    """
    return pd.read_csv(path)

# ----------------------------------------------
# Helper Functions
# ----------------------------------------------

def get_feature_splits(df: pd.DataFrame, data_type: Literal["ICPMS", "XRF"]):
    """
    Helper to slice dataframe and extract metadata column names 
    and numerical column (element) names based entirely on column index.
    """
    if data_type == "ICPMS":
        md_cols = df.columns[:16]
        num_cols = df.columns[16:]
    elif data_type == "XRF":
        md_cols = df.columns[:13]
        num_cols = df.columns[13:]
    else:
        raise ValueError("data_type must be 'ICPMS' or 'XRF'.")

    # Treat ALL columns in the metadata slice as categorical/metadata features
    cat_cols = md_cols.tolist()
    num_feature_cols = num_cols.tolist()
    
    return cat_cols, num_feature_cols

# ----------------------------------------------
# Data Exploration Functions
# ----------------------------------------------

def calculate_categorical_frequencies(
    df: pd.DataFrame, 
    data_type: Literal["ICPMS", "XRF"],
    base_output_path: str,
    data_name: str,
    filename: str = "categorical_freq"
) -> None:
    """
    Slices the dataframe to grab metadata columns based on data_type, 
    and calculates category frequencies for all metadata columns.
    """
    cat_cols, _ = get_feature_splits(df, data_type)
    out = []
    
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False)
        total = vc.sum()
        
        for k, v in vc.items():
            if "WFID Identifier" in df.columns:
                ids = df.loc[df[c] == k, "WFID Identifier"].tolist()
            else:
                ids = df.loc[df[c] == k].index.tolist()
                
            out.append([c, k, v, v / total, ids])

    freq_df = pd.DataFrame(
        out, 
        columns=["column", "category", "frequency", "percentage", "WFID Identifier"]
    )

    freq_num = freq_df.sort_values(["column", "frequency"], ascending=[True, False])  
    freq_alph = freq_df.sort_values(["column", "category"])

    output_num_dir = os.path.join(base_output_path, "frequencies_numeric", data_name)
    output_alph_dir = os.path.join(base_output_path, "frequencies_alphabetic", data_name)
    
    os.makedirs(output_num_dir, exist_ok=True)
    os.makedirs(output_alph_dir, exist_ok=True)

    num_path = os.path.join(output_num_dir, f"{filename}_num.csv")
    alph_path = os.path.join(output_alph_dir, f"{filename}_alph.csv")
    
    freq_num.to_csv(num_path, index=False)
    freq_alph.to_csv(alph_path, index=False)
    
    print(f"Saved numeric-sorted frequencies to: {num_path}")
    print(f"Saved alphabetic-sorted frequencies to: {alph_path}")


def create_general_overview(datasets: dict, out_dir: str) -> None:
    """
    Creates a side-by-side general overview of the datasets.
    datasets: dict mapping target column names to {'df': dataframe, 'type': data_type}
    """
    records = []
    for name, info in datasets.items():
        df = info['df']
        d_type = info['type']
        cat_cols, num_cols = get_feature_splits(df, d_type)
        
        # Formatted for LaTeX/Overleaf rendering via csvsimple
        records.append({
            "Dataset": name,
            "n_{samples}": df.shape[0],
            "n_{cat\_features}": len(cat_cols),
            "n_{num\_features}": len(num_cols)
        })
        
    # Transpose to get datasets as columns and metrics as rows
    overview_df = pd.DataFrame(records).set_index("Dataset").T
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "general_nsamples_overview.csv")
    overview_df.to_csv(out_path)
    print(f"Saved general dataset overview to: {out_path}")

def create_category_distribution_table(df: pd.DataFrame, category_col: str, out_dir: str, file_name: str) -> None:
    """
    Creates a table counting the number of samples per category (e.g., per Genus)
    and saves it to the specified directory.
    """
    if category_col not in df.columns:
        print(f"Column '{category_col}' not found. Cannot create distribution table.")
        return

    # Count samples per category and structure as a clean dataframe
    counts = df[category_col].value_counts(dropna=False).reset_index()
    
    # Column formatted for LaTeX rendering
    counts.columns = [category_col, 'n_{samples}']

    # Ensure directory exists and save
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)
    counts.to_csv(out_path, index=False)
    
    print(f"Saved {category_col} distribution overview to: {out_path}")

def create_individual_feature_lists(df: pd.DataFrame, data_name: str, data_type: str, base_out_dir: str) -> None:
    """
    Creates an individual table detailing all metadata and numerical feature names,
    along with latitude and longitude ranges.
    """
    cat_cols, num_cols = get_feature_splits(df, data_type)
    
    # find latitude and longitude columns
    lat_col = next((col for col in df.columns if col.lower() in ['lat', 'latitude']), None)
    lon_col = next((col for col in df.columns if col.lower() in ['lon', 'longitude']), None)
    
    lat_range = f"{df[lat_col].min()} to {df[lat_col].max()}" if lat_col else "N/A"
    lon_range = f"{df[lon_col].min()} to {df[lon_col].max()}" if lon_col else "N/A"
    
    # Rows formatted for LaTeX rendering
    data = {
        "Attribute": ["cat features", "num features", "lat_{range}", "lon_{range}"],
        "Details": [", ".join(cat_cols), ", ".join(num_cols), lat_range, lon_range]
    }
    feat_df = pd.DataFrame(data)
    
    target_dir = os.path.join(base_out_dir, data_name)
    os.makedirs(target_dir, exist_ok=True)
    
    safe_name = os.path.basename(data_name)
    out_path = os.path.join(target_dir, f"{safe_name}_feature_overview.csv")
    
    feat_df.to_csv(out_path, index=False)
    print(f"Saved feature overview for {safe_name} to: {out_path}")


def create_na_lod_table(df: pd.DataFrame, data_name: str, base_out_dir: str, data_type: Literal["ICPMS", "XRF"]) -> None:
    """
    Creates a table computing the percentage of NA values and LOD values (starting with '<')
    for specifically the numerical features/elements in the dataframe.
    """
    total_samples = len(df)
    if total_samples == 0:
        print(f"DataFrame for {data_name} is empty. Skipping NA/LOD table.")
        return

    # Fetch only the numerical columns based on the data type split
    _, num_cols = get_feature_splits(df, data_type)

    records = []
    for col in num_cols:
        # Prevent key errors if a split column somehow isn't in the dataframe
        if col not in df.columns:
            continue
            
        # Calculate % NA
        na_count = df[col].isna().sum()
        na_pct = (na_count / total_samples) * 100
        
        # Calculate % LOD
        lod_count = df[col].astype(str).str.startswith("<").sum()
        lod_pct = (lod_count / total_samples) * 100
        
        records.append({
            "Feature": col,
            "% NA": round(na_pct, 4),
            "% LOD": round(lod_pct, 4)
        })
        
    na_lod_df = pd.DataFrame(records)
    
    # Target dir inherits structure like 'sI' or 'tX/tX_All_Genera'
    target_dir = os.path.join(base_out_dir, data_name)
    os.makedirs(target_dir, exist_ok=True)
    
    safe_name = os.path.basename(data_name)
    out_path = os.path.join(target_dir, f"{safe_name}_NA_LOD_table.csv")
    
    na_lod_df.to_csv(out_path, index=False)
    print(f"Saved NA/LOD table for {safe_name} to: {out_path}")

# ----------------------------------------------
# Geospatial Plotting Functions
# ----------------------------------------------

def create_geoplot(df: pd.DataFrame, data_name: str, base_out_dir: str, world_map: gpd.GeoDataFrame) -> None:
    """
    Plots the dataset's coordinates on a world map and saves the image to a dedicated subdirectory.
    """
    lat_col = next((col for col in df.columns if col.lower() in ['lat', 'latitude']), None)
    lon_col = next((col for col in df.columns if col.lower() in ['lon', 'longitude']), None)
    
    if not lat_col or not lon_col:
        print(f"Missing latitude or longitude columns in {data_name}. Skipping geoplot.")
        return

    plot_df = df.dropna(subset=[lat_col, lon_col])
    if plot_df.empty:
        print(f"No valid coordinates found for {data_name}. Skipping geoplot.")
        return

    # Add jitter to the coordinates
    # HIDDEN

    gdf = gpd.GeoDataFrame(
        plot_df, 
        geometry=gpd.points_from_xy(plot_df[lon_col] + jitter_lon, plot_df[lat_col] + jitter_lat),
        crs="EPSG:4326" 
    )

    # WIDENED
    fig, ax = plt.subplots(figsize=(16, 8))
    world_map.plot(ax=ax, color='#E0E0E0', edgecolor='white', linewidth=0.5)
    
    # LARGER & MORE TRANSPARENT SAMPLES
    gdf.plot(ax=ax, color='#D32F2F', markersize=250, alpha=0.6, edgecolor='black', linewidth=0.3)

    # Define the mapping for clean titles
    name_map = {
        "sI": r"$\mathit{Glycine\ max}$ (Soy)",
        "cI": r"$\mathit{Theobroma\ cacao}$ (Cocoa)",
        "tX_All_Genera": "Timber (All Genera)"
    }

    # Extract the name and look it up in the map
    safe_name = os.path.basename(data_name)
    clean_name = name_map.get(safe_name, safe_name) # Defaults to the genus name if not in map

    # Apply the title
    ax.set_title(clean_name, fontsize=16, pad=20)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude",  fontsize=12)

    ax.tick_params(axis='both', which='major', labelsize=12)

    minx, miny, maxx, maxy = gdf.total_bounds
    padding = 5 
    ax.set_xlim(minx - padding, maxx + padding)
    ax.set_ylim(miny - padding, maxy + padding)
    ax.grid(True, linestyle='--', alpha=0.5)

    target_dir = os.path.join(base_out_dir, data_name)
    os.makedirs(target_dir, exist_ok=True)
    out_path = os.path.join(target_dir, f"{safe_name}_map.png")

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close() 
    print(f"Saved geoplot for {safe_name} to: {out_path}")


import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def create_genera_subplots(df: pd.DataFrame, selected_genera: list, base_out_dir: str, world_map: gpd.GeoDataFrame) -> None:
    """
    Creates a multi-panel subplot grid for specified genera, dynamically adjusting grid size.
    """
    if not selected_genera:
        print("No genera selected for subplots. Skipping.")
        return

    lat_col = next((col for col in df.columns if col.lower() in ['lat', 'latitude']), None)
    lon_col = next((col for col in df.columns if col.lower() in ['lon', 'longitude']), None)

    if not lat_col or not lon_col:
        print("Missing latitude or longitude columns. Skipping subplots.")
        return

    # Filter dataframe and drop missing coordinates
    plot_df = df[df['Genus'].isin(selected_genera)].dropna(subset=[lat_col, lon_col])

    if plot_df.empty:
        print("No valid coordinates found for selected genera. Skipping subplots.")
        return

    # Determine dynamic grid layout
    num_plots = len(selected_genera)
    cols = 2
    rows = math.ceil(num_plots / cols)

    # WIDENED 
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8 * rows))
    
    # Handle the case where there's only 1 row or 1 plot safely
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, genus in enumerate(selected_genera):
        ax = axes[i]
        genus_df = plot_df[plot_df['Genus'] == genus]
        
        # Plot base map
        world_map.plot(ax=ax, color='#E0E0E0', edgecolor='white', linewidth=0.5)

        if not genus_df.empty:
            # Add jitter
            # HIDDEN

            gdf = gpd.GeoDataFrame(
                genus_df, 
                geometry=gpd.points_from_xy(genus_df[lon_col] + jitter_lon, genus_df[lat_col] + jitter_lat),
                crs="EPSG:4326"
            )
            
            # LARGER & MORE TRANSPARENT SAMPLES 
            gdf.plot(ax=ax, color='#D32F2F', markersize=200, alpha=0.6, edgecolor='black', linewidth=0.3)

            # LARGER NUMBERS: 
            ax.tick_params(axis='both', which='major', labelsize=12)

            # Dynamic bounding box for each specific genus
            minx, miny, maxx, maxy = gdf.total_bounds
            padding = 2.5
            ax.set_xlim(minx - padding, maxx + padding)
            ax.set_ylim(miny - padding, maxy + padding)

        ax.set_title(f"{genus}", fontsize=20, pad=10)
        ax.set_xlabel("Longitude", fontsize=16)
        ax.set_ylabel("Latitude", fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Hide any unused subplots if the number of genera isn't perfectly even
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save 
    target_dir = os.path.join(base_out_dir, "tX", "tX_All_Genera")
    os.makedirs(target_dir, exist_ok=True)
    out_path = os.path.join(target_dir, "genera_comparison_subplots.png")

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved genera subplots grid to: {out_path}")

# ----------------------------------------------
# Main dispatcher
# ----------------------------------------------

def run_all(
    input_path: str, 
    base_output_path: str, 
    data_name: str,
    data_type: Literal["ICPMS", "XRF"]
) -> None:
    """
    Executes the full metadata frequency pipeline.
    """
    print(f"Starting Metadata Frequency Pipeline for {os.path.basename(data_name)}...")
    
    df = load_data(input_path)
    print(f"Loaded data with {df.shape[0]} samples and {df.shape[1]} total columns.")
    
    calculate_categorical_frequencies(df, data_type, base_output_path, data_name)
    
    print("Pipeline execution complete.")
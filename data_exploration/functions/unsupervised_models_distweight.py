"""
unsupervised_pipeline.py
----------------------------
Functions script to perform and compare Principal Component Analysis (PCA), spatial PCA (sPCA), 
and Maximum Autocorrelation Factors (MAF) on spatial trace element datasets.

Input data:
1. Preprocessed multi-element CSV datasets (ICPMS or XRF) containing numerical trace features and spatial coordinates.
2. Local shapefiles for geographical world map overlays.

Generates and saves for every method (PCA, sPCA, MAF):
1. Ranked lists of feature loadings per component in both long and wide formats (CSV).
2. Full feature loading matrices and sample factor scores (CSV).
3. Eigenvalue summaries detailing variance explained, spatial structure types, or spatial difference 
    variance depending on the algorithm used (CSV).
4. Moran's Eigenvector Map (MEM) permutation test results for global and local spatial structures (CSV, sPCA only).
5. Side-by-side biplots combining geographical K-Means cluster maps with component space feature vectors 
    and confidence ellipses (PNG).
6. Method-specific scree plots detailing eigenvalues or variance across components (PNG).
7. Geographical spatial maps displaying sample scores mapped to marker sizes and colors, overlaid with 
    the Gabriel spatial connection graph (PNG).
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, Literal
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.linalg import eigh
import libpysal


# -----------------------------------------------------------------------
# Data loading & preprocessing
# -----------------------------------------------------------------------

def load_and_preprocess_data(
    path: str,
    data_type: Literal["ICPMS", "XRF"],
    transform_type: Literal["standard", "CLR"] = "standard",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads in data based on its type.
    Numerical and metadata cols are split specifically per dataset. 
    Coordinates are extracted. 
    Standard: log transform + standardization.
    """

    df = pd.read_csv(path)

    if data_type == "ICPMS":
        trace_cols = df.columns[16:]
    elif data_type == "XRF":
        trace_cols = df.columns[13:]
    else:
        raise ValueError("data_type must be 'ICPMS' or 'XRF'.")

    X = df[trace_cols].select_dtypes(include="number")
    y = df[["Longitude", "Latitude"]]

    if transform_type == "CLR":
        X_log = np.log1p(X)
        row_means = X_log.mean(axis=1)
        X_processed = X_log.sub(row_means, axis=0)
    else:
        X_log = np.log1p(X) # use log1p to avoid numerical issues with values close to 0
        scaler = StandardScaler()
        X_processed = pd.DataFrame(
            scaler.fit_transform(X_log), columns=X.columns, index=X.index
        )

    return X_processed, y, df


# -----------------------------------------------------------------------
# Spatial graph (Binary, for sPCA)
# -----------------------------------------------------------------------

def create_gabriel_graph(
    coords_df: pd.DataFrame,
    offset_magnitude: float = 1e-4,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Builds a Gabriel Graph and returns the **binary** weight matrix W
    (values in {0, 1}, diagonal 0) plus potentially jittered coordinates.
    """
    coords = coords_df.values.astype(float).copy()

    if coords_df.duplicated(keep=False).any():
        rng = np.random.default_rng(seed=42)
        coords = coords + rng.uniform(-offset_magnitude, offset_magnitude, size=coords.shape)

    w_gabriel = libpysal.weights.Gabriel(coords)
    W_binary, _ = w_gabriel.full()          # binary {0, 1}

    coords_jittered_df = pd.DataFrame(
        coords, index=coords_df.index, columns=coords_df.columns
    )
    return W_binary, coords_jittered_df


def _row_standardise(W_binary: np.ndarray) -> np.ndarray:
    """
    Row-standardise a weight matrix so each row sums to 1.
    """
    row_sums = W_binary.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1       # guard against isolated nodes
    return W_binary / row_sums


# -----------------------------------------------------------------------
# Spatial graph (Inverse-Haversine weighted, for MAF)
# -----------------------------------------------------------------------

def haversine_km(lon1: np.ndarray, lat1: np.ndarray,
                 lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """
    Haversine great-circle distance in kilometres 
    (vectorised for inverse weighted connection network).
    """
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * 2.0 * np.arcsin(np.sqrt(a))

def create_weighted_gabriel_graph(
    coords_df: pd.DataFrame,
    offset_magnitude: float = 1e-4,
    min_distance_km: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Builds an inverse-Haversine weighted Gabriel graph for MAF.
    Returns:
      W_binary        : binary adjacency matrix (for edge enumeration)
      W_row_norm      : row-normalised inverse-distance weight matrix
      coords_jittered : (potentially jittered) coordinate DataFrame
    """
    # Base the topology on the exact same jittered coordinates as the binary graph
    W_binary, coords_jittered_df = create_gabriel_graph(coords_df, offset_magnitude)
    coords = coords_jittered_df.values

    # Pairwise Haversine distances
    lon = coords[:, 0]
    lat = coords[:, 1]
    lon_i, lon_j = np.meshgrid(lon, lon, indexing="ij")
    lat_i, lat_j = np.meshgrid(lat, lat, indexing="ij")
    D_km = haversine_km(lon_i, lat_i, lon_j, lat_j)

    # Inverse-distance weights on Gabriel edges only
    with np.errstate(divide="ignore", invalid="ignore"):
        D_safe = np.where(D_km < min_distance_km, min_distance_km, D_km)
        W_inv  = np.where(W_binary == 1, 1.0 / D_safe, 0.0)
    np.fill_diagonal(W_inv, 0.0)

    # Row normalisation (each row sums to 1, preserving neighbour-mean semantics)
    row_sums = W_inv.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0          # guard isolated nodes
    W_row_norm = W_inv / row_sums

    return W_binary, W_row_norm, coords_jittered_df


# -----------------------------------------------------------------------
# Standard PCA
# -----------------------------------------------------------------------

def perform_pca(
    X_scaled: pd.DataFrame,
    n_components: int = None,
) -> Tuple[PCA, np.ndarray, pd.Index]:
    """
    Standard PCA via sklearn.
    """
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    pca.components_, scores = _resolve_signs(pca.components_, scores)

    return pca, scores, X_scaled.columns


# -----------------------------------------------------------------------
# Spatial PCA  —  Jombart et al. (2008)
# -----------------------------------------------------------------------

def _mem_global_local_test(
    X_val: np.ndarray,
    L: np.ndarray,
    n_permutations: int = 9999,
    rng_seed: int = 42,
) -> dict:
    """
    MEM-based permutation tests for global and local spatial structure.
    Based on Jombart et al. (2008)
    """
    rng = np.random.default_rng(rng_seed)
    n, p = X_val.shape

    H = np.eye(n) - np.ones((n, n)) / n
    L_sym = H @ ((L + L.T) / 2) @ H
    mem_eigenvalues, MEMs = eigh(L_sym)      # ascending order; eigh valid for symmetric
    # Drop the near-zero eigenvalue (constant vector, I ≈ 0)
    nonzero = np.abs(mem_eigenvalues) > 1e-10
    mem_eigenvalues = mem_eigenvalues[nonzero]
    MEMs = MEMs[:, nonzero]

    # Separate into global (I > 0) and local (I < 0) MEMs
    global_mask = mem_eigenvalues > 0
    local_mask  = mem_eigenvalues < 0

    E_pos = MEMs[:, global_mask]   # global MEMs
    E_neg = MEMs[:, local_mask]    # local MEMs

    n_global_mems = int(global_mask.sum())
    n_local_mems  = int(local_mask.sum())

    def _max_mean_r2(X_perm: np.ndarray, E: np.ndarray) -> float:
        if E.shape[1] == 0:
            return 0.0
        # centrer X columns (already centred by StandardScaler, but to be safe)
        Xc = X_perm - X_perm.mean(axis=0)
        # correlations: (n_mems x p)
        E_norm = E / np.linalg.norm(E, axis=0, keepdims=True)
        X_norm = Xc / (np.linalg.norm(Xc, axis=0, keepdims=True) + 1e-15)
        corr   = E_norm.T @ X_norm          # (n_mems x p)
        r2     = corr ** 2                  # (n_mems x p)
        mean_r2_per_mem = r2.mean(axis=1)   # (n_mems,)
        return float(mean_r2_per_mem.max())

    # Observed test statistics 
    obs_global = _max_mean_r2(X_val, E_pos) # all positive eigenvalue MEMs
    obs_local  = _max_mean_r2(X_val, E_neg) # all negative eigenvalue MEMs

    # Permutation distribution under H0
    perm_global = np.empty(n_permutations)
    perm_local  = np.empty(n_permutations)
    idx = np.arange(n)
    for k in range(n_permutations):
        rng.shuffle(idx)
        X_perm       = X_val[idx]
        perm_global[k] = _max_mean_r2(X_perm, E_pos)
        perm_local[k]  = _max_mean_r2(X_perm, E_neg)

    # P-values (proportion of permuted stats >= observed) 
    global_pval = float((perm_global >= obs_global).sum() + 1) / (n_permutations + 1)
    local_pval  = float((perm_local  >= obs_local ).sum() + 1) / (n_permutations + 1)

    return {
        "global_stat":    obs_global,
        "global_pvalue":  global_pval,
        "local_stat":     obs_local,
        "local_pvalue":   local_pval,
        "n_global_mems":  n_global_mems,
        "n_local_mems":   n_local_mems,
    }


def perform_spca(
    X_scaled: pd.DataFrame,
    coords_df: pd.DataFrame,
    n_permutations: int = 9999,
) -> Tuple[SimpleNamespace, np.ndarray, pd.Index]:
    """
    Spatial PCA following Jombart et al. (2008).
    Uses binary row-standardised Gabriel graph.
    """
    X_val = X_scaled.values
    n = X_val.shape[0]

    # Binary Gabriel graph -> symmetrise -> row-standardise once -> L
    W_binary, _ = create_gabriel_graph(coords_df)
    L = _row_standardise(W_binary)

    # Criterion matrix: (1/2n) X'(L + L')X
    criterion = X_val.T @ (L + L.T) @ X_val / (2.0 * n)
    criterion = (criterion + criterion.T) / 2.0   # enforce numerical symmetry

    eigenvalues, V = eigh(criterion)              # ascending order from eigh

    # Descending: global (most +) first, local (most -) last
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, V = eigenvalues[idx], V[:, idx]

    scores = X_val @ V

    comps_fixed, scores = _resolve_signs(V.T, scores)

    # --- MEM-based global / local tests ---
    mem_tests = _mem_global_local_test(X_val, L, n_permutations=n_permutations)

    model = SimpleNamespace()
    model.components_ = comps_fixed        # for same orientation
    model.explained_variance_ratio_ = eigenvalues # THIS IS NOT THE EXPLAINED VARIANCE BUT IS NAMED THIS WAY FOR EASIER HANDLING ACROSS METHODS
    model.n_components_             = V.shape[1]
    model.n_global                  = int((eigenvalues > 0).sum())
    model.n_local                   = int((eigenvalues < 0).sum())
    # attach test results
    model.global_stat               = mem_tests["global_stat"]
    model.global_pvalue             = mem_tests["global_pvalue"]
    model.local_stat                = mem_tests["local_stat"]
    model.local_pvalue              = mem_tests["local_pvalue"]
    model.n_global_mems             = mem_tests["n_global_mems"]
    model.n_local_mems              = mem_tests["n_local_mems"]

    return model, scores, X_scaled.columns


# -----------------------------------------------------------------------
# Maximum Autocorrelation Factors  —  Switzer & Green (1984)
# -----------------------------------------------------------------------

def perform_maf(
    X_scaled: pd.DataFrame,
    coords_df: pd.DataFrame,
    variance_threshold: float = 0.99,
) -> Tuple[SimpleNamespace, np.ndarray, pd.Index]:
    """
    Maximum Autocorrelation Factors (MAF) with inverse-Haversine weighted graph.
    """
    X_val = X_scaled.values
    n = X_val.shape[0]

    # PCA truncation to guarantee non-singular Sigma_0 
    pca = PCA().fit(X_val)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    nf = int(np.where(cum_var >= variance_threshold)[0][0]) + 1

    Z     = pca.transform(X_val)[:, :nf]      # (n x nf) PCA scores
    V_pca = pca.components_.T[:, :nf]         # (p x nf) PCA loadings

    # Gabriel graph: binary adjacency for edge enumeration 
    W_binary, W_row_norm, coords_jittered = create_weighted_gabriel_graph(coords_df)

    rows, cols = np.where(W_binary == 1)
    edge_mask  = rows < cols                   # each undirected edge once
    i_idx      = rows[edge_mask]
    j_idx      = cols[edge_mask]
    diffs      = Z[i_idx] - Z[j_idx]          # (n_edges x nf)
    n_edges    = diffs.shape[0]
    diffs_c = diffs - diffs.mean(axis=0)      # to calculate covariance
    Sigma_h = diffs_c.T @ diffs_c / (n_edges - 1)     # (nf x nf)

    # Sigma_0: overall covariance in PCA space
    Sigma_0 = Z.T @ Z / n                     # (nf x nf)
    Sigma_h = (Sigma_h + Sigma_h.T) / 2.0
    Sigma_0 = (Sigma_0 + Sigma_0.T) / 2.0

    # Generalised eigenvalue problem (ascending = max autocorr first)
    eigenvalues, U = eigh(Sigma_h, Sigma_0)

    # Map GEP eigenvectors back to original feature space 
    loadings = V_pca @ U                      # (p x nf)
    scores   = X_val @ loadings               # (n x nf)

    comps_fixed, scores = _resolve_signs(loadings.T, scores)

    # Pack results
    model = SimpleNamespace()
    model.components_ = comps_fixed   # for same orientation
    model.explained_variance_ratio_ = eigenvalues  # spatial diff variances (ascending)
    model.n_components_             = nf
    model.coords_jittered           = coords_jittered

    return model, scores, X_scaled.columns


# -----------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------

def _make_dirs(out_dir: str):
    os.makedirs(os.path.join(out_dir, "plots"),  exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)


def export_tables(
    model,
    scores: np.ndarray,
    features: pd.Index,
    out_dir: str,
    method: str,
    data_name: str,
):
    """
    outputs:
      ranked_loadings.csv          - loadings sorted by |value| per component (long format)
      {method}_top_features_by_rank.csv - features sorted by loading rank per component (wide format)
      loadings_matrix.csv             - full (features x components) loadings
      scores.csv                - (samples x components) factor scores
      eigenvalue_summary.csv        - per-component eigenvalue + interpretation label
    """
    tables_dir = os.path.join(out_dir, "tables")
    n_comp = model.n_components_
    comp_labels = [f"Comp{i+1}" for i in range(n_comp)]
    prefix = data_name.replace('\\', '_').replace('/', '_') + "_"

    # loadings matrixx
    loadings = model.components_.T
    df_load = pd.DataFrame(loadings, columns=comp_labels, index=features)
    df_load.index.name = "Feature"
    df_load.to_csv(os.path.join(tables_dir, f"{prefix}loadings_matrix.csv"))

    # ranked loadings (long format) AND top features by rank (wide format)
    rows = []
    top_features_dict = {"Rank": list(range(1, len(features) + 1))}
    
    for col in comp_labels:
        sorted_series = df_load[col].abs().sort_values(ascending=False)
        top_features_dict[col] = sorted_series.index.tolist()
        
        for rank, (feat, absval) in enumerate(sorted_series.items(), start=1):
            rows.append({
                "Component":        col,
                "Rank":             rank,
                "Feature":          feat,
                "Absolute_Loading": absval,
                "Actual_Loading":   df_load.loc[feat, col],
            })
            
    # Save long format
    pd.DataFrame(rows).to_csv(
        os.path.join(tables_dir, f"{prefix}ranked_loadings.csv"), index=False
    )
    
    # Save wide format (rows: Rank, cols: Comp1, Comp2...) with method in filename
    pd.DataFrame(top_features_dict).to_csv(
        os.path.join(tables_dir, f"{prefix}{method}_top_features_by_rank.csv"), index=False
    )

    # scores
    df_scores = pd.DataFrame(scores, columns=comp_labels)
    df_scores.index.name = "Sample"
    df_scores.to_csv(os.path.join(tables_dir, f"{prefix}scores.csv"))

    # eigenvalue summary (NOW INCLUDES PERCENTAGES)
    evals = model.explained_variance_ratio_
    
    if method == "PCA":
        pd.DataFrame({
            "Component":                comp_labels,
            "Explained_Variance_Ratio": evals,
            "Explained_Variance_Pct":   evals * 100.0,
            "Cumulative_Variance_Pct":  np.cumsum(evals) * 100.0
        }).to_csv(os.path.join(tables_dir, f"{prefix}eigenvalue_summary.csv"), index=False)

    elif method == "sPCA":
        types = ["global" if e > 0 else ("local" if e < 0 else "neutral")
                 for e in evals]
        
        # Percentage based on absolute spatial eigenvalues (relative structural magnitude)
        abs_evals = np.abs(evals)
        pct = (abs_evals / np.sum(abs_evals)) * 100.0
        
        pd.DataFrame({
            "Component":              comp_labels,
            "Spatial_Eigenvalue":     evals,
            "Absolute_Magnitude_Pct": pct,
            "Structure_Type":         types,
        }).to_csv(os.path.join(tables_dir, f"{prefix}eigenvalue_summary.csv"), index=False)

        # Also write a concise test-level summary
        pd.DataFrame([{
            "Test":         "Global (MEM)",
            "n_MEMs":       model.n_global_mems,
            "Statistic":    model.global_stat,
            "P_Value":      model.global_pvalue,
            "Significant":  model.global_pvalue < 0.05,
        }, {
            "Test":         "Local (MEM)",
            "n_MEMs":       model.n_local_mems,
            "Statistic":    model.local_stat,
            "P_Value":      model.local_pvalue,
            "Significant":  model.local_pvalue < 0.05,
        }]).to_csv(os.path.join(tables_dir, f"{prefix}mem_tests.csv"), index=False)

    elif method == "MAF":
        pd.DataFrame({
            "Component":             comp_labels,
            "Spatial_Diff_Variance": evals,
            "Autocorrelation_Rank":  range(1, len(evals) + 1),
        }).to_csv(os.path.join(tables_dir, f"{prefix}eigenvalue_summary.csv"), index=False)

def _resolve_signs(components: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forces a deterministic sign for eigenvectors, to make plots have same orientation
    components: (n_components, n_features)
    scores: (n_samples, n_components)
    """
    # Find the index of the maximum absolute value in each component (row)
    max_abs_idx = np.argmax(np.abs(components), axis=1)
    
    # Get the sign of that maximum value
    signs = np.sign(components[np.arange(components.shape[0]), max_abs_idx])
    
    # Multiply the component and the score by that sign to force it positive
    components_fixed = components * signs[:, np.newaxis]
    scores_fixed = scores * signs
    
    return components_fixed, scores_fixed

# -----------------------------------------------------------------------
# Confidence ellipse
# -----------------------------------------------------------------------

def _draw_confidence_ellipse(ax, x, y, color, n_std: float = 2.4477):
    """
    95% confidence ellipse (n_std=2.4477 for 2D chi-squared).
    """
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    evals, evecs = np.linalg.eigh(cov)
    order = evals.argsort()[::-1]
    evals, evecs = evals[order], evecs[:, order]
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    w, h  = 2 * n_std * np.sqrt(np.maximum(evals, 0))
    for fc, ls in [(color, "-"), ("none", "--")]:
        ax.add_patch(Ellipse(
            xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle,
            edgecolor=color, facecolor=fc,
            alpha=0.10 if fc != "none" else 1.0,
            linewidth=1.5, linestyle=ls,
        ))


# -----------------------------------------------------------------------
# Axis labels  (method-specific)
# -----------------------------------------------------------------------

def _axis_labels(model, method: str) -> Tuple[str, str]:
    evals = model.explained_variance_ratio_
    if method == "PCA":
        evr = evals
        return (f"PC1 ({evr[0]*100:.1f}% var)",
                f"PC2 ({evr[1]*100:.1f}% var)")
    if method == "sPCA":
        def _tag(e):
            return "global" if e > 0 else ("local" if e < 0 else "neutral")
        return (f"sPCA Comp1 [{_tag(evals[0])}, lambda={evals[0]:.4f}]",
                f"sPCA Comp2 [{_tag(evals[1])}, lambda={evals[1]:.4f}]")
    if method == "MAF":
        return (f"MAF1 (diff var={evals[0]:.4f}, rank=1)",
                f"MAF2 (diff var={evals[1]:.4f}, rank=2)")
    return "Comp1", "Comp2"


# -----------------------------------------------------------------------
# Biplot  (shared by all three methods)
# -----------------------------------------------------------------------

def plot_biplot(
    model,
    scores:       np.ndarray,
    features:     pd.Index,
    coords_df:    pd.DataFrame,
    out_dir:      str,
    data_name:    str,
    method:       str,
    group_labels: np.ndarray,
    map_data_dir: str,
    group_name:   str = "Clusters",
):
    """
    Side-by-side:
      Left  - geographical sample positions coloured by spatial K-Means cluster (overlaid on map).
      Right - biplot (Comp1 x Comp2) with loading arrows.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    unique_groups = np.unique(group_labels)
    # Brighter, colorblind-friendly categorical colormap
    colors = cm.get_cmap("Set2", max(8, len(unique_groups)))
    coords = coords_df.values

    # Add visual jitter for map plot only
    rng = np.random.default_rng(seed=42)
    # map_coords = # HIDDEN

    # LEFT - geographical space (now overlaid on map)
    # Load and plot the world map
    try:
        world = gpd.read_file(map_data_dir)
        world.plot(ax=ax1, color='lightgrey', edgecolor='white')
    except Exception as e:
        raise FileNotFoundError(f"Could not load world map from:\n  {map_data_dir}\nError: {e}") 

    # Plot the samples colored by cluster (using map_coords, larger size, more vague)
    for idx, grp in enumerate(unique_groups):
        mask = group_labels == grp
        ax1.scatter(map_coords[mask, 0], map_coords[mask, 1],
                    alpha=0.5, color=colors(idx), edgecolors="black",
                    s=250, label=f"Cluster {grp}", zorder=2)
                    
    # Zoom / Formatting bounds based on sample coordinates
    pad_x = (map_coords[:, 0].max() - map_coords[:, 0].min()) * 0.1 + 1
    pad_y = (map_coords[:, 1].max() - map_coords[:, 1].min()) * 0.1 + 1
    ax1.set_xlim([map_coords[:, 0].min() - pad_x, map_coords[:, 0].max() + pad_x])
    ax1.set_ylim([map_coords[:, 1].min() - pad_y, map_coords[:, 1].max() + pad_y])

    ax1.set_xlabel("Longitude", fontsize=15, fontweight="bold")
    ax1.set_ylabel("Latitude",  fontsize=15, fontweight="bold")
    
    # Make map axis ticks bigger
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    ax1.legend(title=group_name, loc="best", framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # RIGHT - component space (uses original scores, NO jitter)
    for idx, grp in enumerate(unique_groups):
        mask = group_labels == grp
        xg, yg = scores[mask, 0], scores[mask, 1]
        ax2.scatter(xg, yg, alpha=0.8, color=colors(idx),
                    edgecolors="black", s=60)
        if len(xg) >= 3:
            _draw_confidence_ellipse(ax2, xg, yg, color=colors(idx))

    # Loading arrows - unified scale
    coeff = model.components_[:2, :].T         # (n_features, 2)
    score_range   = max(float(np.abs(scores[:, :2]).max()), 1e-9)
    loading_range = max(float(np.abs(coeff).max()), 1e-9)
    scale = (score_range / loading_range) * 0.8

    for i in range(coeff.shape[0]):
        xv, yv = coeff[i, 0] * scale, coeff[i, 1] * scale
        ax2.arrow(0, 0, xv, yv, color="darkred", alpha=0.8,
                  head_width=score_range * 0.015)
        
        ax2.text(xv * 1.1, yv * 1.1, features[i],
                 color="k", ha="center", va="center", fontsize=12)

    xlabel, ylabel = _axis_labels(model, method)
    ax2.set_xlabel(xlabel, fontsize=16, fontweight="bold")
    ax2.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    
    # Make biplot axis ticks bigger 
    ax2.tick_params(axis='both', which='major', labelsize=15)
    
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax2.axvline(0, color="gray", linewidth=1, linestyle="--")

    prefix = data_name.replace('\\', '_').replace('/', '_') + "_"
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots", f"{prefix}{method.lower()}_biplot_k{len(unique_groups)}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------
# Scree / eigenvalue plot  (shared by all three methods)
# -----------------------------------------------------------------------

def plot_eigenvalues(model, out_dir: str, data_name: str, method: str):
    """
    PCA  : % variance explained (all positive, descending).
    sPCA : raw spatial eigenvalues; blue = global (lambda>0), red = local (lambda<0).
    MAF  : spatial difference variance (ascending - MAF1 is most autocorrelated).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    evals = model.explained_variance_ratio_
    x_pos = range(1, len(evals) + 1)

    if method == "PCA":
        plot_vals  = evals * 100
        bar_colors = ["dodgerblue"] * len(evals)
        ylabel = "Explained Variance (%)"
        title  = f"PCA Scree Plot ({data_name})"

    elif method == "sPCA":
        plot_vals  = evals
        bar_colors = ["dodgerblue" if v > 0 else "darkred" for v in evals]
        ylabel = "Spatial Eigenvalue  (+ global  |  - local)"
        
        # Use getattr to safely retrieve attributes
        n_g = getattr(model, 'n_global', '?')
        n_l = getattr(model, 'n_local', '?')
        gp = getattr(model, 'global_pvalue', float('nan'))
        lp = getattr(model, 'local_pvalue',  float('nan'))
        title  = (
            f"sPCA Eigenvalue Scree Plot ({data_name})\n"
            f"Global (lambda>0): {n_g}  p={gp:.3f}   |   "
            f"Local (lambda<0): {n_l}  p={lp:.3f}"
        )

    else:   # MAF
        plot_vals  = evals
        cmap = cm.get_cmap("Blues_r", len(evals))
        bar_colors = [cmap(i) for i in range(len(evals))]
        ylabel = "Spatial Difference Variance  (smaller = more autocorrelated)"
        title  = (
            f"MAF Scree Plot ({data_name})\n"
            "MAF1 = most spatially autocorrelated factor"
        )

    ax.bar(x_pos, plot_vals, alpha=0.85, color=bar_colors, edgecolor="k")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel(ylabel,       fontsize=12, fontweight="bold")
    ax.set_xlabel("Components", fontsize=12, fontweight="bold")
    ax.set_title(title,         fontsize=13, fontweight="bold")

    prefix = data_name.replace('\\', '_').replace('/', '_') + "_"
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots", f"{prefix}{method.lower()}_scree.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_scores_on_map(
    scores: np.ndarray,
    coords_df: pd.DataFrame,
    out_dir: str,
    data_name: str,
    method: str,
    map_data_dir: str,
    pc_index: int = 0
):
    """
    Plots samples on a geographical map. 
    Marker size is proportional to the absolute score of the specified PC.
    Marker color represents the actual score value.
    Overlays the Gabriel spatial connection graph.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Load and plot the world map
    try:
        world = gpd.read_file(map_data_dir)
        world.plot(ax=ax, color='lightgrey', edgecolor='white')
    except Exception as e:
        raise FileNotFoundError(f"Could not load world map from:\n  {map_data_dir}\nError: {e}") 

    # Get Gabriel graph and coordinates (jittered if there are duplicates)
    W_binary, coords_jittered_df = create_gabriel_graph(coords_df)
    coords = coords_jittered_df.values

    # Visual jitter strictly for the map visualization
    rng = np.random.default_rng(seed=42)
    map_coords = coords # + RANDOM JITTER

    # Plot the connection graph (edges)
    rows, cols = np.where(W_binary == 1)
    for i, j in zip(rows, cols):
        if i < j:
            ax.plot([map_coords[i, 0], map_coords[j, 0]], 
                    [map_coords[i, 1], map_coords[j, 1]], 
                    color='gray', alpha=0.4, linewidth=0.6, zorder=1)

    # Plot the samples (nodes)
    target_scores = scores[:, pc_index]
    abs_scores = np.abs(target_scores)
    
    min_size = 100  # Increased for larger points
    max_size = 800  # Increased for larger points
    if abs_scores.max() == 0:
        sizes = np.full_like(abs_scores, min_size)
    else:
        sizes = min_size + (abs_scores / abs_scores.max()) * (max_size - min_size)

    scatter = ax.scatter(
        map_coords[:, 0], map_coords[:, 1],
        c=target_scores, cmap='plasma',
        s=sizes, alpha=0.5, edgecolors='black', linewidth=0.5, zorder=2  # Lower alpha for 'more vague'
    )

    # Formatting & Zooming
    pad_x = (map_coords[:, 0].max() - map_coords[:, 0].min()) * 0.1 + 1
    pad_y = (map_coords[:, 1].max() - map_coords[:, 1].min()) * 0.1 + 1
    ax.set_xlim([map_coords[:, 0].min() - pad_x, map_coords[:, 0].max() + pad_x])
    ax.set_ylim([map_coords[:, 1].min() - pad_y, map_coords[:, 1].max() + pad_y])

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label(f"Score Value (Component {pc_index + 1})", fontweight="bold")
    
    ax.set_xlabel("Longitude", fontweight="bold")
    ax.set_ylabel("Latitude", fontweight="bold")
    ax.set_title(
        f"{method} - Component {pc_index + 1} Scores on Map ({data_name})\n"
        "Marker Size ~ Absolute Score", 
        fontsize=14, fontweight="bold"
    )
    
    prefix = data_name.replace('\\', '_').replace('/', '_') + "_"
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "plots", f"{prefix}{method.lower()}_scores_on_map_comp{pc_index + 1}.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()


# -----------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------

def run_analysis(
    input_path:         str,
    base_output_path:   str,
    data_name:          str,
    data_type:          Literal["ICPMS", "XRF"],
    method:             Literal["PCA", "sPCA", "MAF"],
    map_data_dir:       str,
    pc_index:           int = 0,
    n_clusters:         int   = 3,
    transform_type:     Literal["standard", "CLR"] = "standard",
    variance_threshold: float = 0.99,
):
    """
    Function that puts together all functions in this script.
    """
    print(f"  [-] {method} | {data_name} | transform={transform_type} | k={n_clusters}")

    X_processed, y, df = load_and_preprocess_data(input_path, data_type, transform_type)

    group_labels = KMeans(
        n_clusters=n_clusters, random_state=1, n_init="auto"
    ).fit_predict(y.values)
    group_name = f"K-Means (k={n_clusters})"

    if method == "PCA":
        model, scores, features = perform_pca(X_processed)
    elif method == "sPCA":
        model, scores, features = perform_spca(X_processed, y)
    elif method == "MAF":
        model, scores, features = perform_maf(
            X_processed, y, variance_threshold=variance_threshold
        )
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose PCA | sPCA | MAF.")

    if transform_type == "CLR":
        out_dir = os.path.join(base_output_path, "CLR_transformed", method, data_name)
    else:
        out_dir = os.path.join(base_output_path, method, data_name)

    _make_dirs(out_dir)
    export_tables(model, scores, features, out_dir, method, data_name)
    
    for k in range(3, 8):
        k_labels = KMeans(
            n_clusters=k, random_state=1, n_init="auto"
        ).fit_predict(y.values)
        plot_biplot(
            model=model, 
            scores=scores, 
            features=features, 
            coords_df=y, 
            out_dir=out_dir, 
            data_name=data_name, 
            method=method,
            group_labels=k_labels, 
            map_data_dir=map_data_dir,
            group_name=f"K-Means (k={k})"
        )
        
    plot_eigenvalues(model, out_dir, data_name, method)

    plot_scores_on_map(
        scores=scores, 
        coords_df=y, 
        out_dir=out_dir, 
        data_name=data_name, 
        method=method, 
        map_data_dir=map_data_dir,
        pc_index=pc_index
    )

    print(f"      -> saved to {out_dir}")
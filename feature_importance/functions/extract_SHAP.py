"""
extract_SHAP.py
----------------------------
Functions script to calculate, extract, and visualize Standard SHAP values for multi-output spatial regression models. 
The pipeline trains models on the full dataset without cross-validation 
to prioritize global interpretability and feature influence.

Input data:
1. Preprocessed multi-element CSV datasets (ICPMS or XRF) containing numerical trace features 
    and spatial coordinate targets (Longitude/Latitude).
2. Precomputed optimal hyperparameter logs (CSV) for configuring the evaluated machine learning algorithms 
    (Random Forest, XGBoost, SVM).

Generates and saves:
1. Comprehensive per-sample SHAP data logs (CSV) containing true geographic coordinates, 
    original feature values, and the calculated longitudinal and latitudinal SHAP "push" per feature.
2. Global SHAP summary beeswarm plots (PNG) visually illustrating the directional and proportional marginal
    impact of each feature on predicted Longitude and Latitude.
3. Tabular global SHAP feature importance logs (CSV) aggregating and ranking the mean absolute 
    spatial impact (in degrees) of all features across both spatial axes.

Changes vs. previous version:
  - Fixed XGBoost SHAP branch: now handles native multi-output XGBRegressor
    directly (no MultiOutputRegressor wrapper required), with the wrapped
    version kept as a fallback (older version doesn't support multi output).
  - KernelExplainer (SVM): subsamples the explained set when N > MAX_EXPLAIN
    to keep runtime manageable, with a printed warning.
  - Added full per-sample SHAP save to create SHAP vector maps: after each model, a wide CSV is written
    containing true coordinates, original feature values, and per-sample SHAP
    values for both longitude and latitude. One file per dataset-model pair.
    Columns: Longitude, Latitude, <features>, shap_lon_<feature>, shap_lat_<feature>.
"""

import os
import ast
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Literal, Tuple, Dict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")


# ----------------------------------------------
# Data & Setup Functions
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
        raise ValueError(
            f"X and y share columns: {overlap}. Check column indexing in load_data."
        )

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    print(f"  -> Loaded {len(X)} samples, {X.shape[1]} features. "
          f"Coord range: Lon [{y['Longitude'].min():.2f}, {y['Longitude'].max():.2f}], "
          f"Lat [{y['Latitude'].min():.2f}, {y['Latitude'].max():.2f}]")

    return X, y


def _make_log_transformer():
    def log_with_epsilon(X):
        epsilon = 1e-6
        X_arr = np.array(X, dtype=float)
        X_arr = np.clip(X_arr, 0.0, None)
        return np.log(X_arr + epsilon)
    return FunctionTransformer(log_with_epsilon, validate=False)


def get_base_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "RandomForest": Pipeline([
            ("log",    _make_log_transformer()),
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200, n_jobs=-1, random_state=random_state
            )),
        ]),
        "XGBoost": Pipeline([
            ("log",    _make_log_transformer()),
            ("scaler", StandardScaler()),
            ("model",  XGBRegressor(
                    n_estimators=200, learning_rate=0.05,
                    n_jobs=-1, random_state=random_state, verbosity=0
                )
            ),
        ]),
        "SVM": Pipeline([
            ("log",    _make_log_transformer()),
            ("scaler", StandardScaler()),
            ("model",  MultiOutputRegressor(SVR(kernel="rbf", C=10, epsilon=0.1))),
        ]),
    }


def get_best_params(csv_path: str, model_name: str) -> dict:
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    row = df[df["model"] == model_name]
    if not row.empty:
        param_str = row["optimal_parameters"].values[0]
        try:
            return ast.literal_eval(param_str)
        except Exception:
            return {}
    return {}


def _prefix_params_for_pipeline(params: dict, step_name: str = "model") -> dict:
    prefixed = {}
    prefix = f"{step_name}__"
    for k, v in params.items():
        if not k.startswith(prefix):
            prefixed[f"{prefix}{k}"] = v
        else:
            prefixed[k] = v
    return prefixed


def _extract_shap_outputs(shap_vals, all_shap_lon, all_shap_lat, context: str):
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        all_shap_lon[:] = shap_vals[0]
        all_shap_lat[:] = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3 and shap_vals.shape[2] == 2:
        all_shap_lon[:] = shap_vals[:, :, 0]
        all_shap_lat[:] = shap_vals[:, :, 1]
    else:
        shape_info = (
            f"list of length {len(shap_vals)}" if isinstance(shap_vals, list)
            else f"ndarray shape {np.array(shap_vals).shape}"
        )
        raise ValueError(
            f"[{context}] Unexpected SHAP output format: {shape_info}. "
            "Expected a list of 2 arrays or a 3-D array with last dim == 2."
        )


# ----------------------------------------------
# NEW: Save full per-sample SHAP data to wide CSV
# ----------------------------------------------

def save_shap_wide_csv(
    X: pd.DataFrame,
    y: pd.DataFrame,
    all_shap_lon: np.ndarray,
    all_shap_lat: np.ndarray,
    explained_idx: np.ndarray,
    feature_names: list,
    output_path: str,
):
    """
    Saves a wide CSV with one row per explained sample containing:
      - True coordinates (Longitude, Latitude)
      - Original feature values
      - Per-sample SHAP push in longitude for each feature (shap_lon_<feature>)
      - Per-sample SHAP push in latitude  for each feature (shap_lat_<feature>)

    Inputs:
    X : pd.DataFrame
        Original (untransformed) feature values, shape (n_samples, n_features).
    y : pd.DataFrame
        True coordinates, shape (n_samples, 2), columns [Longitude, Latitude].
    all_shap_lon : np.ndarray
        Per-sample SHAP values for longitude, shape (n_explained, n_features).
    all_shap_lat : np.ndarray
        Per-sample SHAP values for latitude,  shape (n_explained, n_features).
    explained_idx : np.ndarray
        Row indices into X/y that were actually explained (full range for tree
        models; subsample indices for SVM).
    feature_names : list
        Ordered list of feature column names matching axis-1 of SHAP arrays.
    output_path : str
        Full path to write the CSV.
    """
    # Subset X and y to only the explained samples
    X_exp = X.iloc[explained_idx].reset_index(drop=True)
    y_exp = y.iloc[explained_idx].reset_index(drop=True)

    shap_lon_df = pd.DataFrame(
        all_shap_lon,
        columns=[f"shap_lon_{f}" for f in feature_names],
    )
    shap_lat_df = pd.DataFrame(
        all_shap_lat,
        columns=[f"shap_lat_{f}" for f in feature_names],
    )

    wide_df = pd.concat([y_exp, X_exp, shap_lon_df, shap_lat_df], axis=1)
    wide_df.to_csv(output_path, index=False)
    print(f"     Saved per-sample SHAP data ({len(wide_df)} samples) -> {output_path}")


# ----------------------------------------------
# Core SHAP Logic
# ----------------------------------------------

def run_shap_pipeline(
    data_name: str,
    csv_path: str,
    data_type: str,
    params_file: str,
    output_dir: str,
    random_state: int = 42,
    svm_max_explain: int = 500,      # NEW: cap on SVM explained-set size
):
    """
    Inputs:
    svm_max_explain : int
        Maximum number of samples passed to KernelExplainer.shap_values() for
        SVM models. If the dataset is larger, a random subsample is used and a
        warning is printed. Increase for more complete coverage at the cost of
        runtime.
    """
    print(f"\n{'='*60}")
    print(f"SHAP PIPELINE: {data_name.upper()} ({data_type})")
    print(f"{'='*60}")

    X, y = load_data(csv_path, data_type)
    base_models = get_base_models(random_state)

    ds_output_dir = os.path.join(output_dir, data_name)
    os.makedirs(ds_output_dir, exist_ok=True)

    for base_model_name, base_model_obj in base_models.items():
        print(f"\n  -> Extracting SHAP for: {base_model_name}")

        raw_params   = get_best_params(params_file, base_model_name)
        tuned_params = _prefix_params_for_pipeline(raw_params, step_name="model")

        if tuned_params:
            print(f"     [+] Tuned hyperparameters found and applied for {base_model_name}:")
            for param_name, param_val in tuned_params.items():
                print(f"         - {param_name}: {param_val}")
        else:
            print(f"     [-] Running {base_model_name} with default parameters.")

        model = clone(base_model_obj)
        if tuned_params:
            model.set_params(**tuned_params)

        #  Fit on the full dataset
        model.fit(X, y)

        #  Extract preprocessor and transform full dataset
        preprocessor = Pipeline(model.steps[:-1])
        core_model   = model.named_steps["model"]

        X_transformed = pd.DataFrame(
            preprocessor.transform(X), columns=X.columns, index=X.index
        )

        feature_names = list(X.columns)
        context       = f"{data_name}/{base_model_name}"

        # -------------------------------------------------------
        #  Compute SHAP values
        #    explained_idx tracks which rows were actually explained
        #    (full set for tree models, subsample for SVM).
        # -------------------------------------------------------

        if isinstance(core_model, RandomForestRegressor):
            explained_idx = np.arange(len(X_transformed))
            all_shap_lon  = np.zeros((len(explained_idx), X.shape[1]))
            all_shap_lat  = np.zeros((len(explained_idx), X.shape[1]))

            n_background = min(100, len(X_transformed))
            background   = shap.sample(X_transformed, n_background, random_state=random_state)
            explainer    = shap.TreeExplainer(core_model, background)
            shap_vals    = explainer.shap_values(X_transformed, check_additivity=False)
            _extract_shap_outputs(shap_vals, all_shap_lon, all_shap_lat, context)

        elif isinstance(core_model, XGBRegressor):
            # Native multi-output XGBoost — returns (n_samples, n_features, n_outputs)
            explained_idx = np.arange(len(X_transformed))
            all_shap_lon  = np.zeros((len(explained_idx), X.shape[1]))
            all_shap_lat  = np.zeros((len(explained_idx), X.shape[1]))

            n_background = min(100, len(X_transformed))
            background   = shap.sample(X_transformed, n_background, random_state=random_state)
            explainer    = shap.TreeExplainer(core_model, background)
            shap_vals    = explainer.shap_values(X_transformed, check_additivity=False)

            if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3 and shap_vals.shape[2] == 2:
                all_shap_lon[:] = shap_vals[:, :, 0]
                all_shap_lat[:] = shap_vals[:, :, 1]
            else:
                raise ValueError(
                    f"[{context}] Unexpected XGBoost SHAP shape: {np.array(shap_vals).shape}. "
                    "Expected (n_samples, n_features, 2)."
                )

        elif isinstance(core_model, MultiOutputRegressor) and isinstance(core_model.estimator, XGBRegressor):
            # Fallback: wrapped XGBoost — handle per-estimator
            explained_idx = np.arange(len(X_transformed))
            all_shap_lon  = np.zeros((len(explained_idx), X.shape[1]))
            all_shap_lat  = np.zeros((len(explained_idx), X.shape[1]))

            n_background  = min(100, len(X_transformed))
            background    = shap.sample(X_transformed, n_background, random_state=random_state)
            explainer_lon = shap.TreeExplainer(core_model.estimators_[0], background)
            all_shap_lon[:] = explainer_lon.shap_values(X_transformed, check_additivity=False)
            explainer_lat = shap.TreeExplainer(core_model.estimators_[1], background)
            all_shap_lat[:] = explainer_lat.shap_values(X_transformed, check_additivity=False)

        elif isinstance(core_model, MultiOutputRegressor) and isinstance(core_model.estimator, SVR):
            # Subsample explained set if dataset is large
            if len(X_transformed) > svm_max_explain:
                print(f"     [!] SVM KernelExplainer: dataset has {len(X_transformed)} samples. "
                      f"Subsampling explained set to {svm_max_explain}. "
                      f"Increase svm_max_explain for full coverage.")
                rng          = np.random.default_rng(random_state)
                explained_idx = rng.choice(len(X_transformed), size=svm_max_explain, replace=False)
                explained_idx = np.sort(explained_idx)
            else:
                print(f"     [+] SVM KernelExplainer: explaining full dataset ({len(X_transformed)} samples).")
                explained_idx = np.arange(len(X_transformed))

            X_explain    = X_transformed.iloc[explained_idx]
            all_shap_lon = np.zeros((len(explained_idx), X.shape[1]))
            all_shap_lat = np.zeros((len(explained_idx), X.shape[1]))

            n_centroids = min(50, len(X_transformed))
            background  = shap.kmeans(X_transformed, n_centroids)

            def svm_predict_wrapper(x_data):
                df_x = pd.DataFrame(x_data, columns=X.columns)
                return core_model.predict(df_x)

            explainer = shap.KernelExplainer(svm_predict_wrapper, background)
            shap_vals = explainer.shap_values(X_explain)
            _extract_shap_outputs(shap_vals, all_shap_lon, all_shap_lat, context)

        else:
            raise NotImplementedError(
                f"No SHAP handler implemented for core model type: {type(core_model)}"
            )

        # ---------------------------------------------------------
        # NEW: Save wide per-sample CSV
        # ---------------------------------------------------------
        wide_csv_path = os.path.join(
            ds_output_dir,
            f"{data_name}_{base_model_name}_SHAP_per_sample.csv",
        )
        save_shap_wide_csv(
            X            = X,
            y            = y,
            all_shap_lon = all_shap_lon,
            all_shap_lat = all_shap_lat,
            explained_idx= explained_idx,
            feature_names= feature_names,
            output_path  = wide_csv_path,
        )

        # ---------------------------------------------------------
        # Visualisations
        # ---------------------------------------------------------
        for shap_matrix, axis_label, direction_note in [
            (all_shap_lon, "Longitude", "Positive = Pushes East  |  Negative = Pushes West"),
            (all_shap_lat, "Latitude",  "Positive = Pushes North |  Negative = Pushes South"),
        ]:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_matrix, X.iloc[explained_idx], show=False)
            plt.title(
                f"{axis_label} SHAP: {data_name} ({base_model_name})\n({direction_note})",
                y=1.05,
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    ds_output_dir,
                    f"{data_name}_{base_model_name}_SHAP_{axis_label}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # ---------------------------------------------------------
        # Tabular importance  
        # ---------------------------------------------------------
        mean_abs_lon = np.abs(all_shap_lon).mean(axis=0)
        mean_abs_lat = np.abs(all_shap_lat).mean(axis=0)

        df_imp = pd.DataFrame({
            "Feature":              feature_names,
            "Impact_Longitude_deg": mean_abs_lon,
            "Impact_Latitude_deg":  mean_abs_lat,
            "Total_Magnitude_deg":  np.sqrt(mean_abs_lon**2 + mean_abs_lat**2),
        }).sort_values(by="Total_Magnitude_deg", ascending=False).reset_index(drop=True)

        imp_csv_path = os.path.join(
            ds_output_dir, f"{data_name}_{base_model_name}_SHAP_importance.csv"
        )
        df_imp.to_csv(imp_csv_path, index=False)
        print(f"     Saved plots and tabular data to {ds_output_dir}")

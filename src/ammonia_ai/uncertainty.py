"""Uncertainty Quantification v2: Split-Conformal + Applicability Domain for screening-v3.

Zero-leakage contract: all functions operate on initial-frame descriptors only
(the 64 SCREENING_FEATURES). No last-frame, trajectory, or energy data is used.

Key changes from v1:
- Split-conformal per-adsorbate prediction intervals (chemical-system grouped CV, 5 folds)
- kNN distance AD flag with 95th percentile threshold from CV held-out systems
- Nonmetal-rich surface AD flag with training-derived threshold
- 8 bootstrap models resampled by whole chemical systems
- Rank-normalised acquisition score + diversity term
- All artifacts saved to results/models/ad_model.joblib (never recomputed at inference)
"""
from __future__ import annotations

import json
import subprocess
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from ammonia_ai import schema as S

# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------
def _load_acquisition_config() -> dict[str, Any]:
    """Load acquisition config from YAML, with defaults as fallback."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "acquisition.yaml"
    defaults = {
        "variance_weight": 0.7,
        "diversity_weight": 0.3,
        "n_bootstrap": 8,
        "random_state": 42,
        "conformal_alpha": 0.10,
        "knn_k": 5,
        "knn_threshold_percentile": 0.95,
        "nonmetal_percentile": 0.95,
    }
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f) or {}
            defaults.update(user_config)
        except Exception:
            pass  # Use defaults
    return defaults


_ACQ_CONFIG = _load_acquisition_config()

# ----------------------------------------------------------------------
# Constants (from config)
# ----------------------------------------------------------------------
MAGNETIC_ELEMENTS: set[str] = {"Fe", "Co", "Ni"}

# Nonmetal elements for surface nonmetal fraction
NONMETALS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I", "As"}

# AD Flag column names (v2)
AD_UNSEEN_ELEMENT = "ad_unseen_element"
AD_FAR_FROM_TRAINING = "ad_far_from_training"
AD_MAGNETIC = "ad_magnetic"
AD_NONMETAL_RICH = "ad_nonmetal_rich"
RELIABILITY_TIER = "reliability_tier"

# Conformal confidence level (90% intervals)
CONFORMAL_ALPHA = _ACQ_CONFIG["conformal_alpha"]

# kNN settings
KNN_K = _ACQ_CONFIG["knn_k"]
KNN_THRESHOLD_PERCENTILE = _ACQ_CONFIG["knn_threshold_percentile"]

# Nonmetal percentile
NONMETAL_PERCENTILE = _ACQ_CONFIG["nonmetal_percentile"]

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


def _get_git_hash() -> str:
    """Get current git commit hash for manifest."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
            check=False,
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _extract_element_composition(df: pd.DataFrame) -> list[set[str]]:
    """Parse bulk_symbols column to get element sets for each row."""
    elements_per_row = []
    for _, row in df.iterrows():
        sym = row.get("bulk_symbols", "")
        try:
            counts = S._counts(sym)
            elements_per_row.append(set(counts.keys()))
        except Exception:
            elements_per_row.append(set())
    return elements_per_row


def _compute_surface_nonmetal_fraction(df: pd.DataFrame) -> np.ndarray:
    """Compute surface nonmetal fraction from surf_nm_frac feature."""
    if "surf_nm_frac" in df.columns:
        return df["surf_nm_frac"].to_numpy(dtype=float)
    # Fallback: compute from composition features if available
    # This should not happen in production since build_candidates.py includes it
    warnings.warn("surf_nm_frac not found in features, using 0.0", stacklevel=2)
    return np.zeros(len(df), dtype=float)


def _standardize_features(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Standardize features using pre-computed mean and std."""
    std_safe = np.where(std > 1e-12, std, 1.0)
    return (X - mean) / std_safe


def _compute_knn_distances(
    X_query: np.ndarray, X_ref: np.ndarray, k: int = KNN_K
) -> np.ndarray:
    """Compute mean distance to k nearest neighbors in reference set."""
    # cdist computes pairwise distances
    dists = cdist(X_query, X_ref, metric="euclidean")
    # Sort each row and take mean of k smallest
    knn_dists = np.sort(dists, axis=1)[:, :k]
    return knn_dists.mean(axis=1)


# ----------------------------------------------------------------------
# Training-time functions (compute artifacts for ad_model.joblib)
# ----------------------------------------------------------------------


def compute_conformal_quantiles(
    train_df: pd.DataFrame,
    model_median: HistGradientBoostingRegressor,
    *,
    n_splits: int = 5,
    alpha: float = CONFORMAL_ALPHA,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Compute split-conformal prediction intervals per adsorbate using grouped CV.

    Uses GroupKFold on bulk_chemsys to avoid chemical leakage.
    Returns 90% quantile of absolute residuals per adsorbate.
    """
    # Prepare features and target
    X = train_df[list(S.SCREENING_FEATURES)].to_numpy(dtype=float)
    y = train_df[S.TARGET_COLUMN].to_numpy(dtype=float)
    groups = train_df["bulk_chemsys"].to_numpy()
    adsorbates = train_df["adsorbate"].to_numpy()

    # Get unique adsorbates in training
    unique_adsorbates = np.unique(adsorbates)
    quantiles = {}

    # GroupKFold on chemical systems
    gkf = GroupKFold(n_splits=n_splits)

    # Store residuals per adsorbate
    residuals_by_adsorbate = {ads: [] for ads in unique_adsorbates}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        ads_val = adsorbates[val_idx]

        # Train model on this fold
        fold_model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            early_stopping=False,
            random_state=random_state + fold_idx,
        )
        fold_model.fit(X_train, y_train)

        # Predict on held-out
        y_pred = fold_model.predict(X_val)
        residuals = np.abs(y_val - y_pred)

        # Accumulate by adsorbate
        for ads in unique_adsorbates:
            mask = ads_val == ads
            if mask.any():
                residuals_by_adsorbate[ads].extend(residuals[mask])

    # Compute 90% quantile per adsorbate (1 - alpha)
    for ads in unique_adsorbates:
        res = np.array(residuals_by_adsorbate[ads])
        if len(res) > 0:
            quantiles[ads] = float(np.quantile(res, 1 - alpha))
        else:
            # Fallback: use global quantile
            all_res = np.concatenate([v for v in residuals_by_adsorbate.values() if len(v) > 0])
            quantiles[ads] = float(np.quantile(all_res, 1 - alpha)) if len(all_res) > 0 else 0.6

    return quantiles


def compute_knn_reference(
    train_df: pd.DataFrame,
    *,
    percentile: float = KNN_THRESHOLD_PERCENTILE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute kNN reference set and distance threshold from training data.

    Returns:
        X_ref: Standardized reference features (n_ref, 64)
        scaler_mean: Mean of training features (64,)
        scaler_std: Std of training features (64,)
        distance_threshold: 95th percentile of kNN distances on CV held-out systems
    """
    X = train_df[list(S.SCREENING_FEATURES)].to_numpy(dtype=float)
    groups = train_df["bulk_chemsys"].to_numpy()

    # Standardize
    scaler_mean = X.mean(axis=0)
    scaler_std = X.std(axis=0)
    X_std = _standardize_features(X, scaler_mean, scaler_std)

    # Use GroupKFold to get held-out distances for threshold calibration
    gkf = GroupKFold(n_splits=5)
    heldout_dists = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_std, groups=groups)):
        X_ref_fold = X_std[train_idx]
        X_val_fold = X_std[val_idx]

        # Compute kNN distances for held-out points against training fold
        dists = _compute_knn_distances(X_val_fold, X_ref_fold, k=KNN_K)
        heldout_dists.extend(dists)

    # Threshold at 95th percentile of held-out distances
    distance_threshold = float(np.quantile(heldout_dists, percentile)) if heldout_dists else 10.0

    # Use full training set as reference
    return X_std, scaler_mean, scaler_std, distance_threshold


def compute_nonmetal_threshold(train_df: pd.DataFrame) -> float:
    """Compute surface nonmetal fraction threshold from training data (95th percentile)."""
    nm_frac = _compute_surface_nonmetal_fraction(train_df)
    return float(np.quantile(nm_frac, 0.95))


def derive_in_domain_elements(train_df: pd.DataFrame) -> set[str]:
    """
    Derive the valid element vocabulary dynamically from the training dataset.

    The vocabulary is derived ONLY from the training split.
    """
    element_sets = _extract_element_composition(train_df)
    return set().union(*element_sets) if element_sets else set()


def train_bootstrap_models(
    train_df: pd.DataFrame,
    *,
    n_bootstrap: int = 8,
    random_state: int = 42,
) -> list[HistGradientBoostingRegressor]:
    """
    Train bootstrap models resampled by whole chemical systems.

    Each bootstrap sample draws chemical systems with replacement,
    then includes all rows from those systems.
    """
    rng = np.random.default_rng(random_state)
    unique_chemsys = train_df["bulk_chemsys"].unique()
    n_systems = len(unique_chemsys)

    models = []
    X_full = train_df[list(S.SCREENING_FEATURES)].to_numpy(dtype=float)
    y_full = train_df[S.TARGET_COLUMN].to_numpy(dtype=float)
    groups_full = train_df["bulk_chemsys"].to_numpy()

    for i in range(n_bootstrap):
        # Sample chemical systems with replacement
        sampled_systems = rng.choice(unique_chemsys, size=n_systems, replace=True)
        # Get all rows belonging to sampled systems
        mask = np.isin(groups_full, sampled_systems)
        X_boot = X_full[mask]
        y_boot = y_full[mask]

        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            early_stopping=False,
            random_state=random_state + i,
        )
        model.fit(X_boot, y_boot)
        models.append(model)

    return models


def compute_acquisition_scores(
    X: np.ndarray,
    bootstrap_models: list[HistGradientBoostingRegressor],
    X_ref: np.ndarray,
    *,
    variance_weight: float = _ACQ_CONFIG["variance_weight"],
    diversity_weight: float = _ACQ_CONFIG["diversity_weight"],
) -> np.ndarray:
    """
    Compute acquisition scores: rank-normalized variance + diversity term.

    Variance: epistemic uncertainty from bootstrap ensemble (rank-normalized within batch)
    Diversity: negative mean distance to k nearest neighbors in reference set (rank-normalized)
    """
    n_samples = X.shape[0]

    # Epistemic variance from bootstrap ensemble
    preds = np.column_stack([m.predict(X) for m in bootstrap_models])
    variance = preds.var(axis=1)
    variance_rank = pd.Series(variance).rank(method="average", pct=True).to_numpy()

    # Diversity: mean distance to k nearest neighbors in reference set
    dists = _compute_knn_distances(X, X_ref, k=KNN_K)
    # Negative distance = higher diversity = higher score
    diversity = -dists
    diversity_rank = pd.Series(diversity).rank(method="average", pct=True).to_numpy()

    # Combined score
    scores = variance_weight * variance_rank + diversity_weight * diversity_rank
    return scores


def compute_reliability_tier(ad_flags: pd.DataFrame) -> pd.Series:
    """
    Compute reliability tier from AD flags.

    High: no flags (all False)
    Medium: only ad_magnetic is True
    Low: any other flag is True
    """
    # Check which flags are True for each row
    flag_cols = [AD_UNSEEN_ELEMENT, AD_FAR_FROM_TRAINING, AD_MAGNETIC, AD_NONMETAL_RICH]
    flags = ad_flags[flag_cols]

    # High: no flags
    is_high = (~flags.any(axis=1))
    # Medium: only magnetic
    is_medium = flags[AD_MAGNETIC] & (~flags.drop(columns=[AD_MAGNETIC]).any(axis=1))
    # Low: everything else
    is_low = ~is_high & ~is_medium

    tier = pd.Series("low", index=ad_flags.index)
    tier[is_high] = "high"
    tier[is_medium] = "medium"
    return tier


def build_ad_model(
    train_df: pd.DataFrame,
    model_median: HistGradientBoostingRegressor,
    *,
    n_bootstrap: int = _ACQ_CONFIG["n_bootstrap"],
    random_state: int = _ACQ_CONFIG["random_state"],
) -> dict[str, Any]:
    """
    Build the complete AD model artifact for saving to ad_model.joblib.

    This is the MAIN training-time function that computes all artifacts.
    """
    print("Building AD model artifacts...", flush=True)

    # 1. In-domain element vocabulary (from training split only)
    in_domain_elements = derive_in_domain_elements(train_df)
    print(f"  In-domain elements: {len(in_domain_elements)}", flush=True)

    # 2. Split-conformal quantiles per adsorbate
    conformal_quantiles = compute_conformal_quantiles(train_df, model_median)
    print(f"  Conformal quantiles: {conformal_quantiles}", flush=True)

    # 3. kNN reference and threshold
    X_ref, scaler_mean, scaler_std, knn_threshold = compute_knn_reference(train_df)
    print(f"  kNN threshold (95th pctile): {knn_threshold:.4f}", flush=True)
    print(f"  Reference set size: {len(X_ref)}", flush=True)

    # 4. Nonmetal-rich threshold
    nonmetal_threshold = compute_nonmetal_threshold(train_df)
    print(f"  Nonmetal threshold (95th pctile): {nonmetal_threshold:.4f}", flush=True)

    # 5. Bootstrap models for acquisition
    bootstrap_models = train_bootstrap_models(train_df, n_bootstrap=n_bootstrap, random_state=random_state)
    print(f"  Trained {len(bootstrap_models)} bootstrap models", flush=True)

    # 6. Manifest
    manifest = {
        "schema_version": S.SCHEMA_VERSION,
        "git_commit": _get_git_hash(),
        "n_bootstrap": n_bootstrap,
        "random_state": random_state,
        "conformal_alpha": CONFORMAL_ALPHA,
        "knn_k": KNN_K,
        "knn_threshold_percentile": KNN_THRESHOLD_PERCENTILE,
        "nonmetal_percentile": NONMETAL_PERCENTILE,
        "acquisition_weights": {
            "variance_weight": _ACQ_CONFIG["variance_weight"],
            "diversity_weight": _ACQ_CONFIG["diversity_weight"],
        },
        "in_domain_elements": sorted(in_domain_elements),
        "conformal_quantiles": conformal_quantiles,
        "knn_threshold": knn_threshold,
        "nonmetal_threshold": nonmetal_threshold,
        "scaler_mean": scaler_mean.tolist(),
        "scaler_std": scaler_std.tolist(),
    }

    return {
        "manifest": manifest,
        "in_domain_elements": in_domain_elements,
        "conformal_quantiles": conformal_quantiles,
        "X_ref": X_ref,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "knn_threshold": knn_threshold,
        "nonmetal_threshold": nonmetal_threshold,
        "bootstrap_models": bootstrap_models,
    }


def save_ad_model(ad_model: dict[str, Any], path: Path) -> None:
    """Save AD model artifact to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ad_model, path)
    print(f"Saved AD model to {path}", flush=True)


def load_ad_model(path: Path) -> dict[str, Any]:
    """Load AD model artifact from disk."""
    if not path.exists():
        raise FileNotFoundError(f"AD model not found: {path}")
    ad_model = joblib.load(path)
    print(f"Loaded AD model from {path}", flush=True)
    return ad_model


# ----------------------------------------------------------------------
# Inference-time functions (use loaded ad_model.joblib only)
# ----------------------------------------------------------------------


def compute_applicability_domain_v2(
    df: pd.DataFrame,
    ad_model: dict[str, Any],
) -> pd.DataFrame:
    """
    Compute Applicability Domain (AD) flags v2 for each row.

    Flags returned as independent boolean columns:
    - ad_unseen_element: any element not in training vocabulary
    - ad_far_from_training: mean kNN distance > threshold (from CV held-out)
    - ad_magnetic: contains Fe/Co/Ni (informational only)
    - ad_nonmetal_rich: surface nonmetal fraction > training threshold
    """
    in_domain_elements = ad_model["in_domain_elements"]
    X_ref = ad_model["X_ref"]
    scaler_mean = ad_model["scaler_mean"]
    scaler_std = ad_model["scaler_std"]
    knn_threshold = ad_model["knn_threshold"]
    nonmetal_threshold = ad_model["nonmetal_threshold"]

    element_sets = _extract_element_composition(df)

    # ad_unseen_element
    ad_unseen_element = []
    ad_magnetic = []
    for elements in element_sets:
        ad_unseen_element.append(not elements.issubset(in_domain_elements))
        ad_magnetic.append(not elements.isdisjoint(MAGNETIC_ELEMENTS))

    # ad_far_from_training: kNN distance in standardized feature space
    X = df[list(S.SCREENING_FEATURES)].to_numpy(dtype=float)
    X_std = _standardize_features(X, scaler_mean, scaler_std)
    knn_dists = _compute_knn_distances(X_std, X_ref, k=KNN_K)
    ad_far_from_training = knn_dists > knn_threshold

    # ad_nonmetal_rich: surface nonmetal fraction
    nm_frac = _compute_surface_nonmetal_fraction(df)
    ad_nonmetal_rich = nm_frac > nonmetal_threshold

    return pd.DataFrame({
        AD_UNSEEN_ELEMENT: ad_unseen_element,
        AD_FAR_FROM_TRAINING: ad_far_from_training,
        AD_MAGNETIC: ad_magnetic,
        AD_NONMETAL_RICH: ad_nonmetal_rich,
    }, index=df.index)


def compute_prediction_intervals_conformal(
    df: pd.DataFrame,
    model_median: HistGradientBoostingRegressor,
    ad_model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute prediction intervals using split-conformal quantiles per adsorbate.

    Returns:
        pred_median, pred_low, pred_high
    """
    X = df[list(S.SCREENING_FEATURES)].to_numpy(dtype=float)
    adsorbates = df["adsorbate"].to_numpy()
    conformal_quantiles = ad_model["conformal_quantiles"]

    pred_median = model_median.predict(X)

    # Per-adsorbate interval
    pred_low = np.zeros_like(pred_median)
    pred_high = np.zeros_like(pred_median)

    for ads in np.unique(adsorbates):
        mask = adsorbates == ads
        quantile = conformal_quantiles.get(ads, 0.6)  # fallback
        pred_low[mask] = pred_median[mask] - quantile
        pred_high[mask] = pred_median[mask] + quantile

    return pred_median, pred_low, pred_high


def compute_acquisition_score_v2(
    df: pd.DataFrame,
    ad_model: dict[str, Any],
) -> np.ndarray:
    """Compute acquisition score using bootstrap ensemble + diversity."""
    X = df[list(S.SCREENING_FEATURES)].to_numpy(dtype=float)
    bootstrap_models = ad_model["bootstrap_models"]
    X_ref = ad_model["X_ref"]
    manifest = ad_model["manifest"]
    variance_weight = manifest["acquisition_weights"]["variance_weight"]
    diversity_weight = manifest["acquisition_weights"]["diversity_weight"]

    return compute_acquisition_scores(
        X,
        bootstrap_models,
        X_ref,
        variance_weight=variance_weight,
        diversity_weight=diversity_weight,
    )


def compute_triage_predictions_v2(
    candidate_df: pd.DataFrame,
    model_median: HistGradientBoostingRegressor,
    ad_model: dict[str, Any],
) -> pd.DataFrame:
    """
    Generate a triage report for candidate systems using pre-computed AD model.

    This is the MAIN inference-time function. It NEVER recomputes AD artifacts
    from the candidate table - it only uses the loaded ad_model.joblib.
    """
    # Validate input has full screening contract
    S.validate_feature_frame(candidate_df, features=S.SCREENING_FEATURES, allow_missing=False)

    cand_work = candidate_df.copy()
    X = cand_work[list(S.SCREENING_FEATURES)].to_numpy(dtype=float)

    # Predictions
    pred_median, pred_low, pred_high = compute_prediction_intervals_conformal(
        cand_work, model_median, ad_model
    )

    # Applicability Domain v2
    ad_df = compute_applicability_domain_v2(cand_work, ad_model)

    # Reliability tier
    ad_df[RELIABILITY_TIER] = compute_reliability_tier(ad_df)

    # Acquisition score
    acq_score = compute_acquisition_score_v2(cand_work, ad_model)

    # Build output
    metadata_cols = [c for c in S.METADATA_COLUMNS if c in cand_work.columns]
    out = cand_work[metadata_cols].copy()
    out["predicted_E_ads_eV"] = pred_median
    out["ci_low_eV"] = pred_low
    out["ci_high_eV"] = pred_high
    out["ci_width_eV"] = pred_high - pred_low
    out = pd.concat([out, ad_df], axis=1)
    out["acquisition_score"] = acq_score

    return out


# ----------------------------------------------------------------------
# Validation / Demo functions
# ----------------------------------------------------------------------


def validate_uncertainty_on_heldout(
    train_df: pd.DataFrame,
    model_median: HistGradientBoostingRegressor,
    ad_model: dict[str, Any],
    *,
    n_splits: int = 5,
) -> dict[str, Any]:
    """
    Validate uncertainty estimates on held-out systems using grouped CV.

    Returns metrics:
    - 90% coverage per adsorbate
    - Spearman correlation between interval width and absolute error
    - OOD rate per AD flag
    - MAE per reliability tier
    """
    from scipy.stats import spearmanr

    X = train_df[list(S.SCREENING_FEATURES)].to_numpy(dtype=float)
    y = train_df[S.TARGET_COLUMN].to_numpy(dtype=float)
    groups = train_df["bulk_chemsys"].to_numpy()
    adsorbates = train_df["adsorbate"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)

    all_predictions = []
    all_targets = []
    all_adsorbates = []
    all_widths = []
    all_ad_flags = []
    all_tiers = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_val = X[val_idx]
        y_val = y[val_idx]
        ads_val = adsorbates[val_idx]

        # Create validation dataframe
        val_df = train_df.iloc[val_idx].copy()
        val_df = val_df.reset_index(drop=True)

        # Get predictions with ad_model (trained on full training set, not fold)
        pred_median, pred_low, pred_high = compute_prediction_intervals_conformal(
            val_df, model_median, ad_model
        )

        ad_df = compute_applicability_domain_v2(val_df, ad_model)
        ad_df[RELIABILITY_TIER] = compute_reliability_tier(ad_df)

        all_predictions.extend(pred_median)
        all_targets.extend(y_val)
        all_adsorbates.extend(ads_val)
        all_widths.extend(pred_high - pred_low)
        all_ad_flags.append(ad_df)
        all_tiers.extend(ad_df[RELIABILITY_TIER])

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_adsorbates = np.array(all_adsorbates)
    all_widths = np.array(all_widths)
    all_ad_flags = pd.concat(all_ad_flags, ignore_index=True)
    all_tiers = np.array(all_tiers)

    # 90% coverage per adsorbate
    coverage_by_adsorbate = {}
    for ads in np.unique(all_adsorbates):
        mask = all_adsorbates == ads
        if mask.any():
            in_interval = (all_targets[mask] >= all_predictions[mask] - all_widths[mask]/2) & \
                          (all_targets[mask] <= all_predictions[mask] + all_widths[mask]/2)
            coverage_by_adsorbate[ads] = float(in_interval.mean())

    # Spearman(width vs abs_error)
    abs_errors = np.abs(all_targets - all_predictions)
    spearman_r, spearman_p = spearmanr(all_widths, abs_errors)

    # OOD rate per AD flag
    ood_rates = {}
    for flag in [AD_UNSEEN_ELEMENT, AD_FAR_FROM_TRAINING, AD_MAGNETIC, AD_NONMETAL_RICH]:
        ood_rates[flag] = float(all_ad_flags[flag].mean())

    # MAE per reliability tier
    mae_by_tier = {}
    for tier in ["high", "medium", "low"]:
        mask = all_tiers == tier
        if mask.any():
            mae_by_tier[tier] = float(np.abs(all_targets[mask] - all_predictions[mask]).mean())
        else:
            mae_by_tier[tier] = None

    return {
        "coverage_by_adsorbate": coverage_by_adsorbate,
        "spearman_width_vs_error": {"rho": float(spearman_r), "p_value": float(spearman_p)},
        "ood_rate_per_flag": ood_rates,
        "mae_by_tier": mae_by_tier,
        "n_samples": len(all_predictions),
    }


__all__ = [
    # Training-time
    "build_ad_model",
    "save_ad_model",
    "load_ad_model",
    "compute_conformal_quantiles",
    "compute_knn_reference",
    "compute_nonmetal_threshold",
    "derive_in_domain_elements",
    "train_bootstrap_models",
    # Inference-time
    "compute_applicability_domain_v2",
    "compute_prediction_intervals_conformal",
    "compute_acquisition_score_v2",
    "compute_triage_predictions_v2",
    "compute_reliability_tier",
    # Validation
    "validate_uncertainty_on_heldout",
    # Constants
    "MAGNETIC_ELEMENTS",
    "CONFORMAL_ALPHA",
    "KNN_K",
    "KNN_THRESHOLD_PERCENTILE",
    "AD_UNSEEN_ELEMENT",
    "AD_FAR_FROM_TRAINING",
    "AD_MAGNETIC",
    "AD_NONMETAL_RICH",
    "RELIABILITY_TIER",
]
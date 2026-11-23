#!/usr/bin/env python3
"""Task A5 (Updated): Adaptive (Normalized) Conformal Validation with Per-Adsorbate Calibration.

Fixes NH2/NH3 under-coverage by calibrating normalized conformal quantile
per adsorbate on out-of-group residuals.

Reports: coverage per adsorbate AND per tier; within-adsorbate Spearman;
far-from-training flag rate; MAE per tier. Compares with A4.

Usage:
    .venv/bin/python scripts/validate_uncertainty_a5.py \
        --training-table results/adsorbates/oc20_adsorbates_features.parquet \
        --max-iter 200 \
        --n-bootstrap 4 \
        --time-limit 900
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from ammonia_ai import schema as S
from ammonia_ai.uncertainty import (
    AD_FAR_FROM_TRAINING,
    AD_MAGNETIC,
    AD_NONMETAL_RICH,
    AD_UNSEEN_ELEMENT,
    CONFORMAL_ALPHA,
    KNN_K,
    KNN_THRESHOLD_PERCENTILE,
    MAGNETIC_ELEMENTS,
    RELIABILITY_TIER,
    _compute_surface_nonmetal_fraction,
    _extract_element_composition,
    _standardize_features,
    compute_reliability_tier,
    derive_in_domain_elements,
    train_bootstrap_models,
)


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training table not found: {path}")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows from {path}", file=sys.stderr)

    if "is_clean_primary_regression_row" in df.columns:
        df_clean = df[df["is_clean_primary_regression_row"]].copy()
        print(f"Filtered to {len(df_clean)} clean primary regression rows", file=sys.stderr)
    else:
        df_clean = df.copy()
        print("WARNING: 'is_clean_primary_regression_row' column not found; using all rows", file=sys.stderr)

    required = list(S.SCREENING_FEATURES) + [S.TARGET_COLUMN, "bulk_chemsys", "adsorbate"]
    missing = [c for c in required if c not in df_clean.columns]
    if missing:
        raise ValueError(f"Training table missing required columns: {missing}")

    return df_clean


def compute_knn_distances(X_query: np.ndarray, X_ref: np.ndarray, k: int = KNN_K) -> np.ndarray:
    dists = cdist(X_query, X_ref, metric="euclidean")
    knn_dists = np.sort(dists, axis=1)[:, :k]
    return knn_dists.mean(axis=1)


def calibrate_knn_threshold(X_std_train: np.ndarray, groups_train: np.ndarray, percentile: float = 0.95) -> float:
    gkf = GroupKFold(n_splits=5)
    heldout_dists = []
    for inner_train_idx, inner_val_idx in gkf.split(X_std_train, groups=groups_train):
        X_ref_fold = X_std_train[inner_train_idx]
        X_val_fold = X_std_train[inner_val_idx]
        dists = compute_knn_distances(X_val_fold, X_ref_fold, k=KNN_K)
        heldout_dists.extend(dists)
    return float(np.quantile(heldout_dists, percentile)) if heldout_dists else 10.0


def compute_ad_flags_for_fold(
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    in_domain_elements = derive_in_domain_elements(train_df)

    element_sets = _extract_element_composition(val_df)
    ad_unseen_element = []
    ad_magnetic = []
    for elements in element_sets:
        ad_unseen_element.append(not elements.issubset(in_domain_elements))
        ad_magnetic.append(not elements.isdisjoint(MAGNETIC_ELEMENTS))

    X_train = train_df[features].to_numpy(dtype=float)
    X_val = val_df[features].to_numpy(dtype=float)

    scaler_mean = X_train.mean(axis=0)
    scaler_std = X_train.std(axis=0)

    X_train_std = _standardize_features(X_train, scaler_mean, scaler_std)
    X_val_std = _standardize_features(X_val, scaler_mean, scaler_std)

    groups_train = train_df["bulk_chemsys"].to_numpy()
    knn_threshold = calibrate_knn_threshold(X_train_std, groups_train, percentile=KNN_THRESHOLD_PERCENTILE)

    knn_dists = compute_knn_distances(X_val_std, X_train_std, k=KNN_K)
    ad_far_from_training = knn_dists > knn_threshold

    nm_frac_train = _compute_surface_nonmetal_fraction(train_df)
    nonmetal_threshold = float(np.quantile(nm_frac_train, 0.95))
    nm_frac_val = _compute_surface_nonmetal_fraction(val_df)
    ad_nonmetal_rich = nm_frac_val > nonmetal_threshold

    return pd.DataFrame({
        AD_UNSEEN_ELEMENT: ad_unseen_element,
        AD_FAR_FROM_TRAINING: ad_far_from_training,
        AD_MAGNETIC: ad_magnetic,
        AD_NONMETAL_RICH: ad_nonmetal_rich,
    }, index=val_df.index)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-table", type=Path, required=True)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--n-bootstrap", type=int, default=4, help="Bootstrap models per fold (default 4 for <15min)")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--time-limit", type=int, default=900)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare-a4", type=Path, default=None, help="Path to A4 JSON results for comparison")
    ap.add_argument("--fast", action="store_true", help="Fast mode: 3 folds, 4 bootstrap, max_iter=100")
    a = ap.parse_args(argv)

    if a.fast:
        a.n_splits = 3
        a.n_bootstrap = 4
        a.max_iter = 100
        a.time_limit = 300

    start_time = time.time()
    df = load_training_data(a.training_table)

    features = list(S.SCREENING_FEATURES)
    target = S.TARGET_COLUMN

    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    groups = df["bulk_chemsys"].to_numpy()
    adsorbates = df["adsorbate"].to_numpy()

    print(f"\nRunning {a.n_splits}-fold chemical-system grouped CV (refitting model per fold, max_iter={a.max_iter})...")

    gkf = GroupKFold(n_splits=a.n_splits)
    fold_splits = list(gkf.split(X, y, groups))

    # Phase 1: Fit fold models and get OOF predictions & residuals
    oof_predictions = np.zeros(len(df), dtype=float)
    oof_residuals = np.zeros(len(df), dtype=float)

    fold_models = []
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"  Training fold {fold_idx + 1}/{a.n_splits} on {len(train_idx)} samples...")
        model = HistGradientBoostingRegressor(
            max_iter=a.max_iter,
            learning_rate=0.1,
            early_stopping=False,
            random_state=a.random_state + fold_idx,
        )
        model.fit(X[train_idx], y[train_idx])
        fold_models.append(model)

        preds = model.predict(X[val_idx])
        oof_predictions[val_idx] = preds
        oof_residuals[val_idx] = np.abs(y[val_idx] - preds)

    # Phase 2: Leave-one-fold-out conformal quantiles & AD evaluation per fold
    # Now also train bootstrap models per training fold for adaptive scaling
    val_widths_list = []
    val_targets_list = []
    val_preds_list = []
    val_adsorbates_list = []
    val_ad_flags_list = []
    val_tiers_list = []

    print("\nEvaluating adaptive (normalized) conformal quantiles with PER-ADSORBATE calibration...")
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        val_df = df.iloc[val_idx].copy().reset_index(drop=True)
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)

        # Train bootstrap ensemble on training fold for adaptive scaling
        bootstrap_models = train_bootstrap_models(
            train_df, n_bootstrap=a.n_bootstrap, random_state=a.random_state + fold_idx
        )
        print(f"  Fold {fold_idx + 1}: Trained {len(bootstrap_models)} bootstrap models")

        # Compute bootstrap ensemble std on TRAINING fold for calibration
        X_train = X[train_idx]
        y_train = y[train_idx]
        train_ads = adsorbates[train_idx]
        train_residuals = oof_residuals[train_idx]

        bootstrap_preds_train = np.column_stack([m.predict(X_train) for m in bootstrap_models])
        bootstrap_std_train = bootstrap_preds_train.std(axis=1)
        median_bs_std_train = np.median(bootstrap_std_train)
        if median_bs_std_train > 1e-12:
            normalized_std_train = bootstrap_std_train / median_bs_std_train
        else:
            normalized_std_train = np.ones_like(bootstrap_std_train)

        # Normalized residuals on training fold: |residual| / normalized_std
        normalized_residuals = train_residuals / normalized_std_train

        # Calibrate conformal quantile of NORMALIZED residuals per adsorbate
        unique_ads = np.unique(adsorbates)
        conformal_quantiles_normalized = {}
        for ads in unique_ads:
            mask = train_ads == ads
            if mask.any():
                conformal_quantiles_normalized[ads] = float(np.quantile(normalized_residuals[mask], 1 - CONFORMAL_ALPHA))
            else:
                conformal_quantiles_normalized[ads] = float(np.quantile(normalized_residuals, 1 - CONFORMAL_ALPHA))

        print(f"  Fold {fold_idx + 1} Normalized Conformal Quantiles (90%): {conformal_quantiles_normalized}")

        # Compute bootstrap ensemble std on VALIDATION fold
        X_val = X[val_idx]
        bootstrap_preds_val = np.column_stack([m.predict(X_val) for m in bootstrap_models])
        bootstrap_std_val = bootstrap_preds_val.std(axis=1)
        # Normalize by SAME median from training fold
        if median_bs_std_train > 1e-12:
            normalized_std_val = bootstrap_std_val / median_bs_std_train
        else:
            normalized_std_val = np.ones_like(bootstrap_std_val)

        # Adaptive interval width: normalized conformal quantile * normalized bootstrap std
        val_ads = adsorbates[val_idx]
        val_widths = np.array([2.0 * conformal_quantiles_normalized[ads] for ads in val_ads]) * normalized_std_val

        # AD flags using training fold ONLY
        ad_df = compute_ad_flags_for_fold(val_df, train_df, features)
        ad_df[RELIABILITY_TIER] = compute_reliability_tier(ad_df)

        val_widths_list.extend(val_widths)
        val_targets_list.extend(y[val_idx])
        val_preds_list.extend(oof_predictions[val_idx])
        val_adsorbates_list.extend(val_ads)
        val_ad_flags_list.append(ad_df)
        val_tiers_list.extend(ad_df[RELIABILITY_TIER])

    # Convert aggregated arrays
    all_targets = np.array(val_targets_list)
    all_preds = np.array(val_preds_list)
    all_widths = np.array(val_widths_list)
    all_adsorbates = np.array(val_adsorbates_list)
    all_ad_flags = pd.concat(val_ad_flags_list, ignore_index=True)
    all_tiers = np.array(val_tiers_list)
    abs_errors = np.abs(all_targets - all_preds)

    # 1. Coverage per adsorbate
    coverage_by_adsorbate = {}
    for ads in np.unique(all_adsorbates):
        mask = all_adsorbates == ads
        in_interval = abs_errors[mask] <= (all_widths[mask] / 2.0)
        coverage_by_adsorbate[ads] = float(in_interval.mean())

    # 2. Coverage per tier
    coverage_by_tier = {}
    for tier in ["high", "medium", "low"]:
        mask = all_tiers == tier
        if mask.any():
            in_interval = abs_errors[mask] <= (all_widths[mask] / 2.0)
            coverage_by_tier[tier] = float(in_interval.mean())
        else:
            coverage_by_tier[tier] = None

    # 3. Coverage per adsorbate AND tier
    coverage_by_ads_tier = {}
    for ads in np.unique(all_adsorbates):
        coverage_by_ads_tier[ads] = {}
        for tier in ["high", "medium", "low"]:
            mask = (all_adsorbates == ads) & (all_tiers == tier)
            if mask.any():
                in_interval = abs_errors[mask] <= (all_widths[mask] / 2.0)
                coverage_by_ads_tier[ads][tier] = float(in_interval.mean())
            else:
                coverage_by_ads_tier[ads][tier] = None

    # 4. Spearman per adsorbate
    spearman_by_adsorbate = {}
    for ads in np.unique(all_adsorbates):
        mask = all_adsorbates == ads
        if mask.sum() > 1:
            w_ads = all_widths[mask]
            e_ads = abs_errors[mask]
            if np.std(w_ads) < 1e-12:
                spearman_by_adsorbate[ads] = {"rho": 0.0, "p_value": 1.0, "note": "constant width within adsorbate"}
            else:
                rho, p = spearmanr(w_ads, e_ads)
                spearman_by_adsorbate[ads] = {"rho": float(rho), "p_value": float(p)}

    # Overall Spearman (across all adsorbates)
    overall_rho, overall_p = spearmanr(all_widths, abs_errors)

    # 5. OOD rate per AD flag
    ood_rates = {}
    for flag in [AD_UNSEEN_ELEMENT, AD_FAR_FROM_TRAINING, AD_MAGNETIC, AD_NONMETAL_RICH]:
        ood_rates[flag] = float(all_ad_flags[flag].mean())

    # 6. MAE per reliability tier
    mae_by_tier = {}
    for tier in ["high", "medium", "low"]:
        mask = all_tiers == tier
        if mask.any():
            mae_by_tier[tier] = float(abs_errors[mask].mean())
        else:
            mae_by_tier[tier] = None

    elapsed_time = time.time() - start_time

    # Load A4 results for comparison if provided
    a4_results = None
    if a.compare_a4 and a.compare_a4.exists():
        a4_results = json.loads(a.compare_a4.read_text())

    # Print Report
    print("\n" + "=" * 80)
    print("TASK A5 ADAPTIVE (NORMALIZED) CONFORMAL VALIDATION REPORT (CALIBRATED)")
    print("=" * 80)
    print(f"Total held-out samples: {len(all_targets)}")
    print(f"Execution time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
    print(f"Bootstrap models per fold: {a.n_bootstrap}")

    print("\n--- 90% Coverage per Adsorbate ---")
    for ads, cov in sorted(coverage_by_adsorbate.items()):
        status = "✓" if cov >= 0.88 else "✗"  # Target 0.88-0.92
        a4_cov = a4_results["coverage_by_adsorbate"].get(ads) if a4_results else None
        diff = f" (A4: {a4_cov:.3f})" if a4_cov is not None else ""
        print(f"  {ads:6s}: {cov:.3f} {status}{diff}")
    avg_cov = np.mean(list(coverage_by_adsorbate.values()))
    a4_avg = np.mean(list(a4_results["coverage_by_adsorbate"].values())) if a4_results else None
    diff_avg = f" (A4: {a4_avg:.3f})" if a4_avg is not None else ""
    print(f"  Average: {avg_cov:.3f}{diff_avg}")

    print("\n--- 90% Coverage per Tier ---")
    for tier in ["high", "medium", "low"]:
        cov = coverage_by_tier.get(tier)
        cov_str = f"{cov:.3f}" if cov is not None else "N/A"
        print(f"  {tier:6s}: {cov_str}")

    print("\n--- 90% Coverage per Adsorbate x Tier ---")
    for ads in sorted(coverage_by_ads_tier.keys()):
        print(f"  {ads}:")
        for tier in ["high", "medium", "low"]:
            cov = coverage_by_ads_tier[ads].get(tier)
            cov_str = f"{cov:.3f}" if cov is not None else "N/A (no samples)"
            print(f"    {tier:6s}: {cov_str}")

    print("\n--- Within-Adsorbate Spearman(Interval Width vs |Error|) ---")
    for ads, res in sorted(spearman_by_adsorbate.items()):
        note = f" ({res['note']})" if "note" in res else ""
        a4_rho = a4_results["spearman_by_adsorbate"].get(ads, {}).get("rho") if a4_results else None
        diff_rho = f" (A4: {a4_rho:.4f})" if a4_rho is not None else ""
        print(f"  {ads:6s}: rho = {res['rho']:.4f}, p = {res['p_value']:.4f}{note}{diff_rho}")
    a4_overall_rho = a4_results.get("spearman_overall", {}).get("rho") if a4_results else None
    diff_overall = f" (A4: {a4_overall_rho:.4f})" if a4_overall_rho is not None else ""
    print(f"  Overall (across adsorbates): rho = {overall_rho:.4f}, p = {overall_p:.4f}{diff_overall}")

    print("\n--- OOD Rate per AD Flag ---")
    for flag, rate in ood_rates.items():
        target_str = " (target 3–7%)" if flag == AD_FAR_FROM_TRAINING else ""
        status = " ✓" if (flag == AD_FAR_FROM_TRAINING and 0.03 <= rate <= 0.07) else ""
        a4_rate = a4_results.get("ood_rates", {}).get(flag) if a4_results else None
        diff_rate = f" (A4: {a4_rate:.3f})" if a4_rate is not None else ""
        print(f"  {flag:30s}: {rate * 100:.2f}% ({rate:.3f}){target_str}{diff_rate}{status}")

    print("\n--- MAE per Reliability Tier ---")
    for tier in ["high", "medium", "low"]:
        mae = mae_by_tier.get(tier)
        mae_str = f"{mae:.4f} eV" if mae is not None else "N/A"
        a4_mae = a4_results.get("mae_by_tier", {}).get(tier) if a4_results else None
        diff_mae = f" (A4: {a4_mae:.4f} eV)" if a4_mae is not None else ""
        print(f"  {tier:6s}: {mae_str}{diff_mae}")

    # Summary comparison with A4
    if a4_results:
        print("\n--- A5 vs A4 Summary ---")
        cov_improved = sum(1 for ads in coverage_by_adsorbate if coverage_by_adsorbate[ads] > a4_results["coverage_by_adsorbate"].get(ads, 0))
        cov_same = sum(1 for ads in coverage_by_adsorbate if abs(coverage_by_adsorbate[ads] - a4_results["coverage_by_adsorbate"].get(ads, 0)) < 0.001)
        cov_worse = sum(1 for ads in coverage_by_adsorbate if coverage_by_adsorbate[ads] < a4_results["coverage_by_adsorbate"].get(ads, 0))
        print(f"  Coverage: {cov_improved} improved, {cov_same} same, {cov_worse} worse vs A4")

        spearman_improved = sum(1 for ads in spearman_by_adsorbate
                                if abs(spearman_by_adsorbate[ads]["rho"]) > abs(a4_results["spearman_by_adsorbate"].get(ads, {}).get("rho", 0)))
        print(f"  Within-adsorbate |Spearman|: {spearman_improved}/{len(spearman_by_adsorbate)} improved (higher absolute correlation)")

        overall_improved = "yes" if abs(overall_rho) > abs(a4_overall_rho or 0) else "no"
        print(f"  Overall |Spearman| improved: {overall_improved} (A5: {abs(overall_rho):.4f} vs A4: {abs(a4_overall_rho or 0):.4f})")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "task": "A5",
            "coverage_by_adsorbate": coverage_by_adsorbate,
            "coverage_by_tier": coverage_by_tier,
            "coverage_by_ads_tier": coverage_by_ads_tier,
            "spearman_by_adsorbate": spearman_by_adsorbate,
            "spearman_overall": {"rho": float(overall_rho), "p_value": float(overall_p)},
            "ood_rates": ood_rates,
            "mae_by_tier": mae_by_tier,
            "elapsed_time_seconds": elapsed_time,
            "n_bootstrap": a.n_bootstrap,
            "n_splits": a.n_splits,
            "max_iter": a.max_iter,
        }
        a.out.write_text(json.dumps(out_data, indent=2))
        print(f"\nSaved report to {a.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
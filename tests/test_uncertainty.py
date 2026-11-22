"""Tests for ammonia_ai.uncertainty v2 module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from ammonia_ai import schema as S
from ammonia_ai.uncertainty import (
    build_ad_model,
    compute_applicability_domain_v2,
    compute_prediction_intervals_conformal,
    compute_triage_predictions_v2,
    compute_reliability_tier,
    derive_in_domain_elements,
    load_ad_model,
    save_ad_model,
    validate_uncertainty_on_heldout,
    MAGNETIC_ELEMENTS,
    CONFORMAL_ALPHA,
    KNN_K,
    KNN_THRESHOLD_PERCENTILE,
    AD_UNSEEN_ELEMENT,
    AD_FAR_FROM_TRAINING,
    AD_MAGNETIC,
    AD_NONMETAL_RICH,
    RELIABILITY_TIER,
)


def _synthetic_train_df(n=100, seed=0):
    """Generate a synthetic training table with required columns."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in S.SCREENING_FEATURES})
    elements = ["Fe2", "Ni2", "Ru2", "Cu2", "Co2"]
    df["bulk_symbols"] = [elements[i % len(elements)] for i in range(n)]
    df["system_id"] = [f"train_{i}" for i in range(n)]
    df["bulk_chemsys"] = [S.chemical_system(s) for s in df["bulk_symbols"]]
    df["adsorbate"] = ["N"] * (n // 5) + ["NH"] * (n // 5) + ["NH2"] * (n // 5) + ["NH3"] * (n // 5) + ["N2"] * (n // 5)
    # Target: some function of features + noise
    X = df[list(S.SCREENING_FEATURES)].to_numpy(dtype=float)
    df[S.TARGET_COLUMN] = X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.normal(0, 0.1, size=n)
    df["surf_nm_frac"] = rng.uniform(0, 0.5, size=n)
    # Mark all as clean for testing
    df["is_clean_primary_regression_row"] = True
    return df


def _synthetic_candidate_df(n=20, seed=1):
    """Generate a synthetic candidate table with required columns."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in S.SCREENING_FEATURES})
    elements = ["Fe2", "Ni2", "Ru2", "Au2"]
    df["bulk_symbols"] = [elements[i % len(elements)] for i in range(n)]
    df["system_id"] = [f"cand_{i}" for i in range(n)]
    df["bulk_mpid"] = [f"mp-{i}" for i in range(n)]
    adsorbates = ["N", "NH", "NH2", "NH3"]
    df["adsorbate"] = [adsorbates[i % len(adsorbates)] for i in range(n)]
    df["miller_index"] = [[1, 0, 0]] * n
    df["shift"] = [0.0] * n
    df["ads_id"] = [f"ads_{i}" for i in range(n)]
    df["mapping_adsorbate"] = df["adsorbate"]
    df["surf_nm_frac"] = rng.uniform(0, 0.5, size=n)
    return df


def _train_median_model(train_df):
    """Train a simple median model for testing."""
    X = train_df[list(S.SCREENING_FEATURES)].to_numpy(dtype=float)
    y = train_df[S.TARGET_COLUMN].to_numpy(dtype=float)
    model = HistGradientBoostingRegressor(max_iter=50, learning_rate=0.1, random_state=0)
    model.fit(X, y)
    return model


def test_magnetic_elements_constant():
    """MAGNETIC_ELEMENTS should contain exactly Fe, Co, Ni."""
    assert MAGNETIC_ELEMENTS == {"Fe", "Co", "Ni"}


def test_derive_in_domain_elements():
    """Dynamic element vocabulary derivation should work correctly."""
    df = pd.DataFrame({"bulk_symbols": ["Fe2Ni2", "Ru4", "Au"]})
    vocab = derive_in_domain_elements(df)
    assert vocab == {"Fe", "Ni", "Ru", "Au"}


def test_build_ad_model():
    """Test building the full AD model artifact."""
    train_df = _synthetic_train_df(200)
    model_median = _train_median_model(train_df)

    ad_model = build_ad_model(train_df, model_median, n_bootstrap=4, random_state=42)

    # Check all required keys
    assert "manifest" in ad_model
    assert "in_domain_elements" in ad_model
    assert "conformal_quantiles" in ad_model
    assert "X_ref" in ad_model
    assert "scaler_mean" in ad_model
    assert "scaler_std" in ad_model
    assert "knn_threshold" in ad_model
    assert "nonmetal_threshold" in ad_model
    assert "bootstrap_models" in ad_model

    # Check manifest
    manifest = ad_model["manifest"]
    assert manifest["schema_version"] == S.SCHEMA_VERSION
    assert "git_commit" in manifest
    assert manifest["n_bootstrap"] == 4
    assert len(manifest["in_domain_elements"]) > 0
    assert len(manifest["conformal_quantiles"]) > 0
    assert manifest["knn_threshold"] > 0
    assert manifest["nonmetal_threshold"] > 0

    # Check conformal quantiles per adsorbate
    for ads in train_df["adsorbate"].unique():
        assert ads in ad_model["conformal_quantiles"]
        assert ad_model["conformal_quantiles"][ads] > 0

    # Check reference set
    assert ad_model["X_ref"].shape[1] == len(S.SCREENING_FEATURES)
    assert len(ad_model["X_ref"]) == len(train_df)

    # Check bootstrap models
    assert len(ad_model["bootstrap_models"]) == 4


def test_save_load_ad_model(tmp_path):
    """Test saving and loading AD model artifact."""
    train_df = _synthetic_train_df(100)
    model_median = _train_median_model(train_df)
    ad_model = build_ad_model(train_df, model_median, n_bootstrap=2, random_state=42)

    path = tmp_path / "ad_model.joblib"
    save_ad_model(ad_model, path)
    assert path.exists()

    loaded = load_ad_model(path)
    assert "manifest" in loaded
    assert loaded["manifest"]["schema_version"] == S.SCHEMA_VERSION
    assert len(loaded["bootstrap_models"]) == 2


def test_compute_applicability_domain_v2():
    """Test v2 AD flags on candidate data."""
    train_df = _synthetic_train_df(100)
    model_median = _train_median_model(train_df)
    ad_model = build_ad_model(train_df, model_median, n_bootstrap=2, random_state=42)

    cand_df = _synthetic_candidate_df(10)
    # Make first candidate have unseen element (Au)
    cand_df.loc[0, "bulk_symbols"] = "Au2"
    # Make second candidate magnetic (Fe)
    cand_df.loc[1, "bulk_symbols"] = "Fe2"

    ad_flags = compute_applicability_domain_v2(cand_df, ad_model)

    # Check columns
    expected_cols = [AD_UNSEEN_ELEMENT, AD_FAR_FROM_TRAINING, AD_MAGNETIC, AD_NONMETAL_RICH]
    for col in expected_cols:
        assert col in ad_flags.columns

    # Au is not in training vocabulary
    assert ad_flags.loc[0, AD_UNSEEN_ELEMENT] == True
    # Fe is magnetic
    assert ad_flags.loc[1, AD_MAGNETIC] == True
    # Ru is in-domain and non-magnetic
    assert ad_flags.loc[2, AD_UNSEEN_ELEMENT] == False
    assert ad_flags.loc[2, AD_MAGNETIC] == False


def test_compute_prediction_intervals_conformal():
    """Test conformal prediction intervals per adsorbate."""
    train_df = _synthetic_train_df(100)
    model_median = _train_median_model(train_df)
    ad_model = build_ad_model(train_df, model_median, n_bootstrap=2, random_state=42)

    cand_df = _synthetic_candidate_df(10)
    pred_median, pred_low, pred_high = compute_prediction_intervals_conformal(cand_df, model_median, ad_model)

    assert len(pred_median) == len(cand_df)
    assert len(pred_low) == len(cand_df)
    assert len(pred_high) == len(cand_df)
    # Intervals should be centered on median
    np.testing.assert_allclose(pred_median, (pred_low + pred_high) / 2, rtol=1e-10)
    # Low <= median <= high
    assert np.all(pred_low <= pred_median)
    assert np.all(pred_median <= pred_high)
    # Width should be positive
    assert np.all(pred_high - pred_low > 0)


def test_compute_reliability_tier():
    """Test reliability tier assignment from AD flags."""
    # Create test flags
    flags = pd.DataFrame({
        AD_UNSEEN_ELEMENT: [False, False, True, False, False],
        AD_FAR_FROM_TRAINING: [False, False, False, True, False],
        AD_MAGNETIC: [False, True, False, False, False],
        AD_NONMETAL_RICH: [False, False, False, False, True],
    })

    tiers = compute_reliability_tier(flags)
    assert tiers.iloc[0] == "high"      # No flags
    assert tiers.iloc[1] == "medium"    # Only magnetic
    assert tiers.iloc[2] == "low"       # Unseen element
    assert tiers.iloc[3] == "low"       # Far from training
    assert tiers.iloc[4] == "low"       # Nonmetal rich


def test_compute_triage_predictions_v2():
    """Test full triage prediction pipeline v2."""
    train_df = _synthetic_train_df(200)
    model_median = _train_median_model(train_df)
    ad_model = build_ad_model(train_df, model_median, n_bootstrap=2, random_state=42)

    cand_df = _synthetic_candidate_df(20)
    report_df = compute_triage_predictions_v2(cand_df, model_median, ad_model)

    # Check required columns
    required_cols = [
        "system_id", "bulk_mpid", "adsorbate", "miller_index", "shift",
        "predicted_E_ads_eV", "ci_low_eV", "ci_high_eV", "ci_width_eV",
        AD_UNSEEN_ELEMENT, AD_FAR_FROM_TRAINING, AD_MAGNETIC, AD_NONMETAL_RICH,
        RELIABILITY_TIER, "acquisition_score"
    ]
    for col in required_cols:
        assert col in report_df.columns

    # Check values
    assert len(report_df) == len(cand_df)
    assert np.all(report_df["ci_width_eV"] > 0)
    assert report_df[RELIABILITY_TIER].isin(["high", "medium", "low"]).all()
    assert np.all(report_df["acquisition_score"] >= 0)
    assert np.all(report_df["acquisition_score"] <= 1)


def test_validate_uncertainty_on_heldout():
    """Test validation on held-out systems."""
    train_df = _synthetic_train_df(200)
    model_median = _train_median_model(train_df)
    ad_model = build_ad_model(train_df, model_median, n_bootstrap=2, random_state=42)

    results = validate_uncertainty_on_heldout(train_df, model_median, ad_model, n_splits=3)

    # Check required keys
    assert "coverage_by_adsorbate" in results
    assert "spearman_width_vs_error" in results
    assert "ood_rate_per_flag" in results
    assert "mae_by_tier" in results
    assert "n_samples" in results

    # Check coverage
    for ads, cov in results["coverage_by_adsorbate"].items():
        assert 0 <= cov <= 1

    # Check spearman
    assert "rho" in results["spearman_width_vs_error"]
    assert "p_value" in results["spearman_width_vs_error"]

    # Check OOD rates
    for flag in [AD_UNSEEN_ELEMENT, AD_FAR_FROM_TRAINING, AD_MAGNETIC, AD_NONMETAL_RICH]:
        assert flag in results["ood_rate_per_flag"]
        assert 0 <= results["ood_rate_per_flag"][flag] <= 1

    # Check MAE per tier
    for tier in ["high", "medium", "low"]:
        assert tier in results["mae_by_tier"]


def test_constants():
    """Test that constants have expected values."""
    assert CONFORMAL_ALPHA == 0.10
    assert KNN_K == 5
    assert KNN_THRESHOLD_PERCENTILE == 0.95
    assert AD_UNSEEN_ELEMENT == "ad_unseen_element"
    assert AD_FAR_FROM_TRAINING == "ad_far_from_training"
    assert AD_MAGNETIC == "ad_magnetic"
    assert AD_NONMETAL_RICH == "ad_nonmetal_rich"
    assert RELIABILITY_TIER == "reliability_tier"
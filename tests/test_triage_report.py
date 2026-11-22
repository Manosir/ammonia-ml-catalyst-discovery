"""Tests for scripts/triage_report.py."""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from ammonia_ai import schema as S
from ammonia_ai.uncertainty import build_ad_model, save_ad_model


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


def _synthetic_candidates(n=30, seed=0):
    """Generate synthetic candidate table with the full 64-feature screening contract.

    The frame carries every SCREENING_FEATURES column in canonical order so
    that production triage (which now gates on the full contract) succeeds.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in S.SCREENING_FEATURES})
    df["system_id"] = [f"cand_{i:04d}" for i in range(n)]
    elements = ["Fe2", "Ni2", "Ru2"]
    df["bulk_symbols"] = [elements[i % len(elements)] for i in range(n)]
    return df


def _synthetic_metadata(n=30, seed=0):
    """Generate synthetic metadata (official join)."""
    rng = np.random.default_rng(seed)
    adsorbates = ["N", "N", "N", "N", "N", "N", "N2", "N2", "N2", "N2"]
    df = pd.DataFrame({
        "system_id": [f"cand_{i:04d}" for i in range(n)],
        "bulk_mpid": [f"mp-{i}" for i in range(n)],
        "adsorbate": [adsorbates[i % len(adsorbates)] for i in range(n)],
        "miller_index": [[1, 0, 0]] * n,
        "shift": [0.0] * n,
    })
    return df


def _make_median_model():
    """Create a simple trained median model."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, len(S.SCREENING_FEATURES)))
    y = X[:, 0] * 0.5 + rng.normal(0, 0.1, size=100)
    model = HistGradientBoostingRegressor(max_iter=10, learning_rate=0.1, random_state=0)
    model.fit(X, y)
    return model


def test_triage_report_output_schema(tmp_path):
    """Test that triage_report.py produces valid output schema."""
    cand_df = _synthetic_candidates(30, seed=0)
    meta_df = _synthetic_metadata(30, seed=0)

    cand_path = tmp_path / "candidates.parquet"
    meta_path = tmp_path / "metadata.parquet"
    model_dir = tmp_path / "models"
    out_json = tmp_path / "triage.json"
    out_parquet = tmp_path / "triage.parquet"

    cand_df.to_parquet(cand_path, index=False)
    meta_df.to_parquet(meta_path, index=False)
    model_dir.mkdir()

    model = _make_median_model()
    joblib.dump(model, model_dir / "screening_v3_full.joblib")

    # Need to also create a dummy ad_model.joblib for the test
    # Build a minimal ad_model with the same structure as v2
    from ammonia_ai.uncertainty import build_ad_model
    train_df = _synthetic_train_df(100)
    ad_model = build_ad_model(train_df, model, n_bootstrap=2, random_state=42)
    ad_model_path = model_dir / "ad_model.joblib"
    from ammonia_ai.uncertainty import save_ad_model
    save_ad_model(ad_model, ad_model_path)

    spec = importlib.util.spec_from_file_location("tr", Path(__file__).parents[1] / "scripts" / "triage_report.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)

    args = [
        "--candidates", str(cand_path),
        "--metadata", str(meta_path),
        "--model-dir", str(model_dir),
        "--ad-model", str(ad_model_path),
        "--out", str(out_json),
        "--out-parquet", str(out_parquet),
        "--top-k", "5",
    ]
    assert tr.main(args) == 0

    assert out_json.exists()
    assert out_parquet.exists()

    summary = json.loads(out_json.read_text())
    assert summary["schema_version"] == S.SCHEMA_VERSION
    assert "ad_flags" in summary
    assert "top_candidates" in summary
    assert len(summary["top_candidates"]) == 5

    report_df = pd.read_parquet(out_parquet)
    required_cols = [
        "system_id", "bulk_mpid", "adsorbate", "miller_index", "shift",
        "predicted_E_ads_eV", "ci_low_eV", "ci_high_eV", "ci_width_eV",
        "ad_unseen_element", "ad_far_from_training", "ad_magnetic", "ad_nonmetal_rich",
        "reliability_tier", "acquisition_score"
    ]
    for col in required_cols:
        assert col in report_df.columns

    # Check reliability tiers are valid
    assert report_df["reliability_tier"].isin(["high", "medium", "low"]).all()


def test_triage_report_with_quantile_models(tmp_path):
    """Test triage report with quantile models for proper CIs."""
    cand_df = _synthetic_candidates(20)
    meta_df = _synthetic_metadata(20)

    cand_path = tmp_path / "candidates.parquet"
    meta_path = tmp_path / "metadata.parquet"
    model_dir = tmp_path / "models"
    out_json = tmp_path / "triage.json"
    out_parquet = tmp_path / "triage.parquet"

    cand_df.to_parquet(cand_path, index=False)
    meta_df.to_parquet(meta_path, index=False)
    model_dir.mkdir()

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, len(S.SCREENING_FEATURES)))
    y = X[:, 0] * 0.5 + rng.normal(0, 0.1, size=100)

    model_median = HistGradientBoostingRegressor(max_iter=10, learning_rate=0.1, random_state=0).fit(X, y)
    model_low = HistGradientBoostingRegressor(loss="quantile", quantile=0.025, max_iter=10, learning_rate=0.1, random_state=0).fit(X, y)
    model_high = HistGradientBoostingRegressor(loss="quantile", quantile=0.975, max_iter=10, learning_rate=0.1, random_state=0).fit(X, y)

    joblib.dump(model_median, model_dir / "screening_v3_full.joblib")
    joblib.dump(model_low, model_dir / "screening_v3_full_q0.025.joblib")
    joblib.dump(model_high, model_dir / "screening_v3_full_q0.975.joblib")

    # Build ad_model
    train_df = _synthetic_train_df(100)
    ad_model = build_ad_model(train_df, model_median, n_bootstrap=2, random_state=42)
    ad_model_path = model_dir / "ad_model.joblib"
    save_ad_model(ad_model, ad_model_path)

    spec = importlib.util.spec_from_file_location("tr", Path(__file__).parents[1] / "scripts" / "triage_report.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)

    args = [
        "--candidates", str(cand_path),
        "--metadata", str(meta_path),
        "--model-dir", str(model_dir),
        "--ad-model", str(ad_model_path),
        "--out", str(out_json),
        "--out-parquet", str(out_parquet),
        "--top-k", "3",
    ]
    assert tr.main(args) == 0

    report_df = pd.read_parquet(out_parquet)
    widths = report_df["ci_width_eV"].to_numpy()
    assert np.ptp(widths) > 1e-6


def test_triage_report_missing_candidates_file(tmp_path):
    """Test error handling for missing candidates file."""
    meta_df = _synthetic_metadata(10)
    meta_path = tmp_path / "metadata.parquet"
    meta_df.to_parquet(meta_path, index=False)

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = _make_median_model()
    joblib.dump(model, model_dir / "screening_v3_full.joblib")

    # Build ad_model
    train_df = _synthetic_train_df(100)
    ad_model = build_ad_model(train_df, model, n_bootstrap=2, random_state=42)
    ad_model_path = model_dir / "ad_model.joblib"
    save_ad_model(ad_model, ad_model_path)

    spec = importlib.util.spec_from_file_location("tr", Path(__file__).parents[1] / "scripts" / "triage_report.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)

    args = [
        "--candidates", str(tmp_path / "missing.parquet"),
        "--metadata", str(meta_path),
        "--model-dir", str(model_dir),
        "--ad-model", str(ad_model_path),
        "--out", str(tmp_path / "triage.json"),
    ]
    result = tr.main(args)
    assert result != 0

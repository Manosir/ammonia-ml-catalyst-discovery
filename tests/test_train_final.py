import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ammonia_ai import schema as S

spec = importlib.util.spec_from_file_location("tf", Path(__file__).parents[1] / "scripts" / "train_final.py")
tf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tf)

LIGHT = [c for c in S.SCREENING_FEATURES if c not in set(S.PLACEMENT_SENSITIVE_FEATURES)]


def table(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in S.SCREENING_FEATURES})
    df["anomaly"] = rng.choice([0] * 8 + [1, 2, 3], size=n)
    df["is_clean_primary_regression_row"] = df["anomaly"] == 0
    df["adsorption_energy_eV"] = df["site_en_mean"] + rng.normal(0, 0.1, size=n)
    return df


def run(tmp_path):
    t = tmp_path / "t.parquet"
    table().to_parquet(t, index=False)
    out = tmp_path / "models"
    assert tf.main(["--table", str(t), "--out-dir", str(out), "--registry", "none", "--max-iter", "10"]) == 0
    return out


def load(out, name):
    return joblib.load(out / f"{name}.joblib"), json.loads((out / f"{name}.manifest.json").read_text())


def test_saved_models_match_the_contract(tmp_path):
    out = run(tmp_path)
    full_model, full_manifest = load(out, "screening_v3_full")
    light_model, light_manifest = load(out, "screening_v3_light")
    tf.check_model(full_model, full_manifest, list(S.SCREENING_FEATURES))
    tf.check_model(light_model, light_manifest, LIGHT)
    S.assert_model_matches_schema(full_model, full_manifest)
    assert full_manifest["training_rows"] == int(table().is_clean_primary_regression_row.sum())
    assert light_manifest["placement_sensitive_features"] == []


def test_wrong_manifest_or_model_is_refused(tmp_path):
    out = run(tmp_path)
    full_model, full_manifest = load(out, "screening_v3_full")
    light_model, _ = load(out, "screening_v3_light")
    with pytest.raises(ValueError):
        S.assert_model_matches_schema(full_model, {**full_manifest, "schema_version": "screening-v2-old"})
    with pytest.raises(ValueError):
        tf.check_model(light_model, full_manifest, list(S.SCREENING_FEATURES))  # 55-feature model vs 64-feature contract
    with pytest.raises(ValueError):
        S.assert_model_matches_schema(full_model, {**full_manifest, "feature_columns": LIGHT})


def test_anomaly_classifiers_are_saved_with_all_three_codes(tmp_path):
    out = run(tmp_path)
    classifiers, manifest = load(out, "screening_v3_anomaly")
    assert set(classifiers) == {"dissociated", "desorbed", "surface_changed"}
    X = table()[list(S.SCREENING_FEATURES)].to_numpy(float)[:5]
    for clf in classifiers.values():
        assert clf.predict_proba(X).shape == (5, 2)
        tf.check_model(clf, manifest, list(S.SCREENING_FEATURES))
    assert manifest["training_rows"] == len(table())

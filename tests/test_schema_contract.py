from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ammonia_ai import schema as S

CATEGORIES = {
    "screening": set(S.SCREENING_FEATURES),
    "experimental": set(S.EXPERIMENTAL_FEATURES),
    "audit": set(S.AUDIT_COLUMNS),
    "structure_audit": set(S.STRUCTURE_PASS_AUDIT_COLUMNS),
    "identifier": set(S.IDENTIFIER_COLUMNS),
    "target": set(S.TARGET_COLUMNS),
    "group": set(S.GROUP_KEYS),
    "context": set(S.INPUT_CONTEXT_COLUMNS),
    "provenance": set(S.PROVENANCE_COLUMNS),
    "unverified": set(S.OC20_UNVERIFIED),
    "oc20_metadata": set(S.OC20_METADATA_COLUMNS),
    "derived_audit": set(S.DERIVED_AUDIT_COLUMNS),
}


def _registry_module():
    spec = importlib.util.spec_from_file_location("fa", Path(__file__).parents[1] / "scripts" / "feature_availability.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract_frame(n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.normal(size=n) for c in S.SCREENING_FEATURES})


def test_versioned_contract_is_frozen() -> None:
    assert S.SCHEMA_VERSION.startswith("screening-v3-")
    assert len(S.SCREENING_FEATURES) == 64 and len(set(S.SCREENING_FEATURES)) == 64
    assert len(S.PLACEMENT_SENSITIVE_FEATURES) == 9
    assert set(S.PLACEMENT_SENSITIVE_FEATURES) <= set(S.SITE_DEPENDENT_FEATURES) <= set(S.SCREENING_FEATURES)
    assert S.TARGET_COLUMNS == ("adsorption_energy_eV",)


def test_contract_has_no_element_one_hots_and_no_audit_names() -> None:
    assert not any(n.startswith(("frac_", "siteel_")) for n in S.SCREENING_FEATURES)
    for name in S.SCREENING_FEATURES:
        assert not any(frag in name for frag in S.AUDIT_NAME_FRAGMENTS), name
    for name in ("ads_height", "mean_ads_force", "d_band_center_site"):
        assert name not in S.SCREENING_FEATURES


def test_categories_are_disjoint() -> None:
    for (a, x), (b, y) in itertools.combinations(CATEGORIES.items(), 2):
        assert x.isdisjoint(y), (a, b, x & y)


@pytest.mark.parametrize("name", ["ads_height_final_audit", "mean_ads_force_final_audit", "d_band_center_site", "slab_disp_max"])
def test_leaky_features_rejected_without_registry(name) -> None:
    frame = pd.DataFrame({name: [1.0, 2.0]})
    with pytest.raises(ValueError, match="not permitted"):
        S.validate_feature_frame(frame, features=(name,), registry_path=None)


def test_old_18_feature_set_is_rejected() -> None:
    old = ("d_band_center_slab", "d_band_center_site", "d_band_center_primary", "metallic_radius", "electronegativity",
           "d_electrons", "primary_metal_frac", "tm_frac", "n_distinct_TM", "gcn_initial", "ads_height", "mean_ads_force",
           "facet_roughness", "shift", "n_adsorbate_N", "n_adsorbate_H", "adsorbate_atom_count", "n_surface_atoms")
    with pytest.raises(ValueError):
        S.validate_feature_frame(pd.DataFrame({c: [1.0, 2.0] for c in old}), features=old, registry_path=None)


def test_complete_contract_frame_is_accepted() -> None:
    S.validate_feature_frame(_contract_frame(), registry_path=None)


def test_missing_values_rejected_unless_allowed() -> None:
    frame = _contract_frame()
    frame.loc[0, "cn_slab_mean"] = np.nan
    with pytest.raises(ValueError, match="cn_slab_mean"):
        S.validate_feature_frame(frame, registry_path=None)
    S.validate_feature_frame(frame, registry_path=None, allow_missing=True)


def test_inf_rejected_even_when_missing_allowed() -> None:
    frame = _contract_frame()
    frame.loc[0, "cn_slab_mean"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        S.validate_feature_frame(frame, registry_path=None, allow_missing=True)


def test_non_numeric_rejected() -> None:
    frame = _contract_frame()
    frame["cn_slab_mean"] = ["a"] * len(frame)
    with pytest.raises(TypeError, match="cn_slab_mean"):
        S.validate_feature_frame(frame, registry_path=None)


def test_registry_generator_matches_schema(tmp_path) -> None:
    reg = _registry_module().registry_frame()
    assert reg.name.is_unique
    features = reg[reg.role == "feature"]
    assert set(features.name) == set(S.SCREENING_FEATURES)
    assert (features.implemented == "yes").all() and (features.available_at_screening == "yes").all()
    path = tmp_path / "registry.csv"
    reg.to_csv(path, index=False)
    S.assert_features_allowed(S.SCREENING_FEATURES, path)
    for name in list(S.AUDIT_COLUMNS) + list(S.STRUCTURE_PASS_AUDIT_COLUMNS):
        with pytest.raises(ValueError):
            S.assert_features_allowed([name], path)


def test_generated_registry_file_is_in_sync_when_present() -> None:
    if not S.DEFAULT_REGISTRY.exists():
        pytest.skip("feature registry not generated in this checkout")
    registry = S.load_feature_registry(S.DEFAULT_REGISTRY)
    assert set(registry.loc[registry.role == "feature", "name"]) == set(S.SCREENING_FEATURES), "stale registry: rerun scripts/feature_availability.py"


def test_real_table_columns_are_classified_when_present() -> None:
    path = Path("results/adsorbates/oc20_adsorbates_features.parquet")
    if not path.exists():
        pytest.skip("model table not built")
    known = set().union(*CATEGORIES.values())
    columns = {c for c in pd.read_parquet(path).columns if not c.startswith("qc_")}
    assert not (columns - known), sorted(columns - known)


def test_system_id_and_sid_conflict_is_rejected() -> None:
    with pytest.raises(ValueError):
        S.normalize_system_id_column(pd.DataFrame({"system_id": ["a"], "sid": ["b"]}))

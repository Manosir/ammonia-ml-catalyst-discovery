from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import pytest

from ammonia_ai import schema as S

CATEGORIES = {
    "screening": set(S.SCREENING_FEATURES),
    "experimental": set(S.EXPERIMENTAL_FEATURES),
    "audit": set(S.AUDIT_COLUMNS),
    "identifier": set(S.IDENTIFIER_COLUMNS),
    "target": set(S.TARGET_COLUMNS),
    "group": set(S.GROUP_KEYS),
    "context": set(S.INPUT_CONTEXT_COLUMNS),
    "provenance": set(S.PROVENANCE_COLUMNS),
    "unverified": set(S.OC20_UNVERIFIED),
    "oc20_metadata": set(S.OC20_METADATA_COLUMNS),
    "derived_audit": set(S.DERIVED_AUDIT_COLUMNS),
}


def test_versioned_screening_contract_exists() -> None:
    assert S.SCHEMA_VERSION.startswith("screening-v2-")
    assert S.SCREENING_FEATURES
    assert "ads_height" not in S.SCREENING_FEATURES
    assert "mean_ads_force" not in S.SCREENING_FEATURES
    assert "d_band_center_site" not in S.SCREENING_FEATURES


def test_categories_are_disjoint() -> None:
    for (name_a, values_a), (name_b, values_b) in itertools.combinations(CATEGORIES.items(), 2):
        assert values_a.isdisjoint(values_b), (name_a, name_b, values_a & values_b)


def test_leaky_features_rejected_without_registry() -> None:
    frame = pd.DataFrame({"ads_height_final_audit": [1.0, 2.0], "gcn_initial": [0.1, 0.2]})
    for name in ("ads_height_final_audit", "mean_ads_force_final_audit", "d_band_center_site"):
        candidate = frame.copy()
        candidate[name] = 1.0
        with pytest.raises(ValueError):
            S.validate_feature_frame(candidate, features=(name,), registry_path=None)


def test_old_18_feature_set_is_rejected() -> None:
    old = ("d_band_center_slab", "d_band_center_site", "d_band_center_primary", "metallic_radius", "electronegativity", "d_electrons", "primary_metal_frac", "tm_frac", "n_distinct_TM", "gcn_initial", "ads_height", "mean_ads_force", "facet_roughness", "shift", "n_adsorbate_N", "n_adsorbate_H", "adsorbate_atom_count", "n_surface_atoms")
    frame = pd.DataFrame({column: [1.0, 2.0] for column in old})
    with pytest.raises(ValueError):
        S.validate_feature_frame(frame, features=old)


def test_inf_rejected_even_when_missing_allowed() -> None:
    with pytest.raises(ValueError):
        S.validate_feature_frame(pd.DataFrame({"gcn_initial": [0.1, float("inf")]}), features=("gcn_initial",), registry_path=None, allow_missing=True)


def test_registry_matches_schema() -> None:
    if not S.DEFAULT_REGISTRY.exists():
        pytest.skip("feature registry not generated in this checkout")
    registry = S.load_feature_registry(S.DEFAULT_REGISTRY)
    assert set(registry.loc[registry.role == "feature", "name"]) == set(S.SCREENING_FEATURES)


def test_real_table_columns_are_classified_when_present() -> None:
    path = Path("results/adsorbates/oc20_adsorbates_features.parquet")
    if not path.exists():
        pytest.skip("model table not built")
    known = set().union(*CATEGORIES.values())
    columns = {column for column in pd.read_parquet(path).columns if not column.startswith("qc_")}
    assert not (columns - known), sorted(columns - known)


def test_system_id_and_sid_conflict_is_rejected() -> None:
    with pytest.raises(ValueError):
        S.normalize_system_id_column(pd.DataFrame({"system_id": ["a"], "sid": ["b"]}))

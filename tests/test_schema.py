"""Identifier normalisation, metadata validation and helper functions of the screening-v3 schema."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ammonia_ai import schema as S  # noqa: E402


def metadata(**overrides) -> pd.DataFrame:
    data = {
        "system_id": ["random1", "random2"],
        "bulk_mpid": ["mp-1", "mp-2"],
        "bulk_symbols": ["Fe", "Ru"],
        "bulk_reduced_formula": ["Fe", "Ru"],
        "ads_id": [4, 4],
        "adsorbate": ["N", "N"],
        "mapping_adsorbate": ["N", "N"],
        "miller_index": ["(1, 1, 0)", "(0, 0, 1)"],
        "shift": [0.0, 0.25],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_schema_constants_follow_the_screening_regime() -> None:
    assert S.SCHEMA_VERSION.startswith("screening-v3-")
    assert S.TARGET_COLUMN == "adsorption_energy_eV"
    assert not hasattr(S, "FEATURE_COLUMNS")  # the old leaky list must not come back


def test_group_keys_are_declared_and_required() -> None:
    assert {"bulk_mpid", "bulk_reduced_formula", "bulk_chemsys"} <= set(S.GROUP_KEYS)
    assert {"bulk_mpid", "bulk_reduced_formula"} <= set(S.METADATA_COLUMNS)


def test_legacy_sid_is_normalized() -> None:
    result = S.normalize_system_id_column(pd.DataFrame({"sid": ["random1.extxyz.xz"], "value": [1]}))
    assert list(result["system_id"]) == ["random1"] and "sid" not in result.columns


def test_system_id_and_sid_conflict_is_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting"):
        S.normalize_system_id_column(pd.DataFrame({"system_id": ["random1"], "sid": ["random2"], "value": [1]}))


def test_duplicate_system_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        S.normalize_system_id_column(pd.DataFrame({"system_id": ["random1", "random1"]}))


def test_complete_metadata_is_accepted() -> None:
    assert list(S.validate_metadata_frame(metadata())["system_id"]) == ["random1", "random2"]


@pytest.mark.parametrize("column", list(S.METADATA_COLUMNS))
def test_each_required_metadata_column_is_enforced(column) -> None:
    with pytest.raises(ValueError, match=column):
        S.validate_metadata_frame(metadata().drop(columns=[column]))


def test_mapping_adsorbate_disagreement_is_rejected() -> None:
    with pytest.raises(ValueError, match="mapping"):
        S.validate_metadata_frame(metadata(mapping_adsorbate=["N", "N2"]))


def test_reduced_formula_and_chemical_system() -> None:
    assert S.reduced_formula("Fe") == S.reduced_formula("Fe2") == "Fe"
    assert S.reduced_formula("Sc2Ni2Ga4") == "Ga2NiSc"
    assert S.chemical_system("Sc2Ni2Ga4") == "Ga-Ni-Sc"
    assert S.chemical_system("Fe") == S.chemical_system("Fe2") == "Fe"


def test_adsorbate_counts() -> None:
    assert S.adsorbate_counts("NH3") == (1, 3) and S.adsorbate_counts("N2") == (2, 0) and S.adsorbate_counts("nh") == (1, 1)
    with pytest.raises(KeyError):
        S.adsorbate_counts("CO")


def test_model_must_match_the_contract() -> None:
    class Model:
        n_features_in_ = len(S.SCREENING_FEATURES)

    manifest = {"schema_version": S.SCHEMA_VERSION, "feature_columns": list(S.SCREENING_FEATURES)}
    S.assert_model_matches_schema(Model(), manifest)
    Model.n_features_in_ = 16
    with pytest.raises(ValueError):
        S.assert_model_matches_schema(Model(), manifest)
    with pytest.raises(ValueError):
        S.assert_model_matches_schema(Model(), {**manifest, "schema_version": "screening-v2-old"})


def test_model_feature_count_reads_estimator_and_pipeline_step() -> None:
    class Estimator:
        n_features_in_ = 5

    class Step:
        n_features_in_ = 5

    class Pipeline:
        named_steps = {"model": Step()}

    assert S.model_feature_count(Estimator()) == 5 and S.model_feature_count(Pipeline()) == 5

"""Tests for identifier normalisation, metadata validation and the screening-v2 constants."""

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
    assert S.SCHEMA_VERSION.startswith("screening-v2-")
    assert S.TARGET_COLUMN == "adsorption_energy_eV"
    assert tuple(S.TARGET_COLUMNS) == ("adsorption_energy_eV",)
    assert not hasattr(S, "FEATURE_COLUMNS")  # the old leaky list must not come back


def test_group_keys_are_required_metadata() -> None:
    assert {"bulk_mpid", "bulk_symbols"} <= set(S.METADATA_COLUMNS)


def test_legacy_sid_is_normalized() -> None:
    result = S.normalize_system_id_column(pd.DataFrame({"sid": ["random1.extxyz.xz"], "value": [1]}))
    assert list(result["system_id"]) == ["random1"]
    assert "sid" not in result.columns


def test_system_id_and_sid_conflict_is_rejected() -> None:
    frame = pd.DataFrame({"system_id": ["random1"], "sid": ["random2"], "value": [1]})
    with pytest.raises(ValueError, match="conflicting"):
        S.normalize_system_id_column(frame)


def test_duplicate_system_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        S.normalize_system_id_column(pd.DataFrame({"system_id": ["random1", "random1"]}))


def test_complete_metadata_is_accepted() -> None:
    result = S.validate_metadata_frame(metadata())
    assert list(result["system_id"]) == ["random1", "random2"]


@pytest.mark.parametrize("column", list(S.METADATA_COLUMNS))
def test_each_required_metadata_column_is_enforced(column) -> None:
    with pytest.raises(ValueError, match=column):
        S.validate_metadata_frame(metadata().drop(columns=[column]))


def test_mapping_adsorbate_disagreement_is_rejected() -> None:
    with pytest.raises(ValueError, match="mapping"):
        S.validate_metadata_frame(metadata(mapping_adsorbate=["N", "N2"]))


def test_model_feature_count_reads_estimator() -> None:
    class Estimator:
        n_features_in_ = 5

    assert S.model_feature_count(Estimator()) == 5


def test_model_feature_count_reads_pipeline_step() -> None:
    class Step:
        n_features_in_ = 5

    class Pipeline:
        named_steps = {"model": Step()}

    assert S.model_feature_count(Pipeline()) == 5


def test_reduced_formula_merges_equivalent_representations() -> None:
    assert S.reduced_formula("Fe") == S.reduced_formula("Fe2") == "Fe"
    assert S.reduced_formula("Sc2Ni2Ga4") == "Ga2NiSc"

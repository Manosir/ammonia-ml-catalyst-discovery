"""Tests for the canonical feature and metadata contracts."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ammonia_ai.schema import (  # noqa: E402
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMN,
    model_feature_count,
    normalize_system_id_column,
    validate_feature_frame,
    validate_metadata_frame,
)


def make_features(rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: [float(index + 1) for index in range(rows)]
            for name in FEATURE_COLUMNS
        }
    )


def test_canonical_feature_contract_is_16_ordered_columns() -> None:
    assert len(FEATURE_COLUMNS) == 16
    assert FEATURE_COLUMNS[0] == "d_band_center_slab"
    assert FEATURE_COLUMNS[1] == "d_band_center_site"
    assert FEATURE_COLUMNS[-1] == "n_surface_atoms"
    assert TARGET_COLUMN == "delta_E_ads"


def test_feature_frame_accepts_complete_numeric_data() -> None:
    validate_feature_frame(make_features())


def test_feature_frame_rejects_missing_column() -> None:
    frame = make_features().drop(columns=["gcn_initial"])
    with pytest.raises(ValueError, match="gcn_initial"):
        validate_feature_frame(frame)


def test_feature_frame_rejects_missing_value() -> None:
    frame = make_features()
    frame.loc[0, "gcn_initial"] = float("nan")
    with pytest.raises(ValueError, match="gcn_initial"):
        validate_feature_frame(frame)


def test_feature_frame_rejects_non_numeric_data() -> None:
    frame = make_features()
    frame["shift"] = ["zero", "one"]
    with pytest.raises(TypeError, match="shift"):
        validate_feature_frame(frame)


def test_legacy_sid_is_normalized() -> None:
    frame = pd.DataFrame({"sid": ["random1.extxyz.xz"], "value": [1]})
    normalized = normalize_system_id_column(frame)
    assert list(normalized["system_id"]) == ["random1"]
    assert "sid" not in normalized.columns


def test_system_id_and_sid_conflict_is_rejected() -> None:
    frame = pd.DataFrame(
        {"system_id": ["random1"], "sid": ["random2"], "value": [1]}
    )
    with pytest.raises(ValueError, match="conflicting"):
        normalize_system_id_column(frame)


def test_duplicate_system_ids_are_rejected() -> None:
    frame = pd.DataFrame({"system_id": ["random1", "random1"]})
    with pytest.raises(ValueError, match="duplicate"):
        normalize_system_id_column(frame)


def test_metadata_frame_accepts_current_schema() -> None:
    frame = pd.DataFrame(
        {
            "system_id": ["random1", "random2"],
            "miller_index": ["(1, 1, 1)", "(2, 1, 0)"],
            "shift": [0.0, 0.25],
        }
    )
    result = validate_metadata_frame(frame)
    assert tuple(METADATA_COLUMNS) == ("system_id", "miller_index", "shift")
    assert list(result["system_id"]) == ["random1", "random2"]


def test_metadata_frame_accepts_legacy_sid() -> None:
    frame = pd.DataFrame(
        {
            "sid": ["random1"],
            "miller_index": ["(1, 1, 1)"],
            "shift": [0.0],
        }
    )
    result = validate_metadata_frame(frame)
    assert "system_id" in result.columns


def test_metadata_frame_rejects_missing_required_field() -> None:
    frame = pd.DataFrame(
        {"system_id": ["random1"], "miller_index": ["(1, 1, 1)"]}
    )
    with pytest.raises(ValueError, match="shift"):
        validate_metadata_frame(frame)


def test_metadata_frame_rejects_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "system_id": ["random1"],
            "miller_index": ["(1, 1, 1)"],
            "shift": [float("nan")],
        }
    )
    with pytest.raises(ValueError, match="shift"):
        validate_metadata_frame(frame)


def test_model_feature_count_reads_estimator() -> None:
    class Estimator:
        n_features_in_ = 16

    assert model_feature_count(Estimator()) == 16


def test_model_feature_count_reads_pipeline_step() -> None:
    class Step:
        n_features_in_ = 16

    class Pipeline:
        named_steps = {"model": Step()}

    assert model_feature_count(Pipeline()) == 16

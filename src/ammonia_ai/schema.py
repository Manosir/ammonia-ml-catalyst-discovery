"""Canonical data contracts for the Ammonia-AI N* workflow.

Keep model feature order in one place. Training, prediction, screening, and
benchmarking code should import these constants instead of duplicating lists.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "d_band_center_slab",
    "d_band_center_site",
    "d_band_center_primary",
    "metallic_radius",
    "electronegativity",
    "d_electrons",
    "primary_metal_frac",
    "tm_frac",
    "n_distinct_TM",
    "gcn_initial",
    "ads_height",
    "mean_ads_force",
    "facet_roughness",
    "shift",
    "n_N_ads",
    "n_surface_atoms",
)

TARGET_COLUMN = "delta_E_ads"
SYSTEM_ID_COLUMN = "system_id"
LEGACY_SYSTEM_ID_COLUMN = "sid"
METADATA_COLUMNS: tuple[str, ...] = (
    "system_id",
    "miller_index",
    "shift",
)


def normalize_system_id_column(
    frame: pd.DataFrame,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Return a frame with the canonical ``system_id`` column.

    Both the current ``system_id`` and legacy ``sid`` spelling are accepted.
    Duplicate identifiers are rejected because joins must be one-to-one.
    """
    result = frame.copy() if copy else frame
    has_current = SYSTEM_ID_COLUMN in result.columns
    has_legacy = LEGACY_SYSTEM_ID_COLUMN in result.columns

    if has_current and has_legacy:
        current = result[SYSTEM_ID_COLUMN].astype(str)
        legacy = result[LEGACY_SYSTEM_ID_COLUMN].astype(str)
        if not current.equals(legacy):
            raise ValueError("system_id and sid contain conflicting identifiers")
        result = result.drop(columns=[LEGACY_SYSTEM_ID_COLUMN])
    elif has_legacy:
        result = result.rename(columns={LEGACY_SYSTEM_ID_COLUMN: SYSTEM_ID_COLUMN})
    elif not has_current:
        raise ValueError("table must contain system_id or sid")

    result[SYSTEM_ID_COLUMN] = (
        result[SYSTEM_ID_COLUMN]
        .astype(str)
        .str.replace(r"\.extxyz(?:\.xz)?$", "", regex=True)
    )
    if result[SYSTEM_ID_COLUMN].duplicated().any():
        raise ValueError("table contains duplicate system IDs")
    return result


def validate_feature_frame(
    frame: pd.DataFrame,
    *,
    features: Iterable[str] = FEATURE_COLUMNS,
    require_numeric: bool = True,
    allow_missing: bool = False,
) -> None:
    """Validate the columns and values required by a model input frame."""
    expected = tuple(features)
    missing_columns = [name for name in expected if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"missing feature columns: {missing_columns}")

    values = frame.loc[:, expected]
    if require_numeric:
        non_numeric = [name for name in expected if not pd.api.types.is_numeric_dtype(values[name])]
        if non_numeric:
            raise TypeError(f"non-numeric feature columns: {non_numeric}")
    if not allow_missing and values.isna().any().any():
        missing_values = values.columns[values.isna().any()].tolist()
        raise ValueError(f"missing values in feature columns: {missing_values}")


def validate_metadata_frame(
    frame: pd.DataFrame,
    *,
    required: Iterable[str] = METADATA_COLUMNS,
) -> pd.DataFrame:
    """Validate and return normalized OC20 metadata."""
    result = normalize_system_id_column(frame)
    required_columns = tuple(required)
    missing_columns = [name for name in required_columns if name not in result.columns]
    if missing_columns:
        raise ValueError(f"missing metadata columns: {missing_columns}")
    if result["miller_index"].isna().any():
        raise ValueError("metadata contains missing miller_index values")
    if result["shift"].isna().any():
        raise ValueError("metadata contains missing shift values")
    if not pd.api.types.is_numeric_dtype(result["shift"]):
        raise TypeError("metadata shift column must be numeric")
    return result


def model_feature_count(model: Any) -> int | None:
    """Return a fitted estimator's input feature count when available."""
    value = getattr(model, "n_features_in_", None)
    if value is not None:
        return int(value)
    if hasattr(model, "named_steps"):
        for step in reversed(tuple(model.named_steps.values())):
            value = getattr(step, "n_features_in_", None)
            if value is not None:
                return int(value)
    return None

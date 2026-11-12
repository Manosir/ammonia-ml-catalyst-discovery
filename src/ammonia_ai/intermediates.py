"""Schema helpers for linking ammonia-synthesis intermediates."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

ADSORBATES: tuple[str, ...] = ("N2", "N", "NH", "NH2", "NH3")
LINK_COLUMNS: tuple[str, ...] = ("surface_id", "composition", "facet", "site_id")


def validate_intermediate_frame(frame: pd.DataFrame, *, adsorbate: str | None = None) -> pd.DataFrame:
    """Validate a long-form intermediate prediction table."""
    required = set(LINK_COLUMNS) | {"adsorbate", "prediction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing intermediate columns: {missing}")
    result = frame.copy()
    if adsorbate is not None:
        result["adsorbate"] = adsorbate
    result["adsorbate"] = result["adsorbate"].astype(str).str.upper()
    invalid = sorted(set(result["adsorbate"].dropna()) - set(ADSORBATES))
    if invalid:
        raise ValueError(f"unsupported adsorbates: {invalid}")
    if result[list(LINK_COLUMNS) + ["adsorbate"]].duplicated().any():
        raise ValueError("duplicate surface/site/adsorbate records")
    if not pd.api.types.is_numeric_dtype(result["prediction"]):
        raise TypeError("prediction must be numeric")
    return result


def link_predictions(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Link intermediate predictions into one wide table."""
    checked = [validate_intermediate_frame(frame) for frame in frames]
    if not checked:
        raise ValueError("at least one intermediate frame is required")
    long = pd.concat(checked, ignore_index=True)
    if long.duplicated(list(LINK_COLUMNS) + ["adsorbate"]).any():
        raise ValueError("duplicate linked intermediate records")
    wide = long.pivot(index=list(LINK_COLUMNS), columns="adsorbate", values="prediction").reset_index()
    wide.columns.name = None
    return wide.rename(columns={name: f"prediction_{name}" for name in ADSORBATES if name in wide.columns})

"""Versioned, leakage-safe contracts for thermal Haber-Bosch screening."""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "screening-v2-hb-unrelaxed-2026-11"
DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "results/adsorbates/feature_availability.csv"

NUMERIC_SCREENING_FEATURES: tuple[str, ...] = (
    "gcn_initial", "ads_height_initial", "miller_hkl_max", "miller_norm2",
    "metallic_radius_site_mean", "electronegativity_site_mean", "d_electrons_site_mean",
    "primary_metal_frac", "tm_frac", "n_distinct_TM", "facet_roughness",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("site_ensemble", "bulk_structure_type", "material_class", "nonmetal_flag")
SCREENING_FEATURES: tuple[str, ...] = NUMERIC_SCREENING_FEATURES + CATEGORICAL_FEATURES
EXPERIMENTAL_FEATURES: tuple[str, ...] = ("d_band_proxy_tabulated",)
AUDIT_COLUMNS: tuple[str, ...] = (
    "ads_height_final_audit", "mean_ads_force_final_audit", "energy_initial_eV", "energy_final_eV",
    "relaxation_energy_change_eV", "reference_energy", "anomaly", "official_adsorption_energy_eV",
)
IDENTIFIER_COLUMNS: tuple[str, ...] = ("system_id", "bulk_id", "ads_id")
GROUP_KEYS: tuple[str, ...] = ("bulk_mpid", "bulk_symbols")
INPUT_CONTEXT_COLUMNS: tuple[str, ...] = ("adsorbate",)
TARGET_COLUMNS: tuple[str, ...] = ("adsorption_energy_eV",)
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "status", "error", "source_path", "reference_file", "mapping_adsorbate", "ads_symbols",
    "adsorbate_formula", "adsorbate_class", "adsorbate_mass", "binding_atom",
    "official_adsorption_energy_available", "delta_E_ads", "n_adsorbate_N", "n_adsorbate_H",
    "adsorbate_atom_count", "n_surface_atoms", "shift", "miller_index", "adsorption_energy_available",
)
OC20_UNVERIFIED: tuple[str, ...] = ("split", "top", "adsorption_site", "class")
METADATA_COLUMNS: tuple[str, ...] = ("system_id", "bulk_mpid", "bulk_symbols", "ads_id", "adsorbate", "miller_index", "shift", "mapping_adsorbate")
TARGET_COLUMN = "adsorption_energy_eV"
SYSTEM_ID_COLUMN = "system_id"
LEGACY_SYSTEM_ID_COLUMN = "sid"


def load_feature_registry(path: str | Path = DEFAULT_REGISTRY) -> pd.DataFrame:
    registry = pd.read_csv(path)
    required = {"name", "role", "available_at_screening", "implemented", "leakage_risk"}
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"feature registry missing columns: {sorted(missing)}")
    return registry


def assert_features_allowed(feature_cols: Iterable[str], registry_path: str | Path = DEFAULT_REGISTRY, *, allow_experimental: bool = False) -> None:
    registry = load_feature_registry(registry_path)
    names = set(feature_cols)
    roles = {"feature"} | ({"experimental"} if allow_experimental else set())
    allowed = set(registry.loc[registry.role.isin(roles) & (registry.available_at_screening == "yes") & (registry.implemented == "yes"), "name"])
    unknown = sorted(names - set(registry["name"]))
    banned = sorted(names - allowed)
    if unknown or banned:
        raise ValueError(f"feature registry gate failed; unregistered={unknown}, unavailable_or_unimplemented={banned}")


def normalize_system_id_column(frame: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
    result = frame.copy() if copy else frame
    has_current = SYSTEM_ID_COLUMN in result
    has_legacy = LEGACY_SYSTEM_ID_COLUMN in result
    if has_current and has_legacy:
        current = result[SYSTEM_ID_COLUMN].astype(str).str.strip()
        legacy = result[LEGACY_SYSTEM_ID_COLUMN].astype(str).str.strip()
        if not current.equals(legacy):
            raise ValueError("system_id and sid contain conflicting identifiers")
        result = result.drop(columns=[LEGACY_SYSTEM_ID_COLUMN])
    elif has_legacy:
        result = result.rename(columns={LEGACY_SYSTEM_ID_COLUMN: SYSTEM_ID_COLUMN})
    elif not has_current:
        raise ValueError("table must contain system_id or sid")
    result[SYSTEM_ID_COLUMN] = result[SYSTEM_ID_COLUMN].astype(str).str.strip().str.replace(r"\.extxyz(?:\.xz)?$", "", regex=True)
    if result[SYSTEM_ID_COLUMN].duplicated().any():
        raise ValueError("table contains duplicate system IDs")
    return result


def validate_feature_frame(frame: pd.DataFrame, *, features: Iterable[str] = SCREENING_FEATURES, registry_path: str | Path | None = DEFAULT_REGISTRY, allow_experimental: bool = False, allow_missing: bool = False) -> None:
    expected = tuple(features)
    permitted = set(SCREENING_FEATURES) | (set(EXPERIMENTAL_FEATURES) if allow_experimental else set())
    outside = sorted(set(expected) - permitted)
    if outside:
        raise ValueError(f"not permitted as screening inputs: {outside}")
    if registry_path is not None:
        assert_features_allowed(expected, registry_path, allow_experimental=allow_experimental)
    missing = [name for name in expected if name not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    categorical = set(CATEGORICAL_FEATURES) & set(expected)
    numeric = [name for name in expected if name not in categorical]
    bad = [name for name in numeric if not pd.api.types.is_numeric_dtype(frame[name])]
    if bad:
        raise TypeError(f"non-numeric feature columns: {bad}")
    if numeric and np.isinf(frame[numeric].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("feature table contains infinite values")
    if not allow_missing and frame.loc[:, expected].isna().any().any():
        raise ValueError(f"missing feature values: {frame.loc[:, expected].columns[frame.loc[:, expected].isna().any()].tolist()}")
    constants = [name for name in expected if frame[name].nunique(dropna=True) <= 1]
    if constants:
        warnings.warn(f"all-constant feature columns: {constants}", stacklevel=2)


def validate_metadata_frame(frame: pd.DataFrame, *, required: Iterable[str] = METADATA_COLUMNS) -> pd.DataFrame:
    result = normalize_system_id_column(frame)
    missing = [name for name in required if name not in result.columns]
    if missing:
        raise ValueError(f"missing metadata columns: {missing}")
    if result["mapping_adsorbate"].astype(str).str.upper().ne(result["adsorbate"].astype(str).str.upper()).any():
        raise ValueError("mapping_adsorbate disagrees with folder-derived adsorbate")
    return result


def model_feature_count(model: Any) -> int | None:
    value = getattr(model, "n_features_in_", None)
    if value is not None:
        return int(value)
    if hasattr(model, "named_steps"):
        for step in reversed(tuple(model.named_steps.values())):
            value = getattr(step, "n_features_in_", None)
            if value is not None:
                return int(value)
    return None


def assert_model_schema_version(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"model schema {manifest.get('schema_version')!r} does not match {SCHEMA_VERSION!r}")

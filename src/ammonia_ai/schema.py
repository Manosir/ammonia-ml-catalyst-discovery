"""Versioned, leakage-safe contracts for thermal Haber-Bosch screening (unrelaxed slab + adsorbate)."""
from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "screening-v3-hb-unrelaxed-2026-11"
DEFAULT_REGISTRY = Path(__file__).resolve().parents[2] / "results/adsorbates/feature_availability.csv"

# ---------------------------------------------------------------------------------------------
# Screening contract, frozen from the ablations of 2026-11-14.
# Every name is computed from information available for an UNRELAXED slab + adsorbate (plus the
# adsorption-site coordinate). Bulk composition enters ONLY through tabulated element-property
# statistics: no element one-hot columns (they do not transfer to unseen elements).
# ---------------------------------------------------------------------------------------------
SCOPES: tuple[str, ...] = ("bulk", "surf", "site")
PROPERTY_FAMILIES: tuple[str, ...] = ("en", "group", "period")
PROPERTY_STATS: tuple[str, ...] = ("mean", "min", "max", "rng")

ADSORBATE_FEATURES: tuple[str, ...] = ("ads_nN", "ads_nH")
MILLER_FEATURES: tuple[str, ...] = ("miller_hkl_max", "miller_norm2")
PROPERTY_FEATURES: tuple[str, ...] = tuple(
    f"{scope}_{fam}_{stat}" for scope in SCOPES for fam in PROPERTY_FAMILIES for stat in PROPERTY_STATS
)
COMPOSITION_FEATURES: tuple[str, ...] = tuple(
    f"{scope}_{kind}" for scope in SCOPES for kind in ("tm_frac", "nm_frac", "n_elem")
)
GEOMETRY_FEATURES: tuple[str, ...] = (
    "geo_d_min", "geo_h_nn1", "geo_h_k3", "geo_h_site", "geo_h_top", "geo_h_plane", "geo_min_dist_all",
)
TOPOLOGY_FEATURES: tuple[str, ...] = ("topo_n_site_atoms", "topo_gap2", "topo_gap3")
COORDINATION_FEATURES: tuple[str, ...] = (
    "cn_slab_mean", "cn_gcn_site", "cn_site_mean", "cn_site_max", "cn_site_tag1_frac",
)
# columns produced by scripts/build_candidates.py that belong to the contract
CANDIDATE_COLUMNS: tuple[str, ...] = (
    PROPERTY_FEATURES + COMPOSITION_FEATURES + GEOMETRY_FEATURES + TOPOLOGY_FEATURES + COORDINATION_FEATURES
)
NUMERIC_SCREENING_FEATURES: tuple[str, ...] = ADSORBATE_FEATURES + MILLER_FEATURES + CANDIDATE_COLUMNS
CATEGORICAL_FEATURES: tuple[str, ...] = ()
SCREENING_FEATURES: tuple[str, ...] = NUMERIC_SCREENING_FEATURES + CATEGORICAL_FEATURES
EXPERIMENTAL_FEATURES: tuple[str, ...] = ()
# depend on how the adsorbate was placed (OC20 heuristic): report results with and without them
PLACEMENT_SENSITIVE_FEATURES: tuple[str, ...] = GEOMETRY_FEATURES + ("topo_gap2", "topo_gap3")
SITE_DEPENDENT_FEATURES: tuple[str, ...] = tuple(
    f for f in SCREENING_FEATURES
    if f.startswith(("site_", "geo_", "topo_", "cn_site_")) or f == "cn_gcn_site"
)

ADSORBATE_COMPOSITION: dict[str, tuple[int, int]] = {
    "N": (1, 0), "N2": (2, 0), "NH": (1, 1), "NH2": (1, 2), "NH3": (1, 3),
}

# ---------------------------------------------------------------------------------------------
# Everything that is NOT a screening input
# ---------------------------------------------------------------------------------------------
AUDIT_COLUMNS: tuple[str, ...] = (
    "ads_height_final_audit", "mean_ads_force_final_audit", "energy_initial_eV", "energy_final_eV",
    "relaxation_energy_change_eV", "reference_energy", "anomaly", "official_adsorption_energy_eV",
)
# last-frame / trajectory quantities in results/structure_pass/structure_pass.parquet
STRUCTURE_PASS_AUDIT_COLUMNS: tuple[str, ...] = (
    "n_frames", "slab_disp_max", "slab_disp_rms", "slab_n_disp_gt_0p5", "surf_disp_max",
    "surf_n_disp_gt_0p5", "bind_disp_total", "bind_disp_z", "d_min_ads_slab_final", "h_k3_final",
    "h_top_final", "max_force_final", "e_first", "e_last",
)
AUDIT_NAME_FRAGMENTS: tuple[str, ...] = ("final", "disp", "frames", "force", "e_first", "e_last", "bind_disp")
IDENTIFIER_COLUMNS: tuple[str, ...] = ("system_id", "bulk_id", "ads_id")
GROUP_KEYS: tuple[str, ...] = ("bulk_mpid", "bulk_reduced_formula", "bulk_chemsys")
INPUT_CONTEXT_COLUMNS: tuple[str, ...] = ("adsorbate",)
TARGET_COLUMNS: tuple[str, ...] = ("adsorption_energy_eV",)
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "status", "error", "source_path", "reference_file", "mapping_adsorbate", "ads_symbols",
    "adsorbate_formula", "adsorbate_class", "adsorbate_mass", "binding_atom",
    "official_adsorption_energy_available", "delta_E_ads", "n_adsorbate_N", "n_adsorbate_H",
    "adsorbate_atom_count", "n_surface_atoms", "shift", "adsorption_energy_available",
)
OC20_UNVERIFIED: tuple[str, ...] = ("split",)  # pickle value names differ from the official docs
OC20_METADATA_COLUMNS: tuple[str, ...] = ("bulk_symbols", "miller_index", "top", "adsorption_site", "class")
DERIVED_AUDIT_COLUMNS: tuple[str, ...] = ("anomaly_label", "is_clean_primary_regression_row")
METADATA_COLUMNS: tuple[str, ...] = (
    "system_id", "bulk_mpid", "bulk_reduced_formula", "bulk_symbols", "ads_id", "adsorbate",
    "miller_index", "shift", "mapping_adsorbate",
)
TARGET_COLUMN = "adsorption_energy_eV"
SYSTEM_ID_COLUMN = "system_id"
LEGACY_SYSTEM_ID_COLUMN = "sid"

_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*)")


def _counts(symbols: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element, number in _ELEMENT.findall(str(symbols)):
        counts[element] = counts.get(element, 0) + (int(number) if number else 1)
    return counts


def reduced_formula(symbols: str) -> str:
    """Element-sorted reduced formula from an OC20 bulk_symbols string ('Fe2' -> 'Fe')."""
    counts = _counts(symbols)
    divisor = reduce(gcd, counts.values()) if counts else 1
    parts = []
    for element in sorted(counts):
        n = counts[element] // divisor
        parts.append(element if n == 1 else f"{element}{n}")
    return "".join(parts)


def chemical_system(symbols: str) -> str:
    """Sorted element set joined by '-' ('Sc2Ni2Ga4' -> 'Ga-Ni-Sc'); the grouping key of the evaluations."""
    return "-".join(sorted(_counts(symbols)))


def adsorbate_counts(adsorbate: str) -> tuple[int, int]:
    """(number of N atoms, number of H atoms) of an adsorbate label; KeyError for unknown labels."""
    return ADSORBATE_COMPOSITION[str(adsorbate).strip().upper()]


def load_feature_registry(path: str | Path = DEFAULT_REGISTRY) -> pd.DataFrame:
    registry = pd.read_csv(path)
    required = {"name", "role", "available_at_screening", "implemented", "leakage_risk"}
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"feature registry missing columns: {sorted(missing)}")
    return registry


def assert_features_allowed(
    feature_cols: Iterable[str],
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    allow_experimental: bool = False,
) -> None:
    registry = load_feature_registry(registry_path)
    names = set(feature_cols)
    roles = {"feature"} | ({"experimental"} if allow_experimental else set())
    allowed = set(
        registry.loc[
            registry.role.isin(roles)
            & (registry.available_at_screening == "yes")
            & (registry.implemented == "yes"),
            "name",
        ]
    )
    unknown = sorted(names - set(registry["name"]))
    banned = sorted(names - allowed)
    if unknown or banned:
        raise ValueError(
            f"feature registry gate failed; unregistered={unknown}, unavailable_or_unimplemented={banned}"
        )


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
    result[SYSTEM_ID_COLUMN] = (
        result[SYSTEM_ID_COLUMN].astype(str).str.strip().str.replace(r"\.extxyz(?:\.xz)?$", "", regex=True)
    )
    if result[SYSTEM_ID_COLUMN].duplicated().any():
        raise ValueError("table contains duplicate system IDs")
    return result


def validate_feature_frame(
    frame: pd.DataFrame,
    *,
    features: Iterable[str] = SCREENING_FEATURES,
    registry_path: str | Path | None = DEFAULT_REGISTRY,
    allow_experimental: bool = False,
    allow_missing: bool = False,
) -> None:
    expected = tuple(features)
    permitted = set(SCREENING_FEATURES) | (set(EXPERIMENTAL_FEATURES) if allow_experimental else set())
    outside = sorted(set(expected) - permitted)
    if outside:  # unconditional: does not depend on the registry
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
    block = frame.loc[:, list(expected)]
    if not allow_missing and block.isna().any().any():
        raise ValueError(f"missing feature values: {block.columns[block.isna().any()].tolist()}")
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


def assert_model_matches_schema(model: Any, manifest: dict[str, Any]) -> None:
    """A saved model may be used for screening only if it was trained on the CURRENT contract."""
    assert_model_schema_version(manifest)
    if tuple(manifest.get("feature_columns", ())) != tuple(SCREENING_FEATURES):
        raise ValueError("manifest feature columns differ from the current contract")
    count = model_feature_count(model)
    if count is not None and count != len(SCREENING_FEATURES):
        raise ValueError(f"model expects {count} features, the contract has {len(SCREENING_FEATURES)}")

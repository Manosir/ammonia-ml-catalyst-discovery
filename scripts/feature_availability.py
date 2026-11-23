#!/usr/bin/env python3
"""Generate the feature registry FROM the schema (single source of truth)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ammonia_ai import schema as S  # noqa: E402

COLS = ["name", "role", "source", "available_at_screening", "extra_step_needed", "implemented", "leakage_risk", "note"]
LEGACY_EXCLUDED = ("ads_height_final", "mean_ads_force", "d_band_center_site", "d_band_center_slab_dft", "forces_initial")


def _source(name: str) -> str:
    if name in S.ADSORBATE_FEATURES:
        return "adsorbate identity (numbers of N and H atoms), chosen by the user"
    if name in S.MILLER_FEATURES:
        return "slab Miller index (numeric encoding)"
    if name in S.PROPERTY_FEATURES or name in S.COMPOSITION_FEATURES:
        scope = name.split("_", 1)[0]
        return {
            "bulk": "bulk composition x tabulated element properties",
            "surf": "surface-layer composition of the initial slab (tag 1) x tabulated element properties",
            "site": "slab atoms within d_min + 0.5 A of the adsorbate (initial frame) x tabulated element properties",
        }[scope]
    if name in S.GEOMETRY_FEATURES:
        return "initial-frame adsorbate-slab distances and heights (minimum image)"
    if name in S.TOPOLOGY_FEATURES:
        return "initial-frame nearest slab atoms around the adsorbate"
    if name in S.COORDINATION_FEATURES:
        return "initial-frame slab coordination numbers (real cell, 1.2 x covalent radii)"
    return ""


def registry_frame() -> pd.DataFrame:
    rows: dict[str, dict] = {}

    def add(name, role, source="", avail="no", extra="none", impl="yes", leak="n/a", note=""):
        if name in rows:
            raise ValueError(f"duplicate registry name: {name}")
        rows[name] = dict(
            name=name, role=role, source=source, available_at_screening=avail, extra_step_needed=extra,
            implemented=impl, leakage_risk=leak, note=note,
        )

    placement, site_dep = set(S.PLACEMENT_SENSITIVE_FEATURES), set(S.SITE_DEPENDENT_FEATURES)
    for name in S.SCREENING_FEATURES:
        add(name, "feature", _source(name), "yes", "site enumeration" if name in site_dep else "none", "yes", "low",
            "placement-sensitive: depends on how the adsorbate was positioned" if name in placement else "")
    for name in S.EXPERIMENTAL_FEATURES:
        add(name, "experimental", "", "yes", "none", "no", "low", "")
    for name in S.AUDIT_COLUMNS:
        role = "target" if name == "official_adsorption_energy_eV" else ("label_filter" if name == "anomaly" else "excluded")
        add(name, role, "relaxed or target-derived quantity", "no", "relaxation or DFT", "yes", "high", "audit only")
    for name in S.STRUCTURE_PASS_AUDIT_COLUMNS:
        add(name, "excluded", "last-frame or trajectory quantity from the structure pass", "no", "relaxation", "yes", "high", "audit only")
    for name in LEGACY_EXCLUDED:
        add(name, "excluded", "legacy Stage-1 feature (final-frame or DFT-dependent)", "no", "relaxation or DFT", "yes", "high", "not screening-available")
    for name in S.TARGET_COLUMNS:
        add(name, "target", "energy_final - reference_energy (OC20 official)", "no", "relaxation", "yes", "n/a", "primary supervised target")
    for name in S.IDENTIFIER_COLUMNS:
        add(name, "identifier", "OC20", "no", "none", "yes", "n/a", "never a feature")
    for name in S.GROUP_KEYS:
        add(name, "group_key", "bulk metadata", "no", "none", "yes", "n/a", "grouped split key, never a feature")
    for name in S.INPUT_CONTEXT_COLUMNS:
        add(name, "context", "adsorbate label", "no", "none", "yes", "n/a", "enters through ads_nN / ads_nH")
    return pd.DataFrame(list(rows.values()), columns=COLS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/adsorbates/feature_availability.csv"))
    args = parser.parse_args()
    frame = registry_frame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} registry rows ({int((frame.role == 'feature').sum())} screening features) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

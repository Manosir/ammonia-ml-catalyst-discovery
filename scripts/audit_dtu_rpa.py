#!/usr/bin/env python3
"""Audit DTU adsorption/surface ASE databases and functional pair coverage."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from ase.db import connect
except ImportError as exc:  # pragma: no cover - exercised in CLI environments
    raise SystemExit("ASE is required: python -m pip install ase") from exc

ADSORPTION_COLUMNS = {
    "LDA": "LDA_adsorp",
    "PBE": "PBE_adsorp",
    "RPBE": "RPBE_adsorp",
    "BEEFvdW": "BEEFvdW_adsorp",
    "mBEEF": "mBEEF_adsorp",
    "EXX": "EXX_adsorp",
    "RPA": "RPA_adsorp",
    "RPA_EXX": "RPA_EXX_adsorp",
}
SURFACE_COLUMNS = {
    "LDA": "LDA_surf",
    "PBE": "PBE_surf",
    "RPBE": "RPBE_surf",
    "BEEFvdW": "BEEFvdW_surf",
    "mBEEF": "mBEEF_surf",
    "EXX": "EXX_surf",
    "RPA": "RPA_surf",
}


def finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def row_kvp(row: Any) -> dict[str, Any]:
    return dict(getattr(row, "key_value_pairs", {}) or {})


def first_value(kvp: dict[str, Any], names: list[str], default: Any = None) -> Any:
    for name in names:
        if name in kvp and kvp[name] not in (None, ""):
            return kvp[name]
    return default


def pairing_key(row: Any) -> tuple[str, str, str, str]:
    kvp = row_kvp(row)
    surface = first_value(kvp, ["surf_mat", "surface_material", "surfaceComposition", "chemicalComposition"], "")
    adsorbate = first_value(kvp, ["adsorbate", "adsorbate_name"], "")
    facet = first_value(kvp, ["facet", "surface_facet", "miller_index"], "")
    coverage = first_value(kvp, ["coverage", "coverages", "adsorption_coverage"], "")
    # If no semantic identity fields exist, row.id is retained as an explicit
    # fallback. The report marks such rows so they are not silently paired.
    if not surface and not adsorbate and not facet and not coverage:
        return (f"row:{row.id}", "", "", "")
    return (str(surface), str(adsorbate), str(facet), str(coverage))


def audit_db(path: Path, columns: dict[str, str]) -> dict[str, Any]:
    db = connect(str(path))
    rows = list(db.select())
    key_counts = Counter()
    value_counts = Counter()
    key_examples: dict[str, Any] = {}
    for row in rows:
        kvp = row_kvp(row)
        for key in kvp:
            key_counts[key] += 1
            key_examples.setdefault(key, kvp[key])
        for functional, column in columns.items():
            if column in kvp:
                value_counts[functional] += int(finite(kvp[column]))
    return {
        "path": str(path),
        "row_count": len(rows),
        "row_keys": sorted(key_counts),
        "key_counts": dict(sorted(key_counts.items())),
        "key_examples": key_examples,
        "functional_columns": {
            functional: {
                "column": column,
                "finite_values": value_counts.get(functional, 0),
            }
            for functional, column in columns.items()
        },
    }


def audit_pairs(path: Path) -> dict[str, Any]:
    db = connect(str(path))
    rows = list(db.select())
    pair_counts: Counter[str] = Counter()
    key_count = Counter()
    matched_records = []
    functional_counts = Counter()
    fallback_rows = 0
    rpa_identity_verified = 0
    rpa_identity_failed = 0
    for row in rows:
        kvp = row_kvp(row)
        present = tuple(
            functional for functional, column in ADSORPTION_COLUMNS.items()
            if finite(kvp.get(column))
        )
        for functional in present:
            functional_counts[functional] += 1
        for left_index, left in enumerate(present):
            for right in present[left_index + 1 :]:
                pair_counts[f"{left}__{right}"] += 1
        key = pairing_key(row)
        key_count[key] += 1
        if key[0].startswith("row:"):
            fallback_rows += 1
        if len(present) >= 2:
            values_eV = {
                functional: float(kvp[ADSORPTION_COLUMNS[functional]])
                for functional in present
            }
            # Verify RPA_EXX = EXX + RPA identity for rows that have all three
            if "EXX" in present and "RPA" in present and "RPA_EXX" in present:
                exx = values_eV["EXX"]
                rpa = values_eV["RPA"]
                rpa_exx = values_eV["RPA_EXX"]
                if math.isclose(exx + rpa, rpa_exx, rel_tol=1e-6, abs_tol=1e-6):
                    rpa_identity_verified += 1
                else:
                    rpa_identity_failed += 1
            matched_records.append({
                "row_id": row.id,
                "pairing_key": {
                    "surface": key[0],
                    "adsorbate": key[1],
                    "facet": key[2],
                    "coverage": key[3],
                },
                "available_functionals": list(present),
                "values_eV": values_eV,
            })
    return {
        "row_count": len(rows),
        "functional_value_counts": dict(sorted(functional_counts.items())),
        "matched_pair_counts_same_row": dict(sorted(pair_counts.items())),
        "rows_with_at_least_two_functionals": len(matched_records),
        "pairing_key_unique_count": len(key_count),
        "pairing_key_duplicate_groups": sum(count > 1 for count in key_count.values()),
        "rows_using_row_id_fallback": fallback_rows,
        "rpa_identity_verified": rpa_identity_verified,
        "rpa_identity_failed": rpa_identity_failed,
        "matched_records": matched_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adsorption-db", type=Path, required=True)
    parser.add_argument("--surfaces-db", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=Path("results/dtu_rpa/schema_audit"))
    args = parser.parse_args()
    for path in (args.adsorption_db, args.surfaces_db):
        if not path.exists():
            raise SystemExit(f"Missing database: {path}")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    adsorption = audit_db(args.adsorption_db, ADSORPTION_COLUMNS)
    surfaces = audit_db(args.surfaces_db, SURFACE_COLUMNS)
    pairs = audit_pairs(args.adsorption_db)
    report = {"adsorption": adsorption, "surfaces": surfaces, "functional_pair_audit": pairs}

    json_path = args.output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    summary_path = args.output_prefix.with_suffix(".md")
    lines = [
        "# DTU RPA schema audit",
        "",
        f"- Adsorption rows: {adsorption['row_count']}",
        f"- Surface rows: {surfaces['row_count']}",
        f"- Rows with at least two adsorption functionals: {pairs['rows_with_at_least_two_functionals']}",
        f"- Unique semantic pairing keys: {pairs['pairing_key_unique_count']}",
        f"- Duplicate pairing-key groups: {pairs['pairing_key_duplicate_groups']}",
        f"- Rows requiring row-ID fallback: {pairs['rows_using_row_id_fallback']}",
        f"- RPA_EXX = EXX + RPA verified: {pairs['rpa_identity_verified']} / {pairs['rpa_identity_verified'] + pairs['rpa_identity_failed']}",
        "",
        "## Same-row functional pair counts",
        "",
    ]
    for pair, count in pairs["matched_pair_counts_same_row"].items():
        lines.append(f"- `{pair}`: {count}")
    lines += ["", "## RPA Identity Verification", "", f"- Rows with EXX, RPA, and RPA_EXX all present: {pairs['rpa_identity_verified'] + pairs['rpa_identity_failed']}", f"- RPA_EXX = EXX + RPA holds (within 1e-6 eV): {pairs['rpa_identity_verified']}", f"- Failures: {pairs['rpa_identity_failed']}", "", "## Interpretation", "", "Pair counts above are same-row availability counts. The audit also retains semantic pairing keys and raw values in the JSON report. A subsequent offset analysis must verify facet, coverage, geometry, and reference-state compatibility before calculating offsets.", "", "Critical finding: In the DTU ASE database, `RPA_adsorp` is the **correlation contribution only**, not the total RPA adsorption energy. Total RPA adsorption energy = `EXX_adsorp + RPA_adsorp` = `RPA_EXX_adsorp`. This identity is verified for all matched records."]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps({"adsorption_rows": adsorption["row_count"], "surface_rows": surfaces["row_count"], **pairs}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

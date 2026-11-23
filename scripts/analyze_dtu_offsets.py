#!/usr/bin/env python3
"""Analyze DTU RPA functional adsorption energy offsets and generate summary reports.

Calculates same-row functional offsets from DTU adsorption.db:
- PBE minus RPBE
- RPA-total minus PBE
- RPA-total minus RPBE
- BEEFvdW minus RPBE
- mBEEF minus PBE
- LDA minus PBE
- EXX minus PBE

Where RPA-total = RPA_EXX_adsorp = EXX_adsorp + RPA_adsorp.
RPA_adsorp is the correlation term only and is NOT used as total RPA.

IMPORTANT DISCLAIMER:
These offsets are DTU benchmark offsets computed on DTU slab models.
They MUST NOT be used as universal physical corrections or merged into OC20 training labels.
Note that the DTU dataset lacks ammonia-synthesis intermediates (NH, NH2, NH3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from ase.db import connect
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ASE is required: python -m pip install ase") from exc

MAGNETIC_ELEMENTS = {"Fe", "Co", "Ni"}

TARGET_OFFSETS = [
    "PBE_minus_RPBE",
    "RPA_total_minus_PBE",
    "RPA_total_minus_RPBE",
    "BEEFvdW_minus_RPBE",
    "mBEEF_minus_PBE",
    "LDA_minus_PBE",
    "EXX_minus_PBE",
]

OFFSET_LABELS = {
    "PBE_minus_RPBE": "PBE – RPBE",
    "RPA_total_minus_PBE": "RPA_total – PBE",
    "RPA_total_minus_RPBE": "RPA_total – RPBE",
    "BEEFvdW_minus_RPBE": "BEEFvdW – RPBE",
    "mBEEF_minus_PBE": "mBEEF – PBE",
    "LDA_minus_PBE": "LDA – PBE",
    "EXX_minus_PBE": "EXX – PBE",
}

MISSING_SPECIES_NOTE = (
    "DTU dataset contains adsorbates: CH, CO, H, N, N2, NO, O, OH. "
    "It lacks ammonia-synthesis intermediate species: NH, NH2, NH3."
)

DISCLAIMER_NOTE = (
    "DTU benchmark offsets are empirical differences between electronic structure methods "
    "on DTU slab models. They MUST NOT be used as universal physical corrections or "
    "merged into OC20 training labels."
)


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_stats(
    values: Sequence[float] | np.ndarray,
    seed: int = 42,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Compute summary statistics and 95% bootstrap confidence interval for a 1D array of values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "mad": None,
            "min": None,
            "max": None,
            "ci_95_low": None,
            "ci_95_high": None,
        }

    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    mad_val = float(np.median(np.abs(arr - median_val)))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    if n > 1:
        rng = np.random.default_rng(seed)
        boot_idx = rng.choice(n, size=(n_bootstrap, n), replace=True)
        boot_means = np.mean(arr[boot_idx], axis=1)
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        ci_low_val = float(ci_low)
        ci_high_val = float(ci_high)
    else:
        ci_low_val = mean_val
        ci_high_val = mean_val

    return {
        "count": n,
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "mad": mad_val,
        "min": min_val,
        "max": max_val,
        "ci_95_low": ci_low_val,
        "ci_95_high": ci_high_val,
    }


def extract_row_offsets(kvp: dict[str, Any], row_id: int | None = None) -> dict[str, Any] | None:
    """Extract same-row functional values and compute targets from a single ASE DB row's key-value pairs."""
    pbe = kvp.get("PBE_adsorp")
    rpbe = kvp.get("RPBE_adsorp")
    rpa_exx = kvp.get("RPA_EXX_adsorp")
    rpa_corr = kvp.get("RPA_adsorp")
    exx = kvp.get("EXX_adsorp")
    beef = kvp.get("BEEFvdW_adsorp")
    mbeef = kvp.get("mBEEF_adsorp")
    lda = kvp.get("LDA_adsorp")
    surf = kvp.get("surf_mat", "")
    ads = kvp.get("adsorbate", "")

    # Ensure all target functionals are present and finite
    required = [pbe, rpbe, rpa_exx, exx, beef, mbeef, lda]
    if any(v is None or not math.isfinite(float(v)) for v in required):
        return None

    pbe_val = float(pbe)
    rpbe_val = float(rpbe)
    rpa_exx_val = float(rpa_exx)
    rpa_corr_val = float(rpa_corr) if rpa_corr is not None else None
    exx_val = float(exx)
    beef_val = float(beef)
    mbeef_val = float(mbeef)
    lda_val = float(lda)

    is_mag = str(surf) in MAGNETIC_ELEMENTS

    return {
        "row_id": row_id,
        "surf_mat": str(surf),
        "adsorbate": str(ads),
        "magnetic_group": "magnetic" if is_mag else "non-magnetic",
        "PBE_adsorp": pbe_val,
        "RPBE_adsorp": rpbe_val,
        "RPA_EXX_adsorp": rpa_exx_val,
        "RPA_corr_adsorp": rpa_corr_val,
        "EXX_adsorp": exx_val,
        "BEEFvdW_adsorp": beef_val,
        "mBEEF_adsorp": mbeef_val,
        "LDA_adsorp": lda_val,
        "PBE_minus_RPBE": pbe_val - rpbe_val,
        "RPA_total_minus_PBE": rpa_exx_val - pbe_val,
        "RPA_total_minus_RPBE": rpa_exx_val - rpbe_val,
        "BEEFvdW_minus_RPBE": beef_val - rpbe_val,
        "mBEEF_minus_PBE": mbeef_val - pbe_val,
        "LDA_minus_PBE": lda_val - pbe_val,
        "EXX_minus_PBE": exx_val - pbe_val,
    }


def load_and_extract_offsets(db_path: Path) -> pd.DataFrame:
    """Load DTU adsorption database and compute same-row functional offsets."""
    db = connect(str(db_path))
    records = []
    for row in db.select():
        kvp = dict(getattr(row, "key_value_pairs", {}) or {})
        rec = extract_row_offsets(kvp, row_id=row.id)
        if rec is not None:
            records.append(rec)
    return pd.DataFrame(records)


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics across overall dataset and all required breakdowns."""
    results: dict[str, Any] = {
        "overall": {},
        "by_adsorbate": {},
        "by_surface_element": {},
        "by_magnetic_group": {},
    }

    # Overall statistics for each target offset
    for target in TARGET_OFFSETS:
        results["overall"][target] = compute_stats(df[target])

    # By adsorbate
    for ads, sub in df.groupby("adsorbate"):
        results["by_adsorbate"][str(ads)] = {
            target: compute_stats(sub[target]) for target in TARGET_OFFSETS
        }

    # By surface element
    for surf, sub in df.groupby("surf_mat"):
        results["by_surface_element"][str(surf)] = {
            target: compute_stats(sub[target]) for target in TARGET_OFFSETS
        }

    # By magnetic group
    for mag_group, sub in df.groupby("magnetic_group"):
        results["by_magnetic_group"][str(mag_group)] = {
            target: compute_stats(sub[target]) for target in TARGET_OFFSETS
        }

    return results


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Export detailed per-row functional offsets table to CSV."""
    # Ensure columns match expected export format
    export_cols = [
        "row_id",
        "surf_mat",
        "adsorbate",
        "magnetic_group",
        "PBE_adsorp",
        "RPBE_adsorp",
        "RPA_EXX_adsorp",
        "RPA_corr_adsorp",
        "EXX_adsorp",
        "BEEFvdW_adsorp",
        "mBEEF_adsorp",
        "LDA_adsorp",
        "PBE_minus_RPBE",
        "RPA_total_minus_PBE",
        "RPA_total_minus_RPBE",
        "BEEFvdW_minus_RPBE",
        "mBEEF_minus_PBE",
        "LDA_minus_PBE",
        "EXX_minus_PBE",
    ]
    df_out = df[export_cols].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)


def export_json(
    summary: dict[str, Any],
    db_sha256: str,
    db_path: Path,
    output_path: Path,
    row_count: int,
) -> None:
    """Export metadata and summary statistics to JSON."""
    payload = {
        "metadata": {
            "title": "DTU RPA Functional Adsorption Energy Offsets Analysis",
            "source_database": str(db_path),
            "source_sha256": db_sha256,
            "label_type": "DTU benchmark offsets",
            "usage_restriction": "DTU benchmark offsets only. DO NOT merge into OC20 training labels or use as universal corrections.",
            "missing_species_note": MISSING_SPECIES_NOTE,
            "disclaimer": DISCLAIMER_NOTE,
            "row_count": row_count,
            "target_offsets": TARGET_OFFSETS,
            "rpa_definition": "RPA_total = RPA_EXX_adsorp = EXX_adsorp + RPA_adsorp. RPA_corr_adsorp is correlation only.",
        },
        "summary": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def export_markdown(
    summary: dict[str, Any],
    db_sha256: str,
    db_path: Path,
    output_path: Path,
    row_count: int,
) -> None:
    """Export formatted Markdown summary report."""
    lines = [
        "# DTU RPA Functional Adsorption Energy Offsets Report",
        "",
        "**Benchmark Status:** DTU Benchmark Offsets (Non-Universal)",
        f"**Source Database:** `{db_path}`  ",
        f"**Source SHA256:** `{db_sha256}`  ",
        f"**Paired Records Analyzed:** {row_count}  ",
        "",
        "### Important Notes & Usage Scope",
        f"- **Disclaimer:** {DISCLAIMER_NOTE}",
        f"- **Missing Species:** {MISSING_SPECIES_NOTE}",
        "- **RPA Definition:** Total RPA adsorption energy is `RPA_total = RPA_EXX_adsorp = EXX_adsorp + RPA_adsorp`. `RPA_adsorp` is correlation only.",
        "",
        "---",
        "",
        "## 1. Overall Functional Offsets (All 200 Rows)",
        "",
        "| Offset Target | Count | Mean (eV) | Median (eV) | Std (eV) | MAD (eV) | Min (eV) | Max (eV) | 95% CI (eV) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for target in TARGET_OFFSETS:
        s = summary["overall"][target]
        ci_str = f"[{s['ci_95_low']:+.3f}, {s['ci_95_high']:+.3f}]"
        lines.append(
            f"| `{OFFSET_LABELS[target]}` | {s['count']} | {s['mean']:+.3f} | {s['median']:+.3f} | "
            f"{s['std']:.3f} | {s['mad']:.3f} | {s['min']:+.3f} | {s['max']:+.3f} | {ci_str} |"
        )

    lines += [
        "",
        "## 2. Offsets Breakdown by Magnetic Group",
        "",
        "| Magnetic Group | Target Offset | Count | Mean (eV) | Median (eV) | Std (eV) | MAD (eV) | 95% CI (eV) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for group in ["magnetic", "non-magnetic"]:
        for target in ["RPA_total_minus_RPBE", "RPA_total_minus_PBE", "PBE_minus_RPBE"]:
            s = summary["by_magnetic_group"].get(group, {}).get(target)
            if s:
                ci_str = f"[{s['ci_95_low']:+.3f}, {s['ci_95_high']:+.3f}]"
                lines.append(
                    f"| {group} | `{OFFSET_LABELS[target]}` | {s['count']} | {s['mean']:+.3f} | {s['median']:+.3f} | "
                    f"{s['std']:.3f} | {s['mad']:.3f} | {ci_str} |"
                )

    lines += [
        "",
        "## 3. Offsets Breakdown by Adsorbate",
        "",
        "| Adsorbate | Count | RPA_total – RPBE Mean (eV) | RPA_total – PBE Mean (eV) | PBE – RPBE Mean (eV) | 95% CI (RPA_tot–RPBE) |",
        "|---|---|---|---|---|---|",
    ]

    for ads in sorted(summary["by_adsorbate"].keys()):
        ads_sum = summary["by_adsorbate"][ads]
        rpa_rpbe = ads_sum["RPA_total_minus_RPBE"]
        rpa_pbe = ads_sum["RPA_total_minus_PBE"]
        pbe_rpbe = ads_sum["PBE_minus_RPBE"]
        ci_str = f"[{rpa_rpbe['ci_95_low']:+.3f}, {rpa_rpbe['ci_95_high']:+.3f}]"
        lines.append(
            f"| `{ads}` | {rpa_rpbe['count']} | {rpa_rpbe['mean']:+.3f} | {rpa_pbe['mean']:+.3f} | "
            f"{pbe_rpbe['mean']:+.3f} | {ci_str} |"
        )

    lines += [
        "",
        "## 4. Offsets Breakdown by Surface Element",
        "",
        "| Surface Element | Magnetic | Count | RPA_total – RPBE Mean (eV) | RPA_total – PBE Mean (eV) | PBE – RPBE Mean (eV) |",
        "|---|---|---|---|---|---|",
    ]

    for surf in sorted(summary["by_surface_element"].keys()):
        surf_sum = summary["by_surface_element"][surf]
        is_mag = "Yes" if surf in MAGNETIC_ELEMENTS else "No"
        rpa_rpbe = surf_sum["RPA_total_minus_RPBE"]
        rpa_pbe = surf_sum["RPA_total_minus_PBE"]
        pbe_rpbe = surf_sum["PBE_minus_RPBE"]
        lines.append(
            f"| `{surf}` | {is_mag} | {rpa_rpbe['count']} | {rpa_rpbe['mean']:+.3f} | {rpa_pbe['mean']:+.3f} | "
            f"{pbe_rpbe['mean']:+.3f} |"
        )

    lines += [
        "",
        "---",
        "",
        "*End of DTU functional offsets analysis report.*",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adsorption-db",
        type=Path,
        default=Path("external_raw/dtu_rpa/adsorption.db"),
        help="Path to DTU adsorption.db",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dtu_rpa"),
        help="Directory to output CSV, JSON, and MD reports",
    )
    args = parser.parse_args()

    if not args.adsorption_db.exists():
        raise SystemExit(f"Missing DTU database: {args.adsorption_db}")

    print(f"Loading DTU adsorption database from {args.adsorption_db}...")
    db_sha256 = compute_file_sha256(args.adsorption_db)
    df = load_and_extract_offsets(args.adsorption_db)
    print(f"Loaded {len(df)} paired rows with complete target functional values.")

    print("Computing summary statistics and bootstrap confidence intervals...")
    summary = summarize_dataset(df)

    csv_path = args.output_dir / "functional_offsets.csv"
    json_path = args.output_dir / "functional_offsets.json"
    md_path = args.output_dir / "functional_offsets.md"

    print(f"Writing CSV output to {csv_path}...")
    export_csv(df, csv_path)

    print(f"Writing JSON output to {json_path}...")
    export_json(summary, db_sha256, args.adsorption_db, json_path, row_count=len(df))

    print(f"Writing Markdown output to {md_path}...")
    export_markdown(summary, db_sha256, args.adsorption_db, md_path, row_count=len(df))

    print("\nSummary of DTU Benchmark Offsets (Overall 200 Rows):")
    for target in TARGET_OFFSETS:
        s = summary["overall"][target]
        print(
            f"  {OFFSET_LABELS[target]:20s}: mean={s['mean']:+.3f} eV, std={s['std']:.3f} eV, "
            f"95% CI=[{s['ci_95_low']:+.3f}, {s['ci_95_high']:+.3f}]"
        )

    print(f"\n{DISCLAIMER_NOTE}")
    print(f"{MISSING_SPECIES_NOTE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

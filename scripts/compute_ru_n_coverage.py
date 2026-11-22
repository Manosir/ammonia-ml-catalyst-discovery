#!/usr/bin/env python3
"""Task G1: Correct Ru coverage using actual tag 1 (surface layer) atoms from trajectory frames.

For the 16 pure-Ru N rows, read the trajectory ASE database/file if available,
or compute surface atoms accurately. Uses tag=1 atoms from the first frame of
each trajectory to compute surface coverage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lzma
import ase.io


def get_tag1_count(source_path: str):
    """Read first frame of trajectory and return tag=1 atom count."""
    try:
        with lzma.open(source_path, 'rt') as f:
            atoms = ase.io.read(f, format='extxyz', index=0)
            tags = atoms.get_tags()
            return sum(1 for t in tags if t == 1)
    except Exception as e:
        return None, str(e)


def main():
    path = "results/adsorbates/oc20_adsorbates_features.parquet"
    if not Path(path).exists():
        print(f"Error: {path} not found")
        return

    df = pd.read_parquet(path)
    ru_n = df[(df["bulk_chemsys"] == "Ru") & (df["adsorbate"] == "N")].copy()
    ru_n = ru_n.sort_values("adsorption_energy_eV")

    print(f"Total pure Ru N rows: {len(ru_n)}")

    results = []
    for _, r in ru_n.iterrows():
        system_id = r.get("system_id", "N/A")
        facet = str(r.get("miller_index", "N/A")).replace("[", "").replace("]", "").replace("\n", " ")
        total_slab_atoms = int(r.get("n_surface_atoms", 36))
        n_ads = int(r.get("n_adsorbate_N", 1))
        label = str(r.get("anomaly_label", "none"))
        e_ads = float(r.get("adsorption_energy_eV", 0.0))
        source_path = r.get("source_path", "")

        # Get surface atoms from tag-1 in first trajectory frame
        tag1_result = get_tag1_count(source_path)
        if isinstance(tag1_result, tuple):
            tag1_count = None
            status = f"FAILED: {tag1_result[1]}"
        else:
            tag1_count = tag1_result
            status = "OK"

        if tag1_count is not None and tag1_count > 0:
            surface_atoms = tag1_count
            coverage = n_ads / surface_atoms
        else:
            surface_atoms = np.nan
            coverage = np.nan

        results.append({
            "system_id": system_id,
            "facet": facet,
            "supercell": total_slab_atoms,
            "surface_atoms": surface_atoms,
            "coverage_ML": coverage,
            "anomaly_label": label,
            "E_ads_eV": e_ads,
            "status": status
        })

    res_df = pd.DataFrame(results)

    # Save to results/m3/ru_n_coverage.csv
    out_path = Path("results/m3/ru_n_coverage.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved Ru N coverage table to {out_path}")

    print("\n" + "=" * 120)
    print("TASK G1: Ru N COVERAGE TABLE (surface atoms = tag-1 from first trajectory frame)")
    print("=" * 120)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(res_df.to_string(index=False))
    print("=" * 120)

    # Check how many rows are >= 0.2 ML
    ge_02 = res_df[res_df["coverage_ML"] >= 0.2]
    print(f"\nNumber of rows with coverage >= 0.2 ML: {len(ge_02)}")

    # Summary
    failed = res_df[res_df["status"] != "OK"]
    if len(failed) > 0:
        print(f"\nWARNING: {len(failed)} rows failed to read trajectory:")
        for _, f in failed.iterrows():
            print(f"  {f['system_id']}: {f['status']}")


if __name__ == "__main__":
    main()
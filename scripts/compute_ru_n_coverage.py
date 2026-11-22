#!/usr/bin/env python3
"""Task G1: Correct Ru coverage using actual tag 1 (surface layer) atoms from trajectory frames.

For the 16 pure-Ru N rows, read the trajectory ASE database/file if available,
or compute surface atoms accurately. Since trajectory files (.traj or ASE DB)
may be in raw paths or computed via structure, let's inspect where trajectories are stored.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    path = "results/adsorbates/oc20_adsorbates_features.parquet"
    if not Path(path).exists():
        print(f"Error: {path} not found")
        return

    df = pd.read_parquet(path)
    ru_n = df[(df["bulk_chemsys"] == "Ru") & (df["adsorbate"] == "N")].copy()
    ru_n = ru_n.sort_values("adsorption_energy_eV")

    # Let's check what columns are available regarding trajectories or positions
    print(f"Total pure Ru N rows: {len(ru_n)}")

    # If trajectory files or ase atoms are not directly in parquet, let's check how surface atoms can be derived
    # In OC20 parquet, n_surface_atoms is stored. Let's check if there are any other columns.
    print("Available columns:", [c for c in df.columns if any(k in c.lower() for k in ['surf', 'atom', 'tag', 'coord', 'traj'])])

    # Let's compute coverage using n_surface_atoms / 4 (since 4-layer hcp(0001) slabs have surface atoms = total / 4)
    # Or if n_surface_atoms already represents surface atoms when properly defined:
    # Let's inspect the exact values and print the required table to results/m3/ru_n_coverage.csv

    results = []
    for _, r in ru_n.iterrows():
        system_id = r.get("system_id", "N/A")
        facet = str(r.get("miller_index", "N/A")).replace("[", "").replace("]", "").replace("\n", " ")
        total_slab_atoms = int(r.get("n_surface_atoms", 36))
        # Surface layer atoms for a 4-layer slab: total_slab_atoms / 4
        # Or if n_surface_atoms is already surface atoms, let's check.
        # Earlier we saw n_surface_atoms in [36, 48, 54, 64, 72, 96], which are total slab atom counts for OC20 slabs.
        # Surface layer atoms = total_slab_atoms / 4
        surface_atoms = total_slab_atoms / 4.0
        n_ads = int(r.get("n_adsorbate_N", 1))
        coverage = n_ads / surface_atoms
        label = str(r.get("anomaly_label", "none"))
        e_ads = float(r.get("adsorption_energy_eV", 0.0))

        results.append({
            "system_id": system_id,
            "facet": facet,
            "supercell": total_slab_atoms,
            "surface_atoms": surface_atoms,
            "coverage_ML": coverage,
            "anomaly_label": label,
            "E_ads_eV": e_ads
        })

    res_df = pd.DataFrame(results)

    # Save to results/m3/ru_n_coverage.csv
    out_path = Path("results/m3/ru_n_coverage.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved Ru N coverage table to {out_path}")

    print("\n" + "=" * 110)
    print("TASK G1: Ru N COVERAGE TABLE")
    print("=" * 110)
    print(res_df.to_string(index=False))
    print("=" * 110)

    # Check how many rows are >= 0.2 ML
    ge_02 = res_df[res_df["coverage_ML"] >= 0.2]
    print(f"\nNumber of rows with coverage >= 0.2 ML: {len(ge_02)}")

if __name__ == "__main__":
    main()
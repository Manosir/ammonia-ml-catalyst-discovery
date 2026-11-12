#!/usr/bin/env python3
"""Single source of truth for screening-feature availability."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

COLS = ["name", "role", "source", "available_at_screening", "extra_step_needed", "implemented", "leakage_risk", "note"]
F = [
("gcn_initial","feature","initial slab coordinates + adsorption-site neighbors","yes","site enumeration","no","low","define cutoff and validate on initial frame"),
("ads_height_initial","feature","initial frame, MIC displacement vector z-component","yes","site enumeration","no","low","report variance by adsorbate"),
("site_ensemble","feature","initial site neighbors","yes","site enumeration","no","low","mono-metallic / bimetallic / minority single atom"),
("miller_index","feature","OC20 mapping","yes","parse tuple","no","medium","retain tuple or expanded components"),
("bulk_structure_type","feature","bulk structure + spglib","yes","spglib analysis","no","low","fcc / bcc / hcp / other"),
("material_class","feature","bulk composition","yes","none","no","low","unary / binary / ternary"),
("nonmetal_flag","feature","bulk composition","yes","none","no","low","S, Se, As, etc."),
("metallic_radius_site_mean","feature","tabulated property + site-neighbor aggregation","yes","site enumeration","no","low","aggregation is explicit in name"),
("electronegativity_site_mean","feature","tabulated property + site-neighbor aggregation","yes","site enumeration","no","low","handle nonmetal-containing rows explicitly"),
("d_electrons_site_mean","feature","tabulated property + site-neighbor aggregation","yes","site enumeration","no","low","handle nonmetal-containing rows explicitly"),
("primary_metal_frac","feature","initial composition","yes","define primary element","no","low","most abundant or mapping-defined; document choice"),
("tm_frac","feature","initial composition","yes","none","no","low",""),
("n_distinct_TM","feature","initial composition","yes","none","no","low",""),
("facet_roughness","feature","initial slab coordinates","yes","define metric","no","medium",""),
("d_band_proxy_tabulated","experimental","tabulated composition-averaged proxy","yes","calibrate and compare baselines","no","low","not a true site-projected d-band"),
("shift","identity_key","OC20 mapping","yes","none","yes","medium","do not use as raw numeric feature without grouped splits"),
("n_adsorbate_N","redundant","adsorbate label","yes","none","yes","low","constant within per-adsorbate model"),
("n_adsorbate_H","redundant","adsorbate label","yes","none","yes","low","constant within per-adsorbate model"),
("adsorbate_atom_count","redundant","adsorbate label","yes","none","yes","low","constant within per-adsorbate model"),
("n_surface_atoms","review","initial structure","yes","verify tag semantics","yes","low","rename if it means full slab minus adsorbate"),
("d_band_center_slab_dft","excluded","clean-slab DFT","no","DFT per candidate","no","high","not screening-available"),
("d_band_center_site_dft","excluded","site-projected DOS, usually relaxed","no","DFT per candidate","no","high","not screening-available"),
("ads_height_final","excluded","relaxed frame","no","relaxation","yes","high","audit only"),
("mean_ads_force","excluded","relaxed frame forces","no","relaxation","yes","high","bounded by convergence criterion"),
("energy_initial_eV","excluded","initial-frame DFT","no","DFT per candidate","yes","high","target-side audit"),
("forces_initial","excluded","initial-frame DFT forces","no","DFT per candidate","no","high","not available for a new candidate"),
("energy_final_eV","excluded","relaxed DFT","no","relaxation","yes","high","audit only"),
("relaxation_energy_change_eV","excluded","target-derived","no","relaxation","yes","high","not a screening feature"),
("reference_energy","excluded","OC20 reference / target construction","no","DFT reference","yes","high","target construction only"),
("anomaly","label_filter","OC20 mapping relaxation outcome","no","relaxation","yes","high","policy-controlled QC label"),
("official_adsorption_energy_eV","target","energy_final - reference_energy","no","relaxation","yes","n/a","primary supervised target"),
("system_id","identifier","OC20","no","none","yes","n/a","never a feature"),
("bulk_id","group_key","OC20 mapping","no","none","yes","n/a","grouped split key"),
("ads_id","identifier","OC20 mapping","no","none","yes","n/a","never a feature"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/adsorbates/feature_availability.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(F, columns=COLS).to_csv(args.output, index=False)
    print(f"Wrote {len(F)} feature registry rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

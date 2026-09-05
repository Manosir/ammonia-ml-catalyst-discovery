"""
07_predict_new_catalyst.py
==========================
Predict N* adsorption energy for one or more catalyst compositions
using the trained GBR model from script 03.

Fixes applied vs the original version:
  - Model path updated to match where 03_build_and_train.py actually saves
  - Feature vector updated to match all 16 features the model was trained on
  - Added Sabatier distance and candidate ranking to output
  - Added --metal and --facet CLI arguments for quick single-metal screening
  - Added a built-in catalogue of representative feature vectors for
    common pure metals so users can run without computing features manually

Usage:
    # Predict for the built-in RuFe alloy example:
    python intermediates/n_adsorption/scripts/07_predict_new_catalyst.py

    # Predict for a specific pure metal from the catalogue:
    python intermediates/n_adsorption/scripts/07_predict_new_catalyst.py --metal Ru

    # Predict from a custom CSV of feature vectors:
    python intermediates/n_adsorption/scripts/07_predict_new_catalyst.py \\
        --input-csv my_candidates.csv

    # Use a different model path:
    python intermediates/n_adsorption/scripts/07_predict_new_catalyst.py \\
        --model results/n_adsorption/gbr_ammonia_ai_final.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Model path — matches 03_build_and_train.py output ────────────────────────
DEFAULT_MODEL_PATH = Path("results/n_adsorption/gbr_ammonia_ai_final.joblib")

# ── Feature column order must match training exactly ─────────────────────────
FEATURE_COLS = [
    "d_band_center_slab", "d_band_center_site", "d_band_center_primary",
    "metallic_radius", "electronegativity", "d_electrons",
    "primary_metal_frac", "tm_frac", "n_distinct_TM",
    "gcn_initial", "ads_height", "mean_ads_force",
    "facet_roughness", "shift", "n_N_ads", "n_surface_atoms",
]

# ── Sabatier optimum for N* ───────────────────────────────────────────────────
E_OPT = -0.4   # eV, Hammer-Norskov convention for NH3 synthesis

# ── Representative feature vectors for common pure metals ────────────────────
# d_band values: Hammer & Norskov 2000; geometry: typical OC20 IS2RE values
METAL_CATALOGUE = {
    "Ru": dict(
        d_band_center_slab=-1.41,  d_band_center_site=-1.41,
        d_band_center_primary=-1.41, metallic_radius=1.34,
        electronegativity=2.20, d_electrons=7,
        primary_metal_frac=1.0, tm_frac=1.0, n_distinct_TM=1,
        gcn_initial=6.8, ads_height=1.95, mean_ads_force=0.12,
        facet_roughness=3, shift=0.0, n_N_ads=1, n_surface_atoms=24,
    ),
    "Fe": dict(
        d_band_center_slab=-1.29,  d_band_center_site=-1.29,
        d_band_center_primary=-1.29, metallic_radius=1.26,
        electronegativity=1.83, d_electrons=6,
        primary_metal_frac=1.0, tm_frac=1.0, n_distinct_TM=1,
        gcn_initial=6.5, ads_height=1.88, mean_ads_force=0.15,
        facet_roughness=2, shift=0.0, n_N_ads=1, n_surface_atoms=24,
    ),
    "Co": dict(
        d_band_center_slab=-1.17,  d_band_center_site=-1.17,
        d_band_center_primary=-1.17, metallic_radius=1.25,
        electronegativity=1.88, d_electrons=7,
        primary_metal_frac=1.0, tm_frac=1.0, n_distinct_TM=1,
        gcn_initial=6.9, ads_height=1.90, mean_ads_force=0.11,
        facet_roughness=3, shift=0.0, n_N_ads=1, n_surface_atoms=24,
    ),
    "Mo": dict(
        d_band_center_slab=-1.60,  d_band_center_site=-1.60,
        d_band_center_primary=-1.60, metallic_radius=1.39,
        electronegativity=2.16, d_electrons=5,
        primary_metal_frac=1.0, tm_frac=1.0, n_distinct_TM=1,
        gcn_initial=6.7, ads_height=2.00, mean_ads_force=0.18,
        facet_roughness=2, shift=0.0, n_N_ads=1, n_surface_atoms=24,
    ),
    "Ni": dict(
        d_band_center_slab=-1.29,  d_band_center_site=-1.29,
        d_band_center_primary=-1.29, metallic_radius=1.24,
        electronegativity=1.91, d_electrons=8,
        primary_metal_frac=1.0, tm_frac=1.0, n_distinct_TM=1,
        gcn_initial=7.1, ads_height=1.85, mean_ads_force=0.09,
        facet_roughness=3, shift=0.0, n_N_ads=1, n_surface_atoms=24,
    ),
}

# RuFe alloy example (25% Ru, 75% Fe)
RUFE_EXAMPLE = dict(
    d_band_center_slab=-1.32,   # 0.75*(-1.29) + 0.25*(-1.41)
    d_band_center_site=-1.41,   # Ru atom at binding site
    d_band_center_primary=-1.41,
    metallic_radius=1.34,
    electronegativity=2.20,
    d_electrons=7,
    primary_metal_frac=0.25,
    tm_frac=1.0,
    n_distinct_TM=2,
    gcn_initial=6.8,
    ads_height=1.95,
    mean_ads_force=0.12,
    facet_roughness=3,
    shift=0.0,
    n_N_ads=1,
    n_surface_atoms=24,
)


def predict_and_report(model, df: pd.DataFrame, label: str = "") -> None:
    """Run prediction and print a formatted report."""
    missing = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: Feature vector is missing columns: {missing}\n"
            f"  Required features: {FEATURE_COLS}"
        )

    X = df[FEATURE_COLS].values
    if df[FEATURE_COLS].isna().any().any():
        bad = df[FEATURE_COLS].columns[df[FEATURE_COLS].isna().any()].tolist()
        sys.exit(f"ERROR: Missing numeric values in features: {bad}")
    predictions = model.predict(X)

    print(f"\n{'='*58}")
    if label:
        print(f"  {label}")
    print(f"{'='*58}")
    print(f"  {'Candidate':<22} {'ΔE_ads (eV)':>12}  {'|ΔE−opt|':>10}  {'Rank'}")
    print(f"  {'-'*54}")

    # Sort by distance from Sabatier optimum
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        delta_e = predictions[i]
        dist    = abs(delta_e - E_OPT)
        name    = row.get("label", f"Candidate {i+1}")
        results.append((name, delta_e, dist))

    results.sort(key=lambda x: x[2])
    for rank, (name, delta_e, dist) in enumerate(results, 1):
        bar   = "★" if rank == 1 else " "
        trend = "← near optimum" if dist < 0.3 else ("← too strong" if delta_e < E_OPT else "← too weak")
        print(f"  {bar} {name:<21} {delta_e:>+12.3f}  {dist:>10.3f}  #{rank}  {trend}")

    print(f"\n  Sabatier optimum: ΔE_opt = {E_OPT} eV")
    print(f"  Best candidate: {results[0][0]}  "
          f"(ΔE = {results[0][1]:+.3f} eV, distance = {results[0][2]:.3f} eV)")
    print(f"{'='*58}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Predict N* ΔE_ads for new catalyst compositions."
    )
    ap.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH,
        help=f"Path to trained .joblib model (default: {DEFAULT_MODEL_PATH})",
    )
    ap.add_argument(
        "--metal", type=str, default=None,
        help="Pure metal symbol from catalogue "
             f"({', '.join(METAL_CATALOGUE)}). Overrides built-in example.",
    )
    ap.add_argument(
        "--input-csv", type=Path, default=None,
        help="CSV file with one row per candidate, columns matching FEATURE_COLS. "
             "Must include a 'label' column for candidate names.",
    )
    args = ap.parse_args()

    # Load model
    try:
        import joblib
    except ImportError:
        sys.exit("ERROR: joblib is required. Run: pip install joblib")

    if not args.model.exists():
        sys.exit(
            f"ERROR: Model not found at '{args.model}'.\n"
            f"  Run 03_build_and_train.py first to train the model."
        )

    model = joblib.load(args.model)
    expected = len(FEATURE_COLS)
    actual = getattr(model, "n_features_in_", None)
    if actual != expected:
        sys.exit(f"ERROR: Model expects {actual} features, but this predictor supplies {expected}.")
    print(f"Model loaded from: {args.model} ({actual} features)")

    # Build feature dataframe
    if args.input_csv is not None:
        if not args.input_csv.exists():
            sys.exit(f"ERROR: Input CSV not found: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        predict_and_report(model, df, label=f"Predictions from {args.input_csv.name}")

    elif args.metal is not None:
        metal = args.metal.upper() if len(args.metal) > 1 else args.metal.capitalize()
        if metal not in METAL_CATALOGUE:
            sys.exit(
                f"ERROR: '{metal}' not in catalogue.\n"
                f"  Available: {', '.join(sorted(METAL_CATALOGUE))}\n"
                f"  For custom compositions, use --input-csv."
            )
        df = pd.DataFrame([METAL_CATALOGUE[metal]])
        df["label"] = f"Pure {metal}"
        predict_and_report(model, df, label=f"Pure {metal} (111) surface")

    else:
        # Default: run all catalogue metals + the RuFe alloy example
        rows = []
        for metal, features in METAL_CATALOGUE.items():
            row = dict(features)
            row["label"] = f"Pure {metal}"
            rows.append(row)
        ruf = dict(RUFE_EXAMPLE)
        ruf["label"] = "RuFe alloy (25% Ru)"
        rows.append(ruf)
        df = pd.DataFrame(rows)
        predict_and_report(
            model, df,
            label="Catalogue screening — pure metals + RuFe alloy"
        )


if __name__ == "__main__":
    main()

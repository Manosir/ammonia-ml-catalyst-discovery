"""
13_gcn_leakage_test.py
=======================
Definitive leakage test for the GCN feature.

gemini_01_gcn_utils.py uses `read(extxyz_path, index=-1)` — the FINAL
relaxed frame. This means GCN is computed AFTER adsorption has already
occurred and the geometry has converged to the minimum. The final-frame
geometry partially encodes the adsorption energy (stronger binding →
shorter bond → different neighbour geometry → different GCN).

This script:
  1. Re-computes GCN identically but from the INITIAL frame (index=0),
     which represents what you would know about a NEW, untested catalyst
     before running any DFT — the operationally meaningful quantity.
  2. Runs the same 5-fold CV model comparison as 12_add_gcn_feature.py
     with both initial-frame and final-frame GCN.
  3. Quantifies leakage as: (final-frame improvement) − (initial-frame improvement).

Interpretation:
  - If initial-frame GCN gives similar improvement → GCN is genuinely
    informative structural information. Add it to the model.
  - If initial-frame GCN improvement drops to near zero → the final-frame
    improvement was mostly leakage. Do NOT add final-frame GCN.
  - If initial-frame GCN is somewhere in between → partial leakage.
    Report both values and note the leakage component explicitly.

Usage:
    python scripts2/13_gcn_leakage_test.py --data-dir data/is2re_N/77

This will take ~90 min for all 12,123 files (same as gemini_08).
Use --max-systems 500 for a quick diagnostic on a subset.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── GCN calculation (identical to gemini_01_gcn_utils, but frame-index aware) ──

def calculate_gcn(extxyz_path: Path, frame_index: int = 0, cn_max: int = 12) -> float:
    """
    Computes GCN for the N binding site in an extxyz trajectory.

    frame_index=0  → initial (unrelaxed) structure — what you know before DFT
    frame_index=-1 → final (relaxed) structure   — what gemini_08 computed
    """
    try:
        from ase.io import read
        from ase.neighborlist import neighbor_list, natural_cutoffs
        atoms = read(str(extxyz_path), index=frame_index)
    except Exception:
        return np.nan

    cutoffs = natural_cutoffs(atoms, mult=1.2)
    i_idx, j_idx = neighbor_list("ij", atoms, cutoffs)

    n_indices = [i for i, a in enumerate(atoms) if a.symbol == "N"]
    if not n_indices:
        return np.nan
    n_idx = n_indices[0]

    site_metals = j_idx[i_idx == n_idx]
    if len(site_metals) == 0:
        return np.nan

    gcn_sum = 0.0
    for metal_idx in site_metals:
        metal_nbrs = j_idx[i_idx == metal_idx]
        metal_gcn = 0.0
        for nbr_idx in metal_nbrs:
            if atoms[nbr_idx].symbol != "N":
                nbr_of_nbr = j_idx[i_idx == nbr_idx]
                cn_j = sum(1 for idx in nbr_of_nbr if atoms[idx].symbol != "N")
                metal_gcn += cn_j / cn_max
        gcn_sum += metal_gcn

    return gcn_sum / len(site_metals)


FEATURE_COLS_BASE = [
    "d_band_center_weighted", "d_band_center_primary",
    "metallic_radius", "electronegativity", "d_electrons",
    "n_N_ads", "n_surface_atoms", "n_distinct_TM",
    "primary_metal_frac", "tm_frac", "ads_height", "mean_ads_force",
]
TARGET_COL = "delta_E_ads"


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )),
    ])


def evaluate(X, y, label: str) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_model()
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    cv  = cross_val_score(build_model(), X, y, cv=5, scoring="r2", n_jobs=-1)
    print(f"  {label}")
    print(f"    MAE   : {mae:.3f} eV")
    print(f"    R²    : {r2:.3f}")
    print(f"    CV R² : {cv.mean():.3f} ± {cv.std():.3f}")
    return {"label": label, "mae": mae, "r2": r2,
            "cv_mean": cv.mean(), "cv_std": cv.std()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/is2re_N/77"),
                    help="Directory containing *.extxyz.xz trajectory files")
    ap.add_argument("--delta-csv", type=Path, default=None)
    ap.add_argument("--final-gcn-csv", type=Path,
                    default=Path("results/gcn_results.csv"),
                    help="Pre-computed final-frame GCN from gemini_08 (reused, not recomputed)")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--max-systems", type=int, default=None,
                    help="Process only the first N systems (for quick tests)")
    args = ap.parse_args()

    # Auto-detect delta CSV
    if args.delta_csv is None:
        for c in [Path("results/n_ads_delta_e.csv"),
                  Path("results/is2re/is2re_n_delta_e.csv")]:
            if c.exists():
                args.delta_csv = c; break
    if args.delta_csv is None:
        raise SystemExit("Specify --delta-csv")

    print("Loading feature dataset ...")
    df = pd.read_csv(args.delta_csv)
    if "system_id" in df.columns:
        df = df.rename(columns={"system_id": "sid"})
    df["sid"] = df["sid"].astype(str)
    print(f"  {len(df)} records")

    print("Loading pre-computed final-frame GCN ...")
    gcn_final = pd.read_csv(args.final_gcn_csv)
    gcn_final["sid"] = gcn_final["sid"].str.replace(r"\.extxyz$", "", regex=True)
    gcn_final = gcn_final.rename(columns={"site_gcn": "gcn_final"})
    print(f"  {len(gcn_final)} records")

    # ── Compute initial-frame GCN ─────────────────────────────────────────
    print("\nComputing initial-frame GCN (index=0) ...")
    xyz_files = sorted(args.data_dir.rglob("*.extxyz.xz"))
    if args.max_systems:
        xyz_files = xyz_files[:args.max_systems]
    print(f"  Processing {len(xyz_files)} trajectory files ...")

    records_init = []
    for i, xyz in enumerate(xyz_files, 1):
        sid = xyz.stem.replace(".extxyz", "")
        gcn_val = calculate_gcn(xyz, frame_index=0)
        records_init.append({"sid": sid, "gcn_initial": gcn_val})
        if i % 500 == 0 or i == len(xyz_files):
            print(f"  [{i}/{len(xyz_files)}] done")

    gcn_init = pd.DataFrame(records_init)
    init_csv = args.out_dir / "gcn_initial_frame.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gcn_init.to_csv(init_csv, index=False)
    print(f"  Saved → {init_csv}")

    # ── Merge everything ──────────────────────────────────────────────────
    merged = df.merge(gcn_final[["sid","gcn_final"]], on="sid", how="inner")
    merged = merged.merge(gcn_init[["sid","gcn_initial"]], on="sid", how="inner")
    merged = merged.dropna(subset=FEATURE_COLS_BASE + [TARGET_COL])
    print(f"\nMerged dataset: {len(merged)} records with all features + both GCNs")

    if len(merged) < 30:
        raise SystemExit("Too few records after merge.")

    # ── GCN comparison: initial vs final ─────────────────────────────────
    print("\n" + "="*64)
    print("GCN FRAME COMPARISON")
    print("="*64)
    valid = merged[["gcn_initial","gcn_final",TARGET_COL]].dropna()
    print(f"  initial-frame GCN: mean={valid['gcn_initial'].mean():.3f}  "
         f"std={valid['gcn_initial'].std():.3f}  "
         f"range=[{valid['gcn_initial'].min():.2f},{valid['gcn_initial'].max():.2f}]")
    print(f"  final-frame GCN:   mean={valid['gcn_final'].mean():.3f}  "
         f"std={valid['gcn_final'].std():.3f}  "
         f"range=[{valid['gcn_final'].min():.2f},{valid['gcn_final'].max():.2f}]")
    gcn_shift = (valid["gcn_final"] - valid["gcn_initial"])
    print(f"  GCN shift (final−initial): mean={gcn_shift.mean():.3f}  "
         f"std={gcn_shift.std():.3f}  max_abs={gcn_shift.abs().max():.3f}")
    print(f"  r(initial_GCN, ΔE_ads) = {valid['gcn_initial'].corr(valid[TARGET_COL]):.4f}")
    print(f"  r(final_GCN,   ΔE_ads) = {valid['gcn_final'].corr(valid[TARGET_COL]):.4f}")
    corr_diff = abs(valid['gcn_final'].corr(valid[TARGET_COL])) - \
                abs(valid['gcn_initial'].corr(valid[TARGET_COL]))
    print(f"  Correlation gain from using final vs initial: {corr_diff:+.4f}")
    if corr_diff > 0.05:
        print("  [WARN] Final frame correlates noticeably more with ΔE_ads.")
        print("         This gap is the leakage contribution.")

    # ── Model comparison ──────────────────────────────────────────────────
    print("\n" + "="*64)
    print("MODEL COMPARISON (5-fold CV)")
    print("="*64)

    avail = [f for f in FEATURE_COLS_BASE if f in merged.columns]
    df_m  = merged[avail + ["gcn_initial","gcn_final",TARGET_COL]].dropna()
    y     = df_m[TARGET_COL].values

    print()
    r_base   = evaluate(df_m[avail].values, y, "Baseline (no GCN)")
    r_init   = evaluate(df_m[avail + ["gcn_initial"]].values, y, "Initial-frame GCN")
    r_final  = evaluate(df_m[avail + ["gcn_final"]].values, y, "Final-frame GCN")

    # ── Leakage quantification ────────────────────────────────────────────
    print("\n" + "="*64)
    print("LEAKAGE QUANTIFICATION")
    print("="*64)
    gain_final = r_final["cv_mean"] - r_base["cv_mean"]
    gain_init  = r_init["cv_mean"]  - r_base["cv_mean"]
    leakage    = gain_final - gain_init

    print(f"  CV R² gain — final-frame GCN   : {gain_final:+.4f}")
    print(f"  CV R² gain — initial-frame GCN : {gain_init:+.4f}")
    print(f"  Leakage component              : {leakage:+.4f}")
    print(f"  ({100*leakage/max(abs(gain_final),1e-9):.0f}% of final-frame gain is leakage)")

    print("\n" + "="*64)
    print("VERDICT")
    print("="*64)
    if gain_init > 0.02 and gain_init > 0.5 * gain_final:
        print("  USE initial-frame GCN in production.")
        print(f"  Real signal: CV R² +{gain_init:.3f} (leakage-free).")
        print(f"  Leakage adds {leakage:.3f} on top — do NOT use final-frame GCN")
        print("  for predictions on new catalysts.")
    elif gain_init < 0.01:
        print("  DO NOT USE GCN as a model feature.")
        print(f"  Initial-frame GCN gives CV R² gain of only {gain_init:.3f}.")
        print(f"  The {gain_final:.3f} gain from final-frame GCN is mostly leakage.")
        print("  Adding GCN would make the model appear better in training")
        print("  but fail on genuinely new catalyst structures.")
    else:
        print(f"  PARTIAL signal: initial-frame gives +{gain_init:.3f} CV R².")
        print(f"  Leakage component: {leakage:.3f}.")
        print("  If using GCN, use ONLY the initial-frame version and note the")
        print("  limitation in any publication or report.")

    # Save results table
    summary = pd.DataFrame([r_base, r_init, r_final])
    summary["gain_vs_baseline"] = summary["cv_mean"] - r_base["cv_mean"]
    print("\n  Full results:")
    print(summary[["label","mae","r2","cv_mean","cv_std","gain_vs_baseline"]].to_string(index=False))

    summary_path = args.out_dir / "gcn_leakage_test_results.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n  Results → {summary_path}")


if __name__ == "__main__":
    main()

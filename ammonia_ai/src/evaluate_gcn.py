"""
12_add_gcn_feature.py
======================
Evaluates whether the Generalised Coordination Number (GCN) computed by
gemini_08_save_gcn.py is a genuinely useful feature for the ΔE_ads model,
or whether it introduces leakage / noise without real signal.

The GCN claim (Calle-Vallejo, Nature Chemistry 2015):
  For PURE metal surfaces, GCN of the adsorption site linearly predicts
  ΔE_ads with ~0.1 eV MAE -- better than d-band theory alone for
  geometry-driven variance (terrace vs step vs kink).

The five questions we must answer BEFORE accepting GCN as a feature:
  Q1. Does GCN actually vary across our dataset? (range, std)
  Q2. Does it correlate with ΔE_ads unconditionally? (global Pearson r)
  Q3. Does it correlate within individual metals? (the real Calle-Vallejo claim)
  Q4. Is there final-frame leakage? (GCN from relaxed geometry encodes outcome)
  Q5. Does CV R² improve, not just test-set R²? (overfitting check)

Only if Q1-Q3 pass and Q5 confirms improvement do we accept GCN.
Q4 is flagged as a known limitation regardless.

Inputs:
  - results/gcn_results.csv      : sid, site_gcn  (from gemini_08_save_gcn.py)
  - results/is2re/is2re_n_delta_e.csv  : full feature + target dataset
    OR data/n_ads_delta_e.csv    : fallback to s2ef dataset if IS2RE not ready

Usage:
    python scripts2/12_add_gcn_feature.py
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

FEATURE_COLS_BASE = [
    "d_band_center_weighted",
    "d_band_center_primary",
    "metallic_radius",
    "electronegativity",
    "d_electrons",
    "n_N_ads",
    "n_surface_atoms",
    "n_distinct_TM",
    "primary_metal_frac",
    "tm_frac",
    "ads_height",
    "mean_ads_force",
]
TARGET_COL = "delta_E_ads"


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            min_samples_leaf=5, random_state=42,
        )),
    ])


def evaluate(pipe, X, y, label: str) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)
    cv   = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)
    print(f"  {label}")
    print(f"    MAE:    {mae:.3f} eV")
    print(f"    R²:     {r2:.3f}")
    print(f"    CV R²:  {cv.mean():.3f} ± {cv.std():.3f}")
    return {"label": label, "mae": mae, "r2": r2, "cv_mean": cv.mean(), "cv_std": cv.std()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcn-csv",   type=Path, default=Path("results/gcn_results.csv"))
    ap.add_argument("--delta-csv", type=Path, default=None,
                    help="Path to delta_E_ads CSV. Auto-detected if omitted.")
    ap.add_argument("--out-dir",   type=Path, default=Path("results"))
    args = ap.parse_args()

    # ── Auto-detect delta_E_ads CSV ───────────────────────────────────────
    if args.delta_csv is None:
        candidates = [
            Path("results/is2re/is2re_n_delta_e.csv"),
            Path("results/n_ads_delta_e.csv"),
            Path("data/n_ads_delta_e.csv"),
        ]
        for c in candidates:
            if c.exists():
                args.delta_csv = c
                break
        if args.delta_csv is None:
            raise SystemExit("No delta_E_ads CSV found. Specify --delta-csv.")

    print(f"Loading ΔE_ads dataset : {args.delta_csv}")
    df = pd.read_csv(args.delta_csv)
    print(f"  {len(df)} records, columns: {list(df.columns)}\n")

    # ── Load GCN results ──────────────────────────────────────────────────
    print(f"Loading GCN results    : {args.gcn_csv}")
    gcn = pd.read_csv(args.gcn_csv)
    print(f"  {len(gcn)} records")

    # Normalise sid column name (gemini script uses 'sid'; our pipeline uses
    # 'system_id' in IS2RE and 'sid' in s2ef)
    if "sid" in gcn.columns:
        gcn = gcn.rename(columns={"sid": "sid"})
    if "system_id" in df.columns:
        df = df.rename(columns={"system_id": "sid"})

    # Strip .extxyz suffix from GCN sid if present (stem of filename)
    gcn["sid"] = gcn["sid"].str.replace(r"\.extxyz$", "", regex=True)
    df["sid"]  = df["sid"].astype(str)

    # ── Merge ─────────────────────────────────────────────────────────────
    merged = df.merge(gcn[["sid", "site_gcn"]], on="sid", how="inner")
    print(f"\nAfter merging on sid: {len(merged)} records "
         f"({len(df) - len(merged)} dropped — no GCN match)")

    if len(merged) < 30:
        raise SystemExit(
            "Too few records after merge. Check that sid values match between "
            "the two CSVs. GCN file uses file stems; delta CSV uses system_id/sid."
        )

    # ── Q1: GCN variance ─────────────────────────────────────────────────
    print("\n" + "="*64)
    print("Q1: GCN distribution")
    print("="*64)
    g = merged["site_gcn"]
    print(f"  N      : {g.notna().sum()}")
    print(f"  range  : {g.min():.3f} – {g.max():.3f}")
    print(f"  mean   : {g.mean():.3f}")
    print(f"  std    : {g.std():.3f}")
    print(f"  NaN    : {g.isna().sum()}")

    if g.std() < 0.5:
        print("  [WARN] Low variance — GCN barely changes across the dataset.")
        print("         Adding it will likely not help the model.")
    elif g.std() < 1.0:
        print("  [NOTE] Moderate variance. Signal may be limited.")
    else:
        print("  [OK]   Sufficient variance for a potentially useful feature.")

    # ── Q2: Global correlation ────────────────────────────────────────────
    print("\n" + "="*64)
    print("Q2: Global correlation GCN ↔ ΔE_ads")
    print("="*64)
    valid = merged[["site_gcn", TARGET_COL]].dropna()
    r_global = valid["site_gcn"].corr(valid[TARGET_COL])
    print(f"  Pearson r(GCN, ΔE_ads) = {r_global:.4f}")
    if abs(r_global) < 0.10:
        print("  [WARN] Near-zero global correlation. GCN provides little signal.")
    elif abs(r_global) < 0.20:
        print("  [NOTE] Weak correlation. May still help within-metal if Q3 passes.")
    else:
        print("  [OK]   Meaningful global correlation.")

    # ── Q3: Within-metal correlation (the real test) ──────────────────────
    print("\n" + "="*64)
    print("Q3: Within-metal correlation GCN ↔ ΔE_ads")
    print("="*64)
    print("  (This is the Calle-Vallejo claim: GCN explains geometry-driven")
    print("   variance WITHIN the same metal/alloy family)")
    print()

    within_corrs = []
    for metal, grp in merged.groupby("primary_metal"):
        sub = grp[["site_gcn", TARGET_COL]].dropna()
        if len(sub) < 5:
            continue
        r = sub["site_gcn"].corr(sub[TARGET_COL])
        within_corrs.append((metal, len(sub), r))
        flag = ""
        if abs(r) < 0.10:
            flag = "  ← weak"
        elif abs(r) > 0.40:
            flag = "  ← strong"
        print(f"  {metal:4s}  n={len(sub):>3}  r={r:>7.4f}{flag}")

    if within_corrs:
        mean_abs_r = np.mean([abs(r) for _, _, r in within_corrs])
        print(f"\n  Mean |within-metal r| = {mean_abs_r:.4f}")
        if mean_abs_r < 0.15:
            print("  [WARN] Within-metal correlations are weak — Calle-Vallejo")
            print("         effect is NOT apparent in this dataset. GCN won't")
            print("         explain the within-metal ΔE_ads variance meaningfully.")
        elif mean_abs_r < 0.30:
            print("  [NOTE] Moderate within-metal correlations. GCN may help slightly.")
        else:
            print("  [OK]   Strong within-metal correlations — proceed with GCN.")

    # ── Q4: Leakage flag ──────────────────────────────────────────────────
    print("\n" + "="*64)
    print("Q4: Leakage flag (informational — cannot be resolved here)")
    print("="*64)
    print("  GCN was computed from the FINAL relaxed frame.")
    print("  Final-frame geometry encodes the adsorption outcome:")
    print("    stronger binding  → shorter bond → different neighbour distances")
    print("    → different GCN value compared to the initial unrelaxed structure.")
    print("  This creates partial target leakage: GCN is not fully independent")
    print("  of ΔE_ads. The model will appear to use GCN as a signal even if")
    print("  GCN provides zero causal information about a NEW, unseen catalyst.")
    print()
    print("  FOR A FAIR BENCHMARK: re-compute GCN on the INITIAL frame (frame 0)")
    print("  of each trajectory. If model performance drops substantially, the")
    print("  'improvement' from final-frame GCN was partially due to leakage.")
    print("  gemini_08_save_gcn.py uses rglob on all .extxyz.xz files and calls")
    print("  calculate_site_gcn -- check gemini_01_gcn_utils.py to confirm")
    print("  whether it reads the first or last frame.")

    # ── Q5: Model comparison with proper CV ───────────────────────────────
    print("\n" + "="*64)
    print("Q5: Model comparison — with vs without GCN (proper 5-fold CV)")
    print("="*64)

    # Check which base features are actually present
    available_base = [f for f in FEATURE_COLS_BASE if f in merged.columns]
    missing_base   = [f for f in FEATURE_COLS_BASE if f not in merged.columns]
    if missing_base:
        print(f"  [NOTE] Missing base features (will be skipped): {missing_base}")

    df_model = merged[available_base + ["site_gcn", TARGET_COL]].dropna()
    print(f"  Records with all features: {len(df_model)}")

    if len(df_model) < 30:
        raise SystemExit("Too few complete records for comparison. Check for NaN in GCN.")

    y = df_model[TARGET_COL].values

    print()
    results = []
    results.append(evaluate(build_model(),
                            df_model[available_base].values,
                            y, "Baseline (no GCN)"))
    results.append(evaluate(build_model(),
                            df_model[available_base + ["site_gcn"]].values,
                            y, "With GCN"))

    # ── Summary verdict ───────────────────────────────────────────────────
    print("\n" + "="*64)
    print("SUMMARY VERDICT")
    print("="*64)
    base = results[0]
    with_gcn = results[1]

    delta_cv  = with_gcn["cv_mean"] - base["cv_mean"]
    delta_mae = with_gcn["mae"] - base["mae"]

    print(f"  CV R² change    : {delta_cv:+.3f} (positive = better)")
    print(f"  MAE change      : {delta_mae:+.3f} eV (negative = better)")
    print()

    if delta_cv > 0.02 and delta_mae < -0.05:
        verdict = "ADD GCN — both CV R² and MAE improve meaningfully."
    elif delta_cv > 0.01:
        verdict = "MARGINAL — small CV improvement. GCN helps slightly but check Q4 leakage."
    elif delta_cv < -0.01:
        verdict = "DO NOT ADD — GCN hurts CV performance. It is noise for this dataset."
    else:
        verdict = "NEUTRAL — GCN has no meaningful effect. Not worth the added complexity."

    print(f"  Verdict: {verdict}")
    print()
    if abs(delta_cv) < 0.03:
        print("  NOTE: Given small dataset size, a ±0.03 change in CV R² is within")
        print("  random variation from train/test splitting. Re-run with --chunks -1")
        print("  (full IS2RE dataset) for a statistically meaningful comparison.")

    # ── Save augmented dataset ────────────────────────────────────────────
    out_path = args.out_dir / "n_ads_with_gcn.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\n  Augmented dataset saved → {out_path}")
    print(f"  (Contains all original features + site_gcn for {len(merged)} records)")


if __name__ == "__main__":
    main()

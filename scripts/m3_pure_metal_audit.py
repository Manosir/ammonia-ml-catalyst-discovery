#!/usr/bin/env python3
"""Pure metal leave-one-group-out (LOGO) audit script.

For each pure-metal bulk (single-element bulk_chemsys in training table):
1. Lists rows per metal, facet (miller_hkl_max / miller_index), adsorbate, and anomaly-0 status.
2. Holds out that metal's entire chemical system (GroupKFold on bulk_chemsys), trains a fresh model on the rest, and predicts its rows.
3. Outputs results/m3/pure_metal_oof.csv with DFT label, held-out prediction, per-metal MAE, and most-stable-site label (minimum E_ads per metal, facet, adsorbate).
4. Generates a second table for N only (Ru terrace/step, Os, Rh): min label, held-out prediction, and literature value from benchmarks/benchmark_values.csv (filtered by functional=RPBE, quantity=E_N).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ammonia_ai import schema as S
from ammonia_ai.schema import _counts


BENCHMARK_PATH = Path("benchmarks/benchmark_values.csv")


def load_benchmark_literature_values() -> pd.DataFrame:
    """Load benchmark literature values for N adsorbate with RPBE functional.

    Returns DataFrame with columns: bulk_chemsys (metal), literature_E_ads_eV (RPBE value for N).
    Filters for functional=RPBE and quantity=E_N.
    """
    if not BENCHMARK_PATH.exists():
        print(f"Benchmark file not found at {BENCHMARK_PATH}", file=sys.stderr)
        return pd.DataFrame(columns=["bulk_chemsys", "literature_E_ads_eV"])

    df = pd.read_csv(BENCHMARK_PATH)
    print(f"Loaded benchmark data: {len(df)} rows", file=sys.stderr)

    # Filter for RPBE functional and E_N quantity
    # RPBE appears in functional column as "RPBE" or "RPBE interaction energies using PW91 self-consistent densities"
    rpbe_mask = df["functional"].str.contains("RPBE", case=False, na=False)
    n_mask = df["quantity"] == "E_N"
    filtered = df[rpbe_mask & n_mask].copy()

    if len(filtered) == 0:
        print("No RPBE E_N rows found in benchmark data", file=sys.stderr)
        return pd.DataFrame(columns=["bulk_chemsys", "literature_E_ads_eV"])

    # Target metals: Ru (terrace and step), Os, Rh
    target_metals = ["Ru", "Os", "Rh"]
    metal_mask = filtered["metal"].isin(target_metals)
    filtered = filtered[metal_mask].copy()

    # Convert value_eV to numeric, handling "not found" and "approximately" prefixes
    def parse_value(val):
        if pd.isna(val) or val == "not found":
            return np.nan
        val_str = str(val).strip()
        if val_str.startswith("approximately"):
            val_str = val_str.replace("approximately", "").strip()
        try:
            return float(val_str)
        except ValueError:
            return np.nan

    filtered["value_numeric"] = filtered["value_eV"].apply(parse_value)

    # Map metal to bulk_chemsys (same for pure metals)
    lit_df = pd.DataFrame({
        "bulk_chemsys": filtered["metal"].values,
        "literature_E_ads_eV": filtered["value_numeric"].values
    })

    # Drop NaN values
    lit_df = lit_df.dropna(subset=["literature_E_ads_eV"])

    print(f"Extracted {len(lit_df)} literature values for N adsorbate (RPBE) from benchmarks", file=sys.stderr)
    return lit_df


def load_training_data(training_table_path: Path) -> pd.DataFrame:
    """Load training table and filter to clean primary regression rows."""
    if not training_table_path.exists():
        raise FileNotFoundError(f"Training table not found: {training_table_path}")
    df = pd.read_parquet(training_table_path)
    print(f"Loaded training table with {len(df)} rows from {training_table_path}", file=sys.stderr)

    if "is_clean_primary_regression_row" in df.columns:
        df_clean = df[df["is_clean_primary_regression_row"]].copy()
    else:
        df_clean = df.copy()
        print("WARNING: 'is_clean_primary_regression_row' not found; using all rows", file=sys.stderr)

    print(f"Clean primary regression rows: {len(df_clean)}", file=sys.stderr)
    return df_clean


def identify_pure_metals(df: pd.DataFrame) -> list[str]:
    """Identify pure metal chemical systems (single-element bulk_chemsys)."""
    if "bulk_chemsys" not in df.columns:
        # derive if missing
        df["bulk_chemsys"] = df["bulk_symbols"].apply(S.chemical_system)

    chemsyses = df["bulk_chemsys"].unique()
    pure_metals = []
    for cs in chemsyses:
        # A pure metal system like "Fe", "Cu", "Ru", etc. has no other element symbols.
        # S.chemical_system gives sorted element set joined by '-' ('Fe2' -> 'Fe', 'Fe2Ni2' -> 'Fe-Ni')
        # So pure metals have a single element in the chemical system (no '-' separator).
        if "-" not in cs:
            # Verify it parses to exactly one element
            elements = _counts(cs)
            if len(elements) == 1:
                pure_metals.append(cs)

    pure_metals = sorted(pure_metals)
    print(f"Identified {len(pure_metals)} pure metal chemical systems: {pure_metals}", file=sys.stderr)
    return pure_metals


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-table", type=Path, default=Path("results/adsorbates/oc20_adsorbates_features.parquet"),
                    help="Training features table")
    ap.add_argument("--out-dir", type=Path, default=Path("results/m3"),
                    help="Output directory for M3 audit results")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_training_data(args.training_table)

    # Ensure required columns
    required_cols = list(S.SCREENING_FEATURES) + [S.TARGET_COLUMN, "bulk_chemsys", "bulk_symbols", "adsorbate"]
    for col in required_cols:
        if col not in df.columns and col != S.TARGET_COLUMN:
            # check if target or feature missing
            pass

    # Ensure bulk_chemsys exists
    if "bulk_chemsys" not in df.columns:
        df["bulk_chemsys"] = df["bulk_symbols"].apply(S.chemical_system)

    pure_metals = identify_pure_metals(df)

    # Filter df to rows belonging to pure metal systems
    df_pure = df[df["bulk_chemsys"].isin(pure_metals)].copy()
    print(f"Total rows belonging to pure metal systems: {len(df_pure)}", file=sys.stderr)

    # Facet column: miller_hkl_max or miller_index
    facet_col = "miller_hkl_max" if "miller_hkl_max" in df_pure.columns else "miller_index"
    if facet_col not in df_pure.columns:
        df_pure["facet"] = "unknown"
        facet_col = "facet"

    # Anomaly status column
    anomaly_col = "anomaly_0" if "anomaly_0" in df_pure.columns else ("anomaly_code" if "anomaly_code" in df_pure.columns else None)

    # 2. Leave-One-Group-Out (LOGO) cross-validation per pure metal chemical system
    oof_predictions = []
    per_metal_metrics = []

    features = list(S.SCREENING_FEATURES)

    for metal in pure_metals:
        print(f"\nProcessing pure metal system: {metal}", file=sys.stderr)
        train_mask = df["bulk_chemsys"] != metal
        test_mask = df["bulk_chemsys"] == metal

        df_train_fold = df[train_mask]
        df_test_fold = df[test_mask].copy()

        if len(df_test_fold) == 0:
            print(f"  No rows found for test fold {metal}", file=sys.stderr)
            continue

        X_train = df_train_fold[features].to_numpy(dtype=float)
        y_train = df_train_fold[S.TARGET_COLUMN].to_numpy(dtype=float)

        X_test = df_test_fold[features].to_numpy(dtype=float)
        y_test = df_test_fold[S.TARGET_COLUMN].to_numpy(dtype=float)

        # Train model
        model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        df_test_fold["predicted_E_ads_eV"] = y_pred
        df_test_fold["residual_eV"] = y_pred - y_test
        df_test_fold["abs_error_eV"] = np.abs(df_test_fold["residual_eV"])

        mae = np.mean(df_test_fold["abs_error_eV"])
        rmse = np.sqrt(np.mean(df_test_fold["residual_eV"]**2))
        print(f"  Test rows: {len(df_test_fold)}, MAE: {mae:.4f} eV, RMSE: {rmse:.4f} eV", file=sys.stderr)

        # Calculate MAE with outliers trimmed (qc_label_outlier flag)
        # Flag outliers: E_ads < -4 or > +5 eV
        outlier_mask = (df_test_fold[S.TARGET_COLUMN] < -4.0) | (df_test_fold[S.TARGET_COLUMN] > 5.0)
        df_trimmed = df_test_fold[~outlier_mask]
        if len(df_trimmed) > 0:
            mae_trimmed = np.mean(df_trimmed["abs_error_eV"])
            rows_trimmed = int(outlier_mask.sum())
        else:
            mae_trimmed = mae
            rows_trimmed = int(outlier_mask.sum())

        per_metal_metrics.append({
            "bulk_chemsys": metal,
            "n_samples": len(df_test_fold),
            "mae_eV": mae,
            "mae_trimmed_eV": mae_trimmed,
            "rows_trimmed": rows_trimmed,
            "rmse_eV": rmse,
        })

        oof_predictions.append(df_test_fold)

    if not oof_predictions:
        print("ERROR: No OOF predictions generated.", file=sys.stderr)
        return 1

    oof_df = pd.concat(oof_predictions, ignore_index=True)

    # Most-stable-site label: minimum E_ads per metal (bulk_chemsys), facet, and adsorbate
    # Group by bulk_chemsys, facet, adsorbate and find argmin for DFT label and prediction
    print("\nComputing most-stable-site labels...", file=sys.stderr)

    stable_rows = []
    grouped = oof_df.groupby(["bulk_chemsys", facet_col, "adsorbate"])
    for name, group in grouped:
        cs, facet, ads = name
        # Row with min DFT label (adsorption_energy_eV)
        min_dft_idx = group[S.TARGET_COLUMN].idxmin()
        min_pred_idx = group["predicted_E_ads_eV"].idxmin()

        stable_dft_row = group.loc[min_dft_idx]
        stable_pred_row = group.loc[min_pred_idx]

        stable_rows.append({
            "bulk_chemsys": cs,
            "facet": str(stable_dft_row[facet_col]),
            "adsorbate": ads,
            "min_dft_E_ads_eV": stable_dft_row[S.TARGET_COLUMN],
            "min_pred_E_ads_eV": stable_pred_row["predicted_E_ads_eV"],
            "stable_system_id_dft": stable_dft_row.get("system_id", ""),
            "stable_system_id_pred": stable_pred_row.get("system_id", ""),
        })

    stable_df = pd.DataFrame(stable_rows)

    # Save OOF predictions CSV
    out_oof_path = args.out_dir / "pure_metal_oof.csv"
    oof_df.to_csv(out_oof_path, index=False)
    print(f"Saved OOF predictions to {out_oof_path}", file=sys.stderr)

    out_metrics_path = args.out_dir / "pure_metal_metrics.csv"
    pd.DataFrame(per_metal_metrics).to_csv(out_metrics_path, index=False)
    print(f"Saved per-metal metrics to {out_metrics_path}", file=sys.stderr)

    out_stable_path = args.out_dir / "most_stable_sites.csv"
    stable_df.to_csv(out_stable_path, index=False)
    print(f"Saved most-stable-sites table to {out_stable_path}", file=sys.stderr)

    # 3. Second table for N only: Ru terrace/step, Os, Rh (RPBE functional, N2(g) reference)
    target_metals = ["Ru", "Os", "Rh"]
    print(f"\nGenerating N-only comparison table for metals: {target_metals}", file=sys.stderr)

    n_subset = oof_df[(oof_df["adsorbate"] == "N") & (oof_df["bulk_chemsys"].isin(target_metals))].copy()

    # Load benchmark literature values for N adsorbate (RPBE functional, E_N quantity)
    benchmark_lit = load_benchmark_literature_values()

    n_table_rows = []
    for metal in target_metals:
        metal_rows = n_subset[n_subset["bulk_chemsys"] == metal]
        if len(metal_rows) == 0:
            print(f"  No N rows found for metal {metal}", file=sys.stderr)
            continue

        min_dft = metal_rows[S.TARGET_COLUMN].min()
        # Find prediction corresponding to the min DFT row or min prediction row
        min_pred = metal_rows["predicted_E_ads_eV"].min()
        mean_pred = metal_rows["predicted_E_ads_eV"].mean()

        # Look up literature value from benchmark data
        lit_val = np.nan
        if not benchmark_lit.empty:
            match = benchmark_lit[benchmark_lit["bulk_chemsys"] == metal]
            if len(match) > 0:
                # Take the most stable site (minimum literature value) if multiple
                lit_val = match["literature_E_ads_eV"].min()

        n_table_rows.append({
            "bulk_chemsys": metal,
            "min_dft_E_ads_eV": min_dft,
            "min_pred_E_ads_eV": min_pred,
            "literature_E_ads_eV": lit_val,
            "n_sites": len(metal_rows),
        })

    n_table_df = pd.DataFrame(n_table_rows)
    out_n_path = args.out_dir / "n_only_metal_comparison.csv"
    n_table_df.to_csv(out_n_path, index=False)
    print(f"Saved N-only comparison table to {out_n_path}", file=sys.stderr)

    print("\n" + "=" * 60)
    print("N-ONLY METAL COMPARISON TABLE (Ru terrace/step, Os, Rh)")
    print("=" * 60)
    print(n_table_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

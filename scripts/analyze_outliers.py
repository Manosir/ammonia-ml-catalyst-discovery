#!/usr/bin/env python3
"""Script to analyze the effect of flagging outlier E_ads rows.

Rewritten to:
1. Use 5-fold chemical-system grouped CV
2. Report three MAEs per adsorbate on the SAME unfiltered test rows:
   - full training, all test rows
   - outlier-trimmed training, same test rows
   - full training, outlier-trimmed test rows
3. Real per-metal LOOCV for Fe, Ru, Os, Co, Mo, Rh, Ni, Cu
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from ammonia_ai import schema as S


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where E_ads < -4 or > +5 eV."""
    df = df.copy()
    df["qc_label_outlier"] = (df["adsorption_energy_eV"] < -4.0) | (df["adsorption_energy_eV"] > 5.0)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure bulk_chemsys column exists."""
    if "bulk_chemsys" not in df.columns:
        df = df.copy()
        df["bulk_chemsys"] = df["bulk_symbols"].apply(S.chemical_system)
    return df


def evaluate_on_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target: str,
) -> float:
    """Train on train_df, evaluate on test_df, return MAE."""
    if len(train_df) == 0 or len(test_df) == 0:
        return np.nan

    X_train = train_df[features].to_numpy(dtype=float)
    y_train = train_df[target].to_numpy(dtype=float)
    X_test = test_df[features].to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)

    model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return float(np.mean(np.abs(y_pred - y_test)))


def run_grouped_cv_three_maes(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    n_splits: int = 5,
) -> dict[str, float]:
    """
    Run 5-fold grouped CV and return three MAEs computed on the SAME unfiltered test rows.
    Averages per-fold results.
    """
    df = prepare_data(df)
    gkf = GroupKFold(n_splits=n_splits)
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    groups = df["bulk_chemsys"]

    maes_full_full = []
    maes_trimmed_full = []
    maes_full_trimmed = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        test_df_fixed = df.iloc[test_idx].copy()
        train_df_full = df.iloc[train_idx].copy()

        # Scenario 1: full training, all test rows
        mae_1 = evaluate_on_test(train_df_full, test_df_fixed, features, target)
        maes_full_full.append(mae_1)

        # Scenario 2: outlier-trimmed training, SAME test rows
        train_df_trimmed = train_df_full[~train_df_full["qc_label_outlier"]].copy()
        mae_2 = evaluate_on_test(train_df_trimmed, test_df_fixed, features, target)
        maes_trimmed_full.append(mae_2)

        # Scenario 3: full training, outlier-trimmed test rows (from the fixed test set)
        test_df_trimmed = test_df_fixed[~test_df_fixed["qc_label_outlier"]].copy()
        mae_3 = evaluate_on_test(train_df_full, test_df_trimmed, features, target)
        maes_full_trimmed.append(mae_3)

    return {
        "mae_full_full": float(np.mean(maes_full_full)),
        "mae_trimmed_full": float(np.mean(maes_trimmed_full)),
        "mae_full_trimmed": float(np.mean(maes_full_trimmed)),
    }


def run_loocv_per_metal(
    df: pd.DataFrame,
    metals: list[str],
    features: list[str],
    target: str,
) -> list[dict]:
    """
    Run real per-metal LOOCV for specified metals.
    For each metal: train on all other metals, test on this metal.
    """
    df = prepare_data(df)
    results = []

    for metal in metals:
        train_df = df[df["bulk_chemsys"] != metal].copy()
        test_df = df[df["bulk_chemsys"] == metal].copy()

        if len(test_df) == 0:
            results.append({
                "metal": metal,
                "n_test": 0,
                "mae_full": np.nan,
                "mae_trimmed_train": np.nan,
                "mae_trimmed_test": np.nan,
            })
            continue

        # Scenario 1: full training, full test
        mae_full = evaluate_on_test(train_df, test_df, features, target)

        # Scenario 2: outlier-trimmed training, full test
        train_df_trimmed = train_df[~train_df["qc_label_outlier"]].copy()
        mae_trimmed_train = evaluate_on_test(train_df_trimmed, test_df, features, target)

        # Scenario 3: full training, outlier-trimmed test
        test_df_trimmed = test_df[~test_df["qc_label_outlier"]].copy()
        mae_trimmed_test = evaluate_on_test(train_df, test_df_trimmed, features, target)

        results.append({
            "metal": metal,
            "n_test": len(test_df),
            "mae_full": mae_full,
            "mae_trimmed_train": mae_trimmed_train,
            "mae_trimmed_test": mae_trimmed_test,
        })

    return results


def main():
    path = Path("results/adsorbates/oc20_adsorbates_features.parquet")
    df = pd.read_parquet(path)
    df = flag_outliers(df)

    # Report flagged rows summary
    flagged = df[df["qc_label_outlier"]]
    print(f"Total flagged rows: {len(flagged)}")
    print()
    for ads in ['N', 'N2', 'NH', 'NH2', 'NH3']:
        n_flag = (flagged["adsorbate"] == ads).sum()
        print(f"  {ads}: {n_flag} flagged rows")

    features = list(S.SCREENING_FEATURES)
    target = S.TARGET_COLUMN

    # Three MAEs per adsorbate on SAME unfiltered test rows (5-fold grouped CV)
    print("\n5-fold Grouped CV - Three MAEs per Adsorbate (same test rows)")
    print("=" * 70)
    for ads in ['N', 'N2', 'NH', 'NH2', 'NH3']:
        ads_df = df[df["adsorbate"] == ads].copy()
        maes = run_grouped_cv_three_maes(ads_df, features, target, n_splits=5)
        print(f"\n{ads}:")
        print(f"  Full training, full test:       {maes['mae_full_full']:.4f} eV")
        print(f"  Trimmed training, full test:    {maes['mae_trimmed_full']:.4f} eV")
        print(f"  Full training, trimmed test:    {maes['mae_full_trimmed']:.4f} eV")

    # Real per-metal LOOCV for the 8 target N metals
    print("\n" + "=" * 70)
    print("Per-Metal LOOCV for N on Target Metals (Fe, Ru, Os, Co, Mo, Rh, Ni, Cu)")
    print("=" * 70)
    n_df = df[df["adsorbate"] == "N"].copy()
    target_metals = ["Fe", "Ru", "Os", "Co", "Mo", "Rh", "Ni", "Cu"]
    loocv_results = run_loocv_per_metal(n_df, target_metals, features, target)

    print(f"\n{'Metal':>6} | {'n_test':>6} | {'Full Train':>12} | {'Trim Train':>12} | {'Trim Test':>12}")
    print("-" * 60)
    for r in loocv_results:
        print(f"{r['metal']:>6} | {r['n_test']:>6} | "
              f"{r['mae_full']:>12.4f} | {r['mae_trimmed_train']:>12.4f} | {r['mae_trimmed_test']:>12.4f}")


if __name__ == "__main__":
    main()
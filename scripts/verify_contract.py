#!/usr/bin/env python3
"""Freeze check: the contract table alone (gate-checked) must reproduce the ablation numbers."""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

from ammonia_ai import schema as S

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="results/adsorbates/oc20_adsorbates_features.parquet")
    ap.add_argument("--splits", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    df = pd.read_parquet(a.table)
    S.validate_feature_frame(df, registry_path=S.DEFAULT_REGISTRY)  # registry gate + numeric/NaN/inf checks on every row
    clean = df[df["is_clean_primary_regression_row"]].reset_index(drop=True)
    y = clean["adsorption_energy_eV"].to_numpy()
    cls = clean["class"].to_numpy()
    key = (clean["adsorbate"].astype(str) + "_" + clean["class"].astype(str)).to_numpy()
    groups = clean["bulk_chemsys"].to_numpy()
    full = list(S.SCREENING_FEATURES)
    light = [c for c in full if c not in set(S.PLACEMENT_SENSITIVE_FEATURES)]
    arms = {f"contract, all {len(full)} features": full, f"contract without placement-sensitive ({len(light)})": light}
    print(f"rows {len(df)} | clean {len(clean)} | chemical systems {len(set(groups))}", flush=True)

    res = {"floor (adsorbate, class mean)": [], **{k: [] for k in arms}}
    for tr, te in GroupShuffleSplit(a.splits, test_size=0.2, random_state=0).split(np.zeros(len(clean)), groups=groups):
        mu = pd.Series(y[tr]).groupby(key[tr]).mean()
        ef = np.abs(y[te] - pd.Series(key[te]).map(mu).fillna(y[tr].mean()).to_numpy())
        c0 = cls[te] == 0
        res["floor (adsorbate, class mean)"].append((ef.mean(), ef[c0].mean()))
        for name, cols in arms.items():
            X = clean[cols].to_numpy(dtype=float)
            model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, early_stopping=False, random_state=0)
            e = np.abs(y[te] - model.fit(X[tr], y[tr]).predict(X[te]))
            res[name].append((e.mean(), e[c0].mean()))
        print(f"  split done, {time.time() - t0:.0f}s elapsed", flush=True)
    print("\nchemical-system grouped CV (clean rows)")
    for name, r in res.items():
        r = np.array(r)
        print(f"  {name:46s} MAE {r[:, 0].mean():.3f} ± {r[:, 0].std():.3f} | class 0 {r[:, 1].mean():.3f}")
    names = list(arms)
    d = np.array(res[names[1]])[:, 0] - np.array(res[names[0]])[:, 0]
    print(f"  placement-sensitive features are worth {d.mean():+.3f} eV (worse without in {int((d > 0).sum())}/{len(d)} splits)")
    print("expected from the ablations: contract about 0.576-0.580 (class 0 about 0.465-0.470); without placement-sensitive about 0.585")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""How much does the full contract depend on OC20's adsorbate placement?  (chemical-system grouped CV, clean rows)

Models are always TRAINED on the true features. The TEST rows' placement-sensitive features are then
(a) replaced by the per-adsorbate training median (a site finder that always places at a standard geometry) or
(b) shifted by a common random offset (a different placement height; six adsorbate-distance features).
'light' is the model retrained WITHOUT the placement-sensitive features.
"""
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ammonia_ai import schema as S

warnings.filterwarnings("ignore")
_spec = importlib.util.spec_from_file_location("ev", Path(__file__).with_name("evaluate_screening.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

SIGMAS = (0.1, 0.25, 0.5)
REPEATS, FOLDS = 2, 5


def main():
    t0 = time.time()
    df = ev.load_table("results/adsorbates/oc20_adsorbates_features.parquet", str(S.DEFAULT_REGISTRY))
    full = list(S.SCREENING_FEATURES)
    sensitive = set(S.PLACEMENT_SENSITIVE_FEATURES)
    light = [c for c in full if c not in sensitive]
    pidx = [i for i, c in enumerate(full) if c in sensitive]
    shift_idx = [full.index(c) for c in S.GEOMETRY_FEATURES if c != "geo_min_dist_all"]
    Xf, Xl = df[full].to_numpy(float), df[light].to_numpy(float)
    y = df["adsorption_energy_eV"].to_numpy(float)
    ads = df["adsorbate"].astype(str).to_numpy()
    cls = df["class"].to_numpy()
    groups = df["bulk_chemsys"].astype(str).to_numpy()
    clean = df["is_clean_primary_regression_row"].to_numpy(bool)
    key = (pd.Series(ads) + "_" + pd.Series(cls).astype(str)).to_numpy()
    names = ["floor", "full", "light", "full->median"] + [f"full+N(0,{s})" for s in SIGMAS]
    err = {n: np.zeros(len(y)) for n in names}
    rng = np.random.default_rng(0)
    for r in range(REPEATS):
        fold = ev.assign_folds(groups, FOLDS, 300 + r)
        for f in range(FOLDS):
            te = np.where((fold == f) & clean)[0]
            tr = np.where((fold != f) & clean)[0]
            mf = ev.make_regressor().fit(Xf[tr], y[tr])
            ml = ev.make_regressor().fit(Xl[tr], y[tr])

            def add(name, pred):
                err[name][te] += np.abs(y[te] - pred) / REPEATS

            add("floor", ev.floor_predict(y, key, tr, te))
            add("full", mf.predict(Xf[te]))
            add("light", ml.predict(Xl[te]))
            Xm = Xf[te].copy()
            for a_ in ev.ADSORBATES:
                rows = np.where(ads[te] == a_)[0]
                if len(rows):
                    Xm[np.ix_(rows, pidx)] = np.median(Xf[tr][ads[tr] == a_][:, pidx], axis=0)
            add("full->median", mf.predict(Xm))
            for s in SIGMAS:
                Xn = Xf[te].copy()
                Xn[:, shift_idx] += rng.normal(0, s, size=len(te))[:, None]
                add(f"full+N(0,{s})", mf.predict(Xn))
            print(f"  repeat {r + 1}/{REPEATS}, fold {f + 1}/{FOLDS} ({time.time() - t0:.0f}s)", flush=True)

    pd.set_option("display.width", 250)
    for title, mask in (("clean rows", clean), ("clean class-0 rows", clean & (cls == 0))):
        t = pd.DataFrame({n: err[n][mask] for n in names})
        t["adsorbate"] = ads[mask]
        by = t.groupby("adsorbate").mean().reindex(ev.ADSORBATES)
        by.loc["ALL"] = t.drop(columns="adsorbate").mean()
        print(f"\n{title}: MAE (eV); each row is tested {REPEATS}x with chemical-system folds")
        print(by.round(3).to_string())
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

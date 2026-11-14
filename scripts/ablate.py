#!/usr/bin/env python3
"""Feature-group ablation: chemical-system CV and TRUE leave-element-out. Clean rows (anomaly == 0)."""
import argparse
import ast
import re
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")


def counts(s):
    d = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", str(s)):
        d[el] = d.get(el, 0) + (int(n) if n else 1)
    return d


def hkl(v):
    a = np.abs(np.asarray(ast.literal_eval(v) if isinstance(v, str) else v, dtype=float))
    return a.max(), float((a ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="results/adsorbates/oc20_adsorbates_features.parquet")
    ap.add_argument("--cand", default="results/structure_pass/candidate_features.parquet")
    ap.add_argument("--elements", default="Ru,Fe,Os,Re,Co,Ni,Pd,Pt,Cu,Ag,Au,Ti")
    ap.add_argument("--min-rows", type=int, default=150)
    ap.add_argument("--splits", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()

    tab = pd.read_parquet(a.table, columns=["system_id", "adsorbate", "bulk_symbols", "miller_index", "class", "top",
                                            "anomaly", "adsorption_energy_eV"])
    cand = pd.read_parquet(a.cand)
    bad = [c for c in cand.columns if any(s in c for s in ("final", "disp", "frames", "force", "e_first", "e_last", "bind_disp"))]
    assert not bad, f"audit-only fields among candidates: {bad}"
    m = tab.merge(cand, on="system_id").reset_index(drop=True)
    y = m.adsorption_energy_eV.values
    ads = m.adsorbate.values
    cls = m["class"].values
    els = [frozenset(counts(s)) for s in m.bulk_symbols]
    chem = np.array(["-".join(sorted(s)) for s in els])
    key = np.char.add(ads.astype(str), cls.astype(str))
    idx = np.where((m.anomaly == 0).values)[0]
    print(f"rows {len(m)} | clean {len(idx)} | chemical systems {len(set(chem[idx]))}", flush=True)

    comp = pd.DataFrame([counts(s) for s in m.bulk_symbols]).fillna(0)
    comp = comp.div(comp.sum(axis=1), axis=0)
    comp.columns = ["frac_" + c for c in comp.columns]
    h = np.array([hkl(v) for v in m.miller_index])
    ctx = pd.concat([
        pd.get_dummies(m["class"], prefix="class").astype(float),
        pd.DataFrame({"hkl_max": h[:, 0], "hkl_norm2": h[:, 1], "top": m["top"].astype(float)}),
        pd.get_dummies(m.adsorbate, prefix="ads").astype(float),
        m.adsorbate.map({"N": 1, "N2": 2, "NH": 1, "NH2": 1, "NH3": 1}).astype(float).rename("ads_nN"),
        m.adsorbate.map({"N": 0, "N2": 0, "NH": 1, "NH2": 2, "NH3": 3}).astype(float).rename("ads_nH"),
    ], axis=1)

    def cols(*pref):
        return [c for c in cand.columns if c.startswith(pref)]
    BULK, SURF, SITE, GEO = cols("bulk_"), cols("surf_"), cols("site_"), cols("geo_", "topo_", "cn_")
    arms = {
        "0 ctx only": [ctx],
        "1 ctx+element fractions": [ctx, comp],
        "2 ctx+bulk props": [ctx, m[BULK]],
        "3 ctx+bulk+surface props": [ctx, m[BULK + SURF]],
        "4 +site props": [ctx, m[BULK + SURF + SITE]],
        "5 +geometry/topology/CN (property set)": [ctx, m[BULK + SURF + SITE + GEO]],
        "6 hybrid: fractions + all": [ctx, comp, m[BULK + SURF + SITE + GEO]],
    }
    X = {k: pd.concat(v, axis=1).values.astype(float) for k, v in arms.items()}
    print({k: v.shape[1] for k, v in X.items()}, flush=True)

    def model():
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, early_stopping=False, random_state=0)

    def floor(tr, te):
        mu = pd.Series(y[tr]).groupby(key[tr]).mean()
        return pd.Series(key[te]).map(mu).fillna(y[tr].mean()).values

    # 1. chemical-system grouped CV
    print("\n1. chemical-system grouped CV (clean rows)", flush=True)
    spl = list(GroupShuffleSplit(a.splits, test_size=0.2, random_state=0).split(np.zeros(len(idx)), groups=chem[idx]))
    res = {"floor": []}
    for name in X:
        res[name] = []
    for tr_, te_ in spl:
        tr, te = idx[tr_], idx[te_]
        c0 = cls[te] == 0
        e = np.abs(y[te] - floor(tr, te))
        res["floor"].append((e.mean(), e[c0].mean()))
        for name, Xa in X.items():
            e = np.abs(y[te] - model().fit(Xa[tr], y[tr]).predict(Xa[te]))
            res[name].append((e.mean(), e[c0].mean()))
        print(f"  split done, {time.time() - t0:.0f}s elapsed", flush=True)
    ref = np.array(res["1 ctx+element fractions"])[:, 0]
    for name, r in res.items():
        r = np.array(r)
        dl = r[:, 0] - ref
        print(f"  {name:42s} MAE {r[:, 0].mean():.3f} ± {r[:, 0].std():.3f} | class 0 {r[:, 1].mean():.3f} | vs arm 1: {dl.mean():+.3f} ({(dl < 0).sum()}/{len(dl)} better)")

    # 2. TRUE leave-element-out
    print("\n2. TRUE leave-element-out (every bulk containing the element removed from training)", flush=True)
    recs = []
    for E in [e.strip() for e in a.elements.split(",") if e.strip()]:
        has = np.array([E in s for s in els])
        te, tr = idx[has[idx]], idx[~has[idx]]
        if len(te) < a.min_rows:
            print(f"  {E}: only {len(te)} clean test rows, skipped", flush=True)
            continue
        c0 = cls[te] == 0
        mN = ads[te] == "N"

        def rho(p):
            return spearmanr(p[mN], y[te][mN])[0] if mN.sum() >= 30 else np.nan
        p = floor(tr, te)
        recs.append(dict(el=E, n=len(te), arm="floor", mae=np.abs(y[te] - p).mean(), mae0=np.abs(y[te] - p)[c0].mean(), rhoN=np.nan))
        for name, Xa in X.items():
            p = model().fit(Xa[tr], y[tr]).predict(Xa[te])
            e = np.abs(y[te] - p)
            recs.append(dict(el=E, n=len(te), arm=name, mae=e.mean(), mae0=e[c0].mean(), rhoN=rho(p)))
        print(f"  {E}: {len(te)} test rows done, {time.time() - t0:.0f}s elapsed", flush=True)
    R = pd.DataFrame(recs)
    for col, title in [("mae", "MAE, all test rows"), ("mae0", "MAE, class 0 only"), ("rhoN", "Spearman within N* rows")]:
        pv = R.pivot_table(index="arm", columns="el", values=col)
        pv["MEAN"] = pv.mean(axis=1)
        print(f"\n{title}:")
        print(pv.round(3).to_string())
    print("\ntest rows per element:", R.drop_duplicates("el").set_index("el").n.to_dict())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

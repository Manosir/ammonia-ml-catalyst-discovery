#!/usr/bin/env python3
"""Pre-freeze checks for the screening contract (clean rows; chemical-system CV + true leave-element-out)."""
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
FORBIDDEN = ("final", "disp", "frames", "force", "e_first", "e_last", "bind_disp")


def counts(s):
    d = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", str(s)):
        d[el] = d.get(el, 0) + (int(n) if n else 1)
    return d


def hkl(v):
    a = np.abs(np.asarray(ast.literal_eval(v) if isinstance(v, str) else v, dtype=float))
    return a.max(), float((a ** 2).sum())


def model():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, early_stopping=False, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="results/adsorbates/oc20_adsorbates_features.parquet")
    ap.add_argument("--cand", default="results/structure_pass/candidate_features.parquet")
    ap.add_argument("--elements", default="Ru,Fe,Os,Re,Co,Ni,Pd,Pt,Cu,Ag,Au,Ti")
    ap.add_argument("--splits", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()

    tab = pd.read_parquet(a.table, columns=["system_id", "adsorbate", "bulk_symbols", "miller_index", "class",
                                            "anomaly", "adsorption_energy_eV"])
    cand = pd.read_parquet(a.cand)
    bad = [c for c in cand.columns if any(s in c for s in FORBIDDEN)]
    assert not bad, f"audit-only fields among candidates: {bad}"
    m = tab.merge(cand, on="system_id").reset_index(drop=True)
    assert m.system_id.is_unique and len(m) == len(tab)
    y = m.adsorption_energy_eV.values
    ads = m.adsorbate.values
    cls = m["class"].values
    anom = m.anomaly.values
    els = [frozenset(counts(s)) for s in m.bulk_symbols]
    chem = np.array(["-".join(sorted(s)) for s in els])
    key = np.char.add(ads.astype(str), cls.astype(str))
    idx = np.where(anom == 0)[0]
    print(f"rows {len(m)} | clean {len(idx)} | chemical systems {len(set(chem[idx]))}", flush=True)

    def floor(tr, te):
        mu = pd.Series(y[tr]).groupby(key[tr]).mean()
        return pd.Series(key[te]).map(mu).fillna(y[tr].mean()).values

    def cols(*pref):
        return [c for c in cand.columns if c.startswith(pref)]
    PROP, GEO, TOPO, CN = cols("bulk_", "surf_", "site_"), cols("geo_"), cols("topo_"), cols("cn_")
    GAPS = ["topo_gap2", "topo_gap3"]
    allc = PROP + GEO + TOPO + CN
    assert sorted(allc) == sorted(c for c in cand.columns if c != "system_id"), "unexpected candidate columns"

    def drop_fam(columns, fams):
        return [c for c in columns if not any(re.search(rf"_{p}_(mean|min|max|rng)$", c) for p in fams)]
    light = [c for c in allc if c not in GEO + GAPS]
    three = drop_fam(allc, ("rcov", "nd", "nval"))
    light3 = [c for c in three if c in set(light)]

    h = np.array([hkl(v) for v in m.miller_index])
    P = {"cls": pd.get_dummies(m["class"], prefix="class").astype(float),
         "hkl": pd.DataFrame({"hkl_max": h[:, 0], "hkl_norm2": h[:, 1]}),
         "oh": pd.get_dummies(m.adsorbate, prefix="ads").astype(float),
         "desc": pd.DataFrame({
             "ads_nN": m.adsorbate.map({"N": 1, "N2": 2, "NH": 1, "NH2": 1, "NH3": 1}).astype(float),
             "ads_nH": m.adsorbate.map({"N": 0, "N2": 0, "NH": 1, "NH2": 2, "NH3": 3}).astype(float)})}

    def build(extra, use_cls=True, use_hkl=True, use_oh=False):
        parts = [P["desc"]] + ([P["cls"]] if use_cls else []) + ([P["hkl"]] if use_hkl else []) + ([P["oh"]] if use_oh else [])
        return pd.concat(parts + [m[list(extra)]], axis=1)

    C0, LIGHT = "C0 contract candidate", "placement-light"
    arms = {
        C0: build(allc),
        "C0 + adsorbate one-hot": build(allc, use_oh=True),
        "C0 without class": build(allc, use_cls=False),
        "C0 without hkl": build(allc, use_hkl=False),
        "C0 without class, hkl": build(allc, use_cls=False, use_hkl=False),
        "3 families (en, group, period)": build(three),
        LIGHT: build(light),
        "placement-light + 3 families": build(light3),
    }
    X = {k: v.values.astype(float) for k, v in arms.items()}
    print({k: v.shape[1] for k, v in X.items()}, flush=True)

    print("\n1. chemical-system grouped CV (clean rows)", flush=True)
    spl = list(GroupShuffleSplit(a.splits, test_size=0.2, random_state=0).split(np.zeros(len(idx)), groups=chem[idx]))
    res = {k: [] for k in ["floor"] + list(X)}
    keep = {"floor", C0, LIGHT}
    errs = {k: [] for k in keep}
    for tr_, te_ in spl:
        tr, te = idx[tr_], idx[te_]
        c0 = cls[te] == 0
        ef = np.abs(y[te] - floor(tr, te))
        res["floor"].append((ef.mean(), ef[c0].mean()))
        errs["floor"].append((te, ef))
        for k in X:
            e = np.abs(y[te] - model().fit(X[k][tr], y[tr]).predict(X[k][te]))
            res[k].append((e.mean(), e[c0].mean()))
            if k in keep:
                errs[k].append((te, e))
        print(f"  split done, {time.time() - t0:.0f}s elapsed", flush=True)
    ref = np.array(res[C0])[:, 0]
    for k in ["floor"] + list(X):
        r = np.array(res[k])
        dl = r[:, 0] - ref
        print(f"  {k:34s} MAE {r[:, 0].mean():.3f} ± {r[:, 0].std():.3f} | class 0 {r[:, 1].mean():.3f} | vs C0 {dl.mean():+.3f} ({int((dl > 0).sum())}/{len(dl)} splits worse)")

    E = pd.concat([pd.DataFrame({"arm": k, "ads": ads[te], "c0": cls[te] == 0, "err": e}) for k, lst in errs.items() for te, e in lst])
    print("\n2. MAE by adsorbate")
    print("  all classes:")
    print(E.pivot_table(index="ads", columns="arm", values="err").round(3).to_string())
    print("  class 0 only:")
    print(E[E.c0].pivot_table(index="ads", columns="arm", values="err").round(3).to_string(), flush=True)

    print("\n3. TRUE leave-element-out: C0 vs placement-light vs floor", flush=True)
    recs = []
    for el in [e.strip() for e in a.elements.split(",") if e.strip()]:
        has = np.array([el in s for s in els])
        te, tr = idx[has[idx]], idx[~has[idx]]
        if len(te) < 150:
            print(f"  {el}: {len(te)} clean test rows, skipped")
            continue
        c0 = cls[te] == 0
        mN = ads[te] == "N"
        ef = np.abs(y[te] - floor(tr, te))
        recs.append(dict(el=el, arm="floor", mae=ef.mean(), mae0=ef[c0].mean(), rhoN=np.nan))
        for k in (C0, LIGHT):
            p = model().fit(X[k][tr], y[tr]).predict(X[k][te])
            e = np.abs(y[te] - p)
            recs.append(dict(el=el, arm=k, mae=e.mean(), mae0=e[c0].mean(),
                             rhoN=spearmanr(p[mN], y[te][mN])[0] if mN.sum() >= 30 else np.nan))
        print(f"  {el}: {len(te)} test rows done, {time.time() - t0:.0f}s elapsed", flush=True)
    R = pd.DataFrame(recs)
    for col, title in [("mae", "MAE, all test rows"), ("mae0", "MAE, class 0 only"), ("rhoN", "Spearman within N* rows")]:
        pv = R.pivot_table(index="arm", columns="el", values=col)
        pv["MEAN"] = pv.mean(axis=1)
        print(f"\n{title}:")
        print(pv.round(3).to_string())

    print("\n4. energy distributions by anomaly code (quantiles 5/25/50/75/95 %)")
    for ad in ("N2", "NH3"):
        for code, name in [(0, "clean"), (1, "dissociated"), (2, "desorbed"), (3, "surface changed")]:
            s = y[(ads == ad) & (anom == code)]
            if len(s):
                print(f"  {ad:3s} {name:16s} n={len(s):5d} {np.round(np.quantile(s, [.05, .25, .5, .75, .95]), 2).tolist()}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

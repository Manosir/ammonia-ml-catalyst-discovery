#!/usr/bin/env python3
"""Robustness ablations for the property-based feature set (chemical-system grouped CV; clean rows unless stated)."""
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
    ap.add_argument("--splits", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()

    tab = pd.read_parquet(a.table, columns=["system_id", "adsorbate", "bulk_symbols", "miller_index", "class", "top",
                                            "anomaly", "split", "adsorption_energy_eV"])
    cand = pd.read_parquet(a.cand)
    bad = [c for c in cand.columns if any(s in c for s in FORBIDDEN)]
    assert not bad, f"audit-only fields among candidates: {bad}"
    m = tab.merge(cand, on="system_id").reset_index(drop=True)
    assert m.system_id.is_unique
    y = m.adsorption_energy_eV.values
    ads = m.adsorbate.values
    cls = m["class"].values
    anom = m.anomaly.values
    split = m["split"].values
    chem = np.array(["-".join(sorted(counts(s))) for s in m.bulk_symbols])
    key = np.char.add(ads.astype(str), cls.astype(str))
    clean = anom == 0
    idx = np.where(clean)[0]
    print(f"rows {len(m)} | clean {len(idx)}", flush=True)

    def floor(tr, te):
        mu = pd.Series(y[tr]).groupby(key[tr]).mean()
        return pd.Series(key[te]).map(mu).fillna(y[tr].mean()).values

    comp = pd.DataFrame([counts(s) for s in m.bulk_symbols]).fillna(0)
    comp = comp.div(comp.sum(axis=1), axis=0)
    comp.columns = ["frac_" + c for c in comp.columns]
    h = np.array([hkl(v) for v in m.miller_index])
    core = pd.concat([pd.get_dummies(m["class"], prefix="class").astype(float),
                      pd.DataFrame({"hkl_max": h[:, 0], "hkl_norm2": h[:, 1], "top": m["top"].astype(float)})], axis=1)
    oh = pd.get_dummies(m.adsorbate, prefix="ads").astype(float)
    desc = pd.DataFrame({"ads_nN": m.adsorbate.map({"N": 1, "N2": 2, "NH": 1, "NH2": 1, "NH3": 1}).astype(float),
                         "ads_nH": m.adsorbate.map({"N": 0, "N2": 0, "NH": 1, "NH2": 2, "NH3": 3}).astype(float)})

    def cols(*pref):
        return [c for c in cand.columns if c.startswith(pref)]
    PROP = cols("bulk_", "surf_", "site_")
    GEO, TOPO, CN = cols("geo_"), cols("topo_"), cols("cn_")
    allc = PROP + GEO + TOPO + CN
    compact = [c for c in PROP if c.endswith(("_mean", "_rng", "_tm_frac", "_nm_frac", "_n_elem"))] + GEO + TOPO + CN

    def fam(p):
        return {c for c in PROP if re.search(rf"_{p}_(mean|min|max|rng)$", c)}

    def build(extra, drop_core=()):
        return pd.concat([core.drop(columns=list(drop_core)), oh, desc, m[list(extra)]], axis=1)

    FULL, ARM1, ARM4 = "full property set (arm 5)", "element fractions (arm 1)", "no geo_/topo_/cn_ (arm 4)"
    arms = {
        FULL: build(allc),
        "no geo_ (heights, distances)": build(PROP + TOPO + CN),
        ARM4: build(PROP),
        "no top": build(allc, drop_core=("top",)),
        "no hkl encodings": build(allc, drop_core=("hkl_max", "hkl_norm2")),
        "compact (mean + range only)": build(compact),
    }
    for p in ("rcov", "en", "nd", "nval", "period", "group"):
        arms[f"no {p} family"] = build([c for c in allc if c not in fam(p)])
    arms[ARM1] = pd.concat([core, oh, desc, comp], axis=1)
    X = {k: v.values.astype(float) for k, v in arms.items()}
    print({k: v.shape[1] for k, v in X.items()}, flush=True)

    print("\n1. chemical-system grouped CV (clean rows, %d splits)" % a.splits, flush=True)
    spl = list(GroupShuffleSplit(a.splits, test_size=0.2, random_state=0).split(np.zeros(len(idx)), groups=chem[idx]))
    res = {k: [] for k in ["floor"] + list(X)}
    keep = {"floor", ARM1, ARM4, FULL}
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
            if k in errs:
                errs[k].append((te, e))
        print(f"  split done, {time.time() - t0:.0f}s elapsed", flush=True)
    ref = np.array(res[FULL])[:, 0]
    for k in ["floor"] + list(X):
        r = np.array(res[k])
        dl = r[:, 0] - ref
        print(f"  {k:36s} MAE {r[:, 0].mean():.3f} ± {r[:, 0].std():.3f} | class 0 {r[:, 1].mean():.3f} | vs full {dl.mean():+.3f} ({int((dl > 0).sum())}/{len(dl)} splits worse)")

    print("\n2. MAE by adsorbate (same splits)")
    E = pd.concat([pd.DataFrame({"arm": k, "ads": ads[te], "c0": cls[te] == 0, "err": e}) for k, lst in errs.items() for te, e in lst])
    print("  all classes:")
    print(E.pivot_table(index="ads", columns="arm", values="err").round(3).to_string())
    print("  class 0 only:")
    print(E[E.c0].pivot_table(index="ads", columns="arm", values="err").round(3).to_string(), flush=True)

    print("\n3. clean-trained full-property regressor evaluated on ALL held-out rows, by anomaly code (0 none, 1 dissociated, 2 desorbed, 3 surface changed)")
    allidx = np.arange(len(m))
    out = []
    for tr_, te_ in GroupShuffleSplit(a.splits, test_size=0.2, random_state=1).split(allidx, groups=chem):
        tr = tr_[clean[tr_]]
        p = model().fit(X[FULL][tr], y[tr]).predict(X[FULL][te_])
        out.append(pd.DataFrame({"ads": ads[te_], "code": anom[te_], "err": np.abs(y[te_] - p), "floor_err": np.abs(y[te_] - floor(tr, te_))}))
    O = pd.concat(out)
    print(O.groupby("code").agg(n=("err", "size"), model=("err", "mean"), floor=("floor_err", "mean")).round(3).to_string())
    print(O[O.ads.isin(["N2", "NH3"])].groupby(["ads", "code"]).agg(n=("err", "size"), model=("err", "mean"), floor=("floor_err", "mean")).round(3).to_string(), flush=True)

    print("\n4. NH2 as an unseen adsorbate (train: clean, split == 'train', no NH2; nN/nH inputs, no one-hot)")
    Xd = {"element fractions": pd.concat([core, desc, comp], axis=1).values.astype(float),
          "property set": pd.concat([core, desc, m[allc]], axis=1).values.astype(float)}
    tr = np.where(clean & (ads != "NH2") & (split == "train"))[0]
    mu = pd.Series(y[tr]).groupby(ads[tr]).mean()
    const = (mu["NH"] + mu["NH3"]) / 2
    fits = {k: model().fit(v[tr], y[tr]) for k, v in Xd.items()}
    for spn in ["val_oos_ads", "val_oos_ads_bulk"]:
        te = np.where(clean & (ads == "NH2") & (split == spn))[0]
        te0 = te[cls[te] == 0]
        print(f"  {spn}: {len(te)} rows ({len(te0)} class 0) | constant mean(NH, NH3) MAE {np.abs(y[te] - const).mean():.3f}")
        for k, v in Xd.items():
            p, p0 = fits[k].predict(v[te]), fits[k].predict(v[te0])
            print(f"    {k:18s} Spearman {spearmanr(p, y[te])[0]:.3f} (class 0 {spearmanr(p0, y[te0])[0]:.3f}) | slope {np.polyfit(p, y[te], 1)[0]:.2f} | MAE {np.abs(y[te] - p).mean():.3f} (class 0 {np.abs(y[te0] - p0).mean():.3f})")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

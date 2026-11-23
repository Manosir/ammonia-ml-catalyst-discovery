#!/usr/bin/env python3
"""Evaluation of the frozen screening-v3 contract: chemical-system grouped CV, the official OC20 split,
true leave-element-out, NH2 leave-adsorbate-out and an anomaly classifier.

Conventions
- regression trains on clean rows (anomaly == 0) only; the floor is the (adsorbate, class) mean;
- 'full' = all contract features; 'light' = without the placement-sensitive ones;
- 95% intervals are percentile bootstraps over WHOLE chemical systems; they reflect test-set sampling only,
  not training variability or tuning (models are untuned);
- grouped CV: per-row errors are averaged over --repeats independent fold assignments.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from ammonia_ai import schema as S

warnings.filterwarnings("ignore")

ADSORBATES = ["N", "N2", "NH", "NH2", "NH3"]
ANOMALY_NAMES = {0: "none", 1: "dissociated", 2: "desorbed", 3: "surface changed"}
REQUIRED = ["adsorbate", "class", "anomaly", "split", "bulk_chemsys", "bulk_symbols",
            "adsorption_energy_eV", "is_clean_primary_regression_row"]
_ELEMENT = re.compile(r"[A-Z][a-z]?")


def make_regressor(max_iter: int = 200):
    return HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.1, early_stopping=False, random_state=0)


def assign_folds(groups, k, seed):
    """Whole groups go to one fold; fold sizes are balanced by group count."""
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    np.random.default_rng(seed).shuffle(unique)
    fold_of = {g: i % k for i, g in enumerate(unique)}
    return pd.Series(groups).map(fold_of).to_numpy(dtype=int)


def floor_predict(y, key, train_idx, test_idx):
    """(adsorbate, class) mean of the TRAINING rows; unseen keys fall back to the overall training mean."""
    mu = pd.Series(y[train_idx]).groupby(key[train_idx]).mean()
    return pd.Series(key[test_idx]).map(mu).fillna(float(np.mean(y[train_idx]))).to_numpy()


def group_bootstrap(groups, stat_fn, B, seed):
    """Resample WHOLE groups with replacement. Returns (point, lo, hi) of stat_fn(idx) (scalar or vector)."""
    groups = np.asarray(groups)
    point = np.atleast_1d(np.asarray(stat_fn(np.arange(len(groups))), dtype=float))
    if B <= 0 or len(groups) == 0:
        return point, point.copy(), point.copy()
    codes, _ = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(codes))]
    members = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    rng = np.random.default_rng(seed)
    n_groups = len(members)
    draws = np.empty((B, point.size))
    for b in range(B):
        pick = rng.integers(0, n_groups, size=n_groups)
        draws[b] = np.atleast_1d(np.asarray(stat_fn(np.concatenate([members[i] for i in pick])), dtype=float))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return point, lo, hi


def fmt(p, lo, hi, sign=False):
    f = "{:+.3f}" if sign else "{:.3f}"
    return f"{f.format(p)} [{f.format(lo)}, {f.format(hi)}]"


def spearman_safe(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 10 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])


def slope_safe(p, y):
    p = np.asarray(p, dtype=float)
    return float(np.polyfit(p, y, 1)[0]) if len(p) >= 10 and np.ptp(p) > 0 else float("nan")


def strata_table(title, err, sub, groups, codes, names, B, seed, results, key):
    """MAE of floor / full / light with bootstrap CIs, overall and per stratum, plus paired differences."""
    sub = np.asarray(sub, dtype=int)
    if len(sub) == 0:
        print(f"\n{title}: no rows")
        return
    n = len(names)
    c_sub = np.asarray(codes)[sub].astype(np.intp)
    assert c_sub.min() >= 0, "negative stratum code"
    g_sub = np.asarray(groups)[sub]
    e_floor, e_full, e_light = err["floor"][sub], err["full"][sub], err["light"][sub]
    arrays = (e_floor, e_full, e_light, e_full - e_floor, e_light - e_full)
    counts = np.r_[len(sub), np.bincount(c_sub, minlength=n)]

    def stat(idx):
        c = c_sub[idx]
        den = np.maximum(np.bincount(c, minlength=n), 1)
        out = []
        for v in arrays:
            vv = v[idx]
            out.append(np.r_[vv.mean(), np.bincount(c, weights=vv, minlength=n) / den])
        return np.concatenate(out)

    pt, lo, hi = (x.reshape(5, n + 1) for x in group_bootstrap(g_sub, stat, B, seed))
    print(f"\n{title} | {len(sub):,} rows, {len(set(g_sub)):,} chemical systems (95% CI: {B} bootstrap resamples of whole chemical systems)")
    print(f"  {'':8s} {'rows':>6s} {'floor':>7s}  {'full MAE [CI]':>24s}  {'light MAE [CI]':>24s}  {'full-floor [CI]':>27s}  {'light-full [CI]':>27s}")
    summary = {}
    for j, label in enumerate(["ALL"] + list(names)):
        if counts[j] == 0:
            continue
        print(f"  {label:8s} {counts[j]:6d} {pt[0, j]:7.3f}  {fmt(pt[1, j], lo[1, j], hi[1, j]):>24s}  "
              f"{fmt(pt[2, j], lo[2, j], hi[2, j]):>24s}  {fmt(pt[3, j], lo[3, j], hi[3, j], True):>27s}  "
              f"{fmt(pt[4, j], lo[4, j], hi[4, j], True):>27s}")
        summary[label] = {"rows": int(counts[j]), "floor": float(pt[0, j]),
                          "full": [float(pt[1, j]), float(lo[1, j]), float(hi[1, j])],
                          "light": [float(pt[2, j]), float(lo[2, j]), float(hi[2, j])],
                          "full_minus_floor": [float(pt[3, j]), float(lo[3, j]), float(hi[3, j])],
                          "light_minus_full": [float(pt[4, j]), float(lo[4, j]), float(hi[4, j])]}
    results[key] = summary


def run_grouped(c, X, a, results):
    print("\n==== 1. CHEMICAL-SYSTEM GROUPED CV (train on clean rows; test on all rows of held-out chemical systems) ====", flush=True)
    err = {k: np.zeros(len(c.y)) for k in ("floor", "full", "light")}
    pred = {k: np.zeros(len(c.y)) for k in ("full", "light")}
    t0 = time.time()
    for r in range(a.repeats):
        fold = assign_folds(c.groups, a.folds, a.seed + 100 + r)
        for f in range(a.folds):
            te = np.where(fold == f)[0]
            tr = np.where((fold != f) & c.clean)[0]
            err["floor"][te] += np.abs(c.y[te] - floor_predict(c.y, c.key, tr, te)) / a.repeats
            for name in ("full", "light"):
                p = make_regressor(a.max_iter).fit(X[name][tr], c.y[tr]).predict(X[name][te])
                pred[name][te] += p / a.repeats
                err[name][te] += np.abs(c.y[te] - p) / a.repeats
            print(f"  repeat {r + 1}/{a.repeats}, fold {f + 1}/{a.folds} ({time.time() - t0:.0f}s)", flush=True)
    clean_idx = np.where(c.clean)[0]
    class0 = np.where(c.clean & (c.cls == 0))[0]
    strata_table("Clean rows by adsorbate", err, clean_idx, c.groups, c.ad_codes, ADSORBATES, a.bootstrap, a.seed, results, "grouped_clean_by_adsorbate")
    strata_table("Clean class-0 rows by adsorbate", err, class0, c.groups, c.ad_codes, ADSORBATES, a.bootstrap, a.seed + 1, results, "grouped_clean_class0_by_adsorbate")
    strata_table("Clean rows by OC20 class", err, clean_idx, c.groups, c.cls.astype(int), ["class 0", "class 1", "class 2", "class 3"], a.bootstrap, a.seed + 2, results, "grouped_clean_by_class")

    print("\nAll rows by anomaly code (trained on clean rows only; point estimates, no resampling)")
    for code, name in ANOMALY_NAMES.items():
        m = c.anom == code
        if m.any():
            print(f"  {name:16s} n={int(m.sum()):6d}  floor {err['floor'][m].mean():.3f}  full {err['full'][m].mean():.3f}  light {err['light'][m].mean():.3f}")
    for ad in ("N2", "NH3"):
        for code, name in ANOMALY_NAMES.items():
            m = (c.ads == ad) & (c.anom == code)
            if m.sum() >= 10:
                print(f"  {ad:3s} {name:16s} n={int(m.sum()):5d}  floor {err['floor'][m].mean():.3f}  full {err['full'][m].mean():.3f}  light {err['light'][m].mean():.3f}")

    print("\nSpearman between out-of-fold predictions (full contract) and the DFT labels, within adsorbate")
    B_s = min(a.bootstrap, 300)
    results["grouped_spearman_full"] = {}
    for label, sub in (("clean rows", clean_idx), ("clean class-0 rows", class0)):
        pf, yy, codes = pred["full"][sub], c.y[sub], c.ad_codes[sub]

        def stat(idx, pf=pf, yy=yy, codes=codes):
            cc = codes[idx]
            return [spearman_safe(pf[idx][cc == j], yy[idx][cc == j]) for j in range(len(ADSORBATES))]

        pt, lo, hi = group_bootstrap(c.groups[sub], stat, B_s, a.seed + 7)
        print(f"  {label}: " + "  ".join(f"{ad} {fmt(pt[j], lo[j], hi[j])}" for j, ad in enumerate(ADSORBATES)))
        results["grouped_spearman_full"][label] = {ad: [float(pt[j]), float(lo[j]), float(hi[j])] for j, ad in enumerate(ADSORBATES)}


def run_official(c, X, a, results):
    print("\n==== 2. OFFICIAL OC20 SPLIT ====", flush=True)
    tr = np.where(c.clean & (c.split == "train"))[0]
    print(f"train: {len(tr):,} clean rows with split == 'train' (NH2 never appears here); val_is shares about 95% of its bulks with train, val_oos_bulk shares none")
    models = {name: make_regressor(a.max_iter).fit(X[name][tr], c.y[tr]) for name in ("full", "light")}
    for sp in ("val_is", "val_oos_bulk"):
        te = np.where(c.clean & (c.split == sp))[0]
        if len(te) == 0:
            print(f"{sp}: no clean rows")
            continue
        err = {k: np.zeros(len(c.y)) for k in ("floor", "full", "light")}
        err["floor"][te] = np.abs(c.y[te] - floor_predict(c.y, c.key, tr, te))
        for name, model in models.items():
            err[name][te] = np.abs(c.y[te] - model.predict(X[name][te]))
        strata_table(f"{sp}: clean rows by adsorbate", err, te, c.groups, c.ad_codes, ADSORBATES, a.bootstrap, a.seed, results, f"official_{sp}")
        strata_table(f"{sp}: clean class-0 rows by adsorbate", err, te[c.cls[te] == 0], c.groups, c.ad_codes, ADSORBATES, a.bootstrap, a.seed + 1, results, f"official_{sp}_class0")


def run_leave_element_out(c, X, a, results):
    print("\n==== 3. TRUE LEAVE-ELEMENT-OUT (every bulk containing the element is removed from training) ====", flush=True)
    els = [frozenset(_ELEMENT.findall(s)) for s in c.symbols]
    clean_idx = np.where(c.clean)[0]
    count = Counter(e for i in clean_idx for e in els[i])
    if a.elements == "auto":
        chosen = [e for e, n in count.most_common() if n >= a.min_rows][: a.max_elements]
    else:
        chosen = [e.strip() for e in a.elements.split(",") if count.get(e.strip(), 0) >= 30]
    if not chosen:
        print("no element with enough clean rows")
        return
    print(f"{len(chosen)} elements: {', '.join(chosen)}", flush=True)
    recs, t0 = [], time.time()
    for e in chosen:
        has = np.array([e in s for s in els])
        te, tr = clean_idx[has[clean_idx]], clean_idx[~has[clean_idx]]
        c0, mN = c.cls[te] == 0, c.ads[te] == "N"
        preds = {"floor": floor_predict(c.y, c.key, tr, te)}
        for name in ("full", "light"):
            preds[name] = make_regressor(a.max_iter).fit(X[name][tr], c.y[tr]).predict(X[name][te])
        rec = {"element": e, "rows": len(te), "rows_class0": int(c0.sum())}
        for name, p in preds.items():
            ae = np.abs(c.y[te] - p)
            rec[name] = float(ae.mean())
            rec[name + "_c0"] = float(ae[c0].mean()) if c0.any() else float("nan")
        rec["rho_N_full"] = spearman_safe(preds["full"][mN], c.y[te][mN]) if mN.sum() >= 30 else float("nan")
        recs.append(rec)
        print(f"  {e}: {len(te)} test rows ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(recs)
    print("\nper element (MAE in eV; _c0 = class 0 only; rho_N_full = Spearman within N* rows, full contract):")
    print(R.sort_values("rows", ascending=False).round(3).to_string(index=False))
    rng = np.random.default_rng(a.seed)

    def boot(v):
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        draws = np.array([rng.choice(v, size=len(v)).mean() for _ in range(max(a.bootstrap, 1))])
        return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    print(f"\nmean over {len(R)} elements (95% CI: bootstrap over ELEMENTS; the test sets overlap, so rows are not resampled)")
    summary = {}
    for label, suffix in (("all rows", ""), ("class 0", "_c0")):
        fl, fu, li = R["floor" + suffix], R["full" + suffix], R["light" + suffix]
        out = {"floor": float(fl.mean()), "full": boot(fu), "light": boot(li),
               "full_minus_floor": boot(fu - fl), "light_minus_full": boot(li - fu)}
        summary[label] = out
        print(f"  {label:9s} floor {out['floor']:.3f} | full {fmt(*out['full'])} | light {fmt(*out['light'])} | "
              f"full-floor {fmt(*out['full_minus_floor'], sign=True)} | light-full {fmt(*out['light_minus_full'], sign=True)}")
    print(f"  row-weighted full MAE {(R.full * R.rows).sum() / R.rows.sum():.3f} vs floor {(R.floor * R.rows).sum() / R.rows.sum():.3f}")
    results["leave_element_out"] = {"summary": summary, "per_element": recs}


def run_leave_adsorbate_out(c, X, a, results, held="NH2"):
    print(f"\n==== 4. LEAVE-ADSORBATE-OUT: {held} unseen (train: clean, split == 'train', no {held}; adsorbate described by nN and nH) ====", flush=True)
    tr = np.where(c.clean & (c.split == "train") & (c.ads != held))[0]
    mu = pd.Series(c.y[tr]).groupby(c.ads[tr]).mean()
    const = float((mu["NH"] + mu["NH3"]) / 2)
    models = {name: make_regressor(a.max_iter).fit(X[name][tr], c.y[tr]) for name in ("full", "light")}
    print(f"constant baseline = mean of the NH and NH3 training means = {const:.3f} eV")
    results["leave_adsorbate_out"] = {}
    for sp in ("val_oos_ads", "val_oos_ads_bulk"):
        te_all = np.where(c.clean & (c.split == sp) & (c.ads == held))[0]
        for label, te in (("all classes", te_all), ("class 0", te_all[c.cls[te_all] == 0])):
            if len(te) < 30:
                continue
            pf, pl, yy = models["full"].predict(X["full"][te]), models["light"].predict(X["light"][te]), c.y[te]

            def stat(idx, pf=pf, pl=pl, yy=yy):
                a_f, a_l, a_c = np.abs(yy[idx] - pf[idx]), np.abs(yy[idx] - pl[idx]), np.abs(yy[idx] - const)
                return [a_f.mean(), a_l.mean(), a_c.mean(), (a_f - a_c).mean(), (a_l - a_f).mean(),
                        spearman_safe(pf[idx], yy[idx]), spearman_safe(pl[idx], yy[idx]), slope_safe(pf[idx], yy[idx])]

            pt, lo, hi = group_bootstrap(c.groups[te], stat, min(a.bootstrap, 500), a.seed)
            print(f"\n  {sp} | {label} | {len(te)} rows, {len(set(c.groups[te]))} chemical systems")
            print(f"    MAE full {fmt(pt[0], lo[0], hi[0])} | light {fmt(pt[1], lo[1], hi[1])} | constant {pt[2]:.3f}")
            print(f"    full - constant {fmt(pt[3], lo[3], hi[3], True)} | light - full {fmt(pt[4], lo[4], hi[4], True)}")
            print(f"    Spearman full {fmt(pt[5], lo[5], hi[5])} | light {fmt(pt[6], lo[6], hi[6])} | slope of truth on prediction (full) {fmt(pt[7], lo[7], hi[7])}")
            results["leave_adsorbate_out"][f"{sp}|{label}"] = {
                "rows": int(len(te)), "mae_full": [float(pt[0]), float(lo[0]), float(hi[0])], "mae_constant": float(pt[2]),
                "full_minus_constant": [float(pt[3]), float(lo[3]), float(hi[3])],
                "spearman_full": [float(pt[5]), float(lo[5]), float(hi[5])], "slope_full": [float(pt[7]), float(lo[7]), float(hi[7])]}


def precision_at_recall(y_true, score, recall=0.8):
    prec, rec, _ = precision_recall_curve(y_true, score)
    ok = rec >= recall
    return float(prec[ok].max()) if ok.any() else float("nan")


def run_anomaly(c, X, a, results):
    print("\n==== 5. ANOMALY CLASSIFIER (all rows, chemical-system folds; metrics WITHIN each adsorbate, never pooled) ====", flush=True)
    fold = assign_folds(c.groups, a.folds, a.seed + 200)
    B_c = min(a.bootstrap, 200)
    results["anomaly"] = {}
    for code in (2, 1, 3):
        name = ANOMALY_NAMES[code]
        yb = (c.anom == code).astype(int)
        oof = np.zeros(len(yb))
        for f in range(a.folds):
            te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
            if yb[tr].sum() == 0:
                continue
            clf = HistGradientBoostingClassifier(max_iter=min(a.max_iter, 200), learning_rate=0.1, early_stopping=False, random_state=0)
            oof[te] = clf.fit(X["full"][tr], yb[tr]).predict_proba(X["full"][te])[:, 1]
        print(f"\n  {name}")
        for ad in ADSORBATES:
            m = np.where(c.ads == ad)[0]
            n_pos = int(yb[m].sum())
            if n_pos < 20 or n_pos == len(m):
                continue
            ym, om = yb[m], oof[m]

            def stat(idx, ym=ym, om=om):
                yy = ym[idx]
                return roc_auc_score(yy, om[idx]) if 0 < yy.sum() < len(yy) else float("nan")

            pt, lo, hi = group_bootstrap(c.groups[m], stat, B_c, a.seed + 11)
            note = "  (few positives)" if n_pos < 100 else ""
            print(f"    {ad:3s} positives {n_pos:5d} of {len(m):5d} (base rate {ym.mean():.3f}) | AUC {fmt(pt[0], lo[0], hi[0])} | "
                  f"AP {average_precision_score(ym, om):.3f} | precision at 80% recall {precision_at_recall(ym, om):.3f}{note}")
            results["anomaly"][f"{name}|{ad}"] = {"positives": n_pos, "rows": int(len(m)), "auc": [float(pt[0]), float(lo[0]), float(hi[0])]}


def load_table(path, registry):
    df = pd.read_parquet(path).reset_index(drop=True)
    missing = [col for col in REQUIRED if col not in df.columns]
    if missing:
        raise SystemExit(f"table lacks columns {missing}; rebuild it with scripts/build_features.py")
    S.validate_feature_frame(df, registry_path=registry)  # registry gate + numeric / NaN / inf checks on every row
    if not (df["is_clean_primary_regression_row"].to_numpy() == (df["anomaly"].to_numpy() == 0)).all():
        raise SystemExit("is_clean_primary_regression_row disagrees with anomaly == 0")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="results/adsorbates/oc20_adsorbates_features.parquet")
    ap.add_argument("--out", default="results/evaluation_screening.json")
    ap.add_argument("--mode", default="all", choices=["all", "grouped", "official", "leave-element-out", "leave-adsorbate-out", "anomaly"])
    ap.add_argument("--registry", default=str(S.DEFAULT_REGISTRY), help="registry CSV, or 'none' (tests only)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--min-rows", type=int, default=300)
    ap.add_argument("--max-elements", type=int, default=60)
    ap.add_argument("--elements", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    t0 = time.time()
    df = load_table(a.table, None if str(a.registry).lower() == "none" else a.registry)
    full = list(S.SCREENING_FEATURES)
    light = [col for col in full if col not in set(S.PLACEMENT_SENSITIVE_FEATURES)]
    X = {"full": df[full].to_numpy(dtype=float), "light": df[light].to_numpy(dtype=float)}
    ads = df["adsorbate"].astype(str).to_numpy()
    cls = df["class"].to_numpy()
    codes = pd.Categorical(ads, categories=ADSORBATES).codes.astype(np.intp)
    assert codes.min() >= 0, "unknown adsorbate label"
    c = SimpleNamespace(
        y=df["adsorption_energy_eV"].to_numpy(dtype=float), ads=ads, cls=cls, anom=df["anomaly"].to_numpy().astype(int),
        split=df["split"].astype(str).to_numpy(), groups=df["bulk_chemsys"].astype(str).to_numpy(),
        symbols=df["bulk_symbols"].astype(str).to_numpy(), ad_codes=codes,
        key=(pd.Series(ads) + "_" + pd.Series(cls).astype(str)).to_numpy())
    c.clean = c.anom == 0
    results = {"schema_version": S.SCHEMA_VERSION, "table": str(a.table), "rows": int(len(df)), "clean_rows": int(c.clean.sum()),
               "features_full": len(full), "features_light": len(light),
               "settings": {k: v for k, v in vars(a).items() if k not in ("table", "out", "registry")}}
    print(f"{len(df):,} rows | {int(c.clean.sum()):,} clean | {len(set(c.groups)):,} chemical systems | full {len(full)} features, light {len(light)}", flush=True)

    modes = ["grouped", "official", "leave-element-out", "leave-adsorbate-out", "anomaly"] if a.mode == "all" else [a.mode]
    runners = {"grouped": run_grouped, "official": run_official, "leave-element-out": run_leave_element_out,
               "leave-adsorbate-out": run_leave_adsorbate_out, "anomaly": run_anomaly}
    for mode in modes:
        runners[mode](c, X, a, results)
    out = a.out if hasattr(a.out, "write_text") else __import__("pathlib").Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(f"\nwrote {out} | done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

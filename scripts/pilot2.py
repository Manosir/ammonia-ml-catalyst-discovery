import ast, re, warnings
from collections import Counter
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
warnings.filterwarnings("ignore")

sp = pd.read_parquet("results/structure_pass/structure_pass.parquet")
sp = sp[sp.error.isna()].drop(columns=["adsorbate", "error"])
cols = ["system_id","adsorbate","bulk_mpid","bulk_symbols","miller_index","shift","class","top","anomaly","split","adsorption_energy_eV"]
tab = pd.read_parquet("results/adsorbates/oc20_adsorbates_features.parquet", columns=cols)
m = tab.merge(sp, on="system_id").reset_index(drop=True)
y = m.adsorption_energy_eV.values; ads = m.adsorbate.values; cls = m["class"].values
anom = m.anomaly.values; clean = anom == 0
print(f"rows {len(m)} | clean {int(clean.sum())}", flush=True)

def counts(s):
    d = {}
    for el, k in re.findall(r"([A-Z][a-z]?)(\d*)", str(s)): d[el] = d.get(el, 0) + (int(k) if k else 1)
    return d
comp = pd.DataFrame([counts(s) for s in m.bulk_symbols]).fillna(0)
comp = comp.div(comp.sum(axis=1), axis=0); comp.columns = ["frac_" + c for c in comp.columns]
chem = m.bulk_symbols.map(lambda s: "-".join(sorted(counts(s)))).values
def hkl(v):
    a = np.abs(np.asarray(ast.literal_eval(v) if isinstance(v, str) else v, dtype=float)); return a.max(), float((a ** 2).sum())
h = np.array([hkl(v) for v in m.miller_index])
base = pd.concat([comp, pd.DataFrame({"n_elements": (comp > 0).sum(axis=1), "hkl_max": h[:, 0], "hkl_norm2": h[:, 1], "top": m["top"].astype(float)}),
                  pd.get_dummies(m["class"], prefix="class").astype(float)], axis=1)
rows = []
for r in m.itertuples():                    # INITIAL-FRAME fields only
    d = np.asarray(r.near_dist, dtype=float); sy = np.asarray(r.near_symbols); cn = np.asarray(r.near_cn, dtype=float); tg = np.asarray(r.near_tags)
    k = d <= d[0] + 0.5
    f = {"site_cn_mean": cn[k].mean(), "site_tag1_frac": float((tg[k] == 1).mean()), "gap2": d[1] - d[0], "gap3": d[2] - d[0], "site_n_elem": float(len(set(sy[k])))}
    for e, nn in Counter(sy[k]).items(): f["siteel_" + e] = nn / k.sum()
    rows.append(f)
S = pd.DataFrame(rows); SITEEL = [c for c in S.columns if c.startswith("siteel_")]; S[SITEEL] = S[SITEEL].fillna(0)
GEO = ["d_min_ads_slab_initial","h_nn1_initial","h_k3_initial","h_site_initial","h_top_initial","h_plane_initial","min_dist_all_initial"]
TOPO = ["n_site_atoms","gap2","gap3","site_n_elem"]
CN = ["cn_mean_slab","gcn_site_initial","site_cn_mean","site_tag1_frac"]
F = pd.concat([m[GEO + ["n_site_atoms","cn_mean_slab","gcn_site_initial"]], S], axis=1)
A = pd.concat([base, pd.get_dummies(m.adsorbate, prefix="ads").astype(float)], axis=1)
sets = {"A mapping-only": A,
        "A + geometry (heights, distances)": pd.concat([A, F[GEO]], axis=1),
        "A + topology (gaps, n site atoms)": pd.concat([A, F[TOPO]], axis=1),
        "A + coordination numbers": pd.concat([A, F[CN]], axis=1),
        "A + site element fractions": pd.concat([A, F[SITEEL]], axis=1),
        "B all site features": pd.concat([A, F], axis=1)}
def hgb(): return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, early_stopping=False, random_state=0)

# 1. ablation
idc = np.where(clean)[0]; yc = y[idc]; gc = chem[idc]; clsc = cls[idc]
spl = list(GroupShuffleSplit(5, test_size=0.2, random_state=0).split(np.zeros(len(idc)), yc, gc))
print("\n1. feature-group ablation (clean rows, chemical-system groups, 5 splits)", flush=True)
res = {}
for name, X in sets.items():
    Xv = X.values.astype(float)[idc]; r = []
    for tr, te in spl:
        e = np.abs(yc[te] - hgb().fit(Xv[tr], yc[tr]).predict(Xv[te]))
        r.append((e.mean(), e[clsc[te] == 0].mean()))
    res[name] = np.array(r); d = res[name][:, 0] - res["A mapping-only"][:, 0]
    print(f"  {name:36s} {Xv.shape[1]:4d} feats  MAE {res[name][:,0].mean():.3f} | class 0 {res[name][:,1].mean():.3f} | vs A {d.mean():+.3f} (better in {(d < 0).sum()}/5)", flush=True)

# 2. model vs within-surface oracle on the same rows
print("\n2. model vs within-surface oracle on the same rows", flush=True)
XB = sets["B all site features"].values.astype(float)[idc]; oof = np.zeros(len(idc))
for tr, te in GroupKFold(5).split(XB, yc, gc): oof[te] = hgb().fit(XB[tr], yc[tr]).predict(XB[te])
keys = ["bulk", "ms", "sh", "top", "ads"]
t = pd.DataFrame({"y": yc, "oof": oof, "bulk": m.bulk_mpid.values[idc], "ms": m.miller_index.astype(str).values[idc],
                  "sh": m["shift"].astype(float).round(3).values[idc], "top": m["top"].values[idc], "ads": ads[idc], "cls": clsc,
                  "site": [",".join(f"{v:.1f}" for v in b) for b in m.bind_pos_initial.values[idc]]})
g = t.groupby(keys)["y"]; n = g.transform("size"); s = g.transform("sum"); sel = n >= 2
mm = t[sel].copy(); mm["oracle"] = (s[sel] - mm.y) / (n[sel] - 1)
ns = t.groupby(keys)["site"].transform("nunique")[sel]
print(f"  rows in multi-row groups: {len(mm)} | share in pair groups: {(n[sel] == 2).mean():.2f} | share with all sites distinct: {(ns == n[sel]).mean():.2f}")
for label, d_ in [("all", mm), ("class 0", mm[mm.cls == 0])]:
    print(f"  {label:8s} model (out-of-fold) MAE {np.abs(d_.y - d_.oof).mean():.3f} | oracle (other sites' mean) MAE {np.abs(d_.y - d_.oracle).mean():.3f}", flush=True)

# 3. NH2 as an unseen adsorbate
print("\n3. NH2 as an unseen adsorbate (train without NH2)", flush=True)
nN = m.adsorbate.map({"N": 1, "N2": 2, "NH": 1, "NH2": 1, "NH3": 1}).astype(float)
nH = m.adsorbate.map({"N": 0, "N2": 0, "NH": 1, "NH2": 2, "NH3": 3}).astype(float)
Xd = pd.concat([base, pd.DataFrame({"ads_nN": nN, "ads_nH": nH}), F], axis=1).values.astype(float)
tr_all = np.where(clean & (ads != "NH2"))[0]; tr_N = np.where(clean & (ads == "N"))[0]
mods = {"four-adsorbate model (nN, nH inputs)": hgb().fit(Xd[tr_all], y[tr_all]), "N-only model (scaling-style proxy)": hgb().fit(Xd[tr_N], y[tr_N])}
for spn in ["val_oos_ads", "val_oos_ads_bulk"]:
    te = np.where(clean & (ads == "NH2") & (m["split"].values == spn))[0]; te0 = te[cls[te] == 0]
    print(f"  {spn}: {len(te)} clean NH2 rows ({len(te0)} class 0)")
    for name, md in mods.items():
        p = md.predict(Xd[te]); p0 = md.predict(Xd[te0])
        print(f"    {name:38s} Spearman {spearmanr(p, y[te])[0]:.3f} (class 0 {spearmanr(p0, y[te0])[0]:.3f}) | slope of truth on prediction {np.polyfit(p, y[te], 1)[0]:.2f} | MAE {np.abs(y[te] - p).mean():.3f}", flush=True)

# 4. anomaly classifier on all rows (within-adsorbate AUC is the honest number)
print("\n4. anomaly classifier, all rows, chemical-system groups, 5 folds", flush=True)
XA = sets["B all site features"].values.astype(float)
folds = list(GroupKFold(5).split(XA, anom, chem))
for name, code in [("desorbed", 2), ("dissociated", 1), ("reconstructed", 3)]:
    yb = (anom == code).astype(int); oof = np.zeros(len(yb))
    for tr, te in folds:
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, early_stopping=False, random_state=0).fit(XA[tr], yb[tr])
        oof[te] = clf.predict_proba(XA[te])[:, 1]
    out = [f"{name:13s} base rate {yb.mean():.3f}, AUC {roc_auc_score(yb, oof):.3f} | per adsorbate (rate/AUC):"]
    for a in ["N", "N2", "NH", "NH2", "NH3"]:
        k = ads == a
        if 0 < yb[k].sum() < k.sum(): out.append(f"{a} {yb[k].mean():.3f}/{roc_auc_score(yb[k], oof[k]):.3f}")
    print("  " + "  ".join(out), flush=True)
print("done")

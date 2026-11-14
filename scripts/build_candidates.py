#!/usr/bin/env python3
"""Candidate screening features from the structure pass (INITIAL-frame fields only) + mapping.

Reads results/structure_pass/structure_pass.parquet and the model table; writes
results/structure_pass/candidate_features.parquet. No last-frame field is used.
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import atomic_numbers, covalent_radii

PAULING = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83,
    "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55,
    "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Tc": 1.90,
    "Ru": 2.20, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05,
    "Te": 2.10, "I": 2.66, "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Hf": 1.30, "Ta": 1.50, "W": 2.36,
    "Re": 1.90, "Os": 2.20, "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33,
    "Bi": 2.02,
}
NONMETALS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I", "As"}
PROPS = ("rcov", "en", "nd", "nval", "period", "group")
FORBIDDEN = ("final", "disp", "frames", "force", "e_first", "e_last", "bind_disp")
_cache = {}


def period_group(z):
    """(period, group) from the atomic number; lanthanides are treated as group 3."""
    if z <= 2:
        return 1, (1 if z == 1 else 18)
    if z <= 10:
        return 2, (z - 2 if z <= 4 else z + 8)
    if z <= 18:
        return 3, (z - 10 if z <= 12 else z)
    if z <= 36:
        return 4, z - 18
    if z <= 54:
        return 5, z - 36
    if z <= 56:
        return 6, z - 54
    if z <= 71:
        return 6, 3
    if z <= 86:
        return 6, z - 68
    return 7, float("nan")


def props(el):
    if el not in _cache:
        z = atomic_numbers[el]
        per, grp = period_group(z)
        tm = 3 <= grp <= 12
        _cache[el] = dict(
            rcov=float(covalent_radii[z]), en=PAULING.get(el, np.nan),
            nd=float(grp - 2) if tm else 0.0, nval=float(grp if grp <= 12 else grp - 10),
            period=float(per), group=float(grp), tm=bool(tm), nm=el in NONMETALS)
    return _cache[el]


def parse_formula(s):
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", str(s)):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def parse_comp(s):
    out = {}
    for part in str(s).split(";"):
        if ":" in part:
            el, n = part.split(":")
            out[el] = out.get(el, 0) + int(n)
    return out


def agg(prefix, comp):
    """Property statistics over a composition {element: weight} (weights = counts)."""
    if not comp:
        return {}
    els = list(comp)
    w = np.array([comp[e] for e in els], dtype=float)
    w = w / w.sum()
    out = {}
    for p in PROPS:
        v = np.array([props(e)[p] for e in els], dtype=float)
        ok = ~np.isnan(v)
        if ok.any():
            out[f"{prefix}_{p}_mean"] = float((v[ok] * w[ok]).sum() / w[ok].sum())
            out[f"{prefix}_{p}_min"] = float(v[ok].min())
            out[f"{prefix}_{p}_max"] = float(v[ok].max())
            out[f"{prefix}_{p}_rng"] = float(v[ok].max() - v[ok].min())
    out[f"{prefix}_tm_frac"] = float(sum(wi for e, wi in zip(els, w) if props(e)["tm"]))
    out[f"{prefix}_nm_frac"] = float(sum(wi for e, wi in zip(els, w) if props(e)["nm"]))
    out[f"{prefix}_n_elem"] = float(len(els))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structure", type=Path, default=Path("results/structure_pass/structure_pass.parquet"))
    ap.add_argument("--table", type=Path, default=Path("results/adsorbates/oc20_adsorbates_features.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/structure_pass/candidate_features.parquet"))
    a = ap.parse_args()

    sp = pd.read_parquet(a.structure)
    sp = sp[sp.error.isna()]
    tab = pd.read_parquet(a.table, columns=["system_id", "bulk_symbols"])
    m = sp.merge(tab, on="system_id")
    print(f"{len(m):,} rows", flush=True)

    rows, missing_en = [], Counter()
    for r in m.itertuples(index=False):
        d = np.asarray(r.near_dist, dtype=float)
        sy = np.asarray(r.near_symbols, dtype=object)
        cn = np.asarray(r.near_cn, dtype=float)
        tg = np.asarray(r.near_tags, dtype=float)
        k = d <= d[0] + 0.5
        row = {"system_id": r.system_id}
        row.update(agg("bulk", parse_formula(r.bulk_symbols)))
        row.update(agg("surf", parse_comp(r.surface_comp)))
        row.update(agg("site", Counter(sy[k])))
        for e in set(sy[k]):
            if e not in PAULING:
                missing_en[e] += 1
        row.update({
            "geo_d_min": r.d_min_ads_slab_initial, "geo_h_nn1": r.h_nn1_initial, "geo_h_k3": r.h_k3_initial,
            "geo_h_site": r.h_site_initial, "geo_h_top": r.h_top_initial, "geo_h_plane": r.h_plane_initial,
            "geo_min_dist_all": r.min_dist_all_initial,
            "topo_n_site_atoms": float(k.sum()), "topo_gap2": d[1] - d[0], "topo_gap3": d[2] - d[0],
            "cn_slab_mean": r.cn_mean_slab, "cn_gcn_site": r.gcn_site_initial,
            "cn_site_mean": float(cn[k].mean()), "cn_site_max": float(cn[k].max()),
            "cn_site_tag1_frac": float((tg[k] == 1).mean()),
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c != "system_id"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    bad = [c for c in df.columns if any(s in c for s in FORBIDDEN)]
    assert not bad, f"audit-only names in candidate features: {bad}"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(df):,} rows, {len(num)} feature columns")
    for p in ("bulk_", "surf_", "site_", "geo_", "topo_", "cn_"):
        print(f"  {p:6s} {sum(c.startswith(p) for c in num):3d} columns")
    nan_share = df[num].isna().mean().sort_values(ascending=False)
    print("columns with NaN (share):", {k: round(float(v), 3) for k, v in nan_share[nan_share > 0].head(10).items()})
    print("site-atom elements missing from the electronegativity table (rows):", dict(missing_en) or "none")
    const = [c for c in num if df[c].nunique(dropna=True) <= 1]
    print("constant columns:", const or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

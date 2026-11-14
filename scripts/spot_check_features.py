#!/usr/bin/env python3
"""Independent spot-check of the contract features.

Recomputes the 60 candidate columns of the model table from the FIRST frame of random trajectories with a
different code path (explicit enumeration of periodic images instead of ASE's neighbour list / find_mic) and,
when pymatgen is installed, element data from pymatgen instead of the typed table.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import lzma
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import atomic_numbers, covalent_radii
from ase.io import read

from ammonia_ai import schema as S

try:
    from pymatgen.core import Element as PMGElement
except Exception:  # optional
    PMGElement = None

_spec = importlib.util.spec_from_file_location("bc", Path(__file__).with_name("build_candidates.py"))
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)

NONMETALS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I", "As"}
K, CN_FACTOR, SITE_WINDOW = 8, 1.2, 0.5
_cache: dict = {}


def element_props(el):
    if el in _cache:
        return _cache[el]
    z = atomic_numbers[el]
    if PMGElement is not None:
        e = PMGElement(el)
        try:
            en = float(e.X)
        except Exception:
            en = float("nan")
        grp = 3 if 57 <= z <= 71 else int(e.group)
        per = 6 if 57 <= z <= 71 else int(e.row)
    else:
        p = bc.props(el)
        en, grp, per = p["en"], p["group"], p["period"]
    _cache[el] = {"en": en, "group": float(grp), "period": float(per)}
    return _cache[el]


def counts(s):
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", str(s)):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def shift_grid(cell, pbc, rcut):
    inv = np.linalg.inv(cell)
    widths = 1.0 / np.linalg.norm(inv, axis=0)  # distance between lattice planes along each cell axis
    n = [int(np.ceil(rcut / w)) + 1 if p else 0 for w, p in zip(widths, pbc)]
    ranges = [np.arange(-k, k + 1) for k in n]
    return np.array(np.meshgrid(*ranges, indexing="ij")).reshape(3, -1).T.astype(float)


def image_vectors(ref, others, cell, pbc, rcut):
    """Vectors from ref (3,) to every periodic image of every atom in others: shape (M, S, 3)."""
    inv = np.linalg.inv(cell)
    f = (others - ref) @ inv
    f = f - np.round(f) * np.asarray(pbc, dtype=float)  # wrap periodic axes only
    return (f[:, None, :] + shift_grid(cell, pbc, rcut)[None, :, :]) @ cell


def coordination(pos, numbers, cell, pbc):
    """Number of (atom, image) pairs closer than 1.2 * (r_i + r_j); self at zero shift excluded."""
    r = covalent_radii[numbers] * CN_FACTOR
    rcut = 2.0 * float(r.max())
    cn = np.zeros(len(pos))
    for i in range(len(pos)):
        dist = np.linalg.norm(image_vectors(pos[i], pos, cell, pbc, rcut), axis=2)
        cn[i] = int(((dist < (r[i] + r)[:, None]) & (dist > 1e-8)).sum())
    return cn


def min_pair_distance(pos, cell, pbc):
    best = np.inf
    for i in range(len(pos) - 1):
        best = min(best, float(np.linalg.norm(image_vectors(pos[i], pos[i + 1:], cell, pbc, 6.0), axis=2).min()))
    return best


def aggregates(prefix, comp):
    els = list(comp)
    w = np.array([comp[e] for e in els], dtype=float)
    w /= w.sum()
    out = {}
    for fam in S.PROPERTY_FAMILIES:
        v = np.array([element_props(e)[fam] for e in els], dtype=float)
        ok = ~np.isnan(v)
        if ok.any():
            out[f"{prefix}_{fam}_mean"] = float((v[ok] * w[ok]).sum() / w[ok].sum())
            out[f"{prefix}_{fam}_min"] = float(v[ok].min())
            out[f"{prefix}_{fam}_max"] = float(v[ok].max())
            out[f"{prefix}_{fam}_rng"] = float(v[ok].max() - v[ok].min())
    groups = np.array([element_props(e)["group"] for e in els])
    out[f"{prefix}_tm_frac"] = float(w[(groups >= 3) & (groups <= 12)].sum())
    out[f"{prefix}_nm_frac"] = float(sum(wi for e, wi in zip(els, w) if e in NONMETALS))
    out[f"{prefix}_n_elem"] = float(len(els))
    return out


def expected_row(atoms, bulk_symbols):
    cell, pbc = np.array(atoms.cell), np.array(atoms.pbc, dtype=bool)
    sym, tags, pos = np.array(atoms.get_chemical_symbols()), atoms.get_tags(), atoms.positions
    slab, surf, ads = np.where(tags < 2)[0], np.where(tags == 1)[0], np.where(tags == 2)[0]
    idx = np.arange(len(slab))
    near = {}
    for a in ads:
        vec = image_vectors(pos[a], pos[slab], cell, pbc, 8.0)
        dist = np.linalg.norm(vec, axis=2)
        best = dist.argmin(axis=1)
        near[a] = (dist[idx, best], vec[idx, best])
    cand = [a for a in ads if sym[a] == "N"] or list(ads)
    bind = min(cand, key=lambda a: near[a][0].min())
    d, v = near[bind]
    order = np.argsort(d, kind="stable")[:K]
    dd, dz = d[order], v[order][:, 2]
    site = dd <= dd[0] + SITE_WINDOW
    cn = coordination(pos[slab], atoms.numbers[slab], cell, pbc)
    cn_o = cn[order]
    exp = {
        "geo_d_min": dd[0], "geo_h_nn1": -dz[0], "geo_h_k3": -dz[:3].mean(), "geo_h_site": -dz[site].mean(),
        "geo_h_top": pos[bind, 2] - pos[slab, 2].max(),
        "geo_h_plane": pos[bind, 2] - pos[surf, 2].mean() if len(surf) else np.nan,
        "geo_min_dist_all": min_pair_distance(pos, cell, pbc),
        "topo_n_site_atoms": float(site.sum()), "topo_gap2": dd[1] - dd[0], "topo_gap3": dd[2] - dd[0],
        "cn_slab_mean": cn.mean(), "cn_gcn_site": cn_o[site].sum() / 12.0, "cn_site_mean": cn_o[site].mean(),
        "cn_site_max": cn_o[site].max(), "cn_site_tag1_frac": float((tags[slab][order][site] == 1).mean()),
    }
    exp.update(aggregates("bulk", counts(bulk_symbols)))
    exp.update(aggregates("surf", Counter(sym[surf])))
    exp.update(aggregates("site", Counter(sym[slab][order][site])))
    return exp, Counter(sym[surf])


def first_frame(path):
    with lzma.open(path, "rt") as handle:
        text = handle.read()
    return read(io.StringIO(text), index=0, format="extxyz")


def table_check():
    if PMGElement is None:
        print("pymatgen not installed: the element table is NOT cross-checked (python3 -m pip install pymatgen)")
        return
    bad = []
    for el, x in sorted(bc.PAULING.items()):
        z = atomic_numbers[el]
        e = PMGElement(el)
        try:
            ref = float(e.X)
        except Exception:
            ref = float("nan")
        if not abs(ref - x) <= 0.011:
            bad.append(f"  {el}: typed electronegativity {x} vs pymatgen {ref:.2f}")
        if not 57 <= z <= 71:
            per, grp = bc.period_group(z)
            if per != int(e.row) or grp != int(e.group):
                bad.append(f"  {el}: typed (period {per}, group {grp}) vs pymatgen (period {int(e.row)}, group {int(e.group)})")
    print(f"element table vs pymatgen: {len(bc.PAULING)} elements checked (lanthanides excluded: different convention), {len(bad)} differences")
    for line in bad:
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="results/adsorbates/oc20_adsorbates_features.parquet")
    ap.add_argument("--structure", default="results/structure_pass/structure_pass.parquet")
    ap.add_argument("--per-adsorbate", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"element data source: {'pymatgen' if PMGElement else 'typed table (NOT independent)'}")
    table_check()

    cols = list(S.CANDIDATE_COLUMNS)
    tab = pd.read_parquet(a.table, columns=["system_id", "adsorbate", "bulk_symbols", "source_path"] + cols)
    sp = pd.read_parquet(a.structure, columns=["system_id", "error", "cell_a", "cell_b", "n_atoms", "surface_comp"])
    t = tab.merge(sp[sp.error.isna()].drop(columns="error"), on="system_id")
    picks = pd.concat([g.sample(a.per_adsorbate, random_state=a.seed) for _, g in t.groupby("adsorbate")])
    extra = pd.concat([t.nsmallest(1, "cell_a"), t.nsmallest(1, "cell_b"), t.nlargest(1, "n_atoms")])
    picks = pd.concat([picks, extra]).drop_duplicates("system_id")

    worst = {c: 0.0 for c in cols}
    bad = []
    for r in picks.itertuples(index=False):
        exp, surf_counter = expected_row(first_frame(r.source_path), r.bulk_symbols)
        surf_str = ";".join(f"{e}:{n}" for e, n in sorted(surf_counter.items()))
        if surf_str != r.surface_comp:
            bad.append((r.system_id, "surface_comp", surf_str, r.surface_comp))
        for c in cols:
            e_val, a_val = float(exp.get(c, np.nan)), float(getattr(r, c))
            if np.isnan(e_val) and np.isnan(a_val):
                continue
            diff = np.inf if (np.isnan(e_val) or np.isnan(a_val)) else abs(e_val - a_val)
            worst[c] = max(worst[c], diff)
            if not diff <= 5e-4 + 1e-4 * abs(e_val):
                bad.append((r.system_id, c, e_val, a_val))
        print(f"  checked {r.system_id} ({r.adsorbate})", flush=True)

    print(f"\n{len(picks)} systems ({', '.join(f'{k} {v}' for k, v in picks.adsorbate.value_counts().items())}), {len(cols)} columns each")
    by_group = {}
    for c in cols:
        by_group.setdefault(c.split("_")[0], []).append(worst[c])
    for g, v in by_group.items():
        print(f"  {g:5s}: {len(v)} columns, worst absolute difference {max(v):.2e}")
    if bad:
        print(f"\nMISMATCHES: {len(bad)}")
        for b in bad[:30]:
            print("  ", b)
        return 1
    print("\nSPOT-CHECK PASSED: every column matches the independent recomputation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

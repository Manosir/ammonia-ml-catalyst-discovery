#!/usr/bin/env python3
"""One pass over every OC20 trajectory: initial-frame descriptors plus last-frame audit.

Resumable (JSONL shards). Run with --smoke-only first.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import io
import json
import lzma
import signal
import sys
import time
import traceback
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

K = 8              # nearest slab atoms stored per record
CN_FACTOR = 1.2    # neighbour cutoff = 1.2 * (rcov_i + rcov_j)
SITE_WINDOW = 0.5  # site atoms: within d_min + 0.5 A of the binding atom
TIMEOUT = 240      # seconds per file


def _f(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def _fl(values, nd=4):
    return [_f(v, nd) for v in values]


def _on_alarm(signum, frame):
    raise TimeoutError("file timed out")


def _init_worker():
    os.nice(10)
    signal.signal(signal.SIGALRM, _on_alarm)


def analyse(path):
    from ase.data import covalent_radii
    from ase.geometry import find_mic, get_distances
    from ase.io import read
    from ase.neighborlist import neighbor_list

    with lzma.open(path, "rt") as handle:
        text = handle.read()
    n_frames = text.count("Lattice=")
    first = read(io.StringIO(text), index=0, format="extxyz")
    last = read(io.StringIO(text), index=-1, format="extxyz")
    if len(first) != len(last):
        raise ValueError("first and last frame differ in atom count")
    cell = np.array(first.cell)
    pbc = np.array(first.pbc)
    if abs(np.linalg.det(cell)) < 1e-6 or not pbc.any():
        raise ValueError("missing cell or pbc")
    sym = np.array(first.get_chemical_symbols())
    tags = first.get_tags()
    pos, pos_l = first.positions, last.positions
    slab = np.where(tags < 2)[0]
    ads = np.where(tags == 2)[0]
    surf = np.where(tags == 1)[0]
    if len(slab) == 0 or len(ads) == 0:
        raise ValueError("no slab or no adsorbate atoms")

    # binding atom = the adsorbate N closest to the slab (initial frame, minimum image)
    D, d = get_distances(pos[ads], pos[slab], cell=cell, pbc=pbc)
    cand = np.where(sym[ads] == "N")[0]
    if len(cand) == 0:
        cand = np.arange(len(ads))
    b = int(cand[np.argmin(d[cand].min(axis=1))])
    bind = ads[b]
    order = np.argsort(d[b])[:K]
    dist = d[b][order]
    dz = D[b][order][:, 2]            # vector adsorbate -> slab atom, so height = -dz
    site = dist <= dist[0] + SITE_WINDOW

    # slab coordination numbers with the real cell
    slab_atoms = first[slab]
    rad = list(covalent_radii[slab_atoms.numbers] * CN_FACTOR)
    i_idx, _ = neighbor_list("ij", slab_atoms, rad)
    cn = np.bincount(i_idx, minlength=len(slab)).astype(float)
    pairs_per_atom = len(i_idx) / len(slab)
    if pairs_per_atom > 60:
        raise ValueError(f"neighbour list inflated: {pairs_per_atom:.0f} pairs/atom")

    dall = first.get_all_distances(mic=True)
    np.fill_diagonal(dall, np.inf)

    def comp(idx):
        return ";".join(f"{e}:{n}" for e, n in sorted(Counter(sym[idx]).items()))

    # last-frame audit
    m = tags < 2
    dv, _ = find_mic(pos_l[m] - pos[m], cell, pbc)
    dn = np.linalg.norm(dv, axis=1)
    if len(surf):
        dv1, _ = find_mic(pos_l[surf] - pos[surf], cell, pbc)
        dn1 = np.linalg.norm(dv1, axis=1)
    else:
        dn1 = np.array([np.nan])
    vb, _ = find_mic((pos_l[bind] - pos[bind])[None, :], cell, pbc)
    Dl, dl = get_distances(pos_l[bind][None, :], pos_l[slab], cell=cell, pbc=pbc)
    ol = np.argsort(dl[0])[:3]

    def safe(fn):
        try:
            return fn()
        except Exception:
            return None

    forces = safe(lambda: last.get_forces())
    max_force = None
    if forces is not None:
        free = tags > 0
        max_force = float(np.linalg.norm(forces[free], axis=1).max())

    return {
        "n_atoms": len(first), "n_slab": len(slab), "n_surface": len(surf), "n_ads": len(ads),
        "ads_symbols": "".join(sym[ads]), "bind_symbol": str(sym[bind]),
        "bind_pos_initial": _fl(pos[bind]),
        "cell_a": _f(np.linalg.norm(cell[0])), "cell_b": _f(np.linalg.norm(cell[1])), "cell_c": _f(np.linalg.norm(cell[2])),
        "d_min_ads_slab_initial": _f(dist[0]),
        "h_nn1_initial": _f(-dz[0]), "h_k3_initial": _f(-np.mean(dz[:3])), "h_site_initial": _f(-np.mean(dz[site])),
        "n_site_atoms": int(site.sum()),
        "h_top_initial": _f(pos[bind, 2] - pos[slab, 2].max()),
        "h_plane_initial": _f(pos[bind, 2] - pos[surf, 2].mean()) if len(surf) else None,
        "min_dist_all_initial": _f(dall.min()),
        "cn_mean_slab": _f(cn.mean()), "pairs_per_atom_slab": _f(pairs_per_atom),
        "gcn_site_initial": _f(cn[order][site].sum() / 12.0),
        "near_symbols": sym[slab][order].tolist(), "near_tags": tags[slab][order].tolist(),
        "near_cn": cn[order].astype(int).tolist(), "near_dist": _fl(dist), "near_dz": _fl(dz),
        "slab_comp": comp(slab), "surface_comp": comp(surf),
        "n_frames": n_frames,
        "slab_disp_max": _f(dn.max()), "slab_disp_rms": _f(np.sqrt((dn ** 2).mean())),
        "slab_n_disp_gt_0p5": int((dn > 0.5).sum()),
        "surf_disp_max": _f(np.nanmax(dn1)), "surf_n_disp_gt_0p5": int((dn1 > 0.5).sum()),
        "bind_disp_total": _f(np.linalg.norm(vb[0])), "bind_disp_z": _f(vb[0][2]),
        "d_min_ads_slab_final": _f(dl[0].min()),
        "h_k3_final": _f(-np.mean(Dl[0][ol][:, 2])),
        "h_top_final": _f(pos_l[bind, 2] - pos_l[slab, 2].max()),
        "max_force_final": _f(max_force),
        "e_first": _f(safe(lambda: first.get_potential_energy())),
        "e_last": _f(safe(lambda: last.get_potential_energy())),
    }


def process(task):
    system_id, adsorbate, path = task
    t0 = time.time()
    try:
        signal.alarm(TIMEOUT)
        rec = analyse(path)
        rec["error"] = None
    except Exception as exc:
        rec = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        signal.alarm(0)
    rec.update(system_id=system_id, adsorbate=adsorbate, seconds=round(time.time() - t0, 3))
    return rec


def write_shard(out, name, records):
    path = out / "shards" / f"{name}.jsonl"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as handle:
        for r in records:
            handle.write(json.dumps(r) + "\n")
    tmp.replace(path)


def done_ids(out):
    done = set()
    for p in (out / "shards").glob("*.jsonl"):
        with p.open() as handle:
            for line in handle:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                err = r.get("error")
                if err is None or not str(err).startswith("TimeoutError"):   # timeouts are retried
                    done.add(r["system_id"])
    return done


def merge(out):
    rows = {}
    for p in sorted((out / "shards").glob("*.jsonl")):
        with p.open() as handle:
            for line in handle:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                rows[r["system_id"]] = r          # later shards win
    df = pd.DataFrame(list(rows.values()))
    df.to_parquet(out / "structure_pass.parquet", index=False)
    return df


def summarize(df, table):
    ok = df[df.error.isna()]
    print(f"\nrecords {len(df):,} | ok {len(ok):,} | errors {len(df) - len(ok):,}")
    if len(df) > len(ok):
        print(df.error.dropna().str.slice(0, 90).value_counts().head(6).to_string())
    if len(ok) == 0:
        return
    s = df.seconds
    print(f"seconds per file: median {s.median():.2f}  p95 {s.quantile(.95):.2f}  max {s.max():.1f}")
    g = ok.groupby("adsorbate")
    print(pd.DataFrame({
        "n": g.size(), "h_k3_initial mean": g.h_k3_initial.mean(), "sd": g.h_k3_initial.std(),
        "share<0": g.h_k3_initial.apply(lambda x: (x < 0).mean()),
        "d_min mean": g.d_min_ads_slab_initial.mean(),
        "final h_k3 share<0": g.h_k3_final.apply(lambda x: (x < 0).mean()),
        "median n_frames": g.n_frames.median(),
    }).round(3).to_string())
    t = pd.read_parquet(table, columns=["system_id", "anomaly", "adsorption_energy_eV", "ads_height_final_audit"])
    m = ok.merge(t, on="system_id")
    print(f"\nold final-frame ads_height < 0: {(m.ads_height_final_audit < 0).mean():.3f} | "
          f"MIC-based final h_k3 < 0: {(m.h_k3_final < 0).mean():.3f}")
    print("median slab displacement and n_frames by anomaly code:")
    print(m.groupby("anomaly")[["slab_disp_max", "n_frames"]].median().round(2).to_string())
    lo = m.adsorption_energy_eV < -5
    print(f"E_ads < -5 (n={int(lo.sum())}): median slab_disp_max {m[lo].slab_disp_max.median():.2f} "
          f"vs {m[~lo].slab_disp_max.median():.2f} for the rest")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=Path("results/adsorbates/oc20_adsorbates_features.parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/structure_pass"))
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-hours", type=float, default=4.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    (args.out / "shards").mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        summarize(merge(args.out), args.table)
        return 0

    tab = pd.read_parquet(args.table, columns=["system_id", "adsorbate", "source_path"])
    exists = tab.source_path.map(lambda p: Path(p).exists())
    print(f"{len(tab):,} records in table; {int(exists.sum()):,} source files found", flush=True)
    if exists.mean() < 0.99:
        print("ERROR: source files not found; run from the repository root. Example missing path:",
              tab.source_path[~exists].iloc[0])
        return 2
    done = done_ids(args.out)
    todo = tab[exists & ~tab.system_id.isin(done)].sample(frac=1, random_state=0)
    if args.limit:
        todo = todo.head(args.limit)
    tasks = [(r.system_id, r.adsorbate, r.source_path) for r in todo.itertuples()]
    print(f"already done {len(done):,} | to do {len(tasks):,} | workers {args.workers}", flush=True)
    stamp = int(time.time())

    if not args.skip_smoke and tasks:
        signal.signal(signal.SIGALRM, _on_alarm)
        probe, tasks = tasks[:24], tasks[24:]
        recs = []
        for t in probe:
            r = process(t)
            recs.append(r)
            print(f"  smoke {r['system_id']:>16s} {r['adsorbate']:>4s} {r['seconds']:6.2f}s  {r['error'] or 'ok'}", flush=True)
        write_shard(args.out, f"shard_{stamp}_smoke", recs)
        ok = [r for r in recs if r["error"] is None]
        problems = []
        if len(ok) < 0.9 * len(recs):
            problems.append(f"only {len(ok)}/{len(recs)} smoke files processed")
        if ok:
            dmin = float(np.median([r["d_min_ads_slab_initial"] for r in ok]))
            cnm = float(np.median([r["cn_mean_slab"] for r in ok]))
            if not 1.0 <= dmin <= 3.5:
                problems.append(f"median adsorbate-slab distance {dmin:.2f} A outside 1.0-3.5")
            if not 3.0 <= cnm <= 16.0:
                problems.append(f"median slab coordination number {cnm:.1f} outside 3-16")
        secs = float(np.median([r["seconds"] for r in recs]))
        proj = secs * (len(tasks) + 24) / args.workers * 1.3 / 3600
        print(f"smoke median {secs:.2f} s/file -> projected {proj:.1f} h on {args.workers} workers (x1.3 contention)", flush=True)
        if problems:
            print("SMOKE TEST FAILED:", "; ".join(problems), flush=True)
            return 2
        if args.smoke_only:
            print("smoke test passed", flush=True)
            return 0

    deadline = time.time() + args.max_hours * 3600
    t0 = time.time()
    buf, n_done, n_err, k = [], 0, 0, 0

    def flush():
        nonlocal buf, k
        if buf:
            write_shard(args.out, f"shard_{stamp}_{k:04d}", buf)
            k += 1
            buf = []

    with Pool(args.workers, initializer=_init_worker) as pool:
        try:
            for rec in pool.imap_unordered(process, tasks, chunksize=2):
                buf.append(rec)
                n_done += 1
                n_err += rec.get("error") is not None
                if len(buf) >= args.shard_size:
                    flush()
                if n_done % 200 == 0:
                    rate = n_done / (time.time() - t0)
                    eta = (len(tasks) - n_done) / rate / 3600
                    print(f"[{time.strftime('%H:%M:%S')}] {n_done:,}/{len(tasks):,} | {rate:.2f} files/s | "
                          f"ETA {eta:.1f} h | errors {n_err}", flush=True)
                if time.time() > deadline:
                    print("time budget reached; stopping (rerun the same command to resume)", flush=True)
                    pool.terminate()
                    break
        except KeyboardInterrupt:
            print("interrupted; shards are saved, rerun to resume", flush=True)
            pool.terminate()
        finally:
            flush()

    try:
        summarize(merge(args.out), args.table)
    except Exception:
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())

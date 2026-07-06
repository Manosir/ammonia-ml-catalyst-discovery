"""
14_is2re_final_pipeline.py
===========================
Ammonia-AI: consolidated final pipeline combining all validated improvements.

Confirmed improvements, in order of CV R² impact:
  1. Dataset: full IS2RE *N (12,123 trajectories) vs s2ef_200K slice (372)
  2. Initial-frame GCN (+0.112 CV R²) — leakage test confirmed this is real signal
  3. Site-local d-band descriptor: d-band center of the 3-4 metal atoms
     bonded directly to N (from tags + neighbour list), not the whole-slab average
  4. Facet roughness index: |h|+|k|+|l| as a scalar (cleaner than 3 separate ints)

What this does NOT do:
  - Use final-frame GCN (leakage test showed it's noisier than initial-frame)
  - Use .txt.xz energy (confirmed decoy -- always uses info['energy'] from extxyz)
  - Use oc20_ref.pkl (uses system.txt reference_energy from the IS2RE download)

Usage:
    # Quick inspect (500 systems, ~2 min):
    python 14_is2re_final_pipeline.py --max-systems 500 --inspect-only

    # Full run with pre-computed GCN (fastest -- GCN already in gcn_initial_frame.csv):
    python 14_is2re_final_pipeline.py --gcn-csv results/gcn_initial_frame.csv

    # Full run computing GCN on the fly (slower, ~90 min):
    python 14_is2re_final_pipeline.py
"""

from __future__ import annotations

import argparse
import lzma
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_KV_RE   = re.compile(r'(\w+)=("[^"]*"|\S+)')
_ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

DELTA_E_MIN, DELTA_E_MAX = -8.0, 4.0

TARGET_TM = {
    "Fe", "Ru", "Co", "Mo", "Ni", "Mn", "V", "W", "Re",
    "Os", "Rh", "Ir", "Pd", "Pt", "Cu", "Ag", "Ti", "Zr",
    "Nb", "Ta", "Cr", "Sc", "Hf",
}

D_BAND_CENTER = {
    "Sc": -0.50, "Ti": -0.60, "V":  -1.09, "Cr": -1.32, "Mn": -1.39,
    "Fe": -1.29, "Co": -1.17, "Ni": -1.29, "Cu": -2.67, "Zr": -0.40,
    "Nb": -1.41, "Mo": -1.60, "Ru": -1.41, "Rh": -1.73, "Pd": -1.83,
    "Ag": -4.30, "Hf": -1.10, "Ta": -1.59, "W":  -1.80, "Re": -1.60,
    "Os": -1.40, "Ir": -1.56, "Pt": -2.25,
}
METALLIC_RADII = {
    "Sc": 1.62, "Ti": 1.45, "V":  1.34, "Cr": 1.28, "Mn": 1.29,
    "Fe": 1.26, "Co": 1.25, "Ni": 1.24, "Cu": 1.28, "Zr": 1.60,
    "Nb": 1.46, "Mo": 1.39, "Ru": 1.34, "Rh": 1.34, "Pd": 1.37,
    "Ag": 1.44, "Hf": 1.59, "Ta": 1.46, "W":  1.41, "Re": 1.37,
    "Os": 1.35, "Ir": 1.36, "Pt": 1.39,
}
ELECTRONEGATIVITY = {
    "Sc": 1.36, "Ti": 1.54, "V":  1.63, "Cr": 1.66, "Mn": 1.55,
    "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zr": 1.33,
    "Nb": 1.60, "Mo": 2.16, "Ru": 2.20, "Rh": 2.28, "Pd": 2.20,
    "Ag": 1.93, "Hf": 1.30, "Ta": 1.50, "W":  2.36, "Re": 1.90,
    "Os": 2.20, "Ir": 2.20, "Pt": 2.28,
}
D_ELECTRONS = {
    "Sc": 1,  "Ti": 2,  "V":  3,  "Cr": 5,  "Mn": 5,
    "Fe": 6,  "Co": 7,  "Ni": 8,  "Cu": 10, "Zr": 2,
    "Nb": 4,  "Mo": 5,  "Ru": 7,  "Rh": 8,  "Pd": 10,
    "Ag": 10, "Hf": 2,  "Ta": 3,  "W":  4,  "Re": 5,
    "Os": 6,  "Ir": 7,  "Pt": 9,
}

FEATURE_COLS = [
    # Electronic — whole-slab composition
    "d_band_center_slab",       # composition-weighted d-band over all slab TM atoms
    "d_band_center_site",       # d-band of the 2-4 metal atoms bonded to N (site-local)
    "d_band_center_primary",    # primary TM only (keeps single-metal signal)
    "metallic_radius",
    "electronegativity",
    "d_electrons",
    # Composition
    "primary_metal_frac",
    "tm_frac",
    "n_distinct_TM",
    # Structural — site geometry
    "gcn_initial",              # Generalised Coordination Number, initial frame
    "ads_height",               # N height above surface layer
    "mean_ads_force",           # mean |F| on N in final frame (relaxation progress)
    # Structural — facet
    "facet_roughness",          # |h|+|k|+|l| (scalar, cleaner than 3 separate ints)
    "shift",
    # Adsorbate
    "n_N_ads",
    "n_surface_atoms",
]
TARGET_COL = "delta_E_ads"


# ── Parsers ────────────────────────────────────────────────────────────────────

def parse_comment(c: str) -> dict:
    return {k: v.strip('"') for k, v in _KV_RE.findall(c)}


def load_system_txt(path: Path) -> dict[str, float]:
    refs = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 2:
                try:
                    refs[p[0]] = float(p[1])
                except ValueError:
                    pass
    return refs


def read_frames(xyz_path: Path, indices=(0, -1)) -> dict[int, dict | None]:
    """Read specified frame indices from an extxyz.xz trajectory."""
    try:
        with lzma.open(xyz_path, "rt") as fh:
            text = fh.read()
    except Exception:
        return {i: None for i in indices}

    lines = text.splitlines()
    all_frames = []
    i, n = 0, len(lines)
    while i < n:
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        try:
            natoms = int(lines[i].strip())
        except ValueError:
            i += 1
            continue
        i += 1
        if i >= n:
            break
        comment = lines[i]; i += 1
        if i + natoms > n:
            break
        elements, pos_rows, force_rows, tag_rows = [], [], [], []
        for j in range(natoms):
            p = lines[i + j].split()
            if len(p) >= 9:
                elements.append(p[0])
                pos_rows.append([float(p[1]), float(p[2]), float(p[3])])
                tag_rows.append(int(float(p[5])))
                force_rows.append([float(p[6]), float(p[7]), float(p[8])])
            elif len(p) >= 4:
                elements.append(p[0])
                pos_rows.append([float(p[1]), float(p[2]), float(p[3])])
                tag_rows.append(-1)
                force_rows.append([0.0, 0.0, 0.0])
        i += natoms
        if elements:
            all_frames.append({
                "elements": elements,
                "pos":    np.array(pos_rows,   dtype=np.float32),
                "forces": np.array(force_rows, dtype=np.float32),
                "tags":   np.array(tag_rows,   dtype=np.int8),
                "info":   parse_comment(comment),
            })

    result = {}
    for idx in indices:
        try:
            result[idx] = all_frames[idx] if all_frames else None
        except IndexError:
            result[idx] = None
    return result


# ── GCN ───────────────────────────────────────────────────────────────────────

def compute_gcn(frame: dict, cn_max: int = 12) -> float:
    """GCN of the N binding site from a parsed frame dict."""
    try:
        from ase import Atoms
        from ase.neighborlist import neighbor_list, natural_cutoffs
    except ImportError:
        return np.nan

    elems = frame["elements"]
    pos   = frame["pos"]
    atoms = Atoms(symbols=elems, positions=pos, pbc=True)
    # Infer cell from position range (approximate; real cell from Lattice in comment)
    cutoffs = natural_cutoffs(atoms, mult=1.2)
    try:
        i_idx, j_idx = neighbor_list("ij", atoms, cutoffs)
    except Exception:
        return np.nan

    n_indices = [k for k, e in enumerate(elems) if e == "N"]
    if not n_indices:
        return np.nan
    n_idx = n_indices[0]

    site_metals = j_idx[i_idx == n_idx]
    if len(site_metals) == 0:
        return np.nan

    gcn_sum = 0.0
    for m_idx in site_metals:
        m_nbrs = j_idx[i_idx == m_idx]
        m_gcn  = sum(
            sum(1 for k in j_idx[i_idx == nbr] if elems[k] != "N") / cn_max
            for nbr in m_nbrs if elems[nbr] != "N"
        )
        gcn_sum += m_gcn
    return gcn_sum / len(site_metals)


# ── Feature extraction ─────────────────────────────────────────────────────────

def site_local_d_band(frame: dict, n_idx: int | None = None) -> float:
    """
    d-band center weighted by the metal atoms directly bonded to N
    (the 2-4 atoms forming the adsorption site), NOT the whole slab.
    Falls back to np.nan if ASE not available.
    """
    try:
        from ase import Atoms
        from ase.neighborlist import neighbor_list, natural_cutoffs
    except ImportError:
        return np.nan

    elems = frame["elements"]
    pos   = frame["pos"]
    atoms = Atoms(symbols=elems, positions=pos, pbc=True)
    cutoffs = natural_cutoffs(atoms, mult=1.2)
    try:
        i_idx, j_idx = neighbor_list("ij", atoms, cutoffs)
    except Exception:
        return np.nan

    if n_idx is None:
        ns = [k for k, e in enumerate(elems) if e == "N"]
        if not ns:
            return np.nan
        n_idx = ns[0]

    site_metals = [j_idx[k] for k in range(len(i_idx)) if i_idx[k] == n_idx]
    if not site_metals:
        return np.nan

    dbc_vals = [D_BAND_CENTER[elems[m]] for m in site_metals if elems[m] in D_BAND_CENTER]
    return float(np.mean(dbc_vals)) if dbc_vals else np.nan


def extract_record(system_id: str,
                   frame_init: dict, frame_final: dict,
                   ref_energy: float,
                   gcn_precomputed: float | None = None) -> dict | None:

    if frame_final is None:
        return None

    info  = frame_final["info"]
    elems = frame_final["elements"]
    pos   = frame_final["pos"]
    forces = frame_final["forces"]
    tags  = frame_final["tags"]

    # ── Target ────────────────────────────────────────────────────────────
    e_total = float(info.get("energy", "nan"))
    if np.isnan(e_total):
        return None
    delta_e = e_total - ref_energy
    if not (DELTA_E_MIN < delta_e < DELTA_E_MAX):
        return None

    # ── Adsorbate check ───────────────────────────────────────────────────
    ads_mask  = tags == 2
    surf_mask = tags == 1
    if not ads_mask.any():
        return None
    n_N_ads = sum(1 for e, t in zip(elems, tags) if t == 2 and e == "N")
    if n_N_ads == 0:
        return None

    # ── Metallic slab filter ──────────────────────────────────────────────
    slab_elems = [e for e, t in zip(elems, tags) if t <= 1]
    if not slab_elems:
        return None
    tm_count = sum(1 for e in slab_elems if e in TARGET_TM)
    if tm_count / len(slab_elems) < 0.50:
        return None

    # ── Geometry ──────────────────────────────────────────────────────────
    ads_z  = pos[ads_mask, 2].mean()
    surf_z = pos[surf_mask, 2].mean() if surf_mask.any() else 0.0
    ads_height = float(ads_z - surf_z)
    mean_ads_force = float(np.linalg.norm(forces[ads_mask], axis=1).mean())

    # ── Composition ───────────────────────────────────────────────────────
    comp = Counter(slab_elems)
    tms  = set(comp) & TARGET_TM
    if not tms:
        return None
    total_slab = len(slab_elems)
    primary_tm = max(tms, key=lambda e: comp[e])
    if primary_tm not in D_BAND_CENTER:
        return None

    # Slab-weighted d-band (whole composition)
    d_band_slab = sum(comp[e] * D_BAND_CENTER[e]
                      for e in tms if e in D_BAND_CENTER) / sum(comp[e] for e in tms)

    # Site-local d-band (atoms bonded to N in final frame)
    d_band_site = site_local_d_band(frame_final)

    # ── GCN (initial frame) ───────────────────────────────────────────────
    if gcn_precomputed is not None:
        gcn_init = gcn_precomputed
    elif frame_init is not None:
        gcn_init = compute_gcn(frame_init)
    else:
        gcn_init = np.nan

    return {
        "system_id":          system_id,
        "primary_metal":      primary_tm,
        TARGET_COL:           delta_e,
        "e_total":            e_total,
        "e_ref":              ref_energy,
        # features
        "d_band_center_slab":    d_band_slab,
        "d_band_center_site":    d_band_site,
        "d_band_center_primary": D_BAND_CENTER[primary_tm],
        "metallic_radius":       METALLIC_RADII[primary_tm],
        "electronegativity":     ELECTRONEGATIVITY[primary_tm],
        "d_electrons":           D_ELECTRONS[primary_tm],
        "primary_metal_frac":    comp[primary_tm] / total_slab,
        "tm_frac":               tm_count / total_slab,
        "n_distinct_TM":         len(tms),
        "gcn_initial":           gcn_init,
        "ads_height":            ads_height,
        "mean_ads_force":        mean_ads_force,
        "n_N_ads":               n_N_ads,
        "n_surface_atoms":       int(surf_mask.sum()),
    }


# ── Model ─────────────────────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, r2_score

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    cv  = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)

    print(f"\n{'='*60}")
    print(f"  MAE   : {mae:.3f} eV")
    print(f"  R²    : {r2:.3f}")
    print(f"  CV R² : {cv.mean():.3f} ± {cv.std():.3f}")
    print(f"{'='*60}\n")
    return pipe, (y_te, y_pred), (mae, r2, cv)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_all(df, model, y_te, y_pred, mae, r2, out_dir: Path):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    E_OPT = -0.4

    # Parity
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te, y_pred, alpha=0.3, s=6, c="#1565C0")
    lim = [min(y_te.min(), y_pred.min())-0.3, max(y_te.max(), y_pred.max())+0.3]
    ax.plot(lim, lim, "k--", lw=1)
    ax.text(0.05, 0.92, f"MAE={mae:.3f} eV\nR²={r2:.3f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set(xlim=lim, ylim=lim, xlabel="DFT ΔE_ads(*N) / eV",
           ylabel="Predicted / eV", title="Parity — IS2RE *N")
    ax.grid(True, alpha=0.3)
    plt.savefig(out_dir/"parity.png", dpi=150, bbox_inches="tight"); plt.close()

    # Feature importance
    fi = model.named_steps["gbr"].feature_importances_
    idx = np.argsort(fi)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(fi)), fi[idx], color="#E64A19", edgecolor="k", lw=0.5)
    ax.set_xticks(range(len(fi)))
    ax.set_xticklabels([FEATURE_COLS[i] for i in idx], rotation=40, ha="right", fontsize=8)
    ax.set(ylabel="GBR importance", title="Feature Importances")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir/"feature_importance.png", dpi=150, bbox_inches="tight"); plt.close()

    # Sabatier volcano
    df2 = df.copy()
    df2["e_pred"] = model.predict(df2[FEATURE_COLS].values)
    stats = df2.groupby("primary_metal").agg(
        e_mean=("e_pred","mean"), e_std=("e_pred","std"), n=("e_pred","count")
    ).reset_index()
    stats["activity"] = -np.abs(stats["e_mean"] - E_OPT)
    stats = stats.sort_values("activity", ascending=False)

    norm = plt.Normalize(stats["activity"].min(), stats["activity"].max())
    cmap = plt.cm.RdYlGn
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Ammonia-AI — Sabatier Volcano | OC20 IS2RE *N\n"
                 "Features: d-band (slab+site) + GCN (initial frame) + facet + composition",
                 fontsize=11, fontweight="bold")

    ax = axes[0]
    sc = ax.scatter(stats["e_mean"], stats["activity"],
                    c=stats["activity"], cmap=cmap, norm=norm,
                    s=140, zorder=3, edgecolors="k", lw=0.6)
    for _, row in stats.iterrows():
        ax.annotate(row["primary_metal"], xy=(row["e_mean"], row["activity"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=9)
    xv = np.linspace(stats["e_mean"].min()-0.5, stats["e_mean"].max()+0.5, 300)
    ax.plot(xv, -np.abs(xv-E_OPT), "k--", lw=1, alpha=0.4, label="Sabatier")
    ax.axvline(E_OPT, color="gray", ls=":", lw=1.2, label=f"ΔE_opt={E_OPT} eV")
    ax.set_xlabel("Mean predicted ΔE_ads(*N) / eV", fontsize=11)
    ax.set_ylabel("Activity  −|ΔE − ΔE_opt|", fontsize=11)
    ax.set_title("Sabatier Volcano", fontsize=12)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Activity")

    ax2 = axes[1]
    cols = [cmap(norm(a)) for a in stats["activity"]]
    bars = ax2.barh(stats["primary_metal"], stats["activity"],
                    color=cols, edgecolor="k", lw=0.5)
    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax2.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                 f"n={int(row['n'])}", va="center", fontsize=7)
    ax2.set_xlabel("Activity proxy", fontsize=11); ax2.set_title("Metal Ranking")
    ax2.invert_yaxis(); ax2.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_dir/"sabatier_volcano.png", dpi=150, bbox_inches="tight"); plt.close()

    print(f"  Plots → {out_dir}/")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",   type=Path, default=Path("data/is2re_N/77/77"),
                    help="Directory containing *.extxyz.xz files")
    ap.add_argument("--system-txt", type=Path, default=None,
                    help="system.txt (auto-detected if omitted)")
    ap.add_argument("--gcn-csv",    type=Path, default=Path("results/gcn_initial_frame.csv"),
                    help="Pre-computed initial-frame GCN (skip recomputation if present)")
    ap.add_argument("--out",        type=Path, default=Path("results/final"))
    ap.add_argument("--max-systems",type=int,  default=None)
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args()

    # system.txt lives one level up from the trajectory files
    if args.system_txt is None:
        candidates = [args.data_dir.parent / "system.txt",
                      args.data_dir / "system.txt"]
        for c in candidates:
            if c.exists():
                args.system_txt = c; break
    if args.system_txt is None or not args.system_txt.exists():
        raise SystemExit(f"system.txt not found. Set --system-txt explicitly.")

    print("Ammonia-AI — IS2RE Final Pipeline")
    print(f"  Trajectories : {args.data_dir}")
    print(f"  system.txt   : {args.system_txt}")
    print(f"  GCN source   : {args.gcn_csv if args.gcn_csv.exists() else 'compute on-the-fly'}\n")

    refs = load_system_txt(args.system_txt)
    print(f"Loaded {len(refs):,} reference energies from system.txt")

    # Load pre-computed GCN if available
    gcn_lookup: dict[str, float] = {}
    if args.gcn_csv.exists():
        gcn_df = pd.read_csv(args.gcn_csv)
        gcn_df["sid"] = gcn_df["sid"].str.replace(r"\.extxyz$", "", regex=True)
        gcn_lookup = dict(zip(gcn_df["sid"], gcn_df["gcn_initial"]))
        print(f"Loaded {len(gcn_lookup):,} pre-computed initial-frame GCN values")

    xyz_files = sorted(args.data_dir.rglob("*.extxyz.xz"))
    if args.max_systems:
        xyz_files = xyz_files[:args.max_systems]
    print(f"Processing {len(xyz_files):,} trajectory files ...\n")

    records = []
    n_missing, n_fail, n_filtered = 0, 0, 0

    for i, xyz in enumerate(xyz_files, 1):
        sid = xyz.stem.replace(".extxyz", "")
        if sid not in refs:
            n_missing += 1; continue

        read_idx = (0, -1)
        frames = read_frames(xyz, indices=read_idx)
        if frames[-1] is None:
            n_fail += 1; continue

        gcn_val = gcn_lookup.get(sid, None)  # None triggers on-the-fly computation
        rec = extract_record(sid, frames[0], frames[-1], refs[sid], gcn_val)
        if rec is None:
            n_filtered += 1; continue

        # Add facet roughness from system_id metadata if available
        # (miller index not in IS2RE extxyz -- use oc20_n_metadata.csv join)
        records.append(rec)
        if i % 1000 == 0 or i == len(xyz_files):
            print(f"  [{i}/{len(xyz_files)}] {len(records)} records")

    print(f"\n  Missing ref  : {n_missing}")
    print(f"  Parse fail   : {n_fail}")
    print(f"  Quality filt : {n_filtered}")
    print(f"  Surviving    : {len(records)}")

    if not records:
        raise SystemExit("No records. Check --data-dir path.")

    df = pd.DataFrame(records)

    # Add facet roughness from metadata CSV if available
    meta_path = Path("data/oc20_n_metadata.csv")
    if meta_path.exists():
        meta = pd.read_csv(meta_path).rename(columns={"Unnamed: 0": "system_id"})
        import ast as _ast
        def roughness(s):
            try:
                t = _ast.literal_eval(str(s))
                return sum(abs(x) for x in t)
            except Exception:
                return np.nan
        meta["facet_roughness"] = meta["miller_index"].apply(roughness)
        meta["shift"] = meta["shift"]
        df = df.merge(meta[["system_id","facet_roughness","shift"]],
                      on="system_id", how="left")
    else:
        df["facet_roughness"] = np.nan
        df["shift"] = np.nan

    avail_feats = [f for f in FEATURE_COLS if f in df.columns]
    df_model = df[avail_feats + [TARGET_COL, "primary_metal"]].dropna()
    print(f"\n  Records with all features: {len(df_model)}")

    print(f"\n  Metal distribution:\n{df_model['primary_metal'].value_counts().to_string()}")

    per_metal = df_model.groupby("primary_metal")[TARGET_COL].agg(["mean","std","count"])
    per_metal.columns = ["mean_eV","std_eV","n"]
    print(f"\n  Per-metal ΔE_ads (Sabatier x-axis):")
    print(per_metal.sort_values("mean_eV").to_string())
    print(f"\n  ΔE_ads stats:\n{df_model[TARGET_COL].describe().to_string()}")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "is2re_final_dataset.csv"
    df_model.to_csv(csv_path, index=False)
    print(f"\n  Dataset → {csv_path}")

    if args.inspect_only or len(df_model) < 30:
        print("  Inspect-only. Skipping training.")
        return

    model, (y_te, y_pred), (mae, r2, cv) = train_and_evaluate(df_model)
    plot_all(df_model, model, y_te, y_pred, mae, r2, args.out)

    try:
        import joblib
        model_path = args.out / "gbr_ammonia_ai_final.joblib"
        joblib.dump(model, model_path)
        print(f"  Model → {model_path}")
    except ImportError:
        pass

    print(f"\n{'='*60}")
    print(f"  Ammonia-AI Final Pipeline complete")
    print(f"  n={len(df_model)} | MAE={mae:.3f} eV | R²={r2:.3f} | CV R²={cv.mean():.3f}±{cv.std():.3f}")
    print(f"  Outputs → {args.out}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

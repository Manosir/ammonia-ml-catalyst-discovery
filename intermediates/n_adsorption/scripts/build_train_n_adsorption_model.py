"""
03_build_and_train.py
==========================
Ammonia-AI N* pipeline — fast multiprocessing version.

Key changes vs 03_build_and_train.py:
─────────────────────────────────────
1. ProcessPoolExecutor parallelism
   Files are processed in parallel across all available CPU cores.
   Each worker decompresses and parses one file independently.
   This gives ~6-7x speedup on an 8-core machine.

2. ASE neighbour list removed from the hot path
   d_band_center_site required one ASE call per file (38.6 ms).
   It is now either:
     (a) Loaded from a pre-computed CSV (--site-dband-csv), or
     (b) Skipped entirely (falls back to d_band_center_slab as proxy)
   This alone is a 5x speedup per file.

3. Smart frame reader
   Locates frame boundaries by scanning for integer-only lines
   before parsing, avoiding full re-parse of unused frames.

Combined effect: ~96 min → ~8-12 min on 8 cores without pre-computed
site d-band, ~5-7 min with it pre-computed.

Pre-compute site d-band (one-time, run once, reuse forever):
    python src/precompute_site_dband.py \\
        --data-dir data/is2re_N/77 \\
        --out      results/site_dband.csv

Then run this script with:
    python intermediates/n_adsorption/scripts/03_build_and_train.py \\
        --data-dir data/is2re_N/77 \\
        --gcn-csv  results/gcn_initial_frame.csv \\
        --site-dband-csv results/site_dband.csv \\
        --out      results/n_adsorption/ \\
        --workers  8
"""

from __future__ import annotations

import argparse
import json
import lzma
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_KV_RE   = re.compile(r'(\w+)=("[^"]*"|\S+)')

DELTA_E_MIN, DELTA_E_MAX = -8.0, 4.0

TARGET_TM = {
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu",
    "Zr","Nb","Mo","Ru","Rh","Pd","Ag",
    "Hf","Ta","W","Re","Os","Ir","Pt",
}
D_BAND_CENTER = {
    "Sc":-0.50,"Ti":-0.60,"V": -1.09,"Cr":-1.32,"Mn":-1.39,
    "Fe":-1.29,"Co":-1.17,"Ni":-1.29,"Cu":-2.67,
    "Zr":-0.40,"Nb":-1.41,"Mo":-1.60,"Ru":-1.41,"Rh":-1.73,
    "Pd":-1.83,"Ag":-4.30,
    "Hf":-1.10,"Ta":-1.59,"W": -1.80,"Re":-1.60,
    "Os":-1.40,"Ir":-1.56,"Pt":-2.25,
}
METALLIC_RADII = {
    "Sc":1.62,"Ti":1.45,"V": 1.34,"Cr":1.28,"Mn":1.29,
    "Fe":1.26,"Co":1.25,"Ni":1.24,"Cu":1.28,"Zr":1.60,
    "Nb":1.46,"Mo":1.39,"Ru":1.34,"Rh":1.34,"Pd":1.37,
    "Ag":1.44,"Hf":1.59,"Ta":1.46,"W": 1.41,"Re":1.37,
    "Os":1.35,"Ir":1.36,"Pt":1.39,
}
ELECTRONEGATIVITY = {
    "Sc":1.36,"Ti":1.54,"V": 1.63,"Cr":1.66,"Mn":1.55,
    "Fe":1.83,"Co":1.88,"Ni":1.91,"Cu":1.90,"Zr":1.33,
    "Nb":1.60,"Mo":2.16,"Ru":2.20,"Rh":2.28,"Pd":2.20,
    "Ag":1.93,"Hf":1.30,"Ta":1.50,"W": 2.36,"Re":1.90,
    "Os":2.20,"Ir":2.20,"Pt":2.28,
}
D_ELECTRONS = {
    "Sc":1,"Ti":2,"V": 3,"Cr":5,"Mn":5,"Fe":6,"Co":7,
    "Ni":8,"Cu":10,"Zr":2,"Nb":4,"Mo":5,"Ru":7,"Rh":8,
    "Pd":10,"Ag":10,"Hf":2,"Ta":3,"W": 4,"Re":5,
    "Os":6,"Ir":7,"Pt":9,
}

FEATURE_COLS = [
    "d_band_center_slab","d_band_center_site","d_band_center_primary",
    "metallic_radius","electronegativity","d_electrons",
    "primary_metal_frac","tm_frac","n_distinct_TM",
    "gcn_initial","ads_height","mean_ads_force",
    "facet_roughness","shift","n_N_ads","n_surface_atoms",
]
TARGET_COL = "delta_E_ads"


# ── Fast frame reader ─────────────────────────────────────────────────────────

def _parse_comment(c: str) -> dict:
    return {k: v.strip('"') for k, v in _KV_RE.findall(c)}


def _parse_frame_at(lines: list[str], start: int) -> tuple[dict | None, int]:
    """Parse one frame starting at `start`. Returns (frame_dict, next_start)."""
    try:
        natoms = int(lines[start].strip())
    except (ValueError, IndexError):
        return None, start + 1
    i = start + 1
    if i >= len(lines):
        return None, i
    comment = lines[i]; i += 1
    if i + natoms > len(lines):
        return None, i
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
    if not elements:
        return None, i + natoms
    return {
        "elements": elements,
        "pos":    np.array(pos_rows,   dtype=np.float32),
        "forces": np.array(force_rows, dtype=np.float32),
        "tags":   np.array(tag_rows,   dtype=np.int8),
        "info":   _parse_comment(comment),
    }, i + natoms


def read_first_and_last_frame(xyz_path: Path) -> tuple[dict | None, dict | None]:
    """
    Decompress once, find frame boundary indices cheaply,
    parse only the first and last frames.
    This avoids parsing every intermediate trajectory step.
    """
    try:
        with lzma.open(xyz_path, "rt") as fh:
            lines = fh.read().splitlines()
    except Exception:
        return None, None

    # Frame starts are lines containing only a positive integer
    frame_starts = [
        i for i, l in enumerate(lines)
        if l.strip().isdigit() and int(l.strip()) > 0
    ]
    if not frame_starts:
        return None, None

    frame_0, _  = _parse_frame_at(lines, frame_starts[0])
    if len(frame_starts) == 1:
        return frame_0, frame_0
    frame_N, _  = _parse_frame_at(lines, frame_starts[-1])
    return frame_0, frame_N


# ── Feature extraction (runs in worker process) ───────────────────────────────

def process_one_file(args: tuple) -> dict | None:
    """
    Worker function: processes a single extxyz.xz file.
    Must be a module-level function for multiprocessing pickling.

    args = (xyz_path, ref_energy, gcn_precomputed, site_dband_precomputed)
    """
    xyz_path, ref_energy, gcn_val, site_dband_val = args

    frame_init, frame_final = read_first_and_last_frame(xyz_path)
    if frame_final is None:
        return None

    info   = frame_final["info"]
    elems  = frame_final["elements"]
    pos    = frame_final["pos"]
    forces = frame_final["forces"]
    tags   = frame_final["tags"]

    # ── True ΔE_ads ───────────────────────────────────────────────────────────
    e_total = float(info.get("energy", "nan"))
    if np.isnan(e_total):
        return None
    delta_e = e_total - ref_energy
    if not (DELTA_E_MIN < delta_e < DELTA_E_MAX):
        return None

    # ── Adsorbate check ───────────────────────────────────────────────────────
    ads_mask  = tags == 2
    surf_mask = tags == 1
    if not ads_mask.any():
        return None
    n_N_ads = sum(1 for e, t in zip(elems, tags) if t == 2 and e == "N")
    if n_N_ads == 0:
        return None

    # ── Metallic slab filter ──────────────────────────────────────────────────
    from collections import Counter
    slab_elems = [e for e, t in zip(elems, tags) if t <= 1]
    if not slab_elems:
        return None
    tm_count = sum(1 for e in slab_elems if e in TARGET_TM)
    if tm_count / len(slab_elems) < 0.50:
        return None

    comp       = Counter(slab_elems)
    tms        = set(comp) & TARGET_TM
    if not tms:
        return None
    total_slab = len(slab_elems)
    primary_tm = max(tms, key=lambda e: comp[e])
    if primary_tm not in D_BAND_CENTER:
        return None

    d_band_slab = (sum(comp[e] * D_BAND_CENTER[e] for e in tms if e in D_BAND_CENTER)
                   / sum(comp[e] for e in tms))

    # Require the validated site-local descriptor. Never silently substitute
    # a slab descriptor under the same feature name.
    if site_dband_val is None:
        return None
    d_band_site = float(site_dband_val)

    ads_z  = pos[ads_mask, 2].mean()
    surf_z = pos[surf_mask, 2].mean() if surf_mask.any() else 0.0

    sid = xyz_path.stem.replace(".extxyz", "")
    return {
        "system_id":             sid,
        "primary_metal":         primary_tm,
        TARGET_COL:              delta_e,
        "d_band_center_slab":    d_band_slab,
        "d_band_center_site":    d_band_site,
        "d_band_center_primary": D_BAND_CENTER[primary_tm],
        "metallic_radius":       METALLIC_RADII[primary_tm],
        "electronegativity":     ELECTRONEGATIVITY[primary_tm],
        "d_electrons":           D_ELECTRONS[primary_tm],
        "primary_metal_frac":    comp[primary_tm] / total_slab,
        "tm_frac":               tm_count / total_slab,
        "n_distinct_TM":         len(tms),
        "gcn_initial":           gcn_val if gcn_val is not None else np.nan,
        "ads_height":            float(ads_z - surf_z),
        "mean_ads_force":        float(np.linalg.norm(forces[ads_mask], axis=1).mean()),
        "n_N_ads":               n_N_ads,
        "n_surface_atoms":       int(surf_mask.sum()),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def train_and_evaluate(df: pd.DataFrame):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, r2_score

    avail = [f for f in FEATURE_COLS if f in df.columns]
    X = df[avail].values
    y = df[TARGET_COL].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            min_samples_leaf=5, random_state=42,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    cv  = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)

    print(f"\n{'='*56}")
    print(f"  MAE   : {mae:.3f} eV")
    print(f"  R²    : {r2:.3f}")
    print(f"  CV R² : {cv.mean():.3f} ± {cv.std():.3f}")
    print(f"  Features used: {avail}")
    print(f"{'='*56}\n")
    return pipe, (y_te, y_pred), (mae, r2, cv)


def plot_all(df, model, y_te, y_pred, mae, r2, out_dir):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    E_OPT = -0.4

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te, y_pred, alpha=0.3, s=6, c="#1565C0")
    lim = [min(y_te.min(), y_pred.min())-0.3, max(y_te.max(), y_pred.max())+0.3]
    ax.plot(lim, lim, "k--", lw=1)
    ax.text(0.05, 0.93, f"MAE={mae:.3f} eV\nR²={r2:.3f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set(xlim=lim, ylim=lim,
           xlabel="DFT ΔE$_{ads}$(*N) / eV", ylabel="Predicted / eV",
           title="Parity — IS2RE *N (fast pipeline)")
    ax.grid(True, alpha=0.3)
    plt.savefig(out_dir/"parity.png", dpi=150, bbox_inches="tight")
    plt.close()

    avail = [f for f in FEATURE_COLS if f in df.columns]
    fi  = model.named_steps["gbr"].feature_importances_
    idx = np.argsort(fi)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(fi)), fi[idx], color="#E64A19", edgecolor="k", lw=0.5)
    ax.set_xticks(range(len(fi)))
    ax.set_xticklabels([avail[i] for i in idx], rotation=38, ha="right", fontsize=8.5)
    ax.set(ylabel="GBR importance", title="Feature Importances")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir/"feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    df2 = df.copy()
    df2["e_pred"] = model.predict(df2[avail].values)
    stats = (df2.groupby("primary_metal")
               .agg(e_mean=("e_pred","mean"), n=("e_pred","count"))
               .reset_index()
               .assign(activity=lambda d: -np.abs(d["e_mean"] - E_OPT))
               .sort_values("activity", ascending=False))
    norm = plt.Normalize(stats["activity"].min(), stats["activity"].max())
    cmap = plt.cm.RdYlGn
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(stats["e_mean"], stats["activity"],
                    c=stats["activity"], cmap=cmap, norm=norm,
                    s=100, edgecolors="k", lw=0.5, zorder=3)
    for _, row in stats.iterrows():
        ax.annotate(row["primary_metal"], xy=(row["e_mean"], row["activity"]),
                    xytext=(5,4), textcoords="offset points", fontsize=8.5)
    xv = np.linspace(stats["e_mean"].min()-0.5, stats["e_mean"].max()+0.5, 300)
    ax.plot(xv, -np.abs(xv-E_OPT), "k--", lw=1, alpha=0.4, label="Sabatier")
    ax.axvline(E_OPT, color="gray", ls=":", lw=1.2)
    ax.set_xlabel("Mean predicted ΔE$_{ads}$(*N) / eV")
    ax.set_ylabel("Activity proxy")
    ax.set_title("Sabatier Volcano — *N (fast pipeline)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Activity")
    plt.tight_layout()
    plt.savefig(out_dir/"sabatier_volcano.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plots → {out_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import time
    import os

    ap = argparse.ArgumentParser(
        description="Ammonia-AI N* pipeline — fast multiprocessing version"
    )
    ap.add_argument("--data-dir",    type=Path, default=Path("data/is2re_N/77"))
    ap.add_argument("--system-txt",  type=Path, default=None)
    ap.add_argument("--gcn-csv",     type=Path, required=True,
                    help="Pre-computed initial-frame GCN CSV (system_id or sid, gcn_initial)")
    ap.add_argument("--site-dband-csv", type=Path, required=True,
                    help="Pre-computed site d-band CSV (system_id or sid, d_band_center_site). "
                         "Required for the validated model.")
    ap.add_argument("--out",         type=Path, default=Path("results/n_adsorption"))
    ap.add_argument("--workers",     type=int,  default=min(8, os.cpu_count() or 1),
                    help="Number of parallel workers (default: min(8, CPU count))")
    ap.add_argument("--max-systems", type=int,  default=None)
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args()

    if not args.gcn_csv.exists():
        raise SystemExit(f"GCN CSV not found: {args.gcn_csv}")
    if not args.site_dband_csv.exists():
        raise SystemExit(f"Site d-band CSV not found: {args.site_dband_csv}")

    if args.system_txt is None:
        for c in [args.data_dir.parent/"system.txt", args.data_dir/"system.txt"]:
            if c.exists():
                args.system_txt = c; break
    if not args.system_txt or not args.system_txt.exists():
        raise SystemExit("system.txt not found. Set --system-txt explicitly.")

    print("Ammonia-AI — N* Pipeline (Multiprocessing Fast Version)")
    print(f"  data-dir      : {args.data_dir}")
    print(f"  workers       : {args.workers}")
    print(f"  GCN source    : {args.gcn_csv}")
    print(f"  site d-band   : {args.site_dband_csv}")
    print()

    refs = load_system_txt(args.system_txt)
    print(f"Loaded {len(refs):,} reference energies")

    def normalize_descriptor_table(table: pd.DataFrame, name: str, value_col: str) -> pd.DataFrame:
        """Normalize current system_id and legacy sid schemas to internal sid."""
        if "system_id" in table.columns and "sid" not in table.columns:
            table = table.rename(columns={"system_id": "sid"})
        required = {"sid", value_col}
        if not required.issubset(table.columns):
            raise SystemExit(
                f"{name} CSV must contain system_id or sid plus {value_col}. "
                f"Found columns: {list(table.columns)}"
            )
        table = table.copy()
        table["sid"] = (table["sid"].astype(str)
                         .str.replace(r"\.extxyz$", "", regex=True))
        if table["sid"].duplicated().any():
            dup = int(table["sid"].duplicated().sum())
            raise SystemExit(f"{name} CSV contains {dup} duplicate system IDs")
        table[value_col] = pd.to_numeric(table[value_col], errors="coerce")
        return table

    g = normalize_descriptor_table(
        pd.read_csv(args.gcn_csv), "GCN", "gcn_initial"
    )
    s = normalize_descriptor_table(
        pd.read_csv(args.site_dband_csv), "Site d-band", "d_band_center_site"
    )
    gcn_lookup = dict(zip(g["sid"], g["gcn_initial"]))
    site_dband_lookup = dict(zip(s["sid"], s["d_band_center_site"]))
    print(f"Loaded {len(gcn_lookup):,} GCN rows ({g['gcn_initial'].notna().sum():,} valid)")
    print(f"Loaded {len(site_dband_lookup):,} site d-band rows ({s['d_band_center_site'].notna().sum():,} valid)")

    xyz_files = sorted(args.data_dir.rglob("*.extxyz.xz"))
    if args.max_systems:
        xyz_files = xyz_files[:args.max_systems]

    print(f"\nProcessing {len(xyz_files):,} files across {args.workers} workers ...\n")

    # Build argument tuples for workers
    work_items = []
    missing_gcn = missing_site = 0
    for xyz in xyz_files:
        sid = xyz.stem.replace(".extxyz", "")
        if sid not in refs:
            continue
        gcn_value = gcn_lookup.get(sid)
        site_value = site_dband_lookup.get(sid)
        if gcn_value is None or pd.isna(gcn_value):
            missing_gcn += 1
            continue
        if site_value is None or pd.isna(site_value):
            missing_site += 1
            continue
        work_items.append((
            xyz,
            refs[sid],
            float(gcn_value),
            float(site_value),
        ))

    print(f"  Skipped missing GCN: {missing_gcn:,}; missing site d-band: {missing_site:,}")
    t_start = time.perf_counter()
    records, n_done, n_filtered = [], 0, 0

    try:
        from tqdm import tqdm
        progress = tqdm(total=len(work_items), unit="file",
                       desc="Extracting features", ncols=90)
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one_file, item): item for item in work_items}
        for future in as_completed(futures):
            n_done += 1
            result = future.result()
            if result is not None:
                records.append(result)
            else:
                n_filtered += 1
            if use_tqdm:
                progress.update(1)
            elif n_done % 1000 == 0:
                elapsed = time.perf_counter() - t_start
                rate = n_done / elapsed
                remaining = (len(work_items) - n_done) / rate / 60
                print(f"  [{n_done}/{len(work_items)}] {len(records)} records "
                      f"| {rate:.0f} files/s | ~{remaining:.1f} min left")

    if use_tqdm:
        progress.close()

    elapsed = time.perf_counter() - t_start
    print(f"\n  Processed {n_done:,} files in {elapsed/60:.1f} min "
          f"({n_done/elapsed:.0f} files/sec)")
    print(f"  Surviving records : {len(records)}")
    print(f"  Filtered out      : {n_filtered}")

    if not records:
        raise SystemExit("No records survived. Check --data-dir.")

    df = pd.DataFrame(records)

    meta_path = Path("data/oc20_n_metadata.csv")
    if meta_path.exists():
        import ast as _ast
        meta = pd.read_csv(meta_path).rename(columns={"Unnamed: 0": "system_id"})
        meta["facet_roughness"] = meta["miller_index"].apply(
            lambda s: sum(abs(x) for x in _ast.literal_eval(str(s)))
            if pd.notna(s) else np.nan
        )
        df = df.merge(meta[["system_id","facet_roughness","shift"]],
                      on="system_id", how="left")
    else:
        df["facet_roughness"] = np.nan
        df["shift"] = np.nan

    # facet_roughness and shift are optional metadata features.
    # Fill with 0 when oc20_n_metadata.csv is absent rather than
    # dropping all rows — the model degrades slightly but remains usable.
    if df["facet_roughness"].isna().any() or df["shift"].isna().any():
        raise SystemExit("Metadata coverage incomplete: provide data/oc20_n_metadata.csv before training.")

    # Save FULL dataset before dropna (keeps optional-NaN rows visible)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "is2re_full_dataset.csv", index=False)

    # Core features that must be non-NaN for training
    CORE_FEATURES = [f for f in FEATURE_COLS
                     if f not in ("facet_roughness", "shift")]
    avail = [f for f in FEATURE_COLS if f in df.columns]
    df_model = df[avail + [TARGET_COL, "primary_metal"]].dropna(subset=CORE_FEATURES)
    print(f"\n  Records with all features : {len(df_model)}")

    per_metal = (df_model.groupby("primary_metal")[TARGET_COL]
                 .agg(["mean","std","count"])
                 .rename(columns={"mean":"mean_eV","std":"std_eV","count":"n"})
                 .sort_values("mean_eV"))
    print(f"\n  Per-metal ΔE_ads:")
    print(per_metal.to_string())

    df_model.to_csv(args.out / "is2re_final_dataset.csv", index=False)
    print(f"\n  Dataset → {args.out / 'is2re_final_dataset.csv'}")

    if args.inspect_only or len(df_model) < 30:
        print("  Skipping training.")
        return

    model, (y_te, y_pred), (mae, r2, cv) = train_and_evaluate(df_model)
    plot_all(df_model, model, y_te, y_pred, mae, r2, args.out)

    try:
        import joblib
        model_path = args.out / "gbr_ammonia_ai_final.joblib"
        joblib.dump(model, model_path)
        manifest = {
            "model_file": model_path.name,
            "feature_columns": avail,
            "target": TARGET_COL,
            "model": "StandardScaler + GradientBoostingRegressor",
            "random_state": 42,
            "validation": "80/20 random split plus 5-fold random CV",
            "data_dir": str(args.data_dir),
            "gcn_csv": str(args.gcn_csv),
            "site_dband_csv": str(args.site_dband_csv),
            "records": int(len(df_model)),
            "note": "Site d-band and force/height descriptors are final-frame quantities; this is not a fully pre-DFT model."
        }
        (args.out / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  Model → {model_path}")
        print(f"  Manifest → {args.out/'model_manifest.json'}")
    except ImportError:
        pass

    print(f"\n{'='*56}")
    print(f"  Complete | MAE={mae:.3f} eV | R²={r2:.3f} | CV R²={cv.mean():.3f}±{cv.std():.3f}")
    print(f"  Outputs  → {args.out}/")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    main()

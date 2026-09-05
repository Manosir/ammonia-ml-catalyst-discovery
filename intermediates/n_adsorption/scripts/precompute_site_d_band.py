"""
precompute_site_dband.py
========================
Pre‑compute the site‑local d‑band centre for all IS2RE *N trajectories.

The site‑local d‑band centre is the average d‑band energy of the 2–4
metal atoms directly bonded to the N adsorbate in the final relaxed frame.

This is a one‑time computation that can take ~30–60 minutes on a multi‑core machine.
Once computed, the CSV can be loaded in the fast training script to avoid
expensive ASE neighbour‑list calls during every training run.

Usage:
    python precompute_site_dband.py \
        --data-dir data/is2re_N/77 \
        --out results/d_band_center_site.csv \
        --workers 8
"""

from __future__ import annotations

import argparse
import lzma
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ammonia_ai.features.d_band import site_local_d_band_from_frame

from ase import Atoms
from ase.neighborlist import neighbor_list, natural_cutoffs

# ─── Constants (copy from your training script) ────────────────────────────

D_BAND_CENTER = {
    "Sc": -0.50, "Ti": -0.60, "V": -1.09, "Cr": -1.32, "Mn": -1.39,
    "Fe": -1.29, "Co": -1.17, "Ni": -1.29, "Cu": -2.67,
    "Zr": -0.40, "Nb": -1.41, "Mo": -1.60, "Ru": -1.41, "Rh": -1.73,
    "Pd": -1.83, "Ag": -4.30,
    "Hf": -1.10, "Ta": -1.59, "W": -1.80, "Re": -1.60,
    "Os": -1.40, "Ir": -1.56, "Pt": -2.25,
}

# ─── Parsers ──────────────────────────────────────────────────────────────────

_KV_RE = re.compile(r'(\w+)=("[^"]*"|\S+)')

def parse_comment(comment: str) -> Dict[str, str]:
    """Parse key=value pairs from an extxyz comment line."""
    return {k: v.strip('"') for k, v in _KV_RE.findall(comment)}


def read_frames(xyz_path: Path, indices=(0, -1)) -> Dict[int, Optional[Dict]]:
    """
    Manually parse an LZMA‑compressed extxyz file.
    Returns a dict with frame index -> {elements, pos, forces, tags, info}.
    """
    try:
        with lzma.open(xyz_path, "rt") as fh:
            text = fh.read()
    except Exception:
        return {i: None for i in indices}

    lines = text.splitlines()
    all_frames = []
    i, n = 0, len(lines)

    while i < n:
        # skip blank lines
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

        comment = lines[i]
        i += 1
        if i + natoms > n:
            break

        elements, pos_rows, force_rows, tag_rows = [], [], [], []
        for j in range(natoms):
            parts = lines[i + j].split()
            if len(parts) >= 9:          # full extxyz with forces and tags
                elements.append(parts[0])
                pos_rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
                tag_rows.append(int(float(parts[5])))
                force_rows.append([float(parts[6]), float(parts[7]), float(parts[8])])
            elif len(parts) >= 4:        # minimal: only positions
                elements.append(parts[0])
                pos_rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
                tag_rows.append(-1)
                force_rows.append([0.0, 0.0, 0.0])
        i += natoms

        if elements:
            all_frames.append({
                "elements": elements,
                "pos": np.array(pos_rows, dtype=np.float32),
                "forces": np.array(force_rows, dtype=np.float32),
                "tags": np.array(tag_rows, dtype=np.int8),
                "info": parse_comment(comment),
            })

    result = {}
    for idx in indices:
        try:
            result[idx] = all_frames[idx] if all_frames else None
        except IndexError:
            result[idx] = None
    return result


# ─── Site‑local d‑band computation ──────────────────────────────────────────

def site_local_d_band(frame: Dict) -> float:
    """Compatibility wrapper around the shared site-local d-band module."""
    return site_local_d_band_from_frame(
        frame,
        anchor_symbol="N",
        anchor_tag=2,
        d_band_center=D_BAND_CENTER,
        cutoff_multiplier=1.2,
    )


def process_one(xyz_path: Path) -> Dict[str, float]:
    """Worker function for multiprocessing."""
    sid = xyz_path.stem.replace(".extxyz", "")
    frames = read_frames(xyz_path, indices=(-1,))  # only final frame
    if frames[-1] is None:
        return {"system_id": sid, "d_band_center_site": np.nan}
    dband = site_local_d_band(frames[-1])
    return {"system_id": sid, "d_band_center_site": dband}


def main():
    parser = argparse.ArgumentParser(
        description="Pre‑compute site‑local d‑band centres for all IS2RE *N trajectories."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/is2re_N/77"),
        help="Directory containing .extxyz.xz trajectory files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/d_band_center_site.csv"),
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel processes (default: 8).",
    )
    parser.add_argument(
        "--max-systems",
        type=int,
        default=None,
        help="Limit processing to first N systems (for testing).",
    )
    args = parser.parse_args()

    xyz_files = sorted(args.data_dir.rglob("*.extxyz.xz"))
    if args.max_systems is not None:
        xyz_files = xyz_files[:args.max_systems]

    print(f"Found {len(xyz_files)} trajectory files.")
    print(f"Starting computation with {args.workers} workers...")

    # Run in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Use tqdm if available
        try:
            from tqdm import tqdm
            iterator = tqdm(executor.map(process_one, xyz_files), total=len(xyz_files), desc="Processing")
        except ImportError:
            iterator = executor.map(process_one, xyz_files)

        for res in iterator:
            results.append(res)

    df = pd.DataFrame(results)
    # Ensure system_id is the first column
    cols = ["system_id", "d_band_center_site"]
    df = df[cols]

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} records to {args.out}")


if __name__ == "__main__":
    main()
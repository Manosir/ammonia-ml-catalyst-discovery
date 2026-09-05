"""
precompute_gcn.py
=================
Pre‑compute the Generalised Coordination Number (GCN) of the N adsorption site
from the initial frame (unrelaxed) of each IS2RE *N trajectory.

The GCN is defined as the sum over the metal atoms bonded to N of their
coordination number divided by the maximum coordination (typically 12).
This is the same definition used in the original training pipeline.

Usage:
    python precompute_gcn.py \
        --data-dir data/is2re_N/77 \
        --out results/gcn_initial_frame.csv \
        --workers 12
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

from ammonia_ai.features.gcn import calculate_site_gcn

from ase import Atoms
from ase.neighborlist import neighbor_list, natural_cutoffs

# ─── Parser constants ──────────────────────────────────────────────────────────

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


def compute_gcn(frame: Dict, cn_max: int = 12) -> float:
    """
    Compute the GCN of the N adsorption site from a parsed frame dict.
    Returns np.nan if the N atom is not found or neighbour list fails.

    The GCN is defined as:
      GCN = (1 / N_metals) * Σ_{metal} (Σ_{neighbour of metal} 1 / cn_max)
    where only neighbours that are not N are counted.
    """
    elems = frame["elements"]
    pos = frame["pos"]
    tags = frame["tags"]

    # Find N adsorbate atoms (tag=2)
    n_indices = [i for i, e in enumerate(elems) if e == "N" and tags[i] == 2]
    if not n_indices:
        return np.nan
    n_idx = n_indices[0]

    # Build ASE Atoms object with a large dummy cell to avoid wrapping
    pos_np = np.array(pos)
    cell = np.eye(3) * 50.0
    atoms = Atoms(symbols=elems, positions=pos_np, cell=cell, pbc=False)

    try:
        cutoffs = natural_cutoffs(atoms, mult=1.2)
        i_idx, j_idx = neighbor_list("ij", atoms, cutoffs)
    except Exception:
        return np.nan

    # Find metal neighbours of the N atom
    site_metals = [j_idx[k] for k in range(len(i_idx)) if i_idx[k] == n_idx and elems[j_idx[k]] != "N"]
    if not site_metals:
        return np.nan

    # Compute GCN: for each metal neighbour, count its neighbours (excluding N)
    gcn_sum = 0.0
    for m_idx in site_metals:
        # Neighbours of this metal (excluding itself and N)
        m_neighbours = [j_idx[k] for k in range(len(i_idx)) if i_idx[k] == m_idx and j_idx[k] != m_idx and elems[j_idx[k]] != "N"]
        m_gcn = len(m_neighbours) / cn_max
        gcn_sum += m_gcn

    return gcn_sum / len(site_metals)


def process_one(xyz_path: Path) -> Dict[str, float]:
    """Worker function for multiprocessing using the shared GCN module."""
    sid = xyz_path.stem.replace(".extxyz", "")
    # Explicitly use frame 0: this is the production, initial-frame contract.
    gcn = calculate_site_gcn(xyz_path, frame_index=0, anchor_symbol="N", anchor_tag=2)
    return {"system_id": sid, "gcn_initial": gcn}


def main():
    parser = argparse.ArgumentParser(
        description="Pre‑compute initial‑frame GCN for all IS2RE *N trajectories."
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
        default=Path("results/gcn_initial_frame.csv"),
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
        try:
            from tqdm import tqdm
            iterator = tqdm(executor.map(process_one, xyz_files), total=len(xyz_files), desc="Processing")
        except ImportError:
            iterator = executor.map(process_one, xyz_files)

        for res in iterator:
            results.append(res)

    df = pd.DataFrame(results)
    # Ensure system_id is the first column
    cols = ["system_id", "gcn_initial"]
    df = df[cols]

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} records to {args.out}")


if __name__ == "__main__":
    main()
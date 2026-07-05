"""
09_download_is2re_N.py
=======================
Downloads the OC20 IS2RE *N adsorbate trajectories (adsorbate index 77).

Source: https://github.com/Open-Catalyst-Project/ocp/blob/master/DATASET_PER_ADSORBATE.md

What this gives us (confirmed from the official docs):
  - ~5,700 relaxation trajectories, one per unique (catalyst, facet, site) system
  - Each trajectory: LZMA-compressed extxyz file named <system_id>.extxyz.xz
  - system.txt: system_id, reference_energy  (clean slab + gas *N reference)
  - reference_energy IS the quantity we need: ΔE_ads = E_relaxed - reference_energy
  - Trajectories are TRUE DFT relaxations (not MD), so the final frame IS
    the equilibrium adsorption geometry — no MD-snapshot noise.

Size: 1.1 GB compressed. MD5: bfb6e03d4a687987ff68976f0793cc46

Usage:
    python scripts2/09_download_is2re_N.py [--out-dir data/is2re_N]
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/77.tar"
MD5_EXPECTED = "bfb6e03d4a687987ff68976f0793cc46"
TARNAME = "77.tar"


def compute_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data/is2re_N"),
                    help="Directory to download and extract into")
    ap.add_argument("--skip-md5", action="store_true",
                    help="Skip MD5 verification (faster, less safe)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = args.out_dir / TARNAME

    # ── Download ──────────────────────────────────────────────────────────
    if tar_path.exists():
        print(f"Tar file already exists: {tar_path}")
        print("Skipping download. Delete it to re-download.")
    else:
        print(f"Downloading *N IS2RE trajectories (~1.1 GB)...")
        print(f"  URL: {URL}")
        print(f"  Target: {tar_path}\n")
        result = subprocess.run(
            ["wget", "-q", "--show-progress", URL, "-O", str(tar_path)],
            check=False
        )
        if result.returncode != 0:
            # Try curl as fallback
            print("wget failed, trying curl...")
            result = subprocess.run(
                ["curl", "-L", "--progress-bar", URL, "-o", str(tar_path)],
                check=False
            )
        if result.returncode != 0:
            sys.exit(f"ERROR: Download failed (return code {result.returncode}).\n"
                    f"Try manually: wget '{URL}' -O '{tar_path}'")

    # ── MD5 verification ──────────────────────────────────────────────────
    if not args.skip_md5:
        print("Verifying MD5 checksum...")
        actual_md5 = compute_md5(tar_path)
        if actual_md5 != MD5_EXPECTED:
            sys.exit(
                f"ERROR: MD5 mismatch!\n"
                f"  Expected : {MD5_EXPECTED}\n"
                f"  Got      : {actual_md5}\n"
                f"The file may be corrupted. Delete {tar_path} and re-run."
            )
        print(f"  MD5 OK: {actual_md5}")

    # ── Extract ───────────────────────────────────────────────────────────
    extract_dir = args.out_dir / "77"
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"\nExtract directory already populated: {extract_dir}")
        print("Skipping extraction.")
    else:
        print(f"\nExtracting to {args.out_dir} ...")
        result = subprocess.run(
            ["tar", "-xf", str(tar_path), "-C", str(args.out_dir)],
            check=False
        )
        if result.returncode != 0:
            sys.exit(f"ERROR: Extraction failed (code {result.returncode}).")

    # ── Verify structure ──────────────────────────────────────────────────
    system_txt = extract_dir / "system.txt"
    if not system_txt.exists():
        sys.exit(f"ERROR: Expected {system_txt} not found after extraction.")

    with open(system_txt) as f:
        lines = [l.strip() for l in f if l.strip()]

    n_systems = len(lines)
    sample = lines[:3]
    xyzs = list(extract_dir.glob("*.extxyz.xz"))

    print(f"\n{'='*60}")
    print(f"IS2RE *N download complete")
    print(f"{'='*60}")
    print(f"  Location      : {extract_dir}")
    print(f"  system.txt    : {n_systems} systems")
    print(f"  .extxyz.xz    : {len(xyzs)} trajectory files found")
    print(f"\n  system.txt sample (system_id, reference_energy):")
    for s in sample:
        print(f"    {s}")
    print(f"\n  Next step:")
    print(f"    python scripts2/10_is2re_pipeline.py --data-dir {extract_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

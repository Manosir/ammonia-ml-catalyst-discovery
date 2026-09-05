"""
02_download_is2re_N.py
=======================
Downloads the OC20 IS2RE *N adsorbate trajectories (adsorbate index 77).

Source: https://github.com/Open-Catalyst-Project/ocp/blob/master/DATASET_PER_ADSORBATE.md

What this gives us:
  - ~5,700 relaxation trajectories, one per unique (catalyst, facet, site) system
  - Each trajectory: LZMA-compressed extxyz file named <system_id>.extxyz.xz
  - system.txt: system_id, reference_energy (clean slab + gas *N reference)
  - reference_energy IS the quantity we need: ΔE_ads = E_relaxed - reference_energy

Size: 1.1 GB compressed. MD5: bfb6e03d4a687987ff68976f0793cc46

Usage:
    python intermediates/n_adsorption/scripts/02_download_is2re_N.py [--out-dir data/is2re_N]
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/77.tar"
MD5_EXPECTED = "bfb6e03d4a687987ff68976f0793cc46"


def verify_md5(path: Path, expected: str) -> bool:
    print(f"Verifying MD5 checksum for {path.name} ...")
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest() == expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OC20 IS2RE *N dataset.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/is2re_N"),
        help="Base directory to download and extract data",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = args.out_dir / "77.tar"

    # ── Download ─────────────────────────────────────────────────────────
    if tar_path.exists() and verify_md5(tar_path, MD5_EXPECTED):
        print(f"Valid archive already exists at {tar_path}. Skipping download.")
    else:
        print(f"Downloading IS2RE *N archive from {URL} ...")
        partial = tar_path.with_suffix(".tar.part")
        try:
            subprocess.run(["wget", "--continue", "--tries=3", "--waitretry=5", URL, "-O", str(partial)], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("wget failed or not found. Trying curl...")
            subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--retry-delay", "5", "--continue-at", "-", URL, "--output", str(partial)], check=True)
        partial.replace(tar_path)

        if not verify_md5(tar_path, MD5_EXPECTED):
            sys.exit("ERROR: MD5 checksum mismatch! File might be corrupted.")

    # ── Extract ──────────────────────────────────────────────────────────
    extract_dir = args.out_dir / "77"
    if (extract_dir / "system.txt").exists() and list(extract_dir.glob("*.extxyz.xz")):
        print(f"\nVerified extraction already exists: {extract_dir}")
        print("Skipping extraction.")
    else:
        print(f"\nExtracting to {args.out_dir} ...")
        result = subprocess.run(
            ["tar", "-xf", str(tar_path), "-C", str(args.out_dir)], check=False
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
    xyzs = list(extract_dir.glob("*.extxyz.xz"))

    print(f"\n{'='*60}")
    print(f"IS2RE *N download complete")
    print(f"{'='*60}")
    print(f"  Location      : {extract_dir}")
    print(f"  system.txt    : {n_systems} systems")
    print(f"  .extxyz.xz    : {len(xyzs)} trajectory files found")
    print("\nNext steps:")
    print(
        "  Run the pipeline: python intermediates/n_adsorption/scripts/03_build_and_train.py"
    )
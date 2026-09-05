"""
download_oc20_is2re_n.py
===================
Downloads the OC20 S2EF train 200K slice.

Note: The primary data source for this project is the IS2RE *N dataset
(script 02). S2EF is only needed for exploratory analysis of raw MD
trajectories. Most users should run script 02 instead.

Usage:
    python intermediates/n_adsorption/scripts/01_download_s2ef.py [--out-dir data/oc20_200k]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar"


def download_and_extract(url: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    filepath = target_dir / filename

    print(f"Downloading {filename} (~3.5 GB) ...")
    try:
        subprocess.run(["wget", "-q", "--show-progress", url, "-O", str(filepath)],
                       check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("wget failed or not found. Trying curl ...")
        result = subprocess.run(
            ["curl", "-L", "--progress-bar", url, "-o", str(filepath)],
            check=False
        )
        if result.returncode != 0:
            sys.exit(
                f"ERROR: Download failed with both wget and curl.\n"
                f"  Try manually: wget '{url}' -O '{filepath}'"
            )

    print(f"Extracting to {target_dir} ...")
    result = subprocess.run(
        ["tar", "-xf", str(filepath), "-C", str(target_dir)],
        check=False
    )
    if result.returncode != 0:
        sys.exit(f"ERROR: Extraction failed (code {result.returncode}).")

    if filepath.exists():
        os.remove(filepath)
        print(f"Removed compressed archive: {filepath.name}")

    print("S2EF 200K download and extraction complete.\n")


def main():
    ap = argparse.ArgumentParser(
        description="Download OC20 S2EF 200K data slice."
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path("data/oc20_200k"),
        help="Directory to save and extract data (default: data/oc20_200k)",
    )
    args = ap.parse_args()

    download_and_extract(URL, args.out_dir)

    # Verify extraction
    expected = args.out_dir / "s2ef_train_200K"
    if expected.exists():
        chunks = list(expected.rglob("*.extxyz.xz"))
        print(f"  Verified: {len(chunks)} .extxyz.xz files in {expected}")
    else:
        print(f"  WARNING: expected directory {expected} not found after extraction.")
        print("  Check the tar contents manually.")


if __name__ == "__main__":
    main()

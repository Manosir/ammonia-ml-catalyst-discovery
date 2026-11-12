#!/usr/bin/env python3
"""Download and verify one or more OC20 per-adsorbate trajectory archives.

Examples:
    python scripts/download_adsorbate.py --adsorbate NH2 --output-dir data/oc20
    python scripts/download_adsorbate.py --adsorbate N NH NH2 NH3 N2 --output-dir data/oc20
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

# Allow direct execution from a source checkout without requiring installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ammonia_ai.adsorbates import OC20_ARCHIVES  # noqa: E402


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r") as tar:
        root = destination.resolve()
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tar.extractall(destination)


def process(adsorbate: str, output_dir: Path, *, extract: bool = True) -> Path:
    key = adsorbate.upper()
    if key not in OC20_ARCHIVES:
        raise ValueError(f"unsupported adsorbate {adsorbate!r}; choose from {sorted(OC20_ARCHIVES)}")
    record = OC20_ARCHIVES[key]
    archive = output_dir / "archives" / f"{key}.tar"
    print(f"Downloading {key} from {record.url} ({record.size_gib:.1f} GiB compressed)")
    download(record.url, archive)
    actual = md5sum(archive)
    if actual != record.md5:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {key}: expected {record.md5}, got {actual}")
    print(f"  MD5 verified: {actual}")
    if extract:
        target = output_dir / "trajectories" / key
        extract_archive(archive, target)
        print(f"  Extracted to {target}")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adsorbate", nargs="+", choices=sorted(OC20_ARCHIVES), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/oc20_is2res"))
    parser.add_argument("--no-extract", action="store_true", help="download and verify archives only")
    args = parser.parse_args()
    for adsorbate in args.adsorbate:
        process(adsorbate, args.output_dir, extract=not args.no_extract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

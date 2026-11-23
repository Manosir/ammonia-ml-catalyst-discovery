#!/usr/bin/env python3
"""Download the DTU adsorption/RPA ASE databases with provenance logging."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FILES = {
    "adsorption.db": "https://wiki.fysik.dtu.dk/cmr-files/adsorption.db",
    "surfaces.db": "https://wiki.fysik.dtu.dk/cmr-files/surfaces.db",
}
SOURCE = "https://cmr.fysik.dtu.dk/adsorption/adsorption.html"
LICENSE_NOTE = (
    "Consult the current DTU/CMR source terms and cite Schmidt and Thygesen, "
    "DOI: 10.1021/acs.jpcc.7b12258."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        print(f"Exists, keeping: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out)
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("external_raw/dtu_rpa"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_page": SOURCE,
        "license_note": LICENSE_NOTE,
        "files": {},
    }
    for filename, url in FILES.items():
        path = args.output_dir / filename
        download(url, path, args.overwrite)
        records["files"][filename] = {
            "url": url,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    metadata = args.output_dir / "metadata.json"
    metadata.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote provenance metadata: {metadata}")
    for filename, item in records["files"].items():
        print(f"{filename}: {item['size_bytes']} bytes sha256={item['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

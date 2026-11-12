#!/usr/bin/env python3
"""Extract OC20 adsorbates with append-only checkpointing.

Each completed batch is appended to ``--output``. Re-running the command skips
source files already present in the output, so an interrupted run can resume.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ammonia_ai.adsorbates import ADSORBATE_METADATA  # noqa: E402
from ammonia_ai.features.adsorbate import extract_adsorbate_record  # noqa: E402

SUPPORTED_SUFFIXES = (".extxyz.xz", ".extxyz.gz", ".extxyz", ".xyz")
OUTPUT_COLUMNS = [
    "status", "system_id", "adsorbate", "source_path", "error",
    "energy_initial_eV", "energy_final_eV", "delta_E_ads",
    "ads_height", "mean_ads_force", "n_surface_atoms",
    "adsorbate_formula", "adsorbate_class", "n_adsorbate_N",
    "n_adsorbate_H", "adsorbate_mass", "binding_atom",
    "adsorbate_atom_count",
]


def find_trajectories(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and any(path.name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)
    )


def extract_one(path: Path, adsorbate: str) -> dict:
    try:
        row = extract_adsorbate_record(path, adsorbate)
        row["source_path"] = str(path)
        return {"status": "ok", **row}
    except Exception as exc:
        return {
            "status": "error",
            "adsorbate": adsorbate,
            "source_path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_completed(output: Path) -> set[str]:
    if not output.exists() or output.stat().st_size == 0:
        return set()
    header = pd.read_csv(output, nrows=0)
    if "source_path" not in header.columns:
        raise ValueError(
            f"existing output lacks source_path and cannot be resumed safely: {output}; "
            "move it aside or choose a new output path"
        )
    columns = pd.read_csv(output, usecols=["source_path"], dtype=str)
    return set(columns["source_path"].dropna())


def append_batch(rows: list[dict], output: Path) -> None:
    frame = pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, mode="a", header=not output.exists() or output.stat().st_size == 0, index=False)


def run(data_dir: Path, output: Path, adsorbates: list[str], workers: int, batch_size: int) -> None:
    jobs: list[tuple[Path, str]] = []
    for adsorbate in adsorbates:
        folder = data_dir / adsorbate
        if not folder.exists():
            raise FileNotFoundError(f"missing trajectory folder for {adsorbate}: {folder}")
        found = find_trajectories(folder)
        print(f"{adsorbate}: found {len(found):,} trajectory files", flush=True)
        jobs.extend((path, adsorbate) for path in found)
    if not jobs:
        raise RuntimeError("no supported trajectory files found")

    completed = load_completed(output)
    pending = [(path, adsorbate) for path, adsorbate in jobs if str(path) not in completed]
    print(f"Checkpoint: {len(completed):,} complete; {len(pending):,} pending", flush=True)

    processed = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        if workers == 1:
            rows = [extract_one(path, adsorbate) for path, adsorbate in batch]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(lambda job: extract_one(*job), batch))
        append_batch(rows, output)
        processed += len(rows)
        errors = sum(row.get("status") == "error" for row in rows)
        print(
            f"Checkpoint: wrote {processed:,}/{len(pending):,} pending; "
            f"batch errors: {errors:,}",
            flush=True,
        )

    total = len(completed) + processed
    print(f"Complete: {total:,} records in {output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/oc20_is2res/trajectories"))
    parser.add_argument("--output", type=Path, default=Path("results/adsorbates/oc20_adsorbates.csv"))
    parser.add_argument("--adsorbates", nargs="+", choices=sorted(ADSORBATE_METADATA), default=list(ADSORBATE_METADATA))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        parser.error("--workers and --batch-size must be at least 1")
    run(args.data_dir, args.output, [name.upper() for name in args.adsorbates], args.workers, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

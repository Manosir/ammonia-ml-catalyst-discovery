"""Generate OC20 N* metadata from the official OC20 mapping pickle.

The resulting CSV is used by the N* training pipeline to construct:
    - facet_roughness, derived from miller_index
    - shift

The mapping file must be downloaded from the official OC20 source and
verified with its published MD5 checksum before loading.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import pandas as pd

OC20_MAPPING_MD5 = "71705204c12f8710ff43e71fbc6ba29b"
TARGET_ADS_ID = 77


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_ids(data_dir: Path) -> set[str]:
    """Return system IDs represented by the downloaded trajectory files."""
    ids = set()
    for path in data_dir.rglob("*.extxyz.xz"):
        name = path.name
        if name.endswith(".extxyz.xz"):
            ids.add(name[:-len(".extxyz.xz")])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate metadata for OC20 IS2RE N* trajectories."
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("data/oc20_data_mapping.pkl"),
        help="Official OC20 mapping pickle.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/is2re_N/77/77"),
        help="Directory containing downloaded N* trajectory files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/oc20_n_metadata.csv"),
        help="Output metadata CSV.",
    )
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="Skip MD5 verification only if using a documented local copy.",
    )
    args = parser.parse_args()

    if not args.mapping.exists():
        raise SystemExit(f"Mapping file not found: {args.mapping}")
    if not args.data_dir.exists():
        raise SystemExit(f"Trajectory directory not found: {args.data_dir}")

    if not args.skip_md5:
        actual = md5sum(args.mapping)
        if actual != OC20_MAPPING_MD5:
            raise SystemExit(
                "OC20 mapping MD5 mismatch.\n"
                f"Expected: {OC20_MAPPING_MD5}\n"
                f"Actual:   {actual}"
            )
        print("Mapping MD5 verified.")

    ids = trajectory_ids(args.data_dir)
    if not ids:
        raise SystemExit(f"No .extxyz.xz files found under {args.data_dir}")
    print(f"Trajectory IDs found: {len(ids):,}")

    # This loads the official OC20 mapping only after checksum verification.
    with args.mapping.open("rb") as handle:
        mapping = pickle.load(handle)

    if not isinstance(mapping, dict):
        raise SystemExit("The OC20 mapping pickle does not contain a dictionary.")

    rows = []
    missing_mapping = []
    wrong_adsorbate = []

    for system_id in sorted(ids):
        record = mapping.get(system_id)
        if record is None:
            missing_mapping.append(system_id)
            continue

        ads_id = record.get("ads_id")
        if int(ads_id) != TARGET_ADS_ID:
            wrong_adsorbate.append((system_id, ads_id))
            continue

        miller = record.get("miller_index")
        shift = record.get("shift")
        if miller is None or shift is None:
            continue

        if len(miller) != 3:
            raise SystemExit(
                f"Invalid Miller index for {system_id}: {miller!r}"
            )

        rows.append(
            {
                "system_id": system_id,
                "ads_id": int(ads_id),
                "bulk_id": record.get("bulk_id"),
                "bulk_mpid": record.get("bulk_mpid"),
                "bulk_symbols": record.get("bulk_symbols"),
                "ads_symbols": record.get("ads_symbols"),
                "miller_index": str(tuple(int(x) for x in miller)),
                "shift": float(shift),
                "top": record.get("top"),
                "adsorption_site": str(record.get("adsorption_site")),
            }
        )

    if not rows:
        raise SystemExit("No matching N* metadata records were generated.")

    df = pd.DataFrame(rows).drop_duplicates("system_id")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Generated: {args.out}")
    print(f"Metadata rows: {len(df):,}")
    print(f"Missing mapping IDs: {len(missing_mapping):,}")
    print(f"Unexpected adsorbate IDs: {len(wrong_adsorbate):,}")
    print("Columns:", ", ".join(df.columns))

    if missing_mapping:
        print("First missing mapping IDs:", missing_mapping[:5])
    if wrong_adsorbate:
        print("First unexpected adsorbate IDs:", wrong_adsorbate[:5])


if __name__ == "__main__":
    main()


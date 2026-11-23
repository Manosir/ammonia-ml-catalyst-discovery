#!/usr/bin/env python3
"""Strictly audit and join OC20 mapping/reference assets without changing raw data."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_MAPPING_MD5 = "6b5d485019861f6e7efca38338375b61"
ADSORBATES = ("N", "N2", "NH", "NH2", "NH3")
ANOMALY_NAMES = {0: "none", 1: "dissociated", 2: "desorbed", 3: "surface_reconstructed", 4: "incorrect_CHCOH_placement"}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.replace(r"\.extxyz(?:\.xz)?$", "", regex=True)


def find_one(root: Path, name: str) -> Path | None:
    candidates = sorted((p for p in root.rglob(name) if p.is_file()), key=lambda p: (len(p.parts), str(p)))
    return candidates[0] if candidates else None


def mapping_frame(mapping: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = [{"system_id": str(key), **value} for key, value in mapping.items() if isinstance(value, dict)]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("OC20 mapping contains no dictionary records")
    frame["system_id"] = norm_id(frame["system_id"])
    if frame["system_id"].duplicated().any():
        raise ValueError("OC20 mapping contains duplicate system IDs")
    return frame


def mapping_adsorbate(value: Any) -> str | None:
    text = str(value).strip().replace("*", "").upper()
    for name in sorted(ADSORBATES, key=len, reverse=True):
        if text == name:
            return name
    return None


def read_all_references(root: Path) -> tuple[pd.DataFrame, int, int]:
    frames: list[pd.DataFrame] = []
    for file in sorted(root.rglob("system.txt")):
        adsorbate = next((a for a in ADSORBATES if a in file.parts), None)
        if adsorbate is None:
            continue
        raw = pd.read_csv(file, header=None, names=["system_id", "reference_energy"], dtype=str)
        raw["reference_energy"] = pd.to_numeric(raw["reference_energy"], errors="coerce")
        raw = raw.dropna(subset=["reference_energy"]).copy()
        raw["system_id"] = norm_id(raw["system_id"])
        raw["adsorbate"] = adsorbate
        raw["reference_file"] = str(file)
        frames.append(raw)
    if not frames:
        raise FileNotFoundError("No adsorbate system.txt reference files found")
    result = pd.concat(frames, ignore_index=True)
    duplicate_mask = result.duplicated(["system_id", "adsorbate"], keep=False)
    conflict_count = int(result[duplicate_mask].groupby(["system_id", "adsorbate"])["reference_energy"].nunique().gt(1).sum())
    duplicate_rows = int(duplicate_mask.sum())
    if conflict_count:
        raise ValueError(f"conflicting reference energies found for {conflict_count} system/adsorbate keys")
    return result.drop_duplicates(["system_id", "adsorbate"], keep="first"), duplicate_rows, conflict_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--raw", type=Path, default=Path("results/adsorbates/oc20_adsorbates.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/adsorbates"))
    parser.add_argument("--allow-md5-mismatch", action="store_true")
    args = parser.parse_args()

    mapping_path = find_one(args.data_root, "oc20_data_mapping.pkl")
    if mapping_path is None:
        raise FileNotFoundError("oc20_data_mapping.pkl was not found below --data-root")
    mapping_md5 = md5(mapping_path)
    if mapping_md5 != EXPECTED_MAPPING_MD5 and not args.allow_md5_mismatch:
        raise SystemExit(f"MD5 mismatch before unpickle: {mapping_md5}; expected {EXPECTED_MAPPING_MD5}. Use --allow-md5-mismatch only for a verified release.")
    with mapping_path.open("rb") as handle:
        mapping = pickle.load(handle)
    map_frame = mapping_frame(mapping)
    map_frame["mapping_adsorbate"] = map_frame.get("ads_symbols", pd.Series(pd.NA, index=map_frame.index)).map(mapping_adsorbate)

    raw = pd.read_csv(args.raw)
    if "system_id" not in raw or "adsorbate" not in raw:
        raise ValueError("raw table must contain system_id and adsorbate")
    raw["system_id"] = norm_id(raw["system_id"])
    raw["adsorbate"] = raw["adsorbate"].astype(str).str.upper()
    if raw["system_id"].duplicated().any():
        raise ValueError("raw extraction contains duplicate system IDs")

    refs, duplicate_rows, conflict_count = read_all_references(args.data_root)
    joined = raw.merge(map_frame, on="system_id", how="left", indicator="_mapping_merge", validate="one_to_one", suffixes=("", "_mapping"))
    mapping_missing = int((joined["_mapping_merge"] != "both").sum())
    if mapping_missing:
        raise ValueError(f"mapping coverage is {len(joined)-mapping_missing}/{len(joined)}; missing {mapping_missing} rows")
    joined = joined.drop(columns="_mapping_merge")
    joined["mapping_adsorbate"] = joined["mapping_adsorbate"].fillna("")
    mismatch = joined["mapping_adsorbate"].ne(joined["adsorbate"])
    if mismatch.any():
        raise ValueError(f"mapping/raw adsorbate mismatch in {int(mismatch.sum())} rows")
    joined = joined.merge(refs, on=["system_id", "adsorbate"], how="left", validate="one_to_one")
    if joined["reference_energy"].isna().any():
        raise ValueError(f"reference coverage is {joined['reference_energy'].notna().mean():.3%}")

    joined["official_adsorption_energy_eV"] = joined["energy_final_eV"] - joined["reference_energy"]
    joined["official_adsorption_energy_available"] = True
    relaxation = joined["energy_final_eV"] - joined["energy_initial_eV"]
    joined["relaxation_energy_change_eV"] = relaxation
    joined["qc_negative_height_final_audit"] = pd.to_numeric(joined.get("ads_height"), errors="coerce") < 0
    joined["qc_positive_relaxation"] = relaxation > 0
    joined["qc_large_relaxation"] = relaxation.abs() > 100
    joined["qc_official_outside_initial_window"] = ~joined["official_adsorption_energy_eV"].between(-15, 10)
    anomaly = pd.to_numeric(joined.get("anomaly"), errors="coerce")
    for code, name in ANOMALY_NAMES.items():
        joined[f"qc_anomaly_{name}"] = anomaly.eq(code)
    joined["qc_mapping_anomaly"] = anomaly.notna() & anomaly.ne(0)
    joined["qc_suspect"] = joined[["qc_negative_height_final_audit", "qc_positive_relaxation", "qc_large_relaxation", "qc_mapping_anomaly"]].any(axis=1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out_dir / "oc20_adsorbates_official_join.parquet"
    csv_path = args.out_dir / "oc20_adsorbates_official_join.csv"
    summary_path = args.out_dir / "oc20_asset_audit.json"
    try:
        joined.to_parquet(parquet_path, index=False)
    except ImportError:
        parquet_path = None
    joined.to_csv(csv_path, index=False)

    crosstab = pd.crosstab(joined["adsorbate"], joined["mapping_adsorbate"]).to_dict()
    percentile = {}
    for adsorbate, group in joined.groupby("adsorbate"):
        values = group["official_adsorption_energy_eV"]
        percentile[adsorbate] = {"min": float(values.min()), "p01": float(values.quantile(.01)), "p50": float(values.quantile(.50)), "p99": float(values.quantile(.99)), "max": float(values.max()), "outside_-15_to_10": int((~values.between(-15, 10)).sum())}
    anomaly_by_ads = joined.groupby(["adsorbate", "anomaly"], dropna=False).size().to_dict()
    overlap = {name: int((joined["qc_mapping_anomaly"] & joined[col]).sum()) for name, col in [("negative_height", "qc_negative_height_final_audit"), ("positive_relaxation", "qc_positive_relaxation"), ("large_relaxation", "qc_large_relaxation")]}
    extreme_referenced = int((joined["qc_large_relaxation"] & joined["qc_official_outside_initial_window"]).sum())

    summary = {"mapping_path": str(mapping_path), "mapping_md5": mapping_md5, "mapping_md5_matches_official": mapping_md5 == EXPECTED_MAPPING_MD5, "raw_rows": len(raw), "joined_rows": len(joined), "mapping_missing_rows": mapping_missing, "reference_rows": len(refs), "reference_duplicate_rows": duplicate_rows, "reference_conflict_keys": conflict_count, "mapping_adsorbate_crosstab": crosstab, "official_adsorption_energy_percentiles": percentile, "anomaly_counts_by_adsorbate_and_code": {str(k): int(v) for k, v in anomaly_by_ads.items()}, "anomaly_overlap_counts": overlap, "extreme_relaxation_rows": int(joined["qc_large_relaxation"].sum()), "extreme_rows_still_outside_referenced_window": extreme_referenced, "joined_parquet": str(parquet_path) if parquet_path else None, "joined_csv": str(csv_path), "anomaly_code_meanings": ANOMALY_NAMES}
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

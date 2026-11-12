#!/usr/bin/env python3
"""Build a strict, regime-aware model table without modifying raw extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ammonia_ai.schema import SCHEMA_VERSION, SCREENING_FEATURES, assert_features_allowed


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("results/adsorbates/oc20_adsorbates.csv"))
    parser.add_argument("--official-join", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("results/adsorbates/feature_availability.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/adsorbates/oc20_adsorbates_features.parquet"))
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("results/adsorbates/model_manifest.json"))
    parser.add_argument("--allow-partial-references", action="store_true")
    args = parser.parse_args()

    assert_features_allowed([], args.registry)
    raw = pd.read_csv(args.raw)
    frame = pd.read_parquet(args.official_join) if args.official_join.suffix == ".parquet" else pd.read_csv(args.official_join)
    if len(frame) != len(raw):
        raise ValueError(f"joined row count {len(frame)} differs from raw row count {len(raw)}")
    required = {"system_id", "adsorbate", "bulk_id", "ads_id", "mapping_adsorbate", "reference_energy", "official_adsorption_energy_eV"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"official join missing required columns: {missing}")
    if frame["system_id"].duplicated().any():
        raise ValueError("official join contains duplicate system_id")
    if frame["reference_energy"].isna().any() and not args.allow_partial_references:
        raise ValueError("official join contains missing reference energies")
    if frame["official_adsorption_energy_eV"].isna().any() and not args.allow_partial_references:
        raise ValueError("official join contains missing official adsorption energies")
    if frame["mapping_adsorbate"].astype(str).str.upper().ne(frame["adsorbate"].astype(str).str.upper()).any():
        raise ValueError("mapping adsorbate disagrees with folder-derived adsorbate")

    frame["relaxation_energy_change_eV"] = frame["energy_final_eV"] - frame["energy_initial_eV"]
    frame["adsorption_energy_eV"] = frame["official_adsorption_energy_eV"]
    frame["adsorption_energy_available"] = frame["adsorption_energy_eV"].notna()
    if "ads_height" in frame:
        frame["ads_height_final_audit"] = frame.pop("ads_height")
    if "mean_ads_force" in frame:
        frame["mean_ads_force_final_audit"] = frame.pop("mean_ads_force")
    frame["qc_large_relaxation"] = frame["relaxation_energy_change_eV"].abs() > 100
    frame["qc_positive_relaxation"] = frame["relaxation_energy_change_eV"] > 0
    frame["qc_official_outside_initial_window"] = ~frame["adsorption_energy_eV"].between(-15, 10)
    anomaly = pd.to_numeric(frame["anomaly"], errors="coerce") if "anomaly" in frame else pd.Series(np.nan, index=frame.index)
    frame["qc_mapping_missing"] = anomaly.isna()
    frame["qc_mapping_anomaly"] = anomaly.notna() & anomaly.ne(0)
    for code, name in {0: "none", 1: "dissociated", 2: "desorbed", 3: "surface_reconstructed", 4: "incorrect_CHCOH_placement"}.items():
        frame[f"qc_anomaly_{name}"] = anomaly.eq(code)
    frame["qc_suspect"] = frame[["qc_large_relaxation", "qc_positive_relaxation", "qc_mapping_anomaly", "qc_official_outside_initial_window"]].any(axis=1)

    for feature in SCREENING_FEATURES:
        if feature not in frame:
            frame[feature] = np.nan
    implemented = pd.read_csv(args.registry)
    required_features = implemented.loc[(implemented.role == "feature") & (implemented.available_at_screening == "yes") & (implemented.implemented == "yes"), "name"].tolist()
    unavailable = [feature for feature in required_features if feature not in frame.columns or frame[feature].isna().any()]
    training_ready = not unavailable and not frame["qc_mapping_missing"].any() and frame["adsorption_energy_available"].all()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv_out, index=False)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "screening_regime": "thermal Haber-Bosch",
        "screening_input": "unrelaxed slab + adsorbate",
        "group_key": "bulk_id",
        "split_policy": ["grouped_by_bulk_id", "leave_chemical_system_out", "leave_element_out"],
        "anomaly_policy": "retain flags; compare all rows versus exclude-anomalies sensitivity run",
        "raw_input": str(args.raw),
        "raw_md5": file_md5(args.raw),
        "official_join_input": str(args.official_join),
        "official_join_md5": file_md5(args.official_join),
        "row_count": len(frame),
        "feature_columns": list(SCREENING_FEATURES),
        "implemented_feature_columns": required_features,
        "excluded_columns": ["ads_height_final_audit", "mean_ads_force_final_audit", "energy_initial_eV", "energy_final_eV", "relaxation_energy_change_eV", "reference_energy", "anomaly", "official_adsorption_energy_eV", "system_id", "bulk_id", "ads_id"],
        "git_commit": git_commit,
        "training_ready": training_ready,
        "training_ready_reasons": [] if training_ready else ["feature registry has unimplemented or missing screening features", "run grouped validation only after initial-frame descriptors are complete"],
        "output": str(args.out),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(frame):,} rows to {args.out}")
    print(f"Training-ready: {training_ready}; missing/unimplemented features: {unavailable}")
    print(f"Wrote manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

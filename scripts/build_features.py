#!/usr/bin/env python3
"""Build the screening-v3 model table: audited official join + contract features.

Never touches the raw extraction. training_ready is True only when every contract feature is
registered, implemented, present and finite for every row.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ammonia_ai import schema as S

ANOMALY_NAMES = {0: "none", 1: "dissociated", 2: "desorbed", 3: "surface_reconstructed", 4: "incorrect_CHCOH_placement"}


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return commit, bool(status)
    except Exception:
        return None, None


def _json_default(o):
    if hasattr(o, "item") and getattr(o, "size", 1) == 1:
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _hkl(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    a = np.abs(np.asarray(value, dtype=float))
    return float(a.max()), float((a ** 2).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("results/adsorbates/oc20_adsorbates.csv"))
    parser.add_argument("--official-join", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=Path("results/structure_pass/candidate_features.parquet"))
    parser.add_argument("--registry", type=Path, default=Path("results/adsorbates/feature_availability.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/adsorbates/oc20_adsorbates_features.parquet"))
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("results/adsorbates/model_manifest.json"))
    parser.add_argument("--allow-partial-references", action="store_true")
    args = parser.parse_args()

    # the REAL contract must pass the registry gate (fails loudly on a stale registry)
    try:
        S.assert_features_allowed(S.SCREENING_FEATURES, args.registry)
    except ValueError as exc:
        raise SystemExit(f"{exc}\nRegenerate the registry first: python3 scripts/feature_availability.py") from exc

    raw = pd.read_csv(args.raw)
    frame = pd.read_parquet(args.official_join) if args.official_join.suffix == ".parquet" else pd.read_csv(args.official_join)
    if len(frame) != len(raw):
        raise ValueError(f"joined row count {len(frame)} differs from raw row count {len(raw)}")
    required = {
        "system_id", "adsorbate", "bulk_id", "ads_id", "bulk_mpid", "bulk_symbols", "miller_index",
        "mapping_adsorbate", "reference_energy", "official_adsorption_energy_eV", "energy_final_eV",
        "energy_initial_eV", "anomaly",
    }
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

    # group keys, target, audit columns, QC flags
    frame["bulk_reduced_formula"] = frame["bulk_symbols"].map(S.reduced_formula)
    frame["bulk_chemsys"] = frame["bulk_symbols"].map(S.chemical_system)
    frame = frame.drop(columns=["delta_E_ads"], errors="ignore")  # legacy name; equals the relaxation change
    frame["relaxation_energy_change_eV"] = frame["energy_final_eV"] - frame["energy_initial_eV"]
    frame["adsorption_energy_eV"] = frame["official_adsorption_energy_eV"]
    frame["adsorption_energy_available"] = frame["adsorption_energy_eV"].notna()
    if "ads_height" in frame:
        frame["ads_height_final_audit"] = frame.pop("ads_height")
    if "mean_ads_force" in frame:
        frame["mean_ads_force_final_audit"] = frame.pop("mean_ads_force")
    frame["qc_large_relaxation"] = frame["relaxation_energy_change_eV"].abs() > 100
    frame["qc_positive_relaxation"] = frame["relaxation_energy_change_eV"] > 0
    frame["qc_official_outside_initial_window"] = ~frame["adsorption_energy_eV"].between(-15, 10)  # placeholder window
    anomaly = pd.to_numeric(frame["anomaly"], errors="coerce")
    frame["qc_mapping_missing"] = anomaly.isna()
    frame["qc_mapping_anomaly"] = anomaly.notna() & anomaly.ne(0)
    for code, name in ANOMALY_NAMES.items():
        frame[f"qc_anomaly_{name}"] = anomaly.eq(code)
    frame["qc_suspect"] = frame[
        ["qc_large_relaxation", "qc_positive_relaxation", "qc_mapping_anomaly", "qc_official_outside_initial_window"]
    ].any(axis=1)  # audit only, never a filter
    frame["anomaly_label"] = anomaly.map(ANOMALY_NAMES)
    frame["is_clean_primary_regression_row"] = anomaly.eq(0)

    # contract features computed here: adsorbate descriptors and Miller encodings
    counts = [S.adsorbate_counts(a) for a in frame["adsorbate"]]
    frame["ads_nN"] = np.array([c[0] for c in counts], dtype=float)
    frame["ads_nH"] = np.array([c[1] for c in counts], dtype=float)
    hk = np.array([_hkl(v) for v in frame["miller_index"]])
    frame["miller_hkl_max"], frame["miller_norm2"] = hk[:, 0], hk[:, 1]

    # contract features computed by scripts/build_candidates.py from the initial-frame structure pass
    cand = pd.read_parquet(args.candidates)
    absent = [c for c in S.CANDIDATE_COLUMNS if c not in cand.columns]
    if absent:
        raise SystemExit(f"candidate table lacks contract columns {absent[:6]}...: run scripts/build_candidates.py")
    clash = [c for c in S.CANDIDATE_COLUMNS if c in frame.columns]
    if clash:
        raise ValueError(f"join table already has contract columns: {clash[:6]}")
    cand = cand[["system_id"] + list(S.CANDIDATE_COLUMNS)]
    if cand["system_id"].duplicated().any():
        raise ValueError("candidate table contains duplicate system_id")
    n_before = len(frame)
    frame = frame.merge(cand, on="system_id", how="left", validate="one_to_one")
    if len(frame) != n_before:
        raise ValueError("candidate merge changed the row count")

    # readiness: every contract feature registered, implemented, present, finite
    registry = pd.read_csv(args.registry)
    implemented = set(
        registry.loc[
            (registry.role == "feature") & (registry.available_at_screening == "yes") & (registry.implemented == "yes"),
            "name",
        ]
    )
    reasons = []
    unimplemented = sorted(set(S.SCREENING_FEATURES) - implemented)
    if unimplemented:
        reasons.append(f"contract features not registered as implemented: {unimplemented}")
    absent_cols = [f for f in S.SCREENING_FEATURES if f not in frame.columns]
    if absent_cols:
        reasons.append(f"contract columns absent from the table: {absent_cols}")
    present = [f for f in S.SCREENING_FEATURES if f in frame.columns]
    with_nan = [f for f in present if frame[f].isna().any()]
    if with_nan:
        reasons.append(f"contract columns with missing values: {with_nan}")
    with_inf = [f for f in present if np.isinf(frame[f].to_numpy(dtype=float)).any()]
    if with_inf:
        reasons.append(f"contract columns with infinite values: {with_inf}")
    if frame["qc_mapping_missing"].any():
        reasons.append("rows missing from OC20 mapping")
    if not frame["adsorption_energy_available"].all():
        reasons.append("missing adsorption energies")
    training_ready = not reasons

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv_out, index=False)
    commit, dirty = git_state()
    manifest = {
        "schema_version": S.SCHEMA_VERSION,
        "screening_regime": "thermal Haber-Bosch",
        "screening_input": "unrelaxed slab + adsorbate (+ adsorption-site coordinate)",
        "group_key": "bulk_chemsys",
        "group_keys": list(S.GROUP_KEYS),
        "split_policy": ["grouped_by_bulk_chemsys", "leave_element_out", "leave_adsorbate_out_NH2", "official_oc20_split"],
        "anomaly_policy": "primary regression on anomaly==0; classifier head for anomalies; all-row metrics reported; qc_suspect never filters",
        "raw_input": str(args.raw),
        "raw_md5": file_md5(args.raw),
        "official_join_input": str(args.official_join),
        "official_join_md5": file_md5(args.official_join),
        "candidates_input": str(args.candidates),
        "candidates_md5": file_md5(args.candidates),
        "row_count": len(frame),
        "feature_columns": list(S.SCREENING_FEATURES),
        "placement_sensitive_features": list(S.PLACEMENT_SENSITIVE_FEATURES),
        "implemented_feature_columns": sorted(implemented),
        "excluded_columns": list(S.AUDIT_COLUMNS) + list(S.STRUCTURE_PASS_AUDIT_COLUMNS) + list(S.IDENTIFIER_COLUMNS),
        "git_commit": commit,
        "git_dirty": dirty,
        "training_ready": training_ready,
        "training_ready_reasons": reasons,
        "output": str(args.out),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(f"Wrote {len(frame):,} rows, {frame.shape[1]} columns to {args.out}")
    print(f"Contract features: {len(S.SCREENING_FEATURES)} | clean primary-regression rows: {int(frame['is_clean_primary_regression_row'].sum()):,}")
    print(f"Training-ready: {training_ready}")
    for reason in reasons:
        print("  -", reason[:300])
    print(f"Wrote manifest to {args.manifest} (git dirty: {dirty})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

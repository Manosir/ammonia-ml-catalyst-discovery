#!/usr/bin/env python3
"""Triage Report Generator v2 for commercial catalyst screening.

Input: Candidate feature table (results/structure_pass/candidate_features.parquet
       joined with metadata from the official join).
Output: Structured triage report with predicted E_ads, conformal prediction intervals,
        applicability domain flags (v2), reliability tier, and DFT prioritization ranking.

New in v2:
- Loads pre-computed AD model from results/models/ad_model.joblib (never recomputes)
- Split-conformal per-adsorbate prediction intervals
- 4 independent AD flags: ad_unseen_element, ad_far_from_training, ad_magnetic, ad_nonmetal_rich
- Reliability tier: high / medium / low
- Acquisition score: rank-normalized variance + diversity
- Optional --demo mode: out-of-fold evaluation on model table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ammonia_ai import schema as S
from ammonia_ai.uncertainty import (
    load_ad_model,
    compute_triage_predictions_v2,
    validate_uncertainty_on_heldout,
)


def load_models(model_dir: Path) -> tuple:
    """Load the median model."""
    median_path = model_dir / "screening_v3_full.joblib"
    manifest_path = model_dir / "manifest.json"

    if not median_path.exists():
        raise FileNotFoundError(f"Median model not found: {median_path}")

    # Schema match check
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        S.assert_model_matches_schema(joblib.load(median_path), manifest)

    model_median = joblib.load(median_path)
    print("Loaded median model", file=sys.stderr)
    return model_median


def load_candidate_table(
    candidate_path: Path,
    metadata_path: Path,
) -> pd.DataFrame:
    """Load and merge candidate features with metadata."""
    # Load candidate features
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidate_path}")
    cand = pd.read_parquet(candidate_path)
    # Load metadata (system_id, bulk_mpid, adsorbate, miller_index, shift, etc.)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    meta = pd.read_parquet(metadata_path)

    # Ensure system_id is normalized
    from ammonia_ai.schema import normalize_system_id_column
    cand = normalize_system_id_column(cand)
    meta = normalize_system_id_column(meta)

    # Merge on system_id
    merged = cand.merge(meta, on="system_id", how="inner")
    print(f"Loaded {len(merged)} candidates with metadata", file=sys.stderr)
    return merged


def run_demo_mode(
    model_median,
    ad_model: dict,
    model_table_path: Path,
    out_dir: Path,
) -> dict:
    """Run out-of-fold demo mode on the model table.

    Uses grouped CV on the training table to evaluate uncertainty quality.
    """
    print("\nRunning DEMO mode: out-of-fold evaluation on model table...", file=sys.stderr)

    # Load full model table
    if not model_table_path.exists():
        raise FileNotFoundError(f"Model table not found: {model_table_path}")
    df = pd.read_parquet(model_table_path)

    # Filter to clean primary regression rows
    if "is_clean_primary_regression_row" in df.columns:
        df_clean = df[df["is_clean_primary_regression_row"]].copy()
    else:
        df_clean = df.copy()
        print("WARNING: 'is_clean_primary_regression_row' not found; using all rows", file=sys.stderr)

    print(f"  Clean rows: {len(df_clean)}", file=sys.stderr)

    # Run validation
    results = validate_uncertainty_on_heldout(df_clean, model_median, ad_model)

    # Save demo results
    out_dir.mkdir(parents=True, exist_ok=True)
    demo_json = out_dir / "demo_validation.json"
    demo_parquet = out_dir / "demo_predictions.parquet"

    # Also generate predictions for all clean rows (not just held-out)
    pred_df = compute_triage_predictions_v2(df_clean, model_median, ad_model)
    pred_df.to_parquet(demo_parquet, index=False)

    # Convert numpy types for JSON
    json_results = {
        "schema_version": S.SCHEMA_VERSION,
        "n_samples": results["n_samples"],
        "coverage_by_adsorbate": results["coverage_by_adsorbate"],
        "spearman_width_vs_error": results["spearman_width_vs_error"],
        "ood_rate_per_flag": results["ood_rate_per_flag"],
        "mae_by_tier": results["mae_by_tier"],
    }
    demo_json.write_text(json.dumps(json_results, indent=2))

    print(f"  Demo validation saved to {demo_json}", file=sys.stderr)
    print(f"  Demo predictions saved to {demo_parquet}", file=sys.stderr)

    # Print summary
    print("\n" + "=" * 60)
    print("DEMO MODE RESULTS (Out-of-Fold)")
    print("=" * 60)
    print(f"\nTotal samples: {results['n_samples']}")

    print("\n--- 90% Coverage per Adsorbate ---")
    for ads, cov in sorted(results["coverage_by_adsorbate"].items()):
        status = "✓" if cov >= 0.85 else "✗"
        print(f"  {ads:6s}: {cov:.3f} {status}")
    avg_cov = np.mean(list(results["coverage_by_adsorbate"].values()))
    print(f"  Average: {avg_cov:.3f}")

    print("\n--- Spearman(Width vs |Error|) ---")
    rho = results["spearman_width_vs_error"]["rho"]
    pval = results["spearman_width_vs_error"]["p_value"]
    print(f"  rho = {rho:.4f}, p = {pval:.4f}")

    print("\n--- OOD Rate per AD Flag ---")
    for flag, rate in results["ood_rate_per_flag"].items():
        print(f"  {flag:30s}: {rate:.3f}")

    print("\n--- MAE per Reliability Tier ---")
    for tier in ["high", "medium", "low"]:
        mae = results["mae_by_tier"].get(tier)
        if mae is not None:
            print(f"  {tier:6s}: {mae:.4f} eV")
        else:
            print(f"  {tier:6s}: N/A")

    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, default=Path("results/structure_pass/candidate_features.parquet"),
                    help="Candidate features parquet")
    ap.add_argument("--metadata", type=Path, default=Path("results/adsorbates/oc20_adsorbates_official_join.parquet"),
                    help="Metadata table (official join)")
    ap.add_argument("--model-dir", type=Path, default=Path("results/models"),
                    help="Directory containing trained models")
    ap.add_argument("--ad-model", type=Path, default=Path("results/models/ad_model.joblib"),
                    help="Path to AD model artifact")
    ap.add_argument("--out", type=Path, default=Path("results/triage_report.json"),
                    help="Output triage report (JSON)")
    ap.add_argument("--out-parquet", type=Path, default=Path("results/triage_report.parquet"),
                    help="Output triage report (Parquet)")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Number of top priority candidates to highlight in summary")
    ap.add_argument("--demo", action="store_true",
                    help="Run out-of-fold demo mode on model table (results/adsorbates/oc20_adsorbates_features.parquet)")
    ap.add_argument("--model-table", type=Path, default=Path("results/adsorbates/oc20_adsorbates_features.parquet"),
                    help="Model table for demo mode")
    ap.add_argument("--demo-out", type=Path, default=Path("results/demo"),
                    help="Output directory for demo mode results")
    a = ap.parse_args(argv)

    # Load median model
    try:
        model_median = load_models(a.model_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Load AD model (required - no fallback)
    try:
        ad_model = load_ad_model(a.ad_model)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Run scripts/validate_uncertainty.py --build first to create ad_model.joblib", file=sys.stderr)
        return 1

    # Demo mode: evaluate on held-out training systems
    if a.demo:
        try:
            run_demo_mode(model_median, ad_model, a.model_table, a.demo_out)
        except Exception as e:
            print(f"ERROR in demo mode: {e}", file=sys.stderr)
            return 1
        return 0

    # Normal triage mode
    # Load candidate table
    try:
        df = load_candidate_table(a.candidates, a.metadata)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Production gate: the candidate table must carry the full 64-feature
    # screening contract in the canonical training order.
    S.validate_feature_frame(df, features=S.SCREENING_FEATURES, allow_missing=False)

    # Generate triage predictions using v2 uncertainty (loads from ad_model only)
    report_df = compute_triage_predictions_v2(df, model_median, ad_model)

    # Sort by acquisition score (highest priority first)
    report_df = report_df.sort_values("acquisition_score", ascending=False).reset_index(drop=True)

    # Write outputs
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out_parquet.parent.mkdir(parents=True, exist_ok=True)

    report_df.to_parquet(a.out_parquet, index=False)

    # JSON summary with top-k highlighted
    ad_flag_cols = ["ad_unseen_element", "ad_far_from_training", "ad_magnetic", "ad_nonmetal_rich"]
    summary = {
        "schema_version": S.SCHEMA_VERSION,
        "n_candidates": int(len(report_df)),
        "ad_flags": report_df[ad_flag_cols].sum().to_dict(),
        "reliability_tier": report_df["reliability_tier"].value_counts().to_dict(),
        "top_candidates": report_df.head(a.top_k)[
            ["system_id", "bulk_mpid", "adsorbate", "miller_index", "shift",
             "predicted_E_ads_eV", "ci_low_eV", "ci_high_eV", "ci_width_eV",
             "ad_unseen_element", "ad_far_from_training", "ad_magnetic", "ad_nonmetal_rich",
             "reliability_tier", "acquisition_score"]
        ].to_dict(orient="records"),
        "all_candidates_parquet": str(a.out_parquet),
    }
    a.out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"Triage report written to {a.out} and {a.out_parquet}", file=sys.stderr)
    print(f"Top {a.top_k} candidates by acquisition score:", file=sys.stderr)
    for i, row in report_df.head(a.top_k).iterrows():
        print(f"  {i+1:2d}. {row['system_id']} ({row['adsorbate']}) "
              f"E_ads={row['predicted_E_ads_eV']:.3f}±{(row['ci_width_eV']/2):.3f} "
              f"tier={row['reliability_tier']} "
              f"score={row['acquisition_score']:.3f}",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
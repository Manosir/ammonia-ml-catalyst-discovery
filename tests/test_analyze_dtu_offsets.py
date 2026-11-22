#!/usr/bin/env python3
"""Tests for DTU RPA functional offsets analysis script (scripts/analyze_dtu_offsets.py)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms
from ase.db import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_dtu_offsets import (  # noqa: E402
    DISCLAIMER_NOTE,
    MAGNETIC_ELEMENTS,
    MISSING_SPECIES_NOTE,
    TARGET_OFFSETS,
    compute_stats,
    export_csv,
    export_json,
    export_markdown,
    extract_row_offsets,
    load_and_extract_offsets,
    summarize_dataset,
)


def test_compute_stats_synthetic():
    """Verify compute_stats produces accurate descriptive statistics and bootstrap CI."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = compute_stats(data, seed=42, n_bootstrap=500)
    assert stats["count"] == 5
    assert math.isclose(stats["mean"], 3.0)
    assert math.isclose(stats["median"], 3.0)
    assert math.isclose(stats["std"], np.std(data, ddof=1))
    assert math.isclose(stats["mad"], 1.0)  # median(|[1,2,3,4,5] - 3|) = median([2,1,0,1,2]) = 1.0
    assert math.isclose(stats["min"], 1.0)
    assert math.isclose(stats["max"], 5.0)
    assert stats["ci_95_low"] <= stats["mean"] <= stats["ci_95_high"]


def test_compute_stats_reproducibility():
    """Verify bootstrap CI is reproducible with fixed seed."""
    data = [0.1, -0.4, 0.5, 0.8, -0.2, 0.3]
    s1 = compute_stats(data, seed=123)
    s2 = compute_stats(data, seed=123)
    assert s1["ci_95_low"] == s2["ci_95_low"]
    assert s1["ci_95_high"] == s2["ci_95_high"]


def test_extract_row_offsets_magnetic_and_non_magnetic():
    """Verify offset targets and magnetic classification on a single row dictionary."""
    kvp_fe = {
        "surf_mat": "Fe",
        "adsorbate": "N",
        "PBE_adsorp": -1.0,
        "RPBE_adsorp": -0.8,
        "RPA_EXX_adsorp": -0.6,
        "RPA_adsorp": -0.5,
        "EXX_adsorp": -0.1,
        "BEEFvdW_adsorp": -0.9,
        "mBEEF_adsorp": -0.95,
        "LDA_adsorp": -1.5,
    }
    rec_fe = extract_row_offsets(kvp_fe, row_id=1)
    assert rec_fe is not None
    assert rec_fe["magnetic_group"] == "magnetic"
    # PBE - RPBE = -1.0 - (-0.8) = -0.2
    assert math.isclose(rec_fe["PBE_minus_RPBE"], -0.2)
    # RPA_total - PBE = -0.6 - (-1.0) = +0.4
    assert math.isclose(rec_fe["RPA_total_minus_PBE"], 0.4)
    # RPA_total - RPBE = -0.6 - (-0.8) = +0.2
    assert math.isclose(rec_fe["RPA_total_minus_RPBE"], 0.2)
    # BEEFvdW - RPBE = -0.9 - (-0.8) = -0.1
    assert math.isclose(rec_fe["BEEFvdW_minus_RPBE"], -0.1)
    # mBEEF - PBE = -0.95 - (-1.0) = +0.05
    assert math.isclose(rec_fe["mBEEF_minus_PBE"], 0.05)
    # LDA - PBE = -1.5 - (-1.0) = -0.5
    assert math.isclose(rec_fe["LDA_minus_PBE"], -0.5)
    # EXX - PBE = -0.1 - (-1.0) = +0.9
    assert math.isclose(rec_fe["EXX_minus_PBE"], 0.9)

    kvp_pt = {**kvp_fe, "surf_mat": "Pt"}
    rec_pt = extract_row_offsets(kvp_pt, row_id=2)
    assert rec_pt is not None
    assert rec_pt["magnetic_group"] == "non-magnetic"


def test_rpa_total_separate_from_correlation():
    """Assert RPA_total uses RPA_EXX_adsorp and RPA_adsorp is kept as separate correlation column."""
    kvp = {
        "surf_mat": "Ru",
        "adsorbate": "N",
        "PBE_adsorp": -1.0,
        "RPBE_adsorp": -0.8,
        "EXX_adsorp": -0.0194,
        "RPA_adsorp": -0.5679,
        "RPA_EXX_adsorp": -0.5873,  # EXX + RPA
        "BEEFvdW_adsorp": -0.85,
        "mBEEF_adsorp": -0.92,
        "LDA_adsorp": -1.4,
    }
    rec = extract_row_offsets(kvp, row_id=10)
    assert rec is not None
    assert rec["RPA_EXX_adsorp"] == -0.5873
    assert rec["RPA_corr_adsorp"] == -0.5679
    # Ensure RPA_total_minus_RPBE uses -0.5873, NOT -0.5679
    # -0.5873 - (-0.8) = +0.2127
    assert math.isclose(rec["RPA_total_minus_RPBE"], -0.5873 - (-0.8))
    # Not using correlation term alone (-0.5679 - (-0.8) = +0.2321)
    assert not math.isclose(rec["RPA_total_minus_RPBE"], -0.5679 - (-0.8))


def test_synthetic_ase_db_full_pipeline(tmp_path: Path):
    """Build a synthetic ASE DB and test full pipeline including report generation."""
    db_path = tmp_path / "synthetic_adsorption.db"
    db = connect(str(db_path))

    # Add 4 synthetic records: 2 magnetic (Fe, Co), 2 non-magnetic (Pt, Ru), 2 adsorbates (N, H)
    rows_data = [
        ("Fe", "N", -1.0, -0.8, -0.6, -0.5, -0.1, -0.9, -0.95, -1.5),
        ("Co", "H", -0.5, -0.4, -0.3, -0.25, -0.05, -0.45, -0.48, -0.8),
        ("Pt", "N", -1.2, -1.0, -0.9, -0.7, -0.2, -1.1, -1.15, -1.8),
        ("Ru", "H", -0.8, -0.65, -0.55, -0.45, -0.1, -0.72, -0.75, -1.2),
    ]

    for surf, ads, pbe, rpbe, rpa_exx, rpa_corr, exx, beef, mbeef, lda in rows_data:
        db.write(
            Atoms(surf),
            surf_mat=surf,
            adsorbate=ads,
            PBE_adsorp=pbe,
            RPBE_adsorp=rpbe,
            RPA_EXX_adsorp=rpa_exx,
            RPA_adsorp=rpa_corr,
            EXX_adsorp=exx,
            BEEFvdW_adsorp=beef,
            mBEEF_adsorp=mbeef,
            LDA_adsorp=lda,
        )

    df = load_and_extract_offsets(db_path)
    assert len(df) == 4
    assert set(df["magnetic_group"]) == {"magnetic", "non-magnetic"}

    summary = summarize_dataset(df)
    assert set(summary["overall"].keys()) == set(TARGET_OFFSETS)
    assert "Fe" in summary["by_surface_element"]
    assert "Pt" in summary["by_surface_element"]
    assert "N" in summary["by_adsorbate"]
    assert "H" in summary["by_adsorbate"]

    output_dir = tmp_path / "results"
    csv_path = output_dir / "functional_offsets.csv"
    json_path = output_dir / "functional_offsets.json"
    md_path = output_dir / "functional_offsets.md"

    export_csv(df, csv_path)
    export_json(summary, "dummy_sha256", db_path, json_path, row_count=len(df))
    export_markdown(summary, "dummy_sha256", db_path, md_path, row_count=len(df))

    assert csv_path.exists()
    assert json_path.exists()
    assert md_path.exists()

    # Verify JSON content
    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_data["metadata"]["row_count"] == 4
    assert "NH, NH2, NH3" in json_data["metadata"]["missing_species_note"]
    assert "DO NOT merge into OC20" in json_data["metadata"]["usage_restriction"]

    # Verify MD content
    md_text = md_path.read_text(encoding="utf-8")
    assert "DTU Benchmark Offsets" in md_text
    assert "NH, NH2, NH3" in md_text
    assert "dummy_sha256" in md_text

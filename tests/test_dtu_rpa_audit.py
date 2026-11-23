import math
import sys
from pathlib import Path

import pytest

ase = pytest.importorskip("ase")
from ase import Atoms
from ase.db import connect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_dtu_rpa import (  # noqa: E402
    ADSORPTION_COLUMNS,
    SURFACE_COLUMNS,
    audit_db,
    audit_pairs,
    finite,
    first_value,
    pairing_key,
    row_kvp,
)
from scripts.download_dtu_rpa import download, main as download_main, sha256  # noqa: E402


def test_audit_pairs_detects_same_row_functionals(tmp_path: Path):
    path = tmp_path / "adsorption.db"
    db = connect(str(path))
    db.write(
        Atoms("Fe"),
        surf_mat="Fe",
        adsorbate="N",
        facet=110,
        coverage="1/4",
        PBE_adsorp=-1.0,
        RPBE_adsorp=-0.8,
        LDA_adsorp=-1.2,
    )
    report = audit_pairs(path)
    assert report["row_count"] == 1
    assert report["rows_with_at_least_two_functionals"] == 1
    assert report["matched_pair_counts_same_row"]["PBE__RPBE"] == 1
    assert report["rows_using_row_id_fallback"] == 0


def test_audit_pairs_marks_missing_semantic_key(tmp_path: Path):
    path = tmp_path / "adsorption.db"
    db = connect(str(path))
    db.write(Atoms("Fe"), PBE_adsorp=-1.0, RPBE_adsorp=-0.8)
    report = audit_pairs(path)
    assert report["rows_using_row_id_fallback"] == 1
    assert report["rows_with_at_least_two_functionals"] == 1


def test_audit_pairs_verifies_rpa_identity(tmp_path: Path):
    """Test that RPA_EXX_adsorp = EXX_adsorp + RPA_adsorp is verified for rows with all three columns."""
    path = tmp_path / "adsorption.db"
    db = connect(str(path))
    # Write a row with EXX, RPA, and RPA_EXX where the identity holds
    exx = -0.0194
    rpa = -0.5679
    rpa_exx = exx + rpa  # -0.5873
    db.write(
        Atoms("Fe"),
        surf_mat="Fe",
        adsorbate="N",
        facet=110,
        coverage="1/4",
        EXX_adsorp=exx,
        RPA_adsorp=rpa,
        RPA_EXX_adsorp=rpa_exx,
    )
    report = audit_pairs(path)
    assert report["rpa_identity_verified"] == 1
    assert report["rpa_identity_failed"] == 0
    # Also check the matched record includes RPA_EXX
    assert len(report["matched_records"]) == 1
    assert "RPA_EXX" in report["matched_records"][0]["available_functionals"]
    assert math.isclose(report["matched_records"][0]["values_eV"]["RPA_EXX"], rpa_exx, abs_tol=1e-6)


def test_audit_pairs_detects_rpa_identity_failure(tmp_path: Path):
    """Test that a failure of RPA_EXX = EXX + RPA is detected."""
    path = tmp_path / "adsorption.db"
    db = connect(str(path))
    # Write a row where the identity does NOT hold
    db.write(
        Atoms("Fe"),
        surf_mat="Fe",
        adsorbate="N",
        facet=110,
        coverage="1/4",
        EXX_adsorp=-0.0194,
        RPA_adsorp=-0.5679,
        RPA_EXX_adsorp=-0.1,  # Wrong value, should be -0.5873
    )
    report = audit_pairs(path)
    assert report["rpa_identity_verified"] == 0
    assert report["rpa_identity_failed"] == 1

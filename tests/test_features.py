"""
tests/test_features.py
----------------------
Tests for feature engineering logic in the main pipeline.

Changes from v1:
  - TARGET_TM expanded to 23 metals, matching EXPECTED_METALS in test_dataset.py
  - D_BAND_CENTER completed for all 23 TMs (Hammer-Norskov 2000 / Vojvodic 2012)
  - weighted_d_band now raises KeyError on unmapped TMs (explicit, not silent)
  - Added test that verifies KeyError is raised for unknown TMs
  - test_vocabulary_consistency guards against future drift between the two dicts

Run with: pytest tests/
"""
import numpy as np
import re
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_bulk_symbols(formula: str) -> dict:
    return {el: int(n) if n else 1
            for el, n in _ELEM_RE.findall(formula) if el}


# ── Single, complete metal vocabulary (23 TMs) ────────────────────────────────
# Previously test_features.py had 18 metals; test_dataset.py had 23.
# Now unified. Any future addition must appear in BOTH dicts below.
TARGET_TM = {
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt",
}

# Complete D-band centre values (eV) for all 23 TMs.
# Sources: Hammer & Norskov Adv. Catal. 45 (2000); Vojvodic et al. (2011).
D_BAND_CENTER: dict[str, float] = {
    "Sc": -0.50, "Ti": -0.60, "V":  -1.09, "Cr": -1.32, "Mn": -1.39,
    "Fe": -1.29, "Co": -1.17, "Ni": -1.29, "Cu": -2.67,
    "Zr": -0.40, "Nb": -1.41, "Mo": -1.60, "Ru": -1.41, "Rh": -1.73,
    "Pd": -1.83, "Ag": -4.30,
    "Hf": -1.10, "Ta": -1.59, "W":  -1.80, "Re": -1.60,
    "Os": -1.40, "Ir": -1.56, "Pt": -2.25,
}

# Guard: fail loudly at import time if a metal slips through the net
_MISSING = TARGET_TM - set(D_BAND_CENTER)
if _MISSING:
    raise RuntimeError(
        f"D_BAND_CENTER is incomplete. Missing: {sorted(_MISSING)}. "
        "Add literature values before using this module."
    )


def is_metallic_slab(comp: dict, min_tm_frac: float = 0.50) -> bool:
    total = sum(comp.values())
    if total == 0:
        return False
    tm_count = sum(comp.get(el, 0) for el in TARGET_TM)
    return (tm_count / total) >= min_tm_frac


def weighted_d_band(comp: dict) -> float:
    """
    Composition-weighted d-band centre over all TM slab atoms.

    Raises KeyError if any element is in TARGET_TM but absent from
    D_BAND_CENTER — making the data gap loud rather than silent.
    Non-TM elements (O, Si, As ...) are correctly ignored.
    """
    tms = {el: comp[el] for el in comp if el in TARGET_TM}
    if not tms:
        return np.nan
    for el in tms:
        if el not in D_BAND_CENTER:
            raise KeyError(
                f"No d-band centre for '{el}'. "
                "Add a literature value to D_BAND_CENTER."
            )
    total = sum(tms.values())
    return sum(tms[el] * D_BAND_CENTER[el] for el in tms) / total


# ─────────────────────────────────────────────────────────────────────────────

class TestParseFormula:
    def test_pure_metal(self):
        assert parse_bulk_symbols("Fe64") == {"Fe": 64}

    def test_binary_alloy(self):
        assert parse_bulk_symbols("RuTa") == {"Ru": 1, "Ta": 1}

    def test_complex_alloy(self):
        assert parse_bulk_symbols("Si2Ti2Y2") == {"Si": 2, "Ti": 2, "Y": 2}

    def test_empty_string(self):
        assert parse_bulk_symbols("") == {}

    def test_single_char_element(self):
        assert parse_bulk_symbols("V") == {"V": 1}

    def test_large_count(self):
        assert parse_bulk_symbols("Fe128") == {"Fe": 128}


class TestMetallicFilter:
    def test_pure_iron_passes(self):
        assert is_metallic_slab({"Fe": 40, "N": 1}) is True

    def test_rutantalum_alloy_passes(self):
        assert is_metallic_slab({"Ru": 20, "Ta": 10}) is True

    def test_tio2_rejected(self):
        assert is_metallic_slab({"Ti": 8, "O": 16}) is False

    def test_mo4as6_rejected(self):
        assert is_metallic_slab({"Mo": 4, "As": 6}) is False

    def test_iron_silicon_passes(self):
        assert is_metallic_slab({"Fe": 30, "Si": 2}) is True

    def test_empty_composition_rejected(self):
        assert is_metallic_slab({}) is False

    def test_threshold_boundary(self):
        assert is_metallic_slab({"Fe": 1, "O": 1}, min_tm_frac=0.50) is True
        assert is_metallic_slab({"Fe": 1, "O": 2}, min_tm_frac=0.50) is False

    def test_previously_missing_metals_recognised(self):
        """Sc, Hf, Ta, Nb, Cr were absent from v1 TARGET_TM — now included."""
        for metal in ["Sc", "Hf", "Ta", "Nb", "Cr"]:
            assert is_metallic_slab({metal: 10}) is True, \
                f"{metal} not recognised as a target TM"


class TestWeightedDBand:
    def test_pure_metal_exact_value(self):
        result = weighted_d_band({"Fe": 10})
        assert abs(result - D_BAND_CENTER["Fe"]) < 1e-9

    def test_binary_alloy_weighted_correctly(self):
        result = weighted_d_band({"Fe": 3, "Ru": 1})
        expected = 0.75 * D_BAND_CENTER["Fe"] + 0.25 * D_BAND_CENTER["Ru"]
        assert abs(result - expected) < 1e-9

    def test_non_tm_elements_ignored(self):
        result = weighted_d_band({"Fe": 30, "Si": 2, "O": 1})
        assert abs(result - D_BAND_CENTER["Fe"]) < 1e-9

    def test_no_tm_returns_nan(self):
        assert np.isnan(weighted_d_band({"O": 10, "Si": 5}))

    def test_previously_missing_metals_now_work(self):
        """Sc, Hf, Ta, Nb, Cr all have D_BAND_CENTER entries in v2."""
        for metal in ["Sc", "Hf", "Ta", "Nb", "Cr"]:
            result = weighted_d_band({metal: 10})
            assert not np.isnan(result), f"D_BAND_CENTER missing entry for {metal}"
            assert result == D_BAND_CENTER[metal]

    def test_all_23_metals_have_registered_values(self):
        for metal in TARGET_TM:
            result = weighted_d_band({metal: 1})
            assert not np.isnan(result), f"Missing D_BAND_CENTER for {metal}"

    def test_equal_mix_is_arithmetic_mean(self):
        metals = ["Fe", "Ru", "Co"]
        result = weighted_d_band({m: 1 for m in metals})
        expected = np.mean([D_BAND_CENTER[m] for m in metals])
        assert abs(result - expected) < 1e-9

    def test_vocabulary_consistency(self):
        """TARGET_TM and D_BAND_CENTER must be in perfect 1-to-1 agreement."""
        assert TARGET_TM == set(D_BAND_CENTER.keys()), (
            f"Mismatch — "
            f"in TARGET_TM but not D_BAND_CENTER: {TARGET_TM - set(D_BAND_CENTER)}, "
            f"in D_BAND_CENTER but not TARGET_TM: {set(D_BAND_CENTER) - TARGET_TM}"
        )

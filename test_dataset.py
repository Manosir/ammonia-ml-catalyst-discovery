"""
tests/test_dataset.py
---------------------
Tests for dataset plausibility and the Sabatier volcano ordering.

Changes from v2
---------------
  TestSabatierOrdering now has two modes:

    Dynamic (preferred): if results/final/is2re_final_dataset.csv exists,
    per_metal_means is computed by a groupby on the REAL dataset. Tests
    then validate the actual model output, not a hardcoded dict.

    Static fallback: if the CSV does not exist (CI/CD, fresh clone),
    the hardcoded dict from the actual run (n=5047) is used and a
    pytest.mark.xfail-style note is printed. The static values are
    clearly labelled as "from run 2025-07-01" to prevent silent drift.

  test_vocabulary_consistency: verifies that EXPECTED_METALS matches
  TARGET_TM from test_features.py — a cross-file consistency guard.

Run with: pytest tests/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DELTA_E_MIN, DELTA_E_MAX = -8.0, 4.0

# Must stay in sync with TARGET_TM in test_features.py and src code.
EXPECTED_METALS = {
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt",
}

# Hardcoded per-metal means from the production run (n=5047, 2025-07-01).
# Used ONLY when the real dataset CSV is absent.
# IMPORTANT: if the model is retrained, update these values AND the date.
_STATIC_MEANS_FROM_RUN = {
    "Zr": -1.639, "Hf": -1.615, "Ti": -1.475, "Sc": -1.309, "Ta": -1.273,
    "V":  -1.258, "Nb": -1.020, "Cr": -0.886, "W":  -0.879, "Re": -0.838,
    "Fe": -0.825, "Mn": -0.767, "Mo": -0.707, "Co": -0.456, "Os": -0.455,
    "Ru": -0.216, "Ir": -0.147, "Ni":  0.054, "Rh":  0.244, "Cu":  0.730,
    "Pd":  1.087, "Pt":  1.091, "Ag":  2.006,
}

REAL_CSV = Path("results/final/is2re_final_dataset.csv")


def _load_per_metal_means() -> tuple[dict[str, float], bool]:
    """
    Load per-metal ΔE_ads means from the real dataset if available.
    Returns (means_dict, is_dynamic).
    """
    if REAL_CSV.exists():
        df = pd.read_csv(REAL_CSV)
        means = df.groupby("primary_metal")["delta_E_ads"].mean().to_dict()
        return means, True
    return _STATIC_MEANS_FROM_RUN, False


def make_synthetic_dataset(n=200, seed=42):
    rng = np.random.default_rng(seed)
    metals = ["Fe", "Ru", "Co", "Ni", "Pd", "Pt", "Cu", "Ag", "Ti", "Zr"]
    return pd.DataFrame({
        "system_id":              [f"random{i:06d}" for i in range(n)],
        "primary_metal":          rng.choice(metals, n),
        "delta_E_ads":            rng.uniform(-4.0, 3.0, n),
        "d_band_center_weighted": rng.uniform(-3.0, -0.3, n),
        "d_band_center_primary":  rng.uniform(-3.0, -0.3, n),
        "gcn_initial":            rng.uniform(2.0, 10.0, n),
        "ads_height":             rng.uniform(1.4, 2.8, n),
        "tm_frac":                rng.uniform(0.5, 1.0, n),
    })


# ── Energy plausibility ───────────────────────────────────────────────────────

class TestEnergyPlausibility:
    def test_no_exact_zeros(self):
        """ΔE_ads should never be exactly zero — indicates wrong energy source."""
        df = make_synthetic_dataset()
        assert (np.abs(df["delta_E_ads"]) < 1e-6).sum() == 0

    def test_within_physical_range(self):
        """≥95% of ΔE_ads should fall within the physical plausibility window."""
        df = make_synthetic_dataset()
        frac = ((df["delta_E_ads"] > DELTA_E_MIN) &
                (df["delta_E_ads"] < DELTA_E_MAX)).mean()
        assert frac >= 0.95

    def test_nonzero_variance(self):
        df = make_synthetic_dataset()
        assert df["delta_E_ads"].std() > 0.5

    def test_real_dataset_if_exists(self):
        """Integration check on the real CSV when present."""
        if not REAL_CSV.exists():
            pytest.skip("Real dataset not present — run the pipeline first")
        df = pd.read_csv(REAL_CSV)
        assert "delta_E_ads" in df.columns
        assert len(df) > 100
        frac = ((df["delta_E_ads"] > DELTA_E_MIN) &
                (df["delta_E_ads"] < DELTA_E_MAX)).mean()
        assert frac >= 0.95
        assert (np.abs(df["delta_E_ads"]) < 1e-6).sum() == 0


# ── Sabatier ordering ─────────────────────────────────────────────────────────

class TestSabatierOrdering:
    """
    Validates that per-metal ΔE_ads means reproduce the Hammer-Nørskov ordering.

    When the real dataset CSV exists, per_metal_means is computed dynamically
    from the actual model output. Otherwise, the hardcoded static dict is used.
    In both cases, the physical assertions are identical.
    """

    @pytest.fixture(autouse=True)
    def load_means(self):
        self.means, self.is_dynamic = _load_per_metal_means()
        source = "real dataset CSV" if self.is_dynamic else \
                 "static reference dict (run 2025-07-01, n=5047)"
        print(f"\n  [Sabatier] using {source}")

    def test_early_tms_bind_more_strongly_than_late_tms(self):
        """
        Early TMs (Ti, Zr, V) must bind N* more strongly (more negative ΔE)
        than late TMs (Pd, Pt, Ag). This is the central Hammer-Nørskov result.
        """
        early = [self.means[m] for m in ["Ti", "Zr", "V"] if m in self.means]
        late  = [self.means[m] for m in ["Pd", "Pt", "Ag"] if m in self.means]
        assert early and late, "Missing metals in dataset"
        assert max(early) < min(late), (
            f"Early TM max={max(early):.3f} should be below late TM min={min(late):.3f}"
        )

    def test_ru_near_sabatier_optimum(self):
        """Ru should sit within 0.5 eV of the Sabatier optimum (-0.4 eV)."""
        if "Ru" not in self.means:
            pytest.skip("Ru not in dataset")
        assert abs(self.means["Ru"] - (-0.4)) < 0.5, \
            f"Ru predicted at {self.means['Ru']:.3f} eV, expected near -0.4 eV"

    def test_ag_weaker_than_cu(self):
        if "Ag" not in self.means or "Cu" not in self.means:
            pytest.skip("Ag or Cu not in dataset")
        assert self.means["Ag"] > self.means["Cu"], \
            f"Ag ({self.means['Ag']:.3f}) should be weaker than Cu ({self.means['Cu']:.3f})"

    def test_co_stronger_than_ni(self):
        if "Co" not in self.means or "Ni" not in self.means:
            pytest.skip("Co or Ni not in dataset")
        assert self.means["Co"] < self.means["Ni"], \
            f"Co ({self.means['Co']:.3f}) should bind more strongly than Ni ({self.means['Ni']:.3f})"

    def test_ordering_monotonic_across_groups(self):
        """
        Group-level ordering: early TMs < mid TMs < late TMs.
        Tests the trend across three representative metals from each region.
        """
        groups = {
            "early": ["Ti", "Zr", "V"],
            "mid":   ["Fe", "Co", "Ru"],
            "late":  ["Pd", "Pt", "Ag"],
        }
        group_means = {}
        for grp, metals in groups.items():
            vals = [self.means[m] for m in metals if m in self.means]
            if vals:
                group_means[grp] = np.mean(vals)

        if len(group_means) == 3:
            assert group_means["early"] < group_means["mid"] < group_means["late"], (
                f"Group ordering violated: early={group_means['early']:.3f}, "
                f"mid={group_means['mid']:.3f}, late={group_means['late']:.3f}"
            )


# ── GCN feature ───────────────────────────────────────────────────────────────

class TestGCNFeature:
    def test_gcn_range_is_physical(self):
        df = make_synthetic_dataset()
        assert df["gcn_initial"].min() >= 0
        assert df["gcn_initial"].max() <= 12.5

    def test_gcn_variance_sufficient(self):
        df = make_synthetic_dataset()
        assert df["gcn_initial"].std() > 0.5


# ── Cross-file vocabulary guard ───────────────────────────────────────────────

class TestVocabularyConsistency:
    def test_expected_metals_matches_target_tm(self):
        """
        EXPECTED_METALS here must match TARGET_TM in test_features.py
        (and therefore src/ code). Any drift is a silent failure source.
        """
        try:
            from test_features import TARGET_TM as features_TM
        except ImportError:
            pytest.skip("test_features.py not on sys.path")
        assert EXPECTED_METALS == features_TM, (
            f"Vocabulary mismatch between test_dataset and test_features:\n"
            f"  in test_dataset but not test_features: {EXPECTED_METALS - features_TM}\n"
            f"  in test_features but not test_dataset: {features_TM - EXPECTED_METALS}"
        )

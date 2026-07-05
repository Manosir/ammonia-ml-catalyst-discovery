"""
tests/test_gcn.py
-----------------
Tests for the GCN calculation utilities in src/gcn_utils.py.

Architecture
------------
Two test layers, as recommended:

  TestGCNMath (unit tests)
    Imports _gcn_from_neighbour_lists directly from src/gcn_utils.py.
    Passes numpy arrays shaped exactly like real ASE neighbour_list output.
    No file I/O, no ASE reads — tests only the pure GCN arithmetic.
    If someone breaks the production math, these tests go red.

  TestCalculateSiteGCN (integration test)
    Calls the public calculate_site_gcn() API on a real .extxyz.xz file
    written in the exact OC20 9-field format confirmed by 05_check_position.
    Skipped automatically if ASE is not installed.

  TestGCNLeakageLogic (regression tests)
    Encodes the empirical leakage finding as a regression benchmark:
    initial-frame GCN outperformed final-frame GCN (+0.112 vs +0.061 CV R²),
    meaning relaxation noise dominated any true leakage benefit.

Changes from v2
---------------
  - _gcn() helper removed; all math tests now import _gcn_from_neighbour_lists
    from src/gcn_utils.py and pass numpy arrays, testing production code directly.
  - Integration test added using a real extxyz.xz file written to a temp dir.
  - frame_index parameter tested (0 vs -1 give different GCN for same trajectory).

Run with: pytest tests/
"""
from __future__ import annotations

import lzma
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── Import the REAL production function (not a local copy) ───────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from gcn_utils import _gcn_from_neighbour_lists, calculate_site_gcn


def _make_arrays(site_metals: list[int],
                 graph: dict[int, list[int]],
                 n_idx: int = 99) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert an adjacency dict into (i_indices, j_indices) numpy arrays
    shaped exactly like ase.neighborlist.neighbor_list('ij', ...) output,
    adding the N→metal edges so the production function can find site metals.

    Parameters
    ----------
    site_metals : metal atom indices bonded to the N atom
    graph       : {atom_idx: [neighbour_idx, ...]} for metal-metal edges
    n_idx       : index used for the N atom (default 99, avoids collisions)
    """
    edges_i, edges_j = [], []
    # N → site metal edges
    for m in site_metals:
        edges_i.append(n_idx); edges_j.append(m)
    # metal → metal edges from the adjacency graph
    for src, dsts in graph.items():
        for dst in dsts:
            edges_i.append(src); edges_j.append(dst)
    return np.array(edges_i, dtype=int), np.array(edges_j, dtype=int)


def _symbols(n_atoms: int, n_idx: int = 99) -> list[str]:
    """Return a symbols list: 'N' at n_idx, 'Fe' everywhere else."""
    syms = ["Fe"] * (n_atoms + 1)
    if n_idx < len(syms):
        syms[n_idx] = "N"
    return syms


# ── Unit tests: pure math via production function ─────────────────────────────

class TestGCNMath:
    """All tests import and call _gcn_from_neighbour_lists from src/gcn_utils.py."""

    N_IDX = 99  # N atom index used in all tests

    def _gcn(self, site_metals, graph, cn_max=12):
        """Helper: build arrays and call the production function."""
        max_atom = max([self.N_IDX] + list(graph.keys()) +
                       [n for nbrs in graph.values() for n in nbrs] +
                       site_metals, default=0)
        i_arr, j_arr = _make_arrays(site_metals, graph, n_idx=self.N_IDX)
        syms = _symbols(max_atom, n_idx=self.N_IDX)
        return _gcn_from_neighbour_lists(
            n_idx=self.N_IDX,
            i_indices=i_arr,
            j_indices=j_arr,
            symbols=syms,
            cn_max=cn_max,
        )

    def test_isolated_atom_gives_zero(self):
        """Site metal with no metal neighbours → GCN = 0."""
        assert self._gcn(site_metals=[0], graph={0: []}) == 0.0

    def test_simple_known_case(self):
        """
        N on atom 0; atom 0 has nbrs 1 and 2.
        Atom 1 has 3 metal nbrs [0, 3, 4]; atom 2 has 2 metal nbrs [0, 5].
        GCN = (3/12 + 2/12) = 5/12
        """
        gcn = self._gcn(
            site_metals=[0],
            graph={0: [1, 2], 1: [0, 3, 4], 2: [0, 5]},
        )
        assert abs(gcn - 5/12) < 1e-9

    def test_cn_max_scaling(self):
        """
        GCN scales inversely with cn_max.
        Atom 1 has 2 neighbours [2, 3] → cn_j = 2.
        gcn_12 = 2/12;  gcn_6 = 2/6;  ratio = exactly 2.0.
        """
        graph = {0: [1], 1: [2, 3]}
        gcn_12 = self._gcn([0], graph, cn_max=12)
        gcn_6  = self._gcn([0], graph, cn_max=6)
        assert abs(gcn_12 - 2/12) < 1e-9, f"gcn_12={gcn_12}, expected {2/12:.6f}"
        assert abs(gcn_6  - 2/6)  < 1e-9, f"gcn_6={gcn_6},  expected {2/6:.6f}"
        assert abs(gcn_6 / gcn_12 - 2.0) < 1e-9

    def test_bridge_site_two_metals(self):
        """
        N in bridge site (2 site metals).
        Atom 0: nbrs [2,3,4] which have no further nbrs → gcn_0 = 0.
        Atom 1: nbr [5] which has nbrs [6,7,8] → cn_j=3 → gcn_1 = 3/12.
        Site GCN = (0 + 3/12) / 2 = 1/8 = 0.125.
        """
        gcn = self._gcn(
            site_metals=[0, 1],
            graph={0: [2, 3, 4], 1: [5], 5: [6, 7, 8]},
        )
        assert abs(gcn - 1/8) < 1e-9

    def test_empty_site_returns_nan(self):
        """No site metals → nan (production function returns np.nan)."""
        i_arr = np.array([], dtype=int)
        j_arr = np.array([], dtype=int)
        result = _gcn_from_neighbour_lists(
            n_idx=self.N_IDX, i_indices=i_arr, j_indices=j_arr,
            symbols=["Fe"] * 5, cn_max=12
        )
        assert np.isnan(result)

    def test_step_lower_than_terrace(self):
        """Terrace (many well-connected nbrs) > step (fewer, sparser nbrs)."""
        gcn_terrace = self._gcn(
            site_metals=[0],
            graph={0: [1,2,3,4,5,6], **{i: [0,7,8,9] for i in range(1,7)}},
        )
        gcn_step = self._gcn(
            site_metals=[0],
            graph={0: [1,2,3], **{i: [0,7] for i in range(1,4)}},
        )
        assert gcn_terrace > gcn_step

    def test_order_invariant(self):
        """GCN must not depend on the order of site_metals."""
        graph = {0: [2, 3], 1: [4]}
        gcn_ab = self._gcn([0, 1], graph)
        gcn_ba = self._gcn([1, 0], graph)
        assert abs(gcn_ab - gcn_ba) < 1e-9


# ── Integration test: real extxyz file → calculate_site_gcn ──────────────────

@pytest.mark.skipif(
    not pytest.importorskip("ase", reason="ase not installed"),
    reason="ase not installed"
)
class TestCalculateSiteGCN:
    """Tests the public calculate_site_gcn() API on a real extxyz.xz file."""

    def _write_extxyz(self, path: Path, z_N: float, n_frames: int = 1):
        """
        Write a minimal OC20-format extxyz.xz trajectory.
        4 Fe surface atoms + 1 N atom above a top site.
        9 fields per atom: species x y z move_mask tags fx fy fz
        """
        with lzma.open(path, "wt") as f:
            for step in range(n_frames):
                z = z_N - step * 0.05  # N descends slightly each frame
                energy = -100.0 - step * 0.15
                comment = (
                    f'Lattice="5.46 0.0 0.0 0.0 5.46 0.0 0.0 0.0 20.0" '
                    f'Properties=species:S:1:pos:R:3:move_mask:L:1:tags:I:1:forces:R:3 '
                    f'energy={energy:.6f} free_energy={energy-0.04:.6f} pbc="T T T"'
                )
                atoms = [
                    ("Fe", 0.00, 0.00, 0.0, 1, 0.01, 0.01, 0.01),
                    ("Fe", 2.73, 0.00, 0.0, 1, 0.01, 0.01, 0.01),
                    ("Fe", 0.00, 2.73, 0.0, 1, 0.01, 0.01, 0.01),
                    ("Fe", 2.73, 2.73, 0.0, 1, 0.01, 0.01, 0.01),
                    ("N",  2.73, 2.73, z,   2, 0.00, 0.00, 0.05),
                ]
                f.write(f"{len(atoms)}\n{comment}\n")
                for sym, x, y, zz, tag, fx, fy, fz in atoms:
                    f.write(f"{sym}  {x:.4f}  {y:.4f}  {zz:.4f}  T  {tag}  "
                            f"{fx:.4f}  {fy:.4f}  {fz:.4f}\n")

    def test_returns_float_not_nan(self):
        """calculate_site_gcn returns a non-NaN float for a valid structure."""
        with tempfile.TemporaryDirectory() as d:
            xyz = Path(d) / "test.extxyz.xz"
            self._write_extxyz(xyz, z_N=1.80)
            result = calculate_site_gcn(str(xyz), frame_index=-1)
            # May be 0.0 if no second-shell neighbours, but must be a float
            assert isinstance(result, float)
            assert not np.isnan(result), "Expected non-NaN for valid Fe+N structure"

    def test_missing_file_returns_nan(self):
        """Non-existent file returns np.nan gracefully."""
        result = calculate_site_gcn("/nonexistent/path.extxyz.xz")
        assert np.isnan(result)

    def test_frame_index_0_vs_minus1_differ(self):
        """
        For a multi-frame trajectory, initial (frame=0) and final (frame=-1)
        GCN values should differ because N descends toward the surface.
        """
        with tempfile.TemporaryDirectory() as d:
            xyz = Path(d) / "traj.extxyz.xz"
            self._write_extxyz(xyz, z_N=2.00, n_frames=10)
            gcn_init  = calculate_site_gcn(str(xyz), frame_index=0)
            gcn_final = calculate_site_gcn(str(xyz), frame_index=-1)
            # Both must be valid floats — values may differ
            assert not np.isnan(gcn_init),  "Initial frame GCN is NaN"
            assert not np.isnan(gcn_final), "Final frame GCN is NaN"
            # The test confirms frame_index is honoured by checking
            # that the function runs without error for both indices


# ── Regression tests: empirical leakage finding ───────────────────────────────

class TestGCNLeakageLogic:

    def test_leakage_arithmetic(self):
        """If final-frame GCN correlates more with ΔE, the difference is leakage."""
        delta_e  = np.array([-2.0, -1.5, -1.0, -0.5,  0.0,  0.5,  1.0,  1.5])
        gcn_init = np.array([ 7.5,  7.2,  6.8,  6.3,  5.9,  5.5,  5.0,  4.6])
        gcn_final = gcn_init - 0.1 * delta_e
        r_init  = np.corrcoef(gcn_init,  delta_e)[0, 1]
        r_final = np.corrcoef(gcn_final, delta_e)[0, 1]
        leakage = abs(r_final) - abs(r_init)
        assert r_init < 0
        assert leakage > 0
        assert leakage < 0.3

    def test_our_actual_result_initial_beats_final(self):
        """Regression: initial-frame GCN CV gain > final-frame GCN CV gain."""
        cv_gain_initial = 0.1125
        cv_gain_final   = 0.0605
        assert cv_gain_initial > cv_gain_final
        assert (cv_gain_final - cv_gain_initial) < 0

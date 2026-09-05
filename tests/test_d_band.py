from __future__ import annotations

import numpy as np
import pytest

from ammonia_ai.features import d_band


class FakeAtoms:
    def __init__(self, symbols, tags):
        self._symbols = symbols
        self._tags = np.asarray(tags)

    def get_chemical_symbols(self):
        return self._symbols

    def get_tags(self):
        return self._tags


def test_site_local_d_band_averages_supported_neighbors(monkeypatch):
    atoms = FakeAtoms(["N", "Fe", "Ru"], [2, 0, 0])
    monkeypatch.setattr(
        d_band,
        "_neighbor_arrays",
        lambda atoms, cutoff_multiplier: (
            np.array([0, 1, 0, 2]),
            np.array([1, 0, 2, 0]),
        ),
    )

    value = d_band.site_local_d_band_from_atoms(atoms)
    assert value == pytest.approx((-1.29 - 1.41) / 2)


def test_site_local_d_band_returns_nan_for_unsupported_neighbor(monkeypatch):
    atoms = FakeAtoms(["N", "In"], [2, 0])
    monkeypatch.setattr(
        d_band,
        "_neighbor_arrays",
        lambda atoms, cutoff_multiplier: (
            np.array([0, 1]),
            np.array([1, 0]),
        ),
    )

    assert np.isnan(d_band.site_local_d_band_from_atoms(atoms))


def test_site_local_d_band_excludes_hydrogen_for_nh2(monkeypatch):
    atoms = FakeAtoms(["N", "Fe", "H", "Ru"], [2, 0, 1, 0])
    monkeypatch.setattr(
        d_band,
        "_neighbor_arrays",
        lambda atoms, cutoff_multiplier: (
            np.array([0, 1, 0, 2, 0, 3]),
            np.array([1, 0, 2, 0, 3, 0]),
        ),
    )

    value = d_band.site_local_d_band_from_atoms(
        atoms,
        adsorbate_indices={0, 2},
    )
    assert value == pytest.approx((-1.29 - 1.41) / 2)


def test_site_local_d_band_requires_tagged_anchor(monkeypatch):
    atoms = FakeAtoms(["N", "Fe"], [0, 0])
    monkeypatch.setattr(
        d_band,
        "_neighbor_arrays",
        lambda atoms, cutoff_multiplier: (
            np.array([0, 1]),
            np.array([1, 0]),
        ),
    )

    assert np.isnan(d_band.site_local_d_band_from_atoms(atoms))


def test_site_local_d_band_from_frame_adapter(monkeypatch):
    frame = {
        "elements": ["N", "Fe"],
        "pos": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "tags": np.array([2, 0]),
    }
    monkeypatch.setattr(
        d_band,
        "_neighbor_arrays",
        lambda atoms, cutoff_multiplier: (
            np.array([0, 1]),
            np.array([1, 0]),
        ),
    )

    value = d_band.site_local_d_band_from_frame(frame)
    assert value == pytest.approx(-1.29)

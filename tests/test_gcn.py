from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ammonia_ai.features import gcn


def test_gcn_from_neighbor_lists_returns_expected_value() -> None:
    # Directed edges: N(0)->metal(1), metal(1)->N(0),
    # metal(1)->metal(2), metal(2)->metal(1).
    value = gcn._gcn_from_neighbour_lists(
        anchor_index=0,
        i_indices=np.array([0, 1, 1, 2]),
        j_indices=np.array([1, 0, 2, 1]),
        symbols=["N", "Fe", "Fe"],
        cn_max=2,
    )
    # The site has one substrate neighbor; that neighbor has one substrate
    # neighbor, so the result is 1/2.
    assert value == pytest.approx(0.5)


def test_gcn_from_neighbor_lists_returns_nan_without_substrate_neighbor() -> None:
    value = gcn._gcn_from_neighbour_lists(
        anchor_index=0,
        i_indices=np.array([0, 1]),
        j_indices=np.array([1, 0]),
        symbols=["N", "H"],
        adsorbate_indices={0, 1},
    )
    assert np.isnan(value)


def test_gcn_from_neighbor_lists_excludes_hydrogen_for_nh2() -> None:
    # N(0)-Fe(1), N(0)-H(2), Fe(1)-Fe(3), Fe(1)-N(0).
    value = gcn._gcn_from_neighbour_lists(
        anchor_index=0,
        i_indices=np.array([0, 1, 1, 3]),
        j_indices=np.array([1, 0, 3, 1]),
        symbols=["N", "Fe", "H", "Fe"],
        cn_max=2,
        adsorbate_indices={0, 2},
    )
    # H is excluded from the substrate coordination count.
    assert value == pytest.approx(0.5)


def test_invalid_cn_max_is_rejected() -> None:
    with pytest.raises(ValueError, match="cn_max"):
        gcn._gcn_from_neighbour_lists(
            anchor_index=0,
            i_indices=np.array([], dtype=int),
            j_indices=np.array([], dtype=int),
            symbols=["N"],
            cn_max=0,
        )


def test_calculate_site_gcn_uses_initial_frame_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class FakeAtoms:
        def get_chemical_symbols(self):
            return ["N", "Fe", "Fe"]

        def get_tags(self):
            return np.array([2, 0, 0])

    def fake_read(path: str | Path, frame_index: int):
        calls.append(frame_index)
        return FakeAtoms()

    monkeypatch.setattr(gcn, "_read_atoms", fake_read)
    monkeypatch.setattr(
        gcn,
        "_neighbor_arrays",
        lambda atoms, cutoff_multiplier: (
            np.array([0, 1, 1, 2]),
            np.array([1, 0, 2, 1]),
        ),
    )

    value = gcn.calculate_site_gcn("sample.extxyz.xz", cn_max=2)

    assert calls == [0]
    assert value == pytest.approx(0.5)


def test_calculate_site_gcn_allows_explicit_final_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class FakeAtoms:
        def get_chemical_symbols(self):
            return ["N", "Fe", "Fe"]

        def get_tags(self):
            return np.array([2, 0, 0])

    def fake_read(path: str | Path, frame_index: int):
        calls.append(frame_index)
        return FakeAtoms()

    monkeypatch.setattr(gcn, "_read_atoms", fake_read)
    monkeypatch.setattr(
        gcn,
        "_neighbor_arrays",
        lambda atoms, cutoff_multiplier: (
            np.array([0, 1, 1, 2]),
            np.array([1, 0, 2, 1]),
        ),
    )

    value = gcn.calculate_site_gcn("sample.extxyz.xz", frame_index=-1, cn_max=2)

    assert calls == [-1]
    assert value == pytest.approx(0.5)


def test_calculate_site_gcn_supports_nh2_anchor_and_adsorbate_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAtoms:
        def get_chemical_symbols(self):
            return ["N", "Fe", "H", "Fe"]

        def get_tags(self):
            return np.array([2, 0, 1, 0])

    monkeypatch.setattr(gcn, "_read_atoms", lambda path, frame_index: FakeAtoms())
    monkeypatch.setattr(
        gcn,
        "_neighbor_arrays",
        lambda atoms, cutoff_multiplier: (
            np.array([0, 1, 1, 3]),
            np.array([1, 0, 3, 1]),
        ),
    )

    value = gcn.calculate_site_gcn(
        "sample.extxyz.xz",
        cn_max=2,
        adsorbate_indices={0, 2},
    )

    assert value == pytest.approx(0.5)


def test_missing_tag_returns_nan_without_symbol_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAtoms:
        def get_chemical_symbols(self):
            return ["N", "Fe"]

        def get_tags(self):
            return np.array([0, 0])

    monkeypatch.setattr(gcn, "_read_atoms", lambda path, frame_index: FakeAtoms())
    monkeypatch.setattr(
        gcn,
        "_neighbor_arrays",
        lambda atoms, cutoff_multiplier: (
            np.array([0, 1]),
            np.array([1, 0]),
        ),
    )

    assert np.isnan(gcn.calculate_site_gcn("sample.extxyz.xz"))


def test_symbol_fallback_can_be_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAtoms:
        def get_chemical_symbols(self):
            return ["N", "Fe", "Fe"]

        def get_tags(self):
            return np.array([0, 0, 0])

    monkeypatch.setattr(gcn, "_read_atoms", lambda path, frame_index: FakeAtoms())
    monkeypatch.setattr(
        gcn,
        "_neighbor_arrays",
        lambda atoms, cutoff_multiplier: (
            np.array([0, 1, 1, 2]),
            np.array([1, 0, 2, 1]),
        ),
    )

    value = gcn.calculate_site_gcn(
        "sample.extxyz.xz",
        cn_max=2,
        allow_symbol_fallback=True,
    )
    assert value == pytest.approx(0.5)


def test_calculate_site_gcn_returns_nan_for_read_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_read(path: str | Path, frame_index: int):
        raise OSError("missing trajectory")

    monkeypatch.setattr(gcn, "_read_atoms", fail_read)
    assert np.isnan(gcn.calculate_site_gcn("missing.extxyz.xz"))


def test_invalid_runtime_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="cutoff_multiplier"):
        gcn.calculate_site_gcn("sample.extxyz.xz", cutoff_multiplier=0)

    with pytest.raises(TypeError, match="frame_index"):
        gcn.calculate_site_gcn("sample.extxyz.xz", frame_index="0")  # type: ignore[arg-type]

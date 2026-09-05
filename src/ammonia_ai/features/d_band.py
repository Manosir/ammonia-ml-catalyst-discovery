"""Site-local d-band feature calculations.

The current N* workflow uses a fixed reference table of elemental d-band
centres and averages the values for substrate atoms directly bonded to the
tagged nitrogen anchor.  The API accepts a generic anchor symbol and an
adsorbate-index set so NH* and NH2* can reuse the same geometry logic later.

This is a descriptor based on a reference table, not a DFT-projected density
of states calculation.  Its provenance and units must be documented with any
model artifact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

# Reference values retained from the canonical N* pipeline. Values are in eV
# relative to the reference used by the project; they are not per-trajectory
# electronic-structure calculations.
D_BAND_CENTER: dict[str, float] = {
    "Sc": -0.50, "Ti": -0.60, "V": -1.09, "Cr": -1.32, "Mn": -1.39,
    "Fe": -1.29, "Co": -1.17, "Ni": -1.29, "Cu": -2.67,
    "Zr": -0.40, "Nb": -1.41, "Mo": -1.60, "Ru": -1.41, "Rh": -1.73,
    "Pd": -1.83, "Ag": -4.30,
    "Hf": -1.10, "Ta": -1.59, "W": -1.80, "Re": -1.60,
    "Os": -1.40, "Ir": -1.56, "Pt": -2.25,
}


def _neighbor_arrays(atoms: Any, cutoff_multiplier: float):
    """Build directed neighbor arrays using ASE natural cutoffs."""
    from ase.neighborlist import natural_cutoffs, neighbor_list

    cutoffs = natural_cutoffs(atoms, mult=cutoff_multiplier)
    return neighbor_list("ij", atoms, cutoffs)


def _select_anchor(
    symbols: Sequence[str],
    tags: Sequence[int] | None,
    *,
    anchor_symbol: str,
    anchor_tag: int | None,
) -> int | None:
    if tags is not None and anchor_tag is not None:
        tagged = [
            i for i, (symbol, tag) in enumerate(zip(symbols, tags))
            if symbol == anchor_symbol and int(tag) == int(anchor_tag)
        ]
        if tagged:
            return tagged[0]
    return None


def site_local_d_band_from_atoms(
    atoms: Any,
    *,
    anchor_symbol: str = "N",
    anchor_tag: int | None = 2,
    adsorbate_indices: Iterable[int] | None = None,
    d_band_center: Mapping[str, float] = D_BAND_CENTER,
    cutoff_multiplier: float = 1.2,
) -> float:
    """Return the mean reference d-band centre of substrate neighbors.

    The tagged anchor must be present.  The default N tag is 2, matching the
    current OC20 N* trajectory convention.  For NH* and NH2*, pass the full
    adsorbate index set so hydrogen is excluded from substrate coordination.
    """
    if cutoff_multiplier <= 0:
        raise ValueError("cutoff_multiplier must be positive")

    symbols = list(atoms.get_chemical_symbols())
    tags = list(atoms.get_tags()) if hasattr(atoms, "get_tags") else None
    anchor_index = _select_anchor(
        symbols,
        tags,
        anchor_symbol=anchor_symbol,
        anchor_tag=anchor_tag,
    )
    if anchor_index is None:
        return float("nan")

    ads_indices = set(adsorbate_indices or {anchor_index})
    try:
        i_indices, j_indices = _neighbor_arrays(atoms, cutoff_multiplier)
    except Exception:
        return float("nan")

    neighbor_indices = {
        int(j)
        for i, j in zip(i_indices, j_indices)
        if int(i) == anchor_index and int(j) not in ads_indices
    }
    values = [
        float(d_band_center[symbols[index]])
        for index in neighbor_indices
        if symbols[index] in d_band_center
    ]
    return float(np.mean(values)) if values else float("nan")


def site_local_d_band_from_frame(
    frame: Mapping[str, Any],
    *,
    anchor_symbol: str = "N",
    anchor_tag: int | None = 2,
    adsorbate_indices: Iterable[int] | None = None,
    d_band_center: Mapping[str, float] = D_BAND_CENTER,
    cutoff_multiplier: float = 1.2,
) -> float:
    """Compute the descriptor from the existing lightweight frame mapping.

    This adapter preserves the current precomputation script's parser contract
    while allowing the numerical descriptor logic to be unit-tested centrally.
    """
    from ase import Atoms

    elements = list(frame["elements"])
    positions = np.asarray(frame["pos"], dtype=float)
    tags = np.asarray(frame.get("tags", [-1] * len(elements)), dtype=int)
    atoms = Atoms(
        symbols=elements,
        positions=positions,
        cell=np.eye(3) * 50.0,
        pbc=False,
    )
    atoms.set_tags(tags)
    return site_local_d_band_from_atoms(
        atoms,
        anchor_symbol=anchor_symbol,
        anchor_tag=anchor_tag,
        adsorbate_indices=adsorbate_indices,
        d_band_center=d_band_center,
        cutoff_multiplier=cutoff_multiplier,
    )


__all__ = [
    "D_BAND_CENTER",
    "site_local_d_band_from_atoms",
    "site_local_d_band_from_frame",
]

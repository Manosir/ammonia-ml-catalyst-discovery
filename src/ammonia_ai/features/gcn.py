"""Generalized coordination-number utilities for adsorbate sites.

The public API is intentionally centered on an adsorbate *anchor* atom rather
than on the name of one reaction intermediate.  For the current N* workflow,
use ``anchor_symbol="N"`` and ``anchor_tag=2``.  The same API can later be
used for NH* and NH2* because the nitrogen atom remains the active-site anchor;
future callers may additionally pass all adsorbate atom indices so H atoms are
excluded from substrate coordination counts.

The production screening convention is ``frame_index=0``.  Final relaxed
geometry can be requested explicitly for a diagnostic, but it must not be used
silently for a pre-DFT screening claim.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _read_atoms(path: str | Path, frame_index: int):
    """Read one trajectory frame through ASE.

    Kept as a small wrapper so tests and future data backends can replace the
    file reader without changing the mathematical GCN implementation.
    """
    from ase.io import read

    return read(str(path), index=frame_index)


def _neighbor_arrays(atoms: Any, cutoff_multiplier: float):
    """Return directed neighbor-list arrays for an ASE ``Atoms`` object."""
    from ase.neighborlist import natural_cutoffs, neighbor_list

    cutoffs = natural_cutoffs(atoms, mult=cutoff_multiplier)
    return neighbor_list("ij", atoms, cutoffs)


def _select_anchor_index(
    symbols: Sequence[str],
    tags: Sequence[int] | None,
    *,
    anchor_symbol: str,
    anchor_tag: int | None,
    allow_symbol_fallback: bool,
) -> int | None:
    """Select the tagged adsorbate anchor atom."""
    if tags is not None and anchor_tag is not None:
        tagged = [
            i for i, (symbol, tag) in enumerate(zip(symbols, tags))
            if symbol == anchor_symbol and int(tag) == int(anchor_tag)
        ]
        if tagged:
            return tagged[0]

    if allow_symbol_fallback:
        matches = [i for i, symbol in enumerate(symbols) if symbol == anchor_symbol]
        if matches:
            return matches[0]
    return None


def _gcn_from_neighbour_lists(
    anchor_index: int,
    i_indices: np.ndarray,
    j_indices: np.ndarray,
    symbols: Sequence[str],
    *,
    cn_max: int = 12,
    adsorbate_indices: Iterable[int] | None = None,
    substrate_symbols: set[str] | None = None,
) -> float:
    """Compute GCN from pre-built directed neighbor-list arrays.

    Parameters
    ----------
    anchor_index:
        Index of the adsorbate anchor atom, normally the tagged N atom.
    i_indices, j_indices:
        Directed arrays returned by ``ase.neighborlist.neighbor_list("ij", ...)``.
    symbols:
        Chemical symbols in atom-index order.
    cn_max:
        Reference bulk coordination number; 12 is standard for close-packed
        transition-metal references.
    adsorbate_indices:
        Atom indices belonging to the adsorbate.  For N* this may contain only
        the N anchor.  For NH*/NH2*, pass the N and H indices so H is excluded
        from substrate coordination counts.
    substrate_symbols:
        Optional set of symbols considered substrate atoms.  If omitted, all
        atoms not in ``adsorbate_indices`` are treated as substrate atoms.

    Returns
    -------
    float
        Generalized coordination number, or ``numpy.nan`` if no substrate
        neighbors are found.
    """
    if cn_max <= 0:
        raise ValueError("cn_max must be positive")
    if len(symbols) == 0:
        return float("nan")

    i_indices = np.asarray(i_indices, dtype=int)
    j_indices = np.asarray(j_indices, dtype=int)
    if i_indices.shape != j_indices.shape:
        raise ValueError("i_indices and j_indices must have the same shape")

    ads_indices = set(adsorbate_indices or {anchor_index})

    def is_substrate(index: int) -> bool:
        if index in ads_indices:
            return False
        return substrate_symbols is None or symbols[index] in substrate_symbols

    site_neighbors = [
        int(j) for i, j in zip(i_indices, j_indices)
        if int(i) == anchor_index and is_substrate(int(j))
    ]
    if not site_neighbors:
        return float("nan")

    site_gcn_sum = 0.0
    for metal_index in site_neighbors:
        metal_neighbors = [
            int(k) for i, k in zip(i_indices, j_indices)
            if int(i) == metal_index and is_substrate(int(k))
        ]
        metal_gcn = 0.0
        for neighbor_index in metal_neighbors:
            neighbor_neighbors = [
                int(k) for i, k in zip(i_indices, j_indices)
                if int(i) == neighbor_index and is_substrate(int(k))
            ]
            metal_gcn += len(neighbor_neighbors) / cn_max
        site_gcn_sum += metal_gcn

    return float(site_gcn_sum / len(site_neighbors))


def calculate_site_gcn(
    extxyz_path: str | Path,
    *,
    frame_index: int = 0,
    cn_max: int = 12,
    anchor_symbol: str = "N",
    anchor_tag: int | None = 2,
    allow_symbol_fallback: bool = False,
    adsorbate_indices: Iterable[int] | None = None,
    substrate_symbols: set[str] | None = None,
    cutoff_multiplier: float = 1.2,
) -> float:
    """Calculate the GCN of an adsorbate binding site.

    ``frame_index=0`` is the default because it is the appropriate convention
    for a feature intended to describe a candidate before relaxation.  Use
    ``frame_index=-1`` only for an explicitly labelled relaxed-frame diagnostic.

    The function returns ``numpy.nan`` for an unreadable trajectory, a missing
    tagged anchor, or a site with no recognized substrate neighbors.  Invalid
    configuration arguments raise ``ValueError`` so configuration mistakes are
    not silently hidden.
    """
    if not isinstance(frame_index, int):
        raise TypeError("frame_index must be an integer")
    if cn_max <= 0:
        raise ValueError("cn_max must be positive")
    if cutoff_multiplier <= 0:
        raise ValueError("cutoff_multiplier must be positive")

    try:
        atoms = _read_atoms(extxyz_path, frame_index)
        symbols = list(atoms.get_chemical_symbols())
        tags = list(atoms.get_tags()) if hasattr(atoms, "get_tags") else None
        anchor_index = _select_anchor_index(
            symbols,
            tags,
            anchor_symbol=anchor_symbol,
            anchor_tag=anchor_tag,
            allow_symbol_fallback=allow_symbol_fallback,
        )
        if anchor_index is None:
            return float("nan")
        i_indices, j_indices = _neighbor_arrays(atoms, cutoff_multiplier)
        return _gcn_from_neighbour_lists(
            anchor_index,
            i_indices,
            j_indices,
            symbols,
            cn_max=cn_max,
            adsorbate_indices=adsorbate_indices,
            substrate_symbols=substrate_symbols,
        )
    except (OSError, ValueError, IndexError, TypeError, ImportError):
        return float("nan")
    except Exception:
        # Trajectory parsing and neighbor-list failures are data-quality
        # outcomes for batch preprocessing.  The caller should count and log
        # them; the function must not abort an entire 12k-file run.
        return float("nan")


__all__ = [
    "calculate_site_gcn",
    "_gcn_from_neighbour_lists",
]


# Backward-compatible US spelling for callers that prefer it.
_gcn_from_neighbor_lists = _gcn_from_neighbour_lists

__all__.append("_gcn_from_neighbor_lists")

"""
src/gcn_utils.py
-----------------
Generalised Coordination Number (GCN) utilities for the Ammonia-AI pipeline.

Public API
----------
calculate_site_gcn(extxyz_path, frame_index=-1, cn_max=12)
    Read an extxyz trajectory and return the GCN of the N binding site.
    frame_index: 0 = initial (unrelaxed), -1 = final (relaxed).
    Use frame_index=0 for leakage-free screening of new catalysts.

_gcn_from_neighbour_lists(n_idx, i_indices, j_indices, symbols, cn_max=12)
    Pure mathematical GCN computation from pre-built neighbour lists.
    Separated from file I/O so it can be unit-tested independently.
    Called internally by calculate_site_gcn; also importable directly
    for cases where an ASE Atoms object is already in memory.

Reference
---------
Calle-Vallejo et al., Science 350 (2015) 185-189.
"""

from __future__ import annotations

import numpy as np


def _gcn_from_neighbour_lists(
    n_idx: int,
    i_indices: "np.ndarray",
    j_indices: "np.ndarray",
    symbols: list[str],
    cn_max: int = 12,
) -> float:
    """
    Compute GCN of the N adsorption site from pre-built neighbour lists.

    Parameters
    ----------
    n_idx     : index of the N atom in the Atoms object
    i_indices : 'i' array from ase.neighborlist.neighbor_list('ij', ...)
    j_indices : 'j' array from the same call
    symbols   : list of element symbols (atoms.get_chemical_symbols())
    cn_max    : reference coordination number (12 for FCC/HCP bulk)

    Returns
    -------
    float : GCN value, or np.nan if no metal neighbours found
    """
    site_metal_indices = j_indices[i_indices == n_idx]
    if len(site_metal_indices) == 0:
        return np.nan

    site_gcn_sum = 0.0
    for metal_idx in site_metal_indices:
        metal_neighbours = j_indices[i_indices == metal_idx]
        metal_gcn = 0.0
        for neighbour_idx in metal_neighbours:
            if symbols[neighbour_idx] != "N":
                nbr_of_nbr = j_indices[i_indices == neighbour_idx]
                cn_j = int(np.sum(
                    [symbols[k] != "N" for k in nbr_of_nbr]
                ))
                metal_gcn += cn_j / cn_max
        site_gcn_sum += metal_gcn

    return site_gcn_sum / len(site_metal_indices)


def calculate_site_gcn(
    extxyz_path: str,
    frame_index: int = -1,
    cn_max: int = 12,
) -> float:
    """
    Calculate the GCN of the N binding site from an extxyz trajectory file.

    Parameters
    ----------
    extxyz_path : path to the .extxyz or .extxyz.xz trajectory file
    frame_index : which trajectory frame to use.
                  -1 = final relaxed frame (original behaviour).
                   0 = initial unrelaxed frame (leakage-free, use for screening).
    cn_max      : reference coordination number (default 12)

    Returns
    -------
    float : GCN value, or np.nan on any error
    """
    try:
        from ase.io import read
        from ase.neighborlist import neighbor_list, natural_cutoffs
        atoms = read(str(extxyz_path), index=frame_index)
    except Exception:
        return np.nan

    cutoffs = natural_cutoffs(atoms, mult=1.2)
    try:
        i_indices, j_indices = neighbor_list("ij", atoms, cutoffs)
    except Exception:
        return np.nan

    symbols = atoms.get_chemical_symbols()
    n_indices = [i for i, s in enumerate(symbols) if s == "N"]
    if not n_indices:
        return np.nan

    return _gcn_from_neighbour_lists(
        n_idx=n_indices[0],
        i_indices=i_indices,
        j_indices=j_indices,
        symbols=symbols,
        cn_max=cn_max,
    )

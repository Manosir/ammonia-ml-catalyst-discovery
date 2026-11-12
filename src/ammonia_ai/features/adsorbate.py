"""ASE-based extraction of adsorbate-aware records from OC20 trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ammonia_ai.adsorbates import ADSORBATE_METADATA_COLUMNS, metadata_for


def _energy(atoms: Any, *names: str) -> float | None:
    """Read energy from metadata first, then from an ASE calculator."""
    for name in names:
        value = atoms.info.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    calculator = getattr(atoms, "calc", None)
    if calculator is not None:
        try:
            return float(calculator.get_property("energy", atoms))
        except (KeyError, RuntimeError, TypeError, ValueError):
            pass

        try:
            return float(atoms.get_potential_energy())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    return None


def _system_id(path: Path) -> str:
    name = path.name
    for suffix in (".extxyz.xz", ".extxyz.gz", ".extxyz", ".xyz", ".xz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _adsorbate_indices(atoms: Any, metadata: Any) -> list[int]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    # OC20 trajectories carry an adsorbate tag (the sample N* trajectories use
    # tag 2). Prefer it because surfaces can themselves contain hydrogen.
    try:
        tags = np.asarray(atoms.get_tags())
        tagged = np.flatnonzero(tags == 2).tolist()
        tagged_symbols = symbols[tagged]
        if (
            len(tagged) == metadata.adsorbate_atom_count
            and int(np.sum(tagged_symbols == "N")) == metadata.n_adsorbate_N
            and int(np.sum(tagged_symbols == "H")) == metadata.n_adsorbate_H
        ):
            return tagged
    except (AttributeError, TypeError, ValueError):
        pass

    # Fallback for converted files without OC20 tags. OC20's N-containing
    # adsorbates have N as their binding atom; use the expected composition.
    n_indices = np.flatnonzero(symbols == metadata.binding_atom).tolist()
    if not n_indices:
        return []
    if metadata.adsorbate_formula == "N":
        return [n_indices[-1]]
    if metadata.adsorbate_formula == "N2":
        return n_indices[-metadata.n_adsorbate_N :]
    h_indices = np.flatnonzero(symbols == "H").tolist()
    return n_indices[-1:] + h_indices[-metadata.n_adsorbate_H :]


def extract_adsorbate_record(path: str | Path, adsorbate: str) -> dict[str, Any]:
    """Extract one initial/final-frame record from an OC20 trajectory.

    The function intentionally reads only the first and last frames. It keeps
    missing energies as NaN so the caller can report and filter them explicitly.
    """
    from ase.io import read

    trajectory = Path(path)
    frames = read(str(trajectory), index=":")
    if not frames:
        raise ValueError(f"trajectory contains no frames: {trajectory}")
    initial, final = frames[0], frames[-1]
    metadata = metadata_for(adsorbate)
    ads_indices = _adsorbate_indices(final, metadata)
    if not ads_indices:
        raise ValueError(f"could not identify {adsorbate} in {trajectory}")

    positions = final.get_positions()
    surface_indices = [i for i in range(len(final)) if i not in ads_indices]

    ads_z = positions[ads_indices, 2]
    ads_position = positions[ads_indices].mean(axis=0)

    if surface_indices:
        # Use local surface atoms rather than the globally highest slab atom.
        # OC20 slabs can be tilted, stepped, and periodically wrapped.
        distances = final.get_distances(
            ads_indices[0],
            surface_indices,
            mic=True,
        )

        # Select nearby surface atoms. A 3.5 Å cutoff captures the local
        # coordination environment for the current descriptor.
        local_surface = [
            index
            for index, distance in zip(surface_indices, distances)
            if distance <= 3.5
        ]

        # If the local neighborhood is unusually sparse, retain the nearest
        # three surface atoms rather than producing an artificial missing value.
        if len(local_surface) < 3:
            nearest_order = np.argsort(distances)[: min(3, len(surface_indices))]
            local_surface = [surface_indices[int(i)] for i in nearest_order]

        local_surface_z = positions[local_surface, 2]
        ads_height = float(np.min(ads_z) - np.median(local_surface_z))
    else:
        ads_height = float("nan")


    try:
        forces = final.get_forces(apply_constraint=False)
        ads_force = float(np.linalg.norm(forces[ads_indices], axis=1).mean())
    except (KeyError, RuntimeError, ValueError):
    # Some converted or synthetic trajectories do not contain forces.
    # Preserve the record and make the missing descriptor explicit.
        ads_force = float("nan")

#    forces = final.get_forces(apply_constraint=False) 
#    ads_force = float(np.linalg.norm(forces[ads_indices], axis=1).mean())
    result: dict[str, Any] = {
        "system_id": _system_id(trajectory),
        "adsorbate": metadata.adsorbate_formula,
        "energy_initial_eV": _energy(initial, "energy", "free_energy"),
        "energy_final_eV": _energy(final, "energy", "free_energy"),
        "delta_E_ads": (
            _energy(final, "energy", "free_energy") - _energy(initial, "energy", "free_energy")
            if _energy(initial, "energy", "free_energy") is not None
            and _energy(final, "energy", "free_energy") is not None
            else float("nan")
        ),
        "ads_height": ads_height,
        "mean_ads_force": ads_force,
        "n_surface_atoms": len(surface_indices),
    }
    result.update({name: getattr(metadata, name) for name in ADSORBATE_METADATA_COLUMNS})
    return result

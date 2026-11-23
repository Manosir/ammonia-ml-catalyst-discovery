"""Shared metadata and download registry for OC20 adsorbates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdsorbateMetadata:
    """The seven metadata fields carried by every extracted record."""

    adsorbate_formula: str
    adsorbate_class: str
    n_adsorbate_N: int
    n_adsorbate_H: int
    adsorbate_mass: float
    binding_atom: str
    adsorbate_atom_count: int


# The masses are standard atomic masses in unified atomic mass units.
ADSORBATE_METADATA: dict[str, AdsorbateMetadata] = {
    "N": AdsorbateMetadata("N", "nitrogen_atom", 1, 0, 14.007, "N", 1),
    "N2": AdsorbateMetadata("N2", "nitrogen_molecule", 2, 0, 28.014, "N", 2),
    "NH": AdsorbateMetadata("NH", "nitrogen_hydride", 1, 1, 15.015, "N", 2),
    "NH2": AdsorbateMetadata("NH2", "nitrogen_hydride", 1, 2, 16.023, "N", 3),
    "NH3": AdsorbateMetadata("NH3", "nitrogen_hydride", 1, 3, 17.031, "N", 4),
}


@dataclass(frozen=True)
class OC20Archive:
    """Download information for a per-adsorbate OC20 IS2RE trajectory archive."""

    adsorbate: str
    archive_id: int
    md5: str
    size_gib: float

    @property
    def url(self) -> str:
        return (
            "https://dl.fbaipublicfiles.com/opencatalystproject/data/"
            f"per_adsorbate_is2res/{self.archive_id}.tar"
        )


OC20_ARCHIVES: dict[str, OC20Archive] = {
    "N2": OC20Archive("N2", 69, "cc668fecf679b6edaac8fd8fb9cdd404", 1.5),
    "NH2": OC20Archive("NH2", 71, "c7f383b50faa6244e265c9611466cb8f", 1.6),
    "NH3": OC20Archive("NH3", 72, "2b355741f9300445703270e0e4b8c01c", 1.9),
    "NH": OC20Archive("NH", 74, "7979b9e7ab557d6979b33e352486f0ef", 1.4),
    "N": OC20Archive("N", 77, "bfb6e03d4a687987ff68976f0793cc46", 1.1),
}

ADSORBATE_METADATA_COLUMNS: tuple[str, ...] = (
    "adsorbate_formula",
    "adsorbate_class",
    "n_adsorbate_N",
    "n_adsorbate_H",
    "adsorbate_mass",
    "binding_atom",
    "adsorbate_atom_count",
)


def metadata_for(adsorbate: str) -> AdsorbateMetadata:
    """Return metadata for a canonical OC20 adsorbate name."""
    try:
        return ADSORBATE_METADATA[adsorbate.upper()]
    except KeyError as exc:
        supported = ", ".join(sorted(ADSORBATE_METADATA))
        raise ValueError(f"unsupported adsorbate {adsorbate!r}; choose from {supported}") from exc

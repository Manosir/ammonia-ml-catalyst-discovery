from pathlib import Path

import pytest

from ammonia_ai.adsorbates import ADSORBATE_METADATA_COLUMNS, metadata_for


def test_all_five_adsorbates_have_seven_metadata_fields():
    for name in ("N", "N2", "NH", "NH2", "NH3"):
        metadata = metadata_for(name)
        assert len(ADSORBATE_METADATA_COLUMNS) == 7
        assert all(getattr(metadata, field) is not None for field in ADSORBATE_METADATA_COLUMNS)


def test_unknown_adsorbate_is_rejected():
    with pytest.raises(ValueError):
        metadata_for("NO")


def test_oc20_archive_ids_are_distinct():
    from ammonia_ai.adsorbates import OC20_ARCHIVES

    assert {record.archive_id for record in OC20_ARCHIVES.values()} == {69, 71, 72, 74, 77}


@pytest.mark.parametrize("name, expected_h", [("N", 0), ("N2", 0), ("NH", 1), ("NH2", 2), ("NH3", 3)])
def test_hydrogen_count(name, expected_h):
    assert metadata_for(name).n_adsorbate_H == expected_h

import json
from pathlib import Path

import pytest

from ammonia_ai import schema as S


def test_manifest_is_consistent_with_schema():
    path = Path("results/adsorbates/model_manifest.json")
    if not path.exists():
        pytest.skip("manifest not built")
    m = json.loads(path.read_text())
    assert m["schema_version"] == S.SCHEMA_VERSION
    assert m["group_key"] == "bulk_mpid"
    if m["training_ready"]:
        assert set(m["implemented_feature_columns"]) == set(S.SCREENING_FEATURES)

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("bc", Path(__file__).parents[1] / "scripts" / "build_candidates.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


def test_period_group_known_elements():
    assert bc.period_group(26) == (4, 8)    # Fe
    assert bc.period_group(29) == (4, 11)   # Cu
    assert bc.period_group(44) == (5, 8)    # Ru
    assert bc.period_group(78) == (6, 10)   # Pt
    assert bc.period_group(72) == (6, 4)    # Hf
    assert bc.period_group(7) == (2, 15)    # N
    assert bc.period_group(13) == (3, 13)   # Al
    assert bc.period_group(57) == (6, 3)    # La


def test_element_properties():
    assert bc.props("Ni")["nd"] == 8.0
    assert bc.props("Fe")["nd"] == 6.0
    assert bc.props("N")["nval"] == 5.0 and bc.props("N")["nd"] == 0.0
    assert bc.props("Pt")["tm"] and not bc.props("Al")["tm"]
    assert bc.props("S")["nm"] and not bc.props("Ru")["nm"]


def test_parsers():
    assert bc.parse_formula("Sc2Ni2Ga4") == {"Sc": 2, "Ni": 2, "Ga": 4}
    assert bc.parse_comp("Fe:20;Ru:4") == {"Fe": 20, "Ru": 4}


def test_aggregation():
    a = bc.agg("x", {"Fe": 1, "Ni": 1})
    assert a["x_nd_mean"] == 7.0 and a["x_tm_frac"] == 1.0 and a["x_n_elem"] == 2.0
    assert a["x_nd_rng"] == 2.0
    assert bc.agg("x", {}) == {}
    assert np.isnan(bc.props("Ac")["en"])

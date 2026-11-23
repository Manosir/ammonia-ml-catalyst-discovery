import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ammonia_ai import schema as S

spec = importlib.util.spec_from_file_location("ev", Path(__file__).parents[1] / "scripts" / "evaluate_screening.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def synthetic(n_groups=40, per=25, seed=0):
    rng = np.random.default_rng(seed)
    elements = ["Fe", "Ni", "Cu", "Ru", "Ti", "Pt"]
    rows = []
    for g in range(n_groups):
        chosen = sorted(rng.choice(elements, size=int(rng.integers(1, 3)), replace=False))
        symbols = "".join(f"{e}{int(rng.integers(1, 3))}" for e in chosen)
        for k in range(per):
            rows.append({"bulk_symbols": symbols, "adsorbate": ev.ADSORBATES[(g + k) % 5]})
    df = pd.DataFrame(rows)
    n = len(df)
    df["bulk_chemsys"] = df.bulk_symbols.map(S.chemical_system)
    df["class"] = rng.integers(0, 4, size=n)
    df["anomaly"] = rng.choice([0] * 8 + [1, 2, 3], size=n)
    df["is_clean_primary_regression_row"] = df.anomaly == 0
    counts = [S.adsorbate_counts(a) for a in df.adsorbate]
    feats = {f: rng.normal(size=n) for f in S.SCREENING_FEATURES}
    feats["ads_nN"] = np.array([x[0] for x in counts], dtype=float)
    feats["ads_nH"] = np.array([x[1] for x in counts], dtype=float)
    df = pd.concat([df, pd.DataFrame(feats)], axis=1)
    df["split"] = [rng.choice(["val_oos_ads", "val_oos_ads_bulk"]) if a == "NH2" else rng.choice(["train", "train", "val_is", "val_oos_bulk"])
                   for a in df.adsorbate]
    df["adsorption_energy_eV"] = 0.6 * df["site_en_mean"] - 0.4 * df["ads_nH"] + 0.3 * df["class"] + rng.normal(0, 0.2, size=n)
    return df


def test_assign_folds_keeps_groups_whole_and_uses_every_fold():
    groups = np.repeat([f"g{i}" for i in range(30)], 7)
    fold = ev.assign_folds(groups, 5, seed=1)
    assert set(fold) == set(range(5))
    for g in np.unique(groups):
        assert len(set(fold[groups == g])) == 1
    assert not np.array_equal(fold, ev.assign_folds(groups, 5, seed=2))


def test_floor_predict_uses_training_means_and_falls_back():
    y = np.array([1.0, 3.0, 10.0, 5.0])
    key = np.array(["N_0", "N_0", "NH_1", "N2_2"])
    pred = ev.floor_predict(y, key, np.array([0, 1, 2]), np.array([3]))
    assert pred[0] == np.mean([1.0, 3.0, 10.0])  # unseen key -> overall training mean
    assert ev.floor_predict(y, key, np.array([0, 1, 2]), np.array([0]))[0] == 2.0


def test_group_bootstrap_interval_and_zero_paired_difference():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(50), 10)
    v = rng.normal(1.0, 0.5, size=500)
    point, lo, hi = ev.group_bootstrap(groups, lambda idx: v[idx].mean(), 200, seed=0)
    assert lo[0] <= point[0] <= hi[0] and hi[0] - lo[0] < 0.5
    same = ev.group_bootstrap(groups, lambda idx: (v[idx] - v[idx]).mean(), 50, seed=0)
    assert all(float(x[0]) == 0.0 for x in same)


def test_end_to_end_on_synthetic_data(tmp_path):
    df = synthetic()
    table, out = tmp_path / "t.parquet", tmp_path / "eval.json"
    df.to_parquet(table, index=False)
    args = ["--table", str(table), "--out", str(out), "--registry", "none", "--folds", "3", "--repeats", "1",
            "--bootstrap", "20", "--max-iter", "20", "--min-rows", "30", "--max-elements", "3"]
    assert ev.main(args + ["--mode", "all"]) == 0
    res = json.loads(out.read_text())
    for key in ("grouped_clean_by_adsorbate", "grouped_clean_by_class", "official_val_is", "leave_element_out", "leave_adsorbate_out", "anomaly"):
        assert key in res, key
    row = res["grouped_clean_by_adsorbate"]["ALL"]
    assert row["full"][0] < row["floor"]  # the synthetic signal is learnable: beating the floor is expected
    assert res["features_full"] == 64 and res["features_light"] == 55


def _grouped_mae(df, tmp_path, name):
    table, out = tmp_path / f"{name}.parquet", tmp_path / f"{name}.json"
    df.to_parquet(table, index=False)
    args = ["--table", str(table), "--out", str(out), "--registry", "none", "--folds", "3", "--repeats", "1",
            "--bootstrap", "5", "--max-iter", "20", "--mode", "grouped"]
    assert ev.main(args) == 0
    return json.loads(out.read_text())["grouped_clean_by_adsorbate"]["ALL"]["full"][0]


def test_audit_columns_in_the_table_never_reach_the_model(tmp_path):
    df = synthetic(n_groups=12, per=20)
    base = _grouped_mae(df, tmp_path, "base")
    leaky = df.copy()
    leaky["slab_disp_max"] = leaky["adsorption_energy_eV"]   # a perfect, forbidden predictor
    assert _grouped_mae(leaky, tmp_path, "leaky") == base

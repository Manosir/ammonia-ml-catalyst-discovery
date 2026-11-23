#!/usr/bin/env python3
"""Train and save the screening-v3 models.

- regressors (full contract / without placement-sensitive features) on clean rows (anomaly == 0);
- anomaly classifiers (dissociated, desorbed, surface changed) on all rows, full contract.
Every file gets a manifest. Models are untuned. A regressor is meaningful only for rows an anomaly
classifier calls clean, and none of this is a catalyst ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ammonia_ai import schema as S

ANOMALY_CODES = {1: "dissociated", 2: "desorbed", 3: "surface_changed"}
EVALUATION_REFERENCE = "reports/evaluation_2026-11-14.json"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True, stderr=subprocess.DEVNULL).strip()
        return commit, bool(status)
    except Exception:
        return None, None


def check_model(model, manifest, expected) -> None:
    """A saved model may be used only with a manifest of the current schema and the expected feature list."""
    S.assert_model_schema_version(manifest)
    if tuple(manifest["feature_columns"]) != tuple(expected):
        raise ValueError("manifest feature columns differ from the expected list")
    count = S.model_feature_count(model)
    if count != len(expected):
        raise ValueError(f"model expects {count} features, expected {len(expected)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="results/adsorbates/oc20_adsorbates_features.parquet")
    ap.add_argument("--out-dir", default="results/models")
    ap.add_argument("--registry", default=str(S.DEFAULT_REGISTRY), help="registry CSV, or 'none' (tests only)")
    ap.add_argument("--max-iter", type=int, default=200)
    a = ap.parse_args(argv)

    df = pd.read_parquet(a.table).reset_index(drop=True)
    S.validate_feature_frame(df, registry_path=None if str(a.registry).lower() == "none" else a.registry)
    for col in ("adsorption_energy_eV", "anomaly", "is_clean_primary_regression_row"):
        if col not in df.columns:
            raise SystemExit(f"table lacks {col}; rebuild it with scripts/build_features.py")
    full = list(S.SCREENING_FEATURES)
    light = [c for c in full if c not in set(S.PLACEMENT_SENSITIVE_FEATURES)]
    clean = df["is_clean_primary_regression_row"].to_numpy(bool)
    y = df["adsorption_energy_eV"].to_numpy(float)
    anomaly = df["anomaly"].to_numpy().astype(int)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_state()
    params = dict(max_iter=a.max_iter, learning_rate=0.1, early_stopping=False, random_state=0)
    common = {
        "schema_version": S.SCHEMA_VERSION,
        "table": str(a.table), "table_md5": file_md5(Path(a.table)), "table_rows": int(len(df)),
        "hyperparameters": params,
        "library_versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "git_commit": commit, "git_dirty": dirty,
        "evaluation_reference": EVALUATION_REFERENCE,
        "limitations": [
            "untuned models; error intervals in the evaluation reflect test-set sampling only",
            "regressors are trained on clean rows only: use them for rows the anomaly classifiers call clean",
            "input is an unrelaxed slab with a known adsorption site; features come from the initial frame",
            "OC20 gives low-coverage RPBE adsorption energies: no barriers, coadsorbate effects or free-energy corrections",
            "not a catalyst ranking: N2 dissociation needs a BEP or microkinetic layer",
        ],
    }
    probe = df[full].to_numpy(float)[:50]

    def write(name, features, extra):
        manifest = {**common, "file": f"{name}.joblib", "feature_columns": list(features), **extra}
        (out / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    for kind, cols in (("full", full), ("light", light)):
        name = f"screening_v3_{kind}"
        X = df[cols].to_numpy(float)
        model = HistGradientBoostingRegressor(**params).fit(X[clean], y[clean])
        joblib.dump(model, out / f"{name}.joblib")
        manifest = write(name, cols, {"kind": f"regressor_{kind}", "training_rows": int(clean.sum()),
                                      "trained_on": "clean rows (anomaly == 0)",
                                      "placement_sensitive_features": list(S.PLACEMENT_SENSITIVE_FEATURES) if kind == "full" else []})
        again = joblib.load(out / f"{name}.joblib")
        check_model(again, manifest, cols)
        Xp = probe[:, [full.index(c) for c in cols]]
        assert np.array_equal(model.predict(Xp), again.predict(Xp)), f"{name}: reloaded model predicts differently"
        print(f"saved {name}: {len(cols)} features, {int(clean.sum()):,} training rows")

    X = df[full].to_numpy(float)
    classifiers = {}
    for code, label in ANOMALY_CODES.items():
        yb = (anomaly == code).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb):
            raise SystemExit(f"anomaly code {code} ({label}) is absent or universal in the table")
        classifiers[label] = HistGradientBoostingClassifier(**params).fit(X, yb)
    name = "screening_v3_anomaly"
    joblib.dump(classifiers, out / f"{name}.joblib")
    manifest = write(name, full, {"kind": "anomaly_classifiers", "training_rows": int(len(df)), "trained_on": "all rows",
                                  "targets": {label: code for code, label in ANOMALY_CODES.items()}})
    again = joblib.load(out / f"{name}.joblib")
    for label, clf in again.items():
        check_model(clf, manifest, full)
        assert np.array_equal(classifiers[label].predict_proba(probe), clf.predict_proba(probe)), f"{label}: reloaded classifier differs"
    print(f"saved {name}: {len(classifiers)} classifiers ({', '.join(classifiers)}), {len(df):,} training rows")
    print(f"models and manifests in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

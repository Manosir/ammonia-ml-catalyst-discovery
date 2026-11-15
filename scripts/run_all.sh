#!/usr/bin/env bash
# Raw OC20 files -> audited join -> structure pass -> candidate features -> model table -> evaluation.
# Re-runnable: the structure pass skips finished files. SKIP_STRUCTURE=1 reuses results/structure_pass.
# Environment: PYTHON (default python3), WORKERS (default 6), MAX_HOURS (default 2).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
JOIN=results/adsorbates/oc20_adsorbates_official_join.parquet

echo "== 1/7 audit and official join"
$PY scripts/audit_oc20_assets.py --data-root data --raw results/adsorbates/oc20_adsorbates.csv --out-dir results/adsorbates
echo "== 2/7 feature registry from the schema"
$PY scripts/feature_availability.py --output results/adsorbates/feature_availability.csv
if [ "${SKIP_STRUCTURE:-0}" != "1" ]; then
  echo "== 3/7 structure pass (resumable; about 6 minutes from scratch on 6 workers)"
  $PY -u scripts/structure_pass.py --table "$JOIN" --workers "${WORKERS:-6}" --max-hours "${MAX_HOURS:-2}"
fi
echo "== 4/7 candidate features"
$PY -u scripts/build_candidates.py --table "$JOIN"
echo "== 5/7 model table"
$PY scripts/build_features.py --raw results/adsorbates/oc20_adsorbates.csv --official-join "$JOIN" \
  --registry results/adsorbates/feature_availability.csv --out results/adsorbates/oc20_adsorbates_features.parquet \
  --manifest results/adsorbates/model_manifest.json
echo "== 6/7 evaluation"
$PY -u scripts/evaluate_screening.py --out results/evaluation_screening.json
echo "== 7/7 final models"
$PY scripts/train_final.py
echo "done: results/evaluation_screening.json and results/models/"

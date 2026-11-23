#!/usr/bin/env bash
# Run each OC20 adsorbate extraction sequentially and preserve per-species logs.
set -u

# Keep standard macOS utilities available even if the parent zsh session has a
# damaged PATH. This does not alter the Python virtual environment selection.
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATA_DIR="${DATA_DIR:-data/oc20_is2res/trajectories}"
RESULT_DIR="${RESULT_DIR:-results/adsorbates}"
LOG_DIR="${LOG_DIR:-logs/adsorbates}"
WORKERS="${WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-100}"

ADSORBATES=(N N2 NH NH2 NH3)

mkdir -p "$RESULT_DIR" "$LOG_DIR"

for adsorbate in "${ADSORBATES[@]}"; do
    output="$RESULT_DIR/${adsorbate}.csv"
    log="$LOG_DIR/${adsorbate}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $adsorbate"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] output=$output" >> "$log"

    if ! "$PYTHON_BIN" scripts/extract_adsorbates.py \
        --data-dir "$DATA_DIR" \
        --adsorbates "$adsorbate" \
        --output "$output" \
        --workers "$WORKERS" \
        --batch-size "$BATCH_SIZE" \
        2>&1 | tee -a "$log"; then
        status=${PIPESTATUS[0]}
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED $adsorbate (exit $status)" | tee -a "$log"
        exit "$status"
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished $adsorbate" | tee -a "$log"
done

RESULT_DIR="$RESULT_DIR" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import pandas as pd

result_dir = Path(os.environ["RESULT_DIR"])
files = [result_dir / f"{name}.csv" for name in ("N", "N2", "NH", "NH2", "NH3")]
missing = [str(path) for path in files if not path.exists()]
if missing:
    raise SystemExit(f"Cannot consolidate; missing files: {missing}")

frame = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
output = result_dir / "oc20_adsorbates.csv"
frame.to_csv(output, index=False)
print(f"Wrote {len(frame):,} records to {output}")
print(frame["status"].value_counts(dropna=False).to_string())
print(frame["adsorbate"].value_counts().to_string())
PY

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All adsorbates complete"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/prepare_v3.py
cache="reports/v3/train_cache_dev"
for k in q10 q50 q90 cat50 direction spike_high spike_low; do
  echo "[V3] Training $k"
  python scripts/model_worker_v3.py --data-dir "$cache/matrix" --features "$cache/features.json" --kind "$k" --output "$cache/$k.joblib"
done
python scripts/finalize_v3.py
python scripts/generate_backtest_v3.py
echo "[V3] Development pipeline complete."

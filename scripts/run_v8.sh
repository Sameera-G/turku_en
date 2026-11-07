#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
if [[ ! -f models/v3/model_bundle_v3.joblib ]]; then
  bash scripts/run_v3.sh
fi
python scripts/train_v8.py
python scripts/generate_backtest_v8.py
echo "[V8] Spread-expert pipeline complete."

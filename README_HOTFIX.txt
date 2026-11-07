TE-BalanceAI V11 feature dtype hotfix

Fixes LightGBM error:
  pandas dtypes must be int, float or bool ... object
when compare_v11_candidates.py or V11 retraining reads JSON/SQLite feature snapshots.

Extract directly over the existing project root and replace files.
This does NOT touch .env, the online SQLite database, forecasts, reports, or models.

Then run:
  python -m pytest -q tests\test_v11_system.py tests\test_v10_system.py tests\test_v9_system.py tests\test_v8_system.py tests\test_online_learning.py
  python scripts\compare_v11_candidates.py

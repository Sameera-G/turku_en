# TE-BalanceAI V3

V3 is the next drop-in upgrade after V2.

## Main improvements

- fixes missing 15-minute and 8-hour lag features from V2
- adds cross-border capacity stress and flow-reversal logic
- adds weather/renewable ramp and net-load regime features
- adds Nordic/neighbour market-pressure features
- adds flexible electric-load / electric-boiler market-state proxies
- adds stronger production-consumption and load-forecast-error features
- adds CatBoost as an independent P50 learner
- learns horizon-specific LightGBM/CatBoost late-fusion weights using validation MAE
- keeps P10/P90 and probability calibration
- adds model-disagreement reliability diagnostics
- updates API/dashboard to prefer V3 outputs automatically
- improves starter-package repair when a required file is missing

## Replace V2 files

Use `V3_REPLACE_FILES.txt`.

## Install

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\setup_data.py --zip .\starter_package_v1.zip
```

## Test

```powershell
$env:PYTHONPATH='.'
python -m pytest -q tests\test_v3_constraints.py tests\test_v2_constraints.py
```

## Fast V3 experiment

```powershell
python scripts\train_v3.py
python scripts\generate_backtest_v3.py
python scripts\compare_v2_v3.py
```

## Final candidate

```powershell
python scripts\train_v3.py --refit-all
python scripts\generate_backtest_v3.py
```

## Full one-command pipeline

```powershell
python scripts\run_v3_pipeline.py
```

or final refit:

```powershell
python scripts\run_v3_pipeline.py --refit-all
```

## API / Dashboard

```powershell
uvicorn src.api.app:app --reload --port 8000
streamlit run src\dashboard\app.py
```

## Optional live regional weather

```powershell
python scripts\download_public_context.py --live
```

This is dashboard/live-production context only unless historical forecast-vintage availability is auditable.

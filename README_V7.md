# TE-BalanceAI V7 — Accuracy-visible live forecasting + daily self-learning

V7 preserves the locked V3 competition/backtest engine and adds a production dashboard that makes forecast quality visible.

## What V7 adds

- Official validation metrics directly in the dashboard: MAE, P10/P90 pinball, coverage, spike Brier and direction Brier.
- Additional intuitive diagnostics: median absolute error and shares within ±10, ±20 and ±50 EUR/MWh.
- Accuracy by every one of the 40 quarter-hour forecast steps.
- True live settled accuracy over 24h, 7d and 14d as soon as Fingrid actual outcomes arrive.
- Historical replay with P10/P50/P90 and actual imbalance price overlay.
- Complete 40-row 15-minute forecast table with direction, spike risk, reliability, model disagreement and forecast revision.
- Hourly adaptive learning (24h + 7d + 14d windows).
- Daily Helsinki-time learning checkpoint (default 14:05): performance report + conditional champion/challenger retraining.
- Drift-triggered retraining remains available between daily checkpoints.
- New API endpoints for validation performance and daily-learning state.

## Important distinction

The official challenge has no single "accuracy percentage". It scores lower-is-better MAE, pinball loss and Brier scores. V7 therefore shows those official metrics first. The ±EUR hit rates are extra human-friendly diagnostics only.

## Upgrade from V6

Extract `TE_BalanceAI_V7_Replacement.zip` over your existing V6 folder. Keep `.env`, `data/`, `models/`, `outputs/`, and the official starter package.

Then run:

```powershell
cd "C:\Users\samee\Downloads\turkuen"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
pip install -r requirements.txt
python -m pytest -q tests\test_v7_performance.py tests\test_v6_system.py tests\test_v5_system.py tests\test_online_learning.py tests\test_v3_constraints.py tests\test_v2_constraints.py tests\test_feature_legality.py
```

Expected for the packaged build: `19 passed`.

## See accuracy immediately

```powershell
python scripts\accuracy_report.py
```

The current packaged validation result is approximately:

- MAE: 47.20 EUR/MWh
- P10/P90 pinball: 12.456
- P10-P90 coverage: 79.63%
- Exceptional-price Brier: 0.0973
- Direction Brier: 0.4214
- Within ±10 EUR/MWh: 36.7%
- Within ±20 EUR/MWh: 53.8%
- Within ±50 EUR/MWh: 73.0%

The last three are diagnostics, not official challenge metrics.

## Run a daily checkpoint now

For a report only:

```powershell
python scripts\run_daily_learning.py --force --no-retrain
```

Allow a challenger retrain when enough settled data exists:

```powershell
python scripts\run_daily_learning.py --force
```

Daily reports are written to:

```text
reports\online\daily\latest.json
reports\online\daily\daily_metrics.csv
```

Default daily checkpoint time is 14:00 Europe/Helsinki and can be changed in `.env`:

```text
TE_BALANCEAI_DAILY_LEARNING_LOCAL_HOUR=14
TE_BALANCEAI_DAILY_LEARNING_LOCAL_MINUTE=0
TE_BALANCEAI_DAILY_MIN_EVALUATIONS=32
```

## Start the dashboard

```powershell
cd "C:\Users\samee\Downloads\turkuen"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
python -m streamlit run .\src\dashboard\app.py
```

V7 permanently fixes the `src` import issue inside `app.py`, so the PYTHONPATH line is an extra safeguard rather than a requirement.

Dashboard tabs:

1. **Live forecast** — all 40 × 15-minute periods, uncertainty, direction, spike risk, reliability and hourly revision.
2. **Accuracy & validation** — official validation metrics plus live 24h/7d/14d metrics.
3. **Self-learning** — adaptive windows, drift, daily model checkpoint, promotion/rejection history.
4. **Historical replay** — move the forecast origin into the validation period and compare forecast vs actual.
5. **Data health** — Fingrid/Open-Meteo/FMI/ENTSO-E health and market digital twin.

## Continuous operation

```powershell
python scripts\run_online_daemon.py --delay-seconds 75
```

Every full hour the system:

1. fetches live sources;
2. stores the exact feature snapshot;
3. settles older predictions whose actuals are now known;
4. recomputes 24h/7d/14d adaptive weights/calibration;
5. detects drift;
6. emits a new 40-step next-10-hours forecast;
7. once per Helsinki day, writes the daily accuracy checkpoint and conditionally trains a challenger.

A challenger is promoted only when its chronological evaluation beats the champion without unacceptable regression in the challenge metrics.

## API

```powershell
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

New V7 endpoints:

- `GET /validation/performance`
- `GET /online/performance?hours=24`
- `GET /online/performance?hours=168`
- `GET /online/performance?hours=336`
- `GET /online/daily-learning`
- `GET /forecast/latest`
- `GET /online/market-state`
- `GET /online/source-health`

## Live accuracy warm-up

A new live installation starts with zero evaluations. That is expected. Live metrics become valid only after actual imbalance outcomes have been published and matched to the earlier forecasts. Until then, use the official validation section as the model-quality evidence.

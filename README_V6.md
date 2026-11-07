# TE-BalanceAI V6 - adaptive Finnish imbalance intelligence

V6 keeps the official Turku Energia hackathon backtest frozen and adds a separate production-grade continuous-learning layer inspired by public Finnish energy-industry practice.

## What is new in V6

1. **Three learning windows:** 24 h fast adaptation, 7 d regime adaptation, 14 d stable adaptation.
2. **Horizon-specific bias correction:** recent forecast-vs-actual residuals correct each of the 40 forecast steps.
3. **Drift detection:** compares recent error/coverage with the stable window and can trigger an early challenger retrain.
4. **Reserve Stress Index:** transparent signed summary of mFRR direction/activation, system surplus/deficit and frequency context.
5. **Weather forecast-error context:** FMI observations are compared with current operational weather forecasts; actual-vs-forecast wind/load proxies are also calculated.
6. **Market digital twin:** every hourly cycle writes a compact Finnish power-market state snapshot for UI/API/replay.
7. **Safe champion/challenger:** full retraining is still promoted only after chronological validation and metric guardrails.
8. **Exact online snapshots:** the multi-window state and extra live features are saved in each prediction's `feature_json`, allowing later challenger models to learn from exactly what was known at prediction time.

## Important separation

- **Hackathon/backtest mode:** organizer availability rules, frozen model parameters, reproducible 673 x 40 submission.
- **Production V6 mode:** live APIs + post-hour operational grace window + self-learning. Never use the production grace-window convention to regenerate the official hidden-test backtest.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put `FINGRID_API_KEY` and preferably `ENTSOE_API_TOKEN` into `.env`.

If the starter package is not already present:

```powershell
python scripts\setup_data.py --zip "PATH\starter_package_v1.zip"
```

## Verify

```powershell
python scripts\system_check.py
python -m pytest -q tests\test_v6_system.py tests\test_v5_system.py tests\test_online_learning.py tests\test_v3_constraints.py tests\test_v2_constraints.py tests\test_feature_legality.py
```

## Run one V6 cycle

```powershell
python scripts\run_online_cycle.py
```

Offline smoke test:

```powershell
python scripts\run_online_cycle.py --origin 2026-10-20T12:00:00Z --no-ingest --no-retrain
```

## Run continuously

```powershell
python scripts\run_online_daemon.py --delay-seconds 75
```

Each hourly cycle:

```text
Fingrid + ENTSO-E + Open-Meteo + FMI
                |
                v
        live market snapshot
                |
                v
     settle earlier predictions
                |
                v
  24h + 7d + 14d error learners
                |
                v
 bias / calibration / fusion adaptation
                |
         drift detection
          /           \
      normal       sustained drift
        |              |
        |        challenger retrain
        |              |
        +------ champion gate ------+
                       |
                       v
              40 x 15-min forecast
                       |
              digital-twin snapshot
                       |
                 API + dashboard
```

## Main outputs

- `outputs/online/latest_forecast.csv`
- `outputs/online/latest_diagnostics.csv`
- `data/online/latest_market_state.json`
- `outputs/online/history/forecast_*.csv`
- `outputs/online/history/market_state_*.json`
- `data/online/te_balanceai_online.sqlite3`
- `models/online/champion.joblib`
- `models/online/versions/*.joblib`

## API

```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Useful routes:

- `/forecast/latest`
- `/online/status`
- `/online/performance?hours=24`
- `/online/source-health`
- `/online/adaptation`
- `/online/market-state`
- `/online/history`

## Dashboard

```powershell
streamlit run src\dashboard\app.py
```

The dashboard shows the 10-hour probabilistic forecast, source health, digital-twin state, 24 h / 7 d / 14 d learning windows and drift status.

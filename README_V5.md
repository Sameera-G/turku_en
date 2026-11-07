# TE-BalanceAI V5 — full hackathon + production continuous-learning system

V5 keeps the validated V3 hackathon/backtest path and adds a separate production path that learns continuously from live public data.

## Correct operating cadence

The official challenge forecast origin is **every full hour**. Each origin produces **40 predictions at 15-minute resolution = 10 hours**.
Over a normal day the production service therefore creates up to **24 overlapping forecast batches**. It does **not** extrapolate one model run to 24 hours.

The live daemon runs shortly after each full UTC hour (default +75 seconds), fetches new information, evaluates newly settled forecasts, updates the adaptive layer, optionally runs the daily challenger retrain gate, and produces the 40-step batch for the current origin.

Around **14:00 Europe/Helsinki**, the normal hourly cycle is especially important because next-day day-ahead prices may become available. V5 fetches ENTSO-E day-ahead prices on every cycle, so a newly published curve is automatically picked up without a separate manual job.

## V5 production loop

```text
Fingrid + ENTSO-E + Open-Meteo + FMI
              |
              v
       availability/freshness
              |
              v
      exact origin feature snapshot
              |
              +---------------------------+
              |                           |
              v                           v
      champion forecast           store raw snapshot
              |                   + feature_json
              v
     hourly 24h adaptive learning
   fusion + quantile + probability
              |
              v
       40 x 15-min predictions
              |
              v
      API + dashboard + archive
              |
              v
   when actual outcomes settle
              |
              v
   evaluate MAE / coverage / Brier
              |
              v
  every 24h: train challenger
              |
       compare to champion
              |
        promote or reject
```

## Live data used

### Fingrid
Core Finnish operational inputs include imbalance price, mFRR/aFRR, consumption, production, cross-border flows, updated load/wind/solar forecasts, real-time surplus/deficit, electric-boiler consumption, frequency, wind/hydro/nuclear production.

### ENTSO-E
When `ENTSOE_API_TOKEN` is configured, V5 additionally requests Finnish day-ahead prices, load forecasts/actual load, wind/solar forecasts, generation mix, hydro reservoir state, generation unavailability and transmission-unavailability context.

### Weather
Open-Meteo is the primary regional live weather feed. FMI HARMONIE is an independent Finnish fallback. Representative nodes are Turku, Pori, Kalajoki, Raahe, Vaasa and Oulu.

## Continuous learning policy

**Every hourly cycle**
- compare settled predictions with actuals;
- use the previous 24 hours to update LightGBM/CatBoost fusion weights by horizon;
- adjust P10/P90 residual calibration;
- recalibrate direction and exceptional-price probabilities;
- reduce reliability when live source health is degraded.

**Default every 24 hours**
- build a challenger from leakage-safe historical data plus exact saved live feature snapshots;
- allow new V5 live-only features only after sufficient non-missing online coverage;
- compare champion and challenger on the most recent held-out online origins;
- reject the challenger if the required composite gain is not achieved or any major metric regresses beyond tolerance;
- keep versioned model files so rollback is possible.

Do not blindly retrain the full model every hour. The hourly adaptive layer gives fast learning while the daily champion/challenger gate prevents a single abnormal day from destabilising the model.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`:

```text
FINGRID_API_KEY=...
ENTSOE_API_TOKEN=...
```

Fingrid is the core live source. ENTSO-E is strongly recommended because it gives the public day-ahead price needed to evaluate high/low exceptional-price predictions in production.

## If you still need to install the organizer data

```powershell
python scripts\setup_data.py --zip "C:\path\to\starter_package_v1.zip"
```

Never commit the licensed Nord Pool CSV to a public repository.

## Verify

```powershell
python -m pytest -q tests\test_v5_system.py tests\test_online_learning.py tests\test_v3_constraints.py tests\test_v2_constraints.py tests\test_feature_legality.py
python scripts\system_check.py
```

## One live cycle

Run shortly after a full hour:

```powershell
python scripts\run_online_cycle.py
```

Output:

```text
outputs/online/latest_forecast.csv
outputs/online/latest_diagnostics.csv
outputs/online/latest_context.json
outputs/online/history/forecast_<origin>.csv
```

## Continuous daemon

```powershell
python scripts\run_online_daemon.py --delay-seconds 75
```

or:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_online_daemon.ps1
```

## Status

```powershell
python scripts\online_status.py
```

## API

```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

```text
GET /health
GET /forecast/latest
GET /online/status
GET /online/source-health
GET /online/performance?hours=24
GET /online/history
```

## Dashboard

```powershell
streamlit run src\dashboard\app.py
```

## Hackathon submission mode remains separate

V5 does **not** use continuous retraining to alter the hidden-test replay. Keep using the validated V3/backtest pipeline for the official competition submission. Production V5 is a demonstration of how the same architecture can operate after deployment with real outcomes and real APIs.

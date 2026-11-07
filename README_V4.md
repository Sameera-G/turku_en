# TE-BalanceAI V4 - live Fingrid + continuous learning

V4 keeps the V3 hackathon model and adds the real operational loop requested by the team:

`live data -> evaluate previous forecasts -> learn from last 24h -> adjust weights/calibration -> optional challenger retrain -> next 10h forecast -> repeat`

The hourly loop updates ensemble/calibration parameters from the last 24 hours. A full LightGBM/CatBoost
challenger retrain defaults to once per 24 hours and is promoted only when it beats the current champion.
This separation gives continuous learning without blindly overfitting one unusual day.

## 1. Install / update

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add:

```text
FINGRID_API_KEY=...
ENTSOE_API_TOKEN=...   # optional but recommended
```

## 2. Verify

```powershell
python -m pytest -q tests\test_online_learning.py tests\test_v3_constraints.py tests\test_v2_constraints.py tests\test_feature_legality.py
```

## 3. Run one real cycle

```powershell
python .\scripts\run_online_cycle.py
```

## 4. Inspect status

```powershell
python .\scripts\online_status.py
```

## 5. Run every hour

```powershell
python .\scripts\run_online_daemon.py --minute 55
```

At minute 55 it fetches fresh data and predicts the next full-hour origin. Example: a cycle at 13:55 produces
origin 14:00 and 40 forecasts from 14:00 through 23:45.

## 6. API / dashboard

```powershell
uvicorn src.api.app:app --reload --port 8000
streamlit run src\dashboard\app.py
```

The API automatically prefers `outputs/online/latest_forecast.csv` when live mode is running.

See `docs/V4_CONTINUOUS_LEARNING.md` for architecture, retraining policy, rollback and data-source details.

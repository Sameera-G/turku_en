# TE-BalanceAI V4 - Continuous learning production loop

V4 adds a production-only learning loop while leaving the locked hackathon backtest unchanged.

## Two learning speeds

### Every hour
1. Fetch fresh Fingrid data using the Open Data REST API.
2. Optionally fetch Finnish day-ahead spot prices from ENTSO-E.
3. Evaluate forecasts whose actual imbalance price is now settled.
4. Look back over the most recent 24 hours.
5. Re-learn LightGBM/CatBoost fusion weights for each forecast step.
6. Re-learn small residual corrections for P10/P90 calibration.
7. Re-calibrate direction and spike probabilities when labels are available.
8. Forecast the next 40 quarter-hours.
9. Save the exact feature vector used for every prediction.

This is genuine online adaptation, but it does not change the base tree parameters every hour.
That is deliberate: fully retraining on only one day can overfit an unusual weather or market regime.

### Default every 24 hours
A challenger model is trained using:
- the leakage-safe historical challenge data;
- settled feature snapshots collected by the live system.

The newest 24 online origins are held out for evaluation. The challenger is promoted only if its
weighted validation score improves and no major metric regresses beyond the configured tolerance.
The old champion is retained in `models/online/versions/`, so rollback is possible.

You can set `TE_BALANCEAI_FULL_RETRAIN_HOURS=1` if you deliberately want full retraining every hour,
but the default 24-hour cadence is safer.

## Live sources

Fingrid API access requires `FINGRID_API_KEY` in `.env`. V4 uses individual dataset requests so it
stays within the documented API model and rate limiting. The live registry includes imbalance price,
mFRR direction/activations/prices, aFRR prices, consumption, production, cross-border flows, and the
updated 15-minute consumption/wind/solar forecasts. It also records real-time surplus/deficit,
electric-boiler load, frequency, wind, hydro and nuclear production as operational context.

Fingrid does not redistribute Finnish day-ahead spot price. Set `ENTSOE_API_TOKEN` to enable the
optional ENTSO-E spot-price loader. Without live spot prices, price and direction adaptation still work,
but spike-label learning and full multi-task challenger retraining are held back.

## Run one cycle

```powershell
Copy-Item .env.example .env
# edit .env and add FINGRID_API_KEY (and preferably ENTSOE_API_TOKEN)
python .\scripts\run_online_cycle.py
```

The cycle normally forecasts the next full hour. If launched at 13:55 UTC, the forecast origin is 14:00 UTC.
This avoids pretending that data fetched after 14:00 was known at the 14:00 origin.

## Run continuously

```powershell
python .\scripts\run_online_daemon.py --minute 55
```

Each hourly cycle is launched in a fresh Python process. This isolates native LightGBM/CatBoost memory and
makes long-running operation more robust.

## Files created

- `data/raw/live/fingrid_live_long.parquet` - current live feature cache
- `data/online/te_balanceai_online.sqlite3` - predictions, actuals, evaluations, feature snapshots, model events
- `data/online/adaptive_state.json` - rolling 24h adaptive parameters
- `outputs/online/latest_forecast.csv` - latest 40-step forecast
- `outputs/online/latest_diagnostics.csv` - model disagreement, reliability, model version
- `models/online/champion.joblib` - promoted online champion (when available)
- `models/online/versions/` - immutable promoted model versions

## Safety / model-governance rules

- API keys are environment variables and are not stored in source files.
- Exact feature snapshots are saved at prediction time; past origins are not reconstructed from later forecast revisions.
- A newly trained model is a challenger until it passes the promotion gate.
- The prior champion remains available for rollback.
- Missing external sources reduce available learning signals but do not silently fabricate values.
- This online loop is not used when producing the official locked hackathon backtest CSV.

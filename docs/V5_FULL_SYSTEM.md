# V5 full-system engineering notes

## 1. Two modes

### Competition/backtest
- fixed training chronology;
- `first_usable_origin_utc <= forecast_origin_utc` enforced;
- no online parameter updates on hidden outcomes;
- official validator/scorer remain unchanged.

### Production continuous learning
- current full-hour origin;
- fresh public API snapshot;
- exact feature snapshot persisted;
- 40 quarter-hour predictions;
- actuals evaluated after settlement/publication lag;
- adaptive parameters updated from last 24h;
- full challenger retrain defaults to 24h cadence.

## 2. Source roles

### Fingrid — primary Finnish operations
Fingrid provides the most important current Finnish state and forecast feeds. The client is rate-limited and defensive; a single failed dataset is logged but does not stop the whole cycle.

### ENTSO-E — market/redundancy/structural context
- A44 day-ahead price
- A65 load forecast/actual load
- A69 wind/solar forecast
- A75 generation per type
- A72 reservoir filling
- A80 generation unavailability
- A78 transmission unavailability

These feeds are secondary to Fingrid for overlapping real-time Finnish measurements, but add valuable public market/outage/hydro context.

### Weather
Open-Meteo is the default JSON weather feed. FMI HARMONIE is used as a Finnish fallback. Weather values are attached to the API fetch time, not retroactively assigned to previous forecast origins.

## 3. Why raw snapshots and feature_json both exist

APIs revise forecasts. If the system reconstructed a forecast from today's latest API data, yesterday's model training could accidentally see a newer weather/load forecast than yesterday actually had. V5 prevents this in two ways:

1. every cycle can archive the merged raw live snapshot under `data/online/raw_snapshots/`;
2. every prediction stores the exact feature vector in SQLite `predictions.feature_json`.

The online retrainer learns from the stored feature vector, not from a later reconstruction.

## 4. Learning speeds

### Fast layer — every hour
For each of 40 horizons, recent settled outcomes can change:
- LightGBM/CatBoost fusion weight;
- P10/P90 residual correction;
- UP/DOWN/NONE calibration;
- high/low exceptional-price calibration.

### Slow layer — default every 24h
The challenger trains on leakage-safe history + online feature snapshots. New live-only variables are admitted only when their online non-missing coverage is above the configured threshold. Promotion requires composite improvement and no unacceptable regression in MAE, pinball, direction Brier or spike Brier.

## 5. Reliability and graceful degradation

A live cycle should still produce the mandatory forecast when a secondary API fails. Source health is logged and exposed to the UI/API. Forecast reliability is reduced when the current multi-source snapshot is degraded. The dashboard must tell the operator that a source is stale/unavailable instead of silently hiding the failure.

## 6. 14:00 Finland-time handling

There is no separate once-per-day forecasting regime. The system always forecasts hourly. The normal cycle near 14:00 Europe/Helsinki is simply a high-information cycle because next-day day-ahead prices can become available. Since V5 fetches day-ahead prices every hour, the newly available curve is incorporated automatically.

## 7. Production caution

V5 is decision support. Public-data forecasts should trigger review of trading/production/capacity positions; they should not autonomously execute trades or dispatch Turku Energia assets without internal position, marginal-cost, ramp, heat, contract, ownership and risk-limit data.

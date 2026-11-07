# TE-BalanceAI V8 - market-rule spread experts + continuous learning

V8 keeps the official leakage-safe Turku Energia competition pipeline, but changes the P50 centre forecast to use the market's price structure when a legal day-ahead FI spot price is available.

## What changed

Instead of only learning the full imbalance price directly, V8 learns:

`imbalance_price = spot_price + imbalance_spread`

It uses several models:

- validated V3 direct-price LightGBM/CatBoost ensemble;
- global LightGBM spread model (L1/MAE objective);
- global CatBoost spread model (MAE objective);
- DOWN-regulation LightGBM spread expert;
- UP-regulation LightGBM spread expert;
- NONE expert = zero spread, following the challenge's stated pricing logic;
- the existing calibrated UP/DOWN/NONE probabilities gate the direction experts;
- each of the 40 horizons has its own validation-fitted fusion parameters and robust P50 bias correction.

P10/P90, exceptional-price probabilities and direction probabilities remain on the proven V3 probabilistic heads so the P50 upgrade does not destroy calibration.

## Verified public-validation result

The organizer's exact `validate_submission.py` and `score.py` were used.

- V3 MAE: 47.2047 EUR/MWh
- V8 MAE: 43.4074 EUR/MWh
- V8 P10/P90 pinball: 12.4193
- V8 coverage: 0.7975
- V8 spike Brier: 0.0973
- V8 direction Brier: 0.4214

The organizer public-validation baseline MAE shown in the briefing is 66.20 EUR/MWh. V8 therefore reduces MAE by about 34.4% relative to that baseline and about 8.0% relative to V3.

A chronological robustness check (fit V8 fusion on earlier validation origins, score on later origins) also improved MAE over V3 in all tested splits; the observed relative gains ranged from about 1.5% to 11.7% depending on the split. This is not an official metric, but it is evidence that the V8 gain is not only a same-window fit artifact.

## Run V8 on top of an existing V7/V6 project

```powershell
cd "C:\Users\samee\Downloads\turkuen"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\scripts\run_v8.ps1
```

Then verify:

```powershell
python scripts\compare_v3_v8.py
python scripts\accuracy_report.py
python scripts\smoke_test_self_learning_v8.py --origins 10
```

The final competition-format candidate is:

`outputs\v8\forecast_backtest_v8.csv`

It must report 673 origins, 26,920 rows, 40 rows per origin and 0 validator errors.

## Live V8

Run one live cycle:

```powershell
python scripts\run_online_cycle.py --no-retrain
```

Then start continuous operation:

```powershell
python scripts\run_online_daemon.py --delay-seconds 75
```

V8 prefers live spot sources in this order conceptually:

1. licensed/manual spot curve if provided by the user;
2. ENTSO-E live feed when a token is configured;
3. optional no-key Finnish public spot fallback for hackathon/demo use.

The public fallback is explicitly best-effort and should be replaced by ENTSO-E or licensed Nord Pool in a production trading environment.

If no legal live spot price is available for a delivery interval, V8 automatically falls back to the validated direct-price model rather than fabricating a price.

## Additional live Fingrid context

V8 adds public operational context such as:

- updated 15-minute total production forecast;
- real-time/cumulative surplus-deficit;
- district-heating and industrial CHP;
- BESS discharge;
- electricity-shortage status;
- north-south internal transfer;
- wind forecast capacity and derived capacity-factor signals;
- the previously integrated mFRR/aFRR, consumption/production, cross-border flows, frequency, electric boilers, wind/hydro/nuclear production.

These extra live signals are saved in the exact prediction-time feature snapshot. They are admitted into a future challenger only after enough coverage exists; they are never reconstructed later using revised data.

## Weather

Live mode continues using Open-Meteo regional forecasts plus FMI forecasts/observations. For a future competition experiment, the leakage-safe weather-history source to use is Open-Meteo Single Runs, because it preserves the model run and initialization time. Do not silently use a hindsight-stitched weather series as though it had been available at forecast origin.

## Self-learning proof

`python scripts\smoke_test_self_learning_v8.py --origins 10`

runs an isolated public-validation simulation:

forecast -> save exact features -> insert later actuals -> settle predictions -> compute errors -> learn 24h/7d/14d adaptive state.

It does not touch the production SQLite database. The smoke test must end with `works True` and non-zero evaluations/adapted horizons.

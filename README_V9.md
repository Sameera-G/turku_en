# TE-BalanceAI V9 - ENTSO-E MCP + live conformal calibration + explainability

V9 is a production/live-layer upgrade over the validated V8 model. It deliberately does **not** change the locked Turku Energia historical competition backtest with information reconstructed from today's external archives.

## What V9 changes

1. **ENTSO-E MCP / JSON live ingestion** from `https://entsoemcp.com` (no token required), with fail-safe fallback. High-value live candidates include FI/neighbor day-ahead prices and load forecasts, FI generation forecast, neighbor imbalance prices, scheduled exchanges, day-ahead NTC, Nordic aFRR capacity-market price/volume and FI generation outages. Exact snapshots are archived at prediction time.
2. **Fingrid 209** power-system state and **Fingrid 194** real-time net import/export.
3. **Pooled live conformal calibration** starts after 8 settled outcomes. This fixes the V8 issue where 12 total evaluations could still leave 0 adapted horizons because each of 40 steps needed 8 observations.
4. **Reliability decomposition**: model/ensemble reliability + empirical live performance/calibration + source health + drift penalty.
5. **Explainability dashboard**: global P50/direction feature-family importance, plus separate live plots for production plan, wind, solar, consumption, mFRR, aFRR, frequency, cross-border, CHP, battery, shortage/system state, nuclear, hydro, weather and ENTSO-E MCP signals.
6. Streamlit deprecated `use_container_width` calls were replaced and daemon labels now say V9.

## Important accuracy statement

V9's locked official-validation forecast remains the V8 forecast because the new MCP features are live-only until enough exact prediction-time snapshots exist for a chronological challenger. The organizer-style recheck remains:

- P50 MAE: **43.4074 EUR/MWh**
- P10/P90 mean pinball: **12.4193**
- P10-P90 coverage: **79.75%**
- exceptional-price Brier: **0.0973**
- direction Brier: **0.4214**

Do not claim that MCP has already reduced the official validation MAE. V9 is designed to improve future live performance safely, then promote a challenger only after chronological evidence.

## Upgrade an existing V8 project

Stop Streamlit and the daemon, then extract the V9 replacement ZIP over the project root.

```powershell
cd "C:\Users\samee\Downloads\turkuen"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
```

Test the ENTSO-E MCP source:

```powershell
python scripts\test_entsoe_mcp.py
```

A partial pass is acceptable. V9 continues safely if the third-party MCP cache is unavailable.

Run tests:

```powershell
python -m pytest -q tests\test_v9_system.py tests\test_v8_system.py tests\test_v7_performance.py tests\test_v6_system.py tests\test_online_learning.py tests\test_v5_system.py tests\test_v3_constraints.py tests\test_v2_constraints.py tests\test_feature_legality.py
```

Expected on the packaged code: `25 passed`.

Run one live cycle:

```powershell
python scripts\run_online_cycle.py --no-retrain
python scripts\online_status.py
```

With at least 8 settled live outcomes, look for:

```text
pooled_live_calibration:
  samples: >= 8
  conformal_expand_eur: > 0   # when recent P10-P90 intervals are under-covering
```

Then launch:

```powershell
python -m streamlit run .\src\dashboard\app.py
```

and in another terminal:

```powershell
python scripts\run_online_daemon.py --delay-seconds 75
```

## How live coverage correction works

For each settled forecast, V9 computes a nonconformity miss:

```text
max(P10 - actual, actual - P90, 0)
```

It takes a finite-sample quantile targeted at 80% coverage and gradually expands future P10/P90 intervals. It does **not** change P50, so fixing live coverage does not directly worsen MAE. Per-horizon calibration still takes over later when each step has enough evidence.

## ENTSO-E MCP caveats

`entsoemcp.com` is a third-party service and is not ENTSO-E. V9 treats it as best-effort and records source health. Non-free-reuse datasets (including day-ahead prices, load, generation, forecasts, outages and scheduled exchanges) must not be redistributed casually; keep them local and review the source terms before commercial reuse. For production trading, the preferred long-term source remains direct ENTSO-E access and/or licensed Nord Pool where applicable.

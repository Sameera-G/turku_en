# V6 full-system design notes

## Design goal

TE-BalanceAI V6 is an hourly probabilistic imbalance-price forecasting system with a strict split between competition replay and live continuous learning.

## Learning architecture

### Fast window: 24 hours
Reacts to abrupt changes in market regime. It can alter per-horizon LightGBM/CatBoost fusion, residual bias, interval correction and probability calibration.

### Regime window: 7 days
Reduces overreaction to a single unusual day and captures weekday/weekend/weather regimes.

### Stable window: 14 days
Provides a stable reference for drift detection and longer-term bias.

The configurable default blend is 0.50 / 0.30 / 0.20. These are operational defaults, not scientific constants; validation can tune them.

## Horizon-specific residual correction

For each forecast step h, V6 estimates the median settled residual:

`bias_h = median(actual_h - previous_p50_h)`

The final adaptive centre uses a confidence-weighted correction. P10/P90 residual corrections are learned separately so the interval is not mechanically shifted twice.

## Drift detection

V6 currently flags drift when either:

- 24-hour MAE is materially worse than the 14-day MAE (default ratio 1.35), or
- 24-hour P10-P90 coverage deviates strongly from 0.80 (default absolute deviation 0.12).

Drift lowers reliability and may request an early challenger retrain subject to a cooldown.

## Reserve Stress Index

The signed, bounded `reserve_stress_index` combines legally available signals such as:

- recent mFRR net activation,
- recent balancing direction persistence,
- real-time system surplus/deficit,
- system-frequency deviation.

Positive values represent upward/shortage pressure and negative values represent downward/surplus pressure. The individual raw inputs remain available to the ML model; the index is only a transparent summary feature.

## Forecast-error regime features

V6 adds two forms of forecast-error context:

1. FMI weather observations versus current regional operational forecasts where compatible observation/forecast fields exist.
2. Direct balancing-relevant proxies such as current Finnish wind-generation actual versus current wind forecast, and consumption actual versus current consumption forecast.

This avoids treating weather as a static input only; the system also learns whether the current forecasting environment is systematically missing conditions.

## Digital twin

Each cycle writes a compact state file containing:

- market pressure,
- reserve stress,
- cross-border stress,
- live system stress,
- wind/load forecast-error proxies,
- 24 h / 7 d / 14 d error levels,
- drift status,
- minimum forecast reliability,
- peak direction/spike risks,
- data-source health,
- whether the cycle is in the day-ahead refresh window.

This state is an explainability/operations object, not a physical power-system simulator.

## Champion/challenger retraining

The adaptive layer runs every hourly cycle. Full model retraining is slower and guarded:

1. Train a challenger on bounded historical leakage-safe data plus exact saved live snapshots.
2. Chronologically hold out recent online origins.
3. Compare MAE, pinball, direction Brier and spike Brier.
4. Reject the challenger if weighted gain is insufficient or any protected metric regresses too far.
5. Promote only successful challengers; retain version history for rollback.

## Nominal-origin live grace window

The production daemon intentionally fetches data shortly after a full hour. V6 associates that current live snapshot with the nominal hourly production batch while preserving the real fetched-at timestamp in context/source-health logs.

This convention is **production-only**. It must not be used to bypass the organizer's `first_usable_origin_utc <= forecast_origin_utc` rule in the official hackathon replay.

## Public-industry inspiration

Public Helen/Silo AI material describes adaptive forecasting using historical forecasts, realized outcomes, weather forecasts and time variables, while Helen also publicly describes optimization/flexibility platforms and digital-twin approaches. V6 borrows those architectural ideas only; it does not claim to reproduce Helen's proprietary imbalance-price model.

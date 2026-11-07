# Score summary

- Scorer version: 0.1.1
- Schema version: 1.0
- Submission: forecast_backtest.csv
- Actuals: actuals_validation.csv
- Spike margin: 50.00 EUR/MWh
- Rows in submission: 26880
- Rows scored (price metrics): 26700; skipped (no actual): 180
- Rows scored (exceptional prices): 26700
- Rows scored (direction): 26700

## Overall results

| Metric | Value | Use |
|---|---|---|
| MAE (p50) | 47.2047 | Level 1: accuracy |
| Pinball loss p10 | 6.2822 | |
| Pinball loss p90 | 18.6300 | |
| Mean pinball loss p10 and p90 | 12.4561 | Level 1: calibration |
| P10-P90 coverage | 0.7963 | supporting metric (target about 0.80) |
| Brier p_spike_high | 0.1043 | |
| Brier p_spike_low | 0.0903 | |
| Mean Brier, exceptional prices | 0.0973 | Level 1: exceptional prices |
| Three-class Brier, direction | 0.4214 | Level 1: direction |
| Pinball loss p50 | 23.6023 | for information |
| Observed share of high / low quarter-hours | 0.1213 / 0.1296 | for information |
| Baseline MAE (p50) | n/a | for information |
| MAE difference vs baseline (team - baseline) | n/a | for information |
| MAE ratio vs baseline (team / baseline) | n/a | for information |

## Results by forecast horizon

| Hour | Steps | Rows | MAE | Pinball p10/p90 mean | Coverage | Brier spikes | Brier direction |
|---|---|---|---|---|---|---|---|
| 1 | 1-4 | 2688 | 47.0566 | 12.2959 | 0.7976 | 0.0965 | 0.4199 |
| 2 | 5-8 | 2684 | 47.1481 | 12.3437 | 0.7973 | 0.0962 | 0.4207 |
| 3 | 9-12 | 2680 | 47.1517 | 12.3981 | 0.7970 | 0.0963 | 0.4208 |
| 4 | 13-16 | 2676 | 47.0705 | 12.4404 | 0.7967 | 0.0969 | 0.4200 |
| 5 | 17-20 | 2672 | 47.1144 | 12.4648 | 0.7964 | 0.0973 | 0.4205 |
| 6 | 21-24 | 2668 | 47.2058 | 12.4676 | 0.7961 | 0.0978 | 0.4213 |
| 7 | 25-28 | 2664 | 47.2388 | 12.4783 | 0.7958 | 0.0978 | 0.4227 |
| 8 | 29-32 | 2660 | 47.3476 | 12.5270 | 0.7955 | 0.0977 | 0.4229 |
| 9 | 33-36 | 2656 | 47.3438 | 12.5464 | 0.7952 | 0.0982 | 0.4232 |
| 10 | 37-40 | 2652 | 47.3736 | 12.6027 | 0.7949 | 0.0984 | 0.4222 |

Lower is better for MAE, pinball loss and Brier scores. Points are calculated with `score.py --points` after all submissions have been scored.

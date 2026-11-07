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
| MAE (p50) | 43.4074 | Level 1: accuracy |
| Pinball loss p10 | 6.2822 | |
| Pinball loss p90 | 18.5564 | |
| Mean pinball loss p10 and p90 | 12.4193 | Level 1: calibration |
| P10-P90 coverage | 0.7975 | supporting metric (target about 0.80) |
| Brier p_spike_high | 0.1043 | |
| Brier p_spike_low | 0.0903 | |
| Mean Brier, exceptional prices | 0.0973 | Level 1: exceptional prices |
| Three-class Brier, direction | 0.4214 | Level 1: direction |
| Pinball loss p50 | 21.7037 | for information |
| Observed share of high / low quarter-hours | 0.1213 / 0.1296 | for information |
| Baseline MAE (p50) | n/a | for information |
| MAE difference vs baseline (team - baseline) | n/a | for information |
| MAE ratio vs baseline (team / baseline) | n/a | for information |

## Results by forecast horizon

| Hour | Steps | Rows | MAE | Pinball p10/p90 mean | Coverage | Brier spikes | Brier direction |
|---|---|---|---|---|---|---|---|
| 1 | 1-4 | 2688 | 42.8411 | 12.2586 | 0.7987 | 0.0965 | 0.4199 |
| 2 | 5-8 | 2684 | 43.1553 | 12.3100 | 0.7981 | 0.0962 | 0.4207 |
| 3 | 9-12 | 2680 | 43.2112 | 12.3632 | 0.7981 | 0.0963 | 0.4208 |
| 4 | 13-16 | 2676 | 43.2023 | 12.4087 | 0.7982 | 0.0969 | 0.4200 |
| 5 | 17-20 | 2672 | 43.2714 | 12.4320 | 0.7979 | 0.0973 | 0.4205 |
| 6 | 21-24 | 2668 | 43.4847 | 12.4270 | 0.7976 | 0.0978 | 0.4213 |
| 7 | 25-28 | 2664 | 43.5905 | 12.4364 | 0.7973 | 0.0978 | 0.4227 |
| 8 | 29-32 | 2660 | 43.6934 | 12.4905 | 0.7970 | 0.0977 | 0.4229 |
| 9 | 33-36 | 2656 | 43.8175 | 12.5071 | 0.7959 | 0.0982 | 0.4232 |
| 10 | 37-40 | 2652 | 43.8194 | 12.5630 | 0.7964 | 0.0984 | 0.4222 |

Lower is better for MAE, pinball loss and Brier scores. Points are calculated with `score.py --points` after all submissions have been scored.

# Score summary

- Scorer version: 0.1.1
- Schema version: 1.0
- Submission: forecast_backtest_v3.csv
- Actuals: actuals_validation.csv
- Spike margin: 50.00 EUR/MWh
- Rows in submission: 26920
- Rows scored (price metrics): 0; skipped (no actual): 26920
- Rows scored (exceptional prices): 0
- Rows scored (direction): 0

## Overall results

| Metric | Value | Use |
|---|---|---|
| MAE (p50) | n/a | Level 1: accuracy |
| Pinball loss p10 | n/a | |
| Pinball loss p90 | n/a | |
| Mean pinball loss p10 and p90 | n/a | Level 1: calibration |
| P10-P90 coverage | n/a | supporting metric (target about 0.80) |
| Brier p_spike_high | n/a | |
| Brier p_spike_low | n/a | |
| Mean Brier, exceptional prices | n/a | Level 1: exceptional prices |
| Three-class Brier, direction | n/a | Level 1: direction |
| Pinball loss p50 | n/a | for information |
| Observed share of high / low quarter-hours | n/a / n/a | for information |
| Baseline MAE (p50) | n/a | for information |
| MAE difference vs baseline (team - baseline) | n/a | for information |
| MAE ratio vs baseline (team / baseline) | n/a | for information |

## Results by forecast horizon

| Hour | Steps | Rows | MAE | Pinball p10/p90 mean | Coverage | Brier spikes | Brier direction |
|---|---|---|---|---|---|---|---|
| 1 | 1-4 | 0 | n/a | n/a | n/a | n/a | n/a |
| 2 | 5-8 | 0 | n/a | n/a | n/a | n/a | n/a |
| 3 | 9-12 | 0 | n/a | n/a | n/a | n/a | n/a |
| 4 | 13-16 | 0 | n/a | n/a | n/a | n/a | n/a |
| 5 | 17-20 | 0 | n/a | n/a | n/a | n/a | n/a |
| 6 | 21-24 | 0 | n/a | n/a | n/a | n/a | n/a |
| 7 | 25-28 | 0 | n/a | n/a | n/a | n/a | n/a |
| 8 | 29-32 | 0 | n/a | n/a | n/a | n/a | n/a |
| 9 | 33-36 | 0 | n/a | n/a | n/a | n/a | n/a |
| 10 | 37-40 | 0 | n/a | n/a | n/a | n/a | n/a |

Lower is better for MAE, pinball loss and Brier scores. Points are calculated with `score.py --points` after all submissions have been scored.

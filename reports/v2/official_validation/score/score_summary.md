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
| MAE (p50) | 48.6485 | Level 1: accuracy |
| Pinball loss p10 | 6.3749 | |
| Pinball loss p90 | 18.9307 | |
| Mean pinball loss p10 and p90 | 12.6528 | Level 1: calibration |
| P10-P90 coverage | 0.7966 | supporting metric (target about 0.80) |
| Brier p_spike_high | 0.1038 | |
| Brier p_spike_low | 0.0867 | |
| Mean Brier, exceptional prices | 0.0953 | Level 1: exceptional prices |
| Three-class Brier, direction | 0.4185 | Level 1: direction |
| Pinball loss p50 | 24.3242 | for information |
| Observed share of high / low quarter-hours | 0.1213 / 0.1296 | for information |
| Baseline MAE (p50) | n/a | for information |
| MAE difference vs baseline (team - baseline) | n/a | for information |
| MAE ratio vs baseline (team / baseline) | n/a | for information |

## Results by forecast horizon

| Hour | Steps | Rows | MAE | Pinball p10/p90 mean | Coverage | Brier spikes | Brier direction |
|---|---|---|---|---|---|---|---|
| 1 | 1-4 | 2688 | 48.6895 | 12.5009 | 0.7976 | 0.0947 | 0.4159 |
| 2 | 5-8 | 2684 | 48.7061 | 12.5442 | 0.7973 | 0.0942 | 0.4179 |
| 3 | 9-12 | 2680 | 48.6132 | 12.5822 | 0.7970 | 0.0946 | 0.4173 |
| 4 | 13-16 | 2676 | 48.4858 | 12.6030 | 0.7967 | 0.0947 | 0.4162 |
| 5 | 17-20 | 2672 | 48.5083 | 12.6488 | 0.7964 | 0.0950 | 0.4175 |
| 6 | 21-24 | 2668 | 48.5958 | 12.6804 | 0.7961 | 0.0957 | 0.4190 |
| 7 | 25-28 | 2664 | 48.6470 | 12.6996 | 0.7958 | 0.0957 | 0.4201 |
| 8 | 29-32 | 2660 | 48.7831 | 12.7338 | 0.7955 | 0.0958 | 0.4200 |
| 9 | 33-36 | 2656 | 48.7543 | 12.7600 | 0.7959 | 0.0960 | 0.4208 |
| 10 | 37-40 | 2652 | 48.7031 | 12.7790 | 0.7975 | 0.0960 | 0.4206 |

Lower is better for MAE, pinball loss and Brier scores. Points are calculated with `score.py --points` after all submissions have been scored.

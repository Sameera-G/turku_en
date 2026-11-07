# V7 design: accuracy visibility, 15-minute verification and daily learning

## Why this exists

The Turku Energia challenge evaluates the model with MAE for P50, pinball loss for P10/P90, Brier scores for exceptional-price probabilities and balancing direction, while Level 2 asks for hindsight performance and the ability to move the review time into the past.

V7 exposes the same model-quality concepts directly in the operating UI.

## Accuracy layers

### A. Official validation evidence

Loaded from the locked organizer scoring output. This remains fixed and reproducible.

### B. 40-step / 15-minute horizon diagnostics

For each forecast step 1..40, V7 reports MAE and P10-P90 coverage. This identifies whether short horizons or long horizons are responsible for most error.

### C. Live settled evidence

When a forecast delivery period has passed and the actual Fingrid imbalance outcome is available, V7 stores an evaluation. The dashboard summarizes these over 24 hours, 7 days and 14 days.

## Self-learning cadence

### Every hour

- ingest new data;
- settle old forecasts;
- recompute 24h, 7d and 14d performance windows;
- adapt horizon-specific fusion weights, bias and probability/interval calibration;
- lower reliability when drift is detected;
- forecast the next 40 quarter-hours.

### Once per day

Default: 14:00 Europe/Helsinki, configurable.

- freeze a daily performance checkpoint;
- compare 24h vs 7d vs 14d behaviour;
- write a daily accuracy report;
- if enough settled observations exist, conditionally train a challenger;
- retain the current champion unless the challenger is measurably better.

## Missing live spot prices

Price, interval and direction adaptation can still run without a live ENTSO-E/Nord Pool spot feed. Full multi-task challenger retraining may remain blocked because the exceptional-price labels require spot price. The dashboard reports this explicitly rather than inventing spike labels.

## Dashboard reasoning

The live tab shows every 15-minute delivery period, not an hourly aggregate. The historical-replay tab overlays actual outcomes on the original forecast and therefore directly demonstrates hindsight performance. The accuracy tab separates official metrics from intuitive ±EUR hit rates so the demo does not misrepresent a non-official diagnostic as the challenge score.

# V10 production-refit, reliability and explainability design

## Why reliability fell in V9

V9 combined trained-model reliability with early live empirical performance and a drift penalty. With only 12 settled points and about 33% raw interval coverage, the empirical calibration term and drift penalty could lower a roughly 69% model-confidence score into the high-50s. That was a confidence warning, not evidence that P50 accuracy suddenly became 58%.

V10 makes this sample-aware:

- coverage-drift trigger requires at least 24 settled samples;
- empirical evidence weight ramps from 0 to 30% over roughly 48 settled observations;
- coverage used inside reliability is shrunk toward the historical 80% target with a 24-sample prior;
- raw live coverage remains visible and is never hidden;
- asymmetric conformal interval correction can begin after 8 total settled observations.

## MAE strategy

V10 does not add weak learners for diversity alone. Experiments found XGBoost absolute-error (~47.21 public-validation MAE), XGBoost L2 (~46.34), stronger spread LightGBM (~44.02) and a stronger direction LightGBM (Brier ~0.4465) did not beat the existing V8 locked evidence (43.41 MAE / 0.4214 direction Brier).

The robust gain came from recency-aware refitting. A strict chronological split trained the same architecture on development + earlier validation, calibrated on the next block and evaluated on the final unseen block. The direct expert improved from about 32.25 to 30.91 MAE. A neutral 50/50 direct/spread production centre then achieved about 30.82 MAE versus 34.16 for the fair old-V8 pipeline on the same unseen block.

The final V10 production model is therefore refit through 4 Oct and starts with a neutral direct/spread centre. Settled live errors then move the blend and bias.

## New live candidate signals

V10 extends the exact-vintage live feature store with:

- FI intraday auction price;
- FI wind and solar forecasts from ENTSO-E MCP;
- FI imbalance volume;
- realized activated aFRR/mFRR energy prices;
- forecast revisions since the previous hourly snapshot for spot, intraday price, load, wind, solar, production and weather;
- existing V9 MCP day-ahead prices, neighboring load/imbalance, schedules, NTC, aFRR capacity and outages;
- Fingrid production plan, system state, net import/export, frequency, reserves, CHP, battery, shortage status, nuclear, hydro and cross-border flows;
- FMI/Open-Meteo forecast and observation context.

These live-only signals are not silently injected into the historical challenge backtest. They enter a challenger only after enough prediction-time snapshots and settled labels exist.

## Explainability

V10 exposes four layers:

1. global feature-family importance from the actual trained tree models;
2. signed LightGBM evidence for a selected live 15-minute step (direct price and spread expert separately);
3. separate time-series graphs for each contributing source family;
4. operational explanation fields: direct vs spread prediction, model disagreement, source health, reliability components, forecast revisions and top transparent drivers.

Signed tree contributions are associative model evidence, not causal proof, and the dashboard says so explicitly.

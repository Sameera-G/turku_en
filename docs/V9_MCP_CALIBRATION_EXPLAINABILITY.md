# V9 design: MCP market context, live calibration and explainability

## Objective

Improve the live production system without weakening the challenge's anti-leakage guarantees. V9 therefore separates:

- **locked competition evidence**: V8 organizer-style validation remains 43.4074 EUR/MWh MAE;
- **live production learning**: external sources are captured as exact vintages and may influence future challengers only after settled outcomes exist.

## New live sources

### ENTSO-E MCP / JSON cache

V9 queries the public `entsoemcp.com` series interface in fail-safe mode. Candidate features include:

- FI and neighboring day-ahead prices;
- FI and neighboring load forecasts;
- FI generation forecast;
- SE1/SE3/EE/NO4 imbalance prices;
- scheduled FI-border commercial exchanges;
- day-ahead FI-border NTC;
- Nordic aFRR capacity-market clearing price and procured MW for FI;
- published FI generation-unit outages.

The current V8 spread model can use an MCP FI day-ahead curve immediately because it is mapped to the canonical `fi_spot_price`. Other MCP signals are saved as challenger features, not injected by unvalidated hand-written coefficients.

### Fingrid additions

- dataset 209: power-system state (1-5), 3-minute feed;
- dataset 194: net import/export of electricity, 3-minute feed.

## Pooled conformal live calibration

V8 required 8 examples for each individual step before a step could adapt. With 40 horizons this delayed correction. V9 learns a global production-only conformal expansion after 8 total settled outcomes.

For actual `y` and interval `[p10,p90]`:

`score = max(p10-y, y-p90, 0)`.

The target finite-sample quantile is derived from the configured 80% target. Expansion is gradually blended in until 48 samples; the maximum expansion is capped. When enough step-level data exists, the existing step-specific correction remains active.

This changes uncertainty intervals only. P50 remains governed by the price/spread ensemble and adaptive bias/fusion logic.

## Reliability

V9 exposes:

- `reliability_model`: interval width, model disagreement, direction entropy, missingness and OOD envelope from the trained model;
- `reliability_empirical`: recent MAE and observed interval coverage, confidence-weighted by live sample count;
- `reliability_calibration`: closeness of observed live coverage to 80%;
- `reliability_source`: live-source health;
- final `reliability`: a bounded blend with a drift penalty.

Reliability is not accuracy. It is a warning/triage score.

## Explainability

The dashboard's Signals & explainability tab provides:

1. normalized global feature-family importance for the P50 ensemble;
2. normalized global importance for the UP/DOWN/NONE classifier;
3. individual live time-series charts grouped by source and split by unit;
4. row-level forecast decomposition: P50, interval, direction probabilities, disagreement, reliability and transparent top-driver text.

Feature importance is association, not causal proof. Live-only sources that have not yet entered a promoted challenger correctly show zero trained-model importance even though their raw signal charts are visible.

## Model experiments rejected

V9 does not add a model merely to increase model count. A second CatBoost direction ensemble and an additional histogram-gradient spread learner were tested during development; neither showed sufficiently robust chronological improvement to justify production inclusion. The safer path is to collect high-quality live vintages, calibrate uncertainty, and let champion/challenger evaluation decide whether new features improve real MAE.

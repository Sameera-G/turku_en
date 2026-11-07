# TE-BalanceAI V11: active-model transparency and adaptive challenger tournament

## Why the dashboard can still show 43.41 after installing V10/V11

The 43.4074 EUR/MWh value is the frozen organizer-style public-validation MAE. It is intentionally not rewritten after a production model is refit using labels through 2026-10-04. The V10 refit has separate chronological holdout evidence of about 30.8181 EUR/MWh on 6,720 later unseen rows. V11 shows both values separately.

A second issue can make a replacement appear to have no effect: the online runtime prefers `models/online/champion.joblib` whenever it exists. V11 exposes the exact active path/version in `scripts/online_status.py` and the Data Health dashboard. `scripts/activate_v10_refit.py` safely backs up an older champion and activates the packaged V10 refit without deleting online history or adaptive state.

## What the V11 model search found

The remaining error is strongly regime-dependent. UP-regulation periods are much harder than DOWN periods, and an oracle direction gate would lower price error substantially. However, new binary-UP classifiers, extra XGBoost/HistGradientBoosting models, horizon specialists, recency weighting, residual stacking and upper-quantile tilts did not improve consistently across chronological periods. V11 therefore does not add weak ensemble members merely to increase model count.

## V11 tournament

Once at least 48 settled forecast origins exist, the daily learner uses three chronological partitions:

1. older settled origins: training;
2. newer origins: calibration;
3. newest origins: untouched promotion evaluation.

It compares the current champion against a standard market-structured challenger and an optional stronger MAE-focused P50 LightGBM/CatBoost challenger. Promotion requires:

- improvement in P50 MAE itself;
- improvement in the challenge-style automatic-metric composite;
- no unacceptable regression in pinball, direction Brier or spike Brier.

This prevents calibration data from also serving as promotion evidence.

## New live features

Exact prediction-time MCP, forecast-revision, weather and extra Fingrid features are archived at each forecast origin. They enter a challenger only after sufficient non-missing prediction-time coverage. Current archives are never backfilled into the locked hackathon validation as though they had been historically available.

## Recommended operating sequence

Run `python scripts/online_status.py` immediately after installing V11. If the active model is an old online champion, and not `10.1-production-refit-robust-fusion` or a later `v11_*` champion, run `python scripts/activate_v10_refit.py`. Then run one online cycle and check status again.

Do not expect old stored live errors to change: they belong to forecasts that were already issued. Only future forecasts can demonstrate a lower live MAE.

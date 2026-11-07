# TE-BalanceAI V10 - production refit, robust fusion, forecast revisions and sample-aware reliability

V10 is a live/production upgrade over the organizer-validated V8/V9 system. It keeps the locked competition evidence unchanged while improving the post-cutoff production model using newer legal labels and safer online calibration.

## What changed

1. **Production P50 refit on 120,660 labeled rows.** The direct LightGBM/CatBoost P50 learners and the global/UP/DOWN spread learners are refit using the original development set plus all public-validation labels through 4 Oct 2026. P10/P90, direction and spike heads remain the independently validated V8 heads.
2. **Robust 50/50 centre fusion instead of stale horizon bias.** A strict chronological test showed that reusing the old final P50 step calibration after refitting could overfit market regimes. V10 keeps the market-structure direction/gating logic but starts from a neutral 50/50 direct-vs-spread centre and zero static P50 intercept. The online learner then moves the weight using settled outcomes.
3. **Chronological evidence.** On a final unseen validation quarter, a fair old-V8 pipeline scored 34.16 EUR/MWh MAE while the V10 robust refit centre scored 30.82 EUR/MWh, a 9.77% reduction. This is not the official competition score; it is a chronological production-refit experiment.
4. **Sample-aware reliability.** A tiny live sample no longer overwhelms historical model evidence. Coverage drift does not trigger before 24 settled samples; live evidence weight ramps gradually, and coverage is shrunk toward the 80% historical target for reliability reporting while raw coverage remains visible.
5. **Asymmetric conformal calibration.** Lower-tail and upper-tail misses are calibrated separately so live P10/P90 can widen on the side that is actually failing.
6. **Forecast revisions.** V10 stores current-vs-previous-hour revisions for wind, load, production, spot/intraday price and weather forecasts. Revision features are timestamped at the new fetch time and are eligible only for future challenger learning.
7. **More ENTSO-E MCP candidates.** In addition to V9 series, V10 attempts FI intraday auction price, FI wind/solar forecast, FI imbalance volume and realized activated aFRR/mFRR energy prices. These are live-vintage features only.
8. **Stronger explainability.** Dashboard shows global feature-family importance, separate raw source graphs, reliability decomposition, and signed LightGBM evidence for a selected 15-minute forecast step.
9. **Alternative-model benchmark.** XGBoost, histogram-gradient boosting, a stronger spread LightGBM and a stronger direction LightGBM were tested. They did not consistently beat the market-structured champion, so they are not added merely for model count.

## Accuracy: what can and cannot be claimed

The locked organizer-style public-validation evidence remains:

- P50 MAE: **43.4074 EUR/MWh**
- P10/P90 mean pinball: **12.4193**
- P10-P90 coverage: **79.75%**
- exceptional-price Brier: **0.0973**
- direction Brier: **0.4214**

The final V10 production model is trained on the public-validation labels, so scoring V10 on that same window would be in-sample and must **not** be presented as unbiased validation. The useful new evidence is the strict chronological experiment recorded in `reports/v10/production_refit_summary.json`:

- fair old-V8 MAE on final unseen block: **34.16 EUR/MWh**
- V10 robust-refit MAE on same unseen block: **30.82 EUR/MWh**
- relative reduction: **9.77%**

Hidden-test MAE is still unknowable without organizer actuals.

## Upgrade an existing V9 project

Stop Streamlit and the daemon, then extract the V10 replacement ZIP over the project root.

```powershell
cd "C:\Users\samee\Downloads\turkuen"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
```

Run tests:

```powershell
python -m pytest -q `
tests\test_v10_system.py `
tests\test_v9_system.py `
tests\test_v8_system.py `
tests\test_v7_performance.py `
tests\test_v6_system.py `
tests\test_online_learning.py `
tests\test_v5_system.py `
tests\test_v3_constraints.py `
tests\test_v2_constraints.py `
tests\test_feature_legality.py
```

Test the expanded ENTSO-E MCP source:

```powershell
python scripts\test_entsoe_mcp.py
```

Run one live cycle:

```powershell
python scripts\run_online_cycle.py --no-retrain
python scripts\online_status.py
```

Look for:

- `mode = V10-production-full-refit-revisions-sample-aware-learning`
- `pooled_live_calibration.samples`
- `conformal_expand_low_eur` / `conformal_expand_high_eur`
- `drift_detected` should not trigger from coverage alone before 24 settled samples
- `latest_diagnostics.csv` model version `10.1-production-refit-robust-fusion`

Launch dashboard:

```powershell
python -m streamlit run .\src\dashboard\app.py
```

Start daemon in another terminal:

```powershell
python scripts\run_online_daemon.py --delay-seconds 75
```

## Reproduce the V10 production refit

A prebuilt V10 bundle is included. To retrain it locally from your existing V3/V8 artifacts:

```powershell
python scripts\train_v10_production.py
```

This deliberately does not overwrite the locked public-validation score. It creates `models/v10/model_bundle_v10.joblib` and a refit report.

## Data governance

ENTSO-E MCP is a third-party best-effort service. Exact live snapshots are stored locally and are not backfilled into the locked competition replay. Keep raw MCP/licensed market data out of public repositories. For production trading, direct ENTSO-E and/or licensed Nord Pool feeds remain preferable.

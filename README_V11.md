# TE-BalanceAI V11 — adaptive tournament + active-model transparency

V11 does **not** falsify the locked organizer validation score. The frozen public-validation evidence remains **43.4074 €/MWh MAE**. The production refit has separate chronological holdout evidence of **30.8181 €/MWh MAE** on 6,720 later unseen rows; this is shown separately in the dashboard.

## Why V11 exists

V10 could be installed while an older `models/online/champion.joblib` remained active. V11 makes the exact active model path/version visible in `online_status.py`, `latest_diagnostics.csv`, and the Data Health dashboard. It also upgrades full retraining from a two-way train/validation decision to a three-way chronological tournament:

1. train on historical + older settled live snapshots;
2. calibrate on a newer live block;
3. evaluate promotion on an untouched newest live block.

Two challengers are tested: the standard market-structured V8/V10 family and a stronger MAE-focused P50 LightGBM/CatBoost challenger. A challenger must improve **MAE itself**, improve the challenge-style composite score, and avoid unacceptable pinball/Brier regressions before promotion.

## New data

V11 continues V10's exact-vintage ingestion from Fingrid, FMI/Open-Meteo and entsoe-mcp. MCP, forecast-revision, weather and extra Fingrid features enter challenger training only when enough saved prediction-time snapshots contain them. No current archive is backfilled into the locked hackathon validation as if it had existed historically.

## Install

Stop Streamlit and the daemon. Extract `TE_BalanceAI_V11_Replacement.zip` directly over the project root and replace matching files. Preserve `.env`, `data/online`, and your outputs/history.

Then run:

```powershell
cd "C:\Users\samee\Downloads\turkuen"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path

python scripts\online_status.py
python -m pytest -q tests\test_v11_system.py tests\test_v10_system.py tests\test_v9_system.py tests\test_v8_system.py tests\test_online_learning.py
python scripts\run_online_cycle.py --no-retrain
python scripts\online_status.py
python scripts\compare_v11_candidates.py
```

If `online_status.py` says the active model is an older online champion rather than `10.1-production-refit-robust-fusion` (or a later `v11_*` champion), activate the packaged V10 refit safely:

```powershell
python scripts\activate_v10_refit.py
python scripts\run_online_cycle.py --no-retrain
python scripts\online_status.py
```

`activate_v10_refit.py` backs up the old champion before replacing only the model file. It does not delete the SQLite history or adaptive state.

After at least 48 settled origins, trigger the V11 tournament:

```powershell
python scripts\run_daily_learning.py --force
python scripts\online_status.py
```

The daily learner will train/calibrate/evaluate multiple challengers and promote only a qualifying winner.

## Dashboard

```powershell
python -m streamlit run .\src\dashboard\app.py
```

Accuracy now separates:

- locked organizer validation MAE (43.41);
- production chronological holdout MAE (~30.82);
- live settled MAE (changes as real outcomes arrive).

Data Health also shows the exact active model version and path.

## Daemon

```powershell
python scripts\run_online_daemon.py --delay-seconds 75
```

The console label is `[V11 daemon]`.

## Technical notes

See `docs/V11_ADAPTIVE_TOURNAMENT.md` for the active-model precedence rule, model-search conclusions, three-way chronological promotion test, and live-feature admission rules.

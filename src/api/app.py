from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.config import LIVE_SOURCE_HEALTH_JSON, ONLINE_OUTPUTS_DIR, ONLINE_MARKET_STATE_JSON
from src.online.store import OnlineStore
from src.online.adaptive import AdaptiveState
from src.online.continuous import default_model_path
from src.explainability import grouped_model_importance
from src.performance import official_validation_summary, official_validation_by_hour, validation_step_performance, validation_detail, validation_intuitive_accuracy, latest_daily_report, daily_history

ROOT = Path(__file__).resolve().parents[2]
ONLINE_FORECAST = ONLINE_OUTPUTS_DIR / "latest_forecast.csv"
ONLINE_DIAG = ONLINE_OUTPUTS_DIR / "latest_diagnostics.csv"
V3_FORECAST = ROOT / "outputs" / "v3" / "forecast_backtest_v3.csv"
V3_DIAG = ROOT / "outputs" / "v3" / "forecast_diagnostics_v3.csv"
V2_FORECAST = ROOT / "outputs" / "v2" / "forecast_backtest_v2.csv"
V1_FORECAST = ROOT / "outputs" / "forecast_backtest.csv"

API_KEY = os.getenv("TE_BALANCEAI_API_KEY", "").strip()
CORS_ORIGINS = [x.strip() for x in os.getenv(
    "TE_BALANCEAI_CORS_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501",
).split(",") if x.strip()]

app = FastAPI(title="TE-BalanceAI API", version="0.9.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


def require_api_key(x_api_key: str | None) -> None:
    if API_KEY and (not x_api_key or not secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(401, "Invalid or missing API key")


def active_forecast() -> Path:
    for p in [ONLINE_FORECAST, V3_FORECAST, V2_FORECAST, V1_FORECAST]:
        if p.exists(): return p
    return ONLINE_FORECAST


def active_diag() -> Path | None:
    path=active_forecast()
    if path==ONLINE_FORECAST and ONLINE_DIAG.exists(): return ONLINE_DIAG
    if path==V3_FORECAST and V3_DIAG.exists(): return V3_DIAG
    return None


def load() -> pd.DataFrame:
    path=active_forecast()
    if not path.exists(): raise HTTPException(503,"Forecast file not generated yet")
    return pd.read_csv(path)


def load_diag() -> pd.DataFrame | None:
    p=active_diag(); return pd.read_csv(p) if p and p.exists() else None


@app.get('/health')
def health():
    path=active_forecast()
    source_health={}
    if LIVE_SOURCE_HEALTH_JSON.exists():
        try: source_health=json.loads(LIVE_SOURCE_HEALTH_JSON.read_text(encoding='utf-8'))
        except Exception: pass
    return {
        'status':'ok','version':'0.9.0','forecast_ready':path.exists(),
        'api_key_required':bool(API_KEY),'active_forecast':str(path.relative_to(ROOT)) if path.exists() else None,
        'online_mode':ONLINE_FORECAST.exists(),'source_healthy_fraction':source_health.get('healthy_fraction'),
    }


@app.get('/online/status')
def online_status(x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    return OnlineStore().status()


@app.get('/online/source-health')
def source_health(x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    if not LIVE_SOURCE_HEALTH_JSON.exists(): return {'status':'not-yet-ingested'}
    return json.loads(LIVE_SOURCE_HEALTH_JSON.read_text(encoding='utf-8'))


@app.get('/online/performance')
def online_performance(hours: int = 24, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    hours=max(1,min(int(hours),24*30))
    df=OnlineStore().recent_joined(pd.Timestamp.now(tz='UTC')-pd.Timedelta(hours=hours))
    if df.empty: return {'hours':hours,'samples':0}
    by_step=(df.groupby('step').agg(samples=('abs_error','size'),mae=('abs_error','mean'),coverage=('covered','mean')).reset_index())
    return {
        'hours':hours,'samples':len(df),'mae':float(df.abs_error.mean()),'coverage':float(df.covered.mean()),
        'direction_brier':None if df.direction_brier.dropna().empty else float(df.direction_brier.mean()),
        'spike_high_brier':None if df.spike_high_brier.dropna().empty else float(df.spike_high_brier.mean()),
        'spike_low_brier':None if df.spike_low_brier.dropna().empty else float(df.spike_low_brier.mean()),
        'by_step':by_step.to_dict(orient='records'),
    }


@app.get('/online/market-state')
def market_state(x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    if not ONLINE_MARKET_STATE_JSON.exists():
        return {'status':'not-yet-generated'}
    return json.loads(ONLINE_MARKET_STATE_JSON.read_text(encoding='utf-8'))


@app.get('/online/adaptation')
def adaptation(x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    state=AdaptiveState.load()
    return {
        'updated_at_utc':state.updated_at_utc,
        'windows':{k:{'hours':v.hours,'samples':v.samples,'mae':v.rolling_mae,'coverage':v.rolling_coverage,'adapted_horizons':len(v.steps)} for k,v in state.windows.items()},
        'drift_score':state.drift_score,
        'drift_detected':state.drift_detected,
        'drift_reasons':state.drift_reasons,
        'blended_horizons':len(state.blended_steps),
        'pooled_live_calibration':{
            'samples':state.blended_global.n,
            'empirical_coverage':state.blended_global.empirical_coverage,
            'conformal_expand_eur':state.blended_global.conformal_expand,
            'target_coverage':0.80,
        },
    }


@app.get('/online/history')
def online_history(x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    d=ONLINE_OUTPUTS_DIR/'history'
    files=sorted(d.glob('forecast_*.csv')) if d.exists() else []
    return {'count':len(files),'files':[x.name for x in files[-72:]]}


@app.get('/validation/performance')
def validation_performance(x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    summary=official_validation_summary()
    intuitive=validation_intuitive_accuracy()
    by_step=validation_step_performance()
    return {
        'official':summary,
        'intuitive_diagnostics':intuitive,
        'by_15min_step':by_step.to_dict(orient='records') if not by_step.empty else [],
    }


@app.get('/online/daily-learning')
def daily_learning_status(x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    latest=latest_daily_report()
    hist=daily_history()
    return {
        'latest':latest,
        'history':hist.to_dict(orient='records') if not hist.empty else [],
    }



@app.get('/online/explainability')
def online_explainability(x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    try:
        imp=grouped_model_importance(default_model_path())
        return {'global_group_importance':imp.to_dict(orient='records'), 'note':'Normalized global model importance; association, not causal proof.'}
    except Exception as exc:
        raise HTTPException(503, f'Explainability unavailable: {exc}')

@app.get('/metadata')
def metadata(x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    df=load()
    return {
        'system':'TE-BalanceAI','version':'0.9.0',
        'forecast_origins':int(df['forecast_origin_utc'].nunique()),
        'steps_per_origin':int(df.groupby('forecast_origin_utc').size().median()),
        'latest_origin':df['forecast_origin_utc'].max(),
        'continuous_learning':ONLINE_FORECAST.exists(),
        'forecast_cadence':'one batch each full hour; 40 x 15-minute steps; 10-hour horizon',
        'security':'read-only API; optional X-API-Key; CORS allowlist; no-store responses',
    }


@app.get('/origins')
def origins(x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    df=load(); return {'origins':df['forecast_origin_utc'].drop_duplicates().tolist()}


@app.get('/forecast/latest')
def latest(x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    df=load(); origin=df['forecast_origin_utc'].max(); rows=df[df.forecast_origin_utc==origin].copy()
    diag=load_diag()
    if diag is not None:
        rows=rows.merge(diag[diag.forecast_origin_utc==origin],on=['forecast_origin_utc','delivery_start_utc','step'],how='left')
    return {'forecast_origin_utc':origin,'rows':rows.to_dict(orient='records')}


@app.get('/forecast/{origin}')
def forecast(origin: str, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    df=load(); rows=df[df.forecast_origin_utc==origin].copy()
    if rows.empty: raise HTTPException(404,'Origin not found')
    diag=load_diag()
    if diag is not None:
        rows=rows.merge(diag[diag.forecast_origin_utc==origin],on=['forecast_origin_utc','delivery_start_utc','step'],how='left')
    return {'forecast_origin_utc':origin,'rows':rows.to_dict(orient='records')}

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..config import (
    LIVE_LONG_PARQUET, LIVE_CONTEXT_JSON, ONLINE_OUTPUTS_DIR, ONLINE_EVAL_HOURS,
    ONLINE_LOOKBACK_HOURS, ONLINE_FORECAST_HOURS, ONLINE_SETTLEMENT_LAG_MINUTES,
    ONLINE_CHAMPION, MODELS_DIR, ONLINE_ENABLE_FULL_RETRAIN, ONLINE_FULL_RETRAIN_HOURS,
    ONLINE_DAYAHEAD_REFRESH_LOCAL_HOUR, ONLINE_ADAPT_WINDOWS_HOURS,
    ONLINE_DRIFT_RETRAIN_COOLDOWN_HOURS,
)
from ..data import clear_data_cache
from ..features import build_features, business_driver_text
from ..models import ModelBundleV3, predict_bundle
from ..models_v8 import ModelBundleV8, load_any_bundle, predict_any_bundle
from .adaptive import (
    AdaptiveState, apply_adaptive, learn_multiwindow_state, attach_adaptive_meta_features,
)
from .sources import LiveSourceOrchestrator
from .store import OnlineStore
from .retrain import maybe_retrain
from .market_state import build_market_state, save_market_state
from .daily import run_daily_learning


def _easter_sunday(year: int) -> date:
    a=year%19; b=year//100; c=year%100; d=b//4; e=b%4
    f=(b+8)//25; g=(b-f+1)//3; h=(19*a+b-d-g+15)%30
    i=c//4; k=c%4; l=(32+2*e+2*i-h-k)%7; m=(a+11*h+22*l)//451
    month=(h+l-7*m+114)//31; day=((h+l-7*m+114)%31)+1
    return date(year,month,day)


def _fi_holidays(year: int) -> set[date]:
    easter=_easter_sunday(year)
    fixed={date(year,1,1),date(year,1,6),date(year,5,1),date(year,12,6),date(year,12,24),date(year,12,25),date(year,12,26)}
    movable={easter-timedelta(days=2),easter,easter+timedelta(days=1),easter+timedelta(days=39)}
    midsummer=next(date(year,6,d) for d in range(19,26) if date(year,6,d).weekday()==4)
    return fixed|movable|{midsummer}


def _calendar_long(origin: pd.Timestamp, end: pd.Timestamp, fetched_at: pd.Timestamp) -> pd.DataFrame:
    times=pd.date_range(origin.floor('15min'),end.ceil('15min'),freq='15min',tz='UTC')
    helsinki=ZoneInfo('Europe/Helsinki')
    rows=[]
    for ts in times:
        local=ts.to_pydatetime().astimezone(helsinki)
        vals={
            'cal_weekday_local':local.weekday(),
            'cal_hour_local':local.hour,
            'cal_is_holiday_fi':int(local.date() in _fi_holidays(local.year)),
            'cal_is_dst':int(bool(local.dst() and local.dst().total_seconds()!=0)),
        }
        for series,value in vals.items():
            rows.append({'series_id':series,'source':'calendar_live','dataset_id':'','timestamp_utc':ts,'value':float(value),
                         'unit':'','first_usable_origin_utc':fetched_at,'kind':'forecast'})
    return pd.DataFrame(rows)



REVISION_BASE_SERIES = {
    "fi_spot_price", "fi_consumption_fc_da", "fi_wind_fc_da", "fi_solar_fc_da",
    "fi_production_fc_rt", "mcp_fi_day_ahead_price", "mcp_fi_intraday_price",
    "mcp_fi_load_forecast", "mcp_fi_generation_forecast",
    "mcp_fi_wind_forecast", "mcp_fi_solar_forecast",
    "weather_wind_100m_weighted", "weather_temperature_mean",
}

def _forecast_revision_rows(current: pd.DataFrame, previous: pd.DataFrame, fetched_at: pd.Timestamp) -> pd.DataFrame:
    """Difference between this forecast snapshot and the previous hourly snapshot.

    Forecast *revisions* are often more predictive than the raw forecast level:
    a sudden downward wind revision or upward load revision is direct evidence
    that the expected system balance has changed.  The revision itself becomes
    usable only at the current fetch time, so its first-usable timestamp is the
    new snapshot time.
    """
    cols=["series_id","source","dataset_id","timestamp_utc","value","unit","first_usable_origin_utc","kind"]
    if current is None or previous is None or current.empty or previous.empty:
        return pd.DataFrame(columns=cols)
    c=current[current["series_id"].isin(REVISION_BASE_SERIES)].copy()
    p=previous[previous["series_id"].isin(REVISION_BASE_SERIES)].copy()
    if c.empty or p.empty:
        return pd.DataFrame(columns=cols)
    for d in (c,p):
        d["timestamp_utc"]=pd.to_datetime(d["timestamp_utc"],utc=True,errors="coerce")
        d["value"]=pd.to_numeric(d["value"],errors="coerce")
    c=c.dropna(subset=["timestamp_utc","value"]).sort_values("first_usable_origin_utc").drop_duplicates(["series_id","timestamp_utc"],keep="last")
    p=p.dropna(subset=["timestamp_utc","value"]).sort_values("first_usable_origin_utc").drop_duplicates(["series_id","timestamp_utc"],keep="last")
    m=c[["series_id","timestamp_utc","value","unit"]].merge(
        p[["series_id","timestamp_utc","value"]], on=["series_id","timestamp_utc"], how="inner", suffixes=("_new","_old")
    )
    if m.empty:
        return pd.DataFrame(columns=cols)
    out=pd.DataFrame({
        "series_id":"revision__"+m["series_id"].astype(str),
        "source":"forecast_revision",
        "dataset_id":"previous_hour_snapshot",
        "timestamp_utc":m["timestamp_utc"],
        "value":m["value_new"]-m["value_old"],
        "unit":m["unit"],
        "first_usable_origin_utc":pd.Timestamp(fetched_at),
        "kind":"forecast",
    })
    return out.replace([np.inf,-np.inf],np.nan).dropna(subset=["value"])

def current_full_hour(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Current hourly forecast origin.

    Production cycles run shortly *after* the full hour so information that is
    published exactly at the origin can be included. This matches the challenge
    definition more closely than forecasting the next origin at minute 55.
    """
    t=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if t.tzinfo is None: t=t.tz_localize('UTC')
    else: t=t.tz_convert('UTC')
    return t.floor('h')


def default_model_path() -> Path:
    for p in [ONLINE_CHAMPION, MODELS_DIR/'v10'/'model_bundle_v10.joblib', MODELS_DIR/'v8'/'model_bundle_v8.joblib', MODELS_DIR/'v3'/'model_bundle_v3.joblib', MODELS_DIR/'v3'/'model_bundle_v3_dev.joblib']:
        if p.exists(): return p
    raise FileNotFoundError('No V10/V8/V3 model bundle found. Run V3/V8 first, then python scripts/train_v10_production.py.')


def _context_snapshot(live_long: pd.DataFrame, fetched_at: pd.Timestamp, health: dict, origin: pd.Timestamp) -> dict:
    helsinki=ZoneInfo('Europe/Helsinki')
    local_origin=origin.to_pydatetime().astimezone(helsinki)
    context={
        'fetched_at_utc':fetched_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'origin_utc':origin.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'origin_helsinki':local_origin.isoformat(),
        'day_ahead_refresh_window': local_origin.hour == ONLINE_DAYAHEAD_REFRESH_LOCAL_HOUR,
        'source_health': health,
    }
    watch=[
        'fi_surplus_deficit_rt','fi_electric_boiler_rt','fi_frequency_rt','fi_wind_generation_rt',
        'fi_hydro_generation_rt','fi_nuclear_generation_rt','fi_power_system_state_rt','fi_net_import_export_rt','entsoe_hydro_reservoir',
        'entsoe_generation_unavailable_mw','entsoe_transmission_outage_count',
        'weather_wind_100m_weighted','weather_wind_10m_weighted','weather_wind_gust_max','weather_temperature_mean',
        'weather_obs_wind_10m_weighted','weather_obs_temperature_mean','mcp_fi_day_ahead_price',
        'mcp_fi_generation_unavailable_mw','mcp_fi_afrr_up_capacity_price','mcp_fi_afrr_down_capacity_price',
    ]
    for series in watch:
        g=live_long[live_long['series_id']==series]
        if not g.empty:
            r=g.sort_values('timestamp_utc').iloc[-1]
            context[series]={'value':float(r['value']),'timestamp_utc':pd.Timestamp(r['timestamp_utc']).strftime('%Y-%m-%dT%H:%M:%SZ'),'source':str(r.get('source',''))}
    return context


class ContinuousLearningService:
    def __init__(self):
        self.store=OnlineStore()
        self.sources=LiveSourceOrchestrator()

    def ingest(self, origin: pd.Timestamp) -> tuple[pd.DataFrame,dict]:
        now=pd.Timestamp.now(tz='UTC')
        actual_start=now-pd.Timedelta(hours=ONLINE_LOOKBACK_HOURS)
        forecast_end=origin+pd.Timedelta(hours=ONLINE_FORECAST_HOURS)
        # Load the previous exact snapshot before overwriting the live cache so
        # V10 can derive forecast-revision features (wind/load/production/price).
        previous=pd.DataFrame()
        if LIVE_LONG_PARQUET.exists():
            try:
                previous=pd.read_parquet(LIVE_LONG_PARQUET)
            except Exception:
                previous=pd.DataFrame()
        bundle=self.sources.collect(origin,actual_start,now,forecast_end)
        live=bundle.data
        revisions=_forecast_revision_rows(live,previous,now)
        if not revisions.empty:
            live=pd.concat([live,revisions],ignore_index=True)
            bundle.log.append({"source":"forecast_revision","status":"ok","rows":int(len(revisions)),"series":int(revisions["series_id"].nunique())})
        live=pd.concat([live,_calendar_long(origin,forecast_end,now)],ignore_index=True)
        # Current-origin multi-source cache only. Historical online retraining
        # uses feature_json saved at prediction time, preventing revised data
        # from leaking backwards.
        LIVE_LONG_PARQUET.parent.mkdir(parents=True,exist_ok=True)
        live.drop(columns=['kind'],errors='ignore').to_parquet(LIVE_LONG_PARQUET,index=False)
        clear_data_cache()
        self.store.upsert_actuals_from_long(live)
        status='ok' if bundle.health.get('items_ok',0)>0 else 'degraded'
        self.store.log_ingestion('multi_source_live',status,{'origin':str(origin),'health':bundle.health})
        context=_context_snapshot(live,now,bundle.health,origin)
        LIVE_CONTEXT_JSON.parent.mkdir(parents=True,exist_ok=True)
        LIVE_CONTEXT_JSON.write_text(json.dumps(context,indent=2,default=str),encoding='utf-8')
        return live,{'datasets':bundle.log,'health':bundle.health,'context':context}

    def update_learning(self) -> AdaptiveState:
        cutoff=pd.Timestamp.now(tz='UTC')-pd.Timedelta(minutes=ONLINE_SETTLEMENT_LAG_MINUTES)
        evaluated=self.store.evaluate_settled(cutoff)
        now=pd.Timestamp.now(tz='UTC')
        frames={int(h): self.store.recent_joined(now-pd.Timedelta(hours=int(h))) for h in ONLINE_ADAPT_WINDOWS_HOURS}
        state=learn_multiwindow_state(frames)
        state.save()
        if evaluated or state.drift_detected:
            self.store.log_ingestion('online_learning','ok',{
                'newly_evaluated':evaluated,
                'windows_hours':list(ONLINE_ADAPT_WINDOWS_HOURS),
                'rolling_mae_24h':state.rolling_mae,
                'rolling_coverage_24h':state.rolling_coverage,
                'drift_score':state.drift_score,
                'drift_detected':state.drift_detected,
                'drift_reasons':state.drift_reasons,
            })
        return state

    def maybe_full_retrain(self, model_path: Path, state: AdaptiveState) -> dict:
        if not ONLINE_ENABLE_FULL_RETRAIN:
            return {'attempted':False,'reason':'disabled'}
        last=self.store.last_model_event('retrain_promoted') or self.store.last_model_event('retrain_rejected')
        if last:
            elapsed=pd.Timestamp.now(tz='UTC')-pd.Timestamp(last['event_time_utc'])
            required=ONLINE_DRIFT_RETRAIN_COOLDOWN_HOURS if state.drift_detected else ONLINE_FULL_RETRAIN_HOURS
            if elapsed < pd.Timedelta(hours=required):
                return {'attempted':False,'reason':f'last retrain attempt {elapsed} ago; required {required}h'}
        result=asdict(maybe_retrain(self.store,model_path))
        if state.drift_detected:
            result['trigger']='drift'
            result['drift_score']=state.drift_score
            result['drift_reasons']=state.drift_reasons
        else:
            result['trigger']='scheduled'
        return result

    def forecast(self, origin: pd.Timestamp, context: dict | None = None, state: AdaptiveState | None = None) -> tuple[pd.DataFrame,pd.DataFrame]:
        model_path=default_model_path()
        bundle=load_any_bundle(model_path)
        features=build_features(pd.DatetimeIndex([origin]),include_labels=False,include_online_context=True).reset_index(drop=True)
        state=state or AdaptiveState.load()
        # Save the exact multi-window error state into every forecast snapshot.
        # Old champions ignore these columns; future V6 challengers may learn them.
        features=attach_adaptive_meta_features(features,state)
        base=predict_any_bundle(bundle,features).reset_index(drop=True)
        pred=apply_adaptive(base,features['step'],state)
        health=float((context or {}).get('source_health',{}).get('healthy_fraction',1.0))
        pred['reliability_source'] = health
        if 'reliability' in pred:
            pred['reliability_pre_source'] = pred['reliability'].to_numpy(float)
            pred['reliability']=np.clip(pred['reliability'].to_numpy(float)*(0.75+0.25*health),0,1)
        cov = state.rolling_coverage
        cov_text = 'not enough settled outcomes yet' if cov is None else f'live interval coverage {100*cov:.1f}%'
        evidence_n = int(getattr(state.blended_global, "n", 0) or 0)
        sample_conf = min(1.0, evidence_n / 48.0)
        pred['reliability_reason'] = [
            f"model uncertainty + disagreement; {cov_text}; {evidence_n} settled live samples ({100*sample_conf:.0f}% evidence maturity); source health {100*health:.0f}%; drift {state.drift_score:.2f}"
        ] * len(pred)
        drivers=business_driver_text(features)
        model_version=getattr(bundle,'version',model_path.stem)
        self.store.save_predictions(features,pred,drivers,model_version,context=context)
        cutoff=pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
        out=pd.DataFrame({
            'forecast_origin_utc':features['forecast_origin_utc'].dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'delivery_start_utc':features['delivery_start_utc'].dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'step':features['step'].astype(int),
            'p10':pred['p10'],'p50':pred['p50'],'p90':pred['p90'],
            'p_up':pred['p_up'],'p_down':pred['p_down'],'p_none':pred['p_none'],
            'p_spike_high':pred['p_spike_high'],'p_spike_low':pred['p_spike_low'],
            'data_cutoff_utc':cutoff,'top_drivers':drivers,
        })
        ONLINE_OUTPUTS_DIR.mkdir(parents=True,exist_ok=True)
        (ONLINE_OUTPUTS_DIR/'history').mkdir(parents=True,exist_ok=True)
        out.to_csv(ONLINE_OUTPUTS_DIR/'latest_forecast.csv',index=False)
        stamp=origin.strftime('%Y%m%dT%H%MZ')
        out.to_csv(ONLINE_OUTPUTS_DIR/'history'/f'forecast_{stamp}.csv',index=False)
        diag=pd.DataFrame({
            'forecast_origin_utc':out['forecast_origin_utc'],'delivery_start_utc':out['delivery_start_utc'],'step':out['step'],
            'model_version':[model_version]*len(out),'model_path':[str(model_path)]*len(out),
            'p50_lgb':pred.get('p50_lgb'),'p50_cat':pred.get('p50_cat'),
            'p50_direct':pred.get('p50_direct',pred.get('p50')),'p50_spread':pred.get('p50_spread'),
            'v8_spot_available':pred.get('v8_spot_available'),
            'model_disagreement':pred.get('model_disagreement'),'reliability':pred.get('reliability'),
            'reliability_model':pred.get('reliability_model'),'reliability_empirical':pred.get('reliability_empirical'),
            'reliability_calibration':pred.get('reliability_calibration'),'reliability_source':pred.get('reliability_source'),
            'reliability_coverage_raw':pred.get('reliability_coverage_raw'),
            'reliability_coverage_shrunk':pred.get('reliability_coverage_shrunk'),
            'reliability_live_sample_confidence':pred.get('reliability_live_sample_confidence'),
            'reliability_reason':pred.get('reliability_reason'),
            'adaptive_applied':pred.get('adaptive_applied',False),'adaptive_bias':pred.get('adaptive_bias',0.0),
            'global_p50_bias_applied':pred.get('global_p50_bias_applied',0.0),
            'global_p50_mix_strength':pred.get('global_p50_mix_strength',0.0),
            'conformal_expand_eur':pred.get('conformal_expand_eur',0.0),
            'conformal_expand_low_eur':pred.get('conformal_expand_low_eur',0.0),
            'conformal_expand_high_eur':pred.get('conformal_expand_high_eur',0.0),
            'drift_score':pred.get('drift_score',state.drift_score),'model_version':model_version,
            'source_health':health,
        })
        diag.to_csv(ONLINE_OUTPUTS_DIR/'latest_diagnostics.csv',index=False)
        diag.to_csv(ONLINE_OUTPUTS_DIR/'history'/f'diagnostics_{stamp}.csv',index=False)
        (ONLINE_OUTPUTS_DIR/'latest_context.json').write_text(json.dumps(context or {},indent=2,default=str),encoding='utf-8')
        twin=build_market_state(features,pred,state,context=context)
        save_market_state(twin,origin)
        return out,diag

    def run_cycle(self, origin: pd.Timestamp | None = None, ingest: bool = True, allow_retrain: bool = True) -> dict:
        origin=current_full_hour() if origin is None else pd.Timestamp(origin)
        if origin.tzinfo is None: origin=origin.tz_localize('UTC')
        else: origin=origin.tz_convert('UTC')
        context={}; ingest_summary={'skipped':True}
        if ingest:
            _,summary=self.ingest(origin); context=summary.get('context',{}); ingest_summary=summary
        state=self.update_learning()
        # Hourly cycles always adapt weights/bias/calibration. Full base-model retraining
        # is reserved for drift or the once-per-day learning checkpoint.
        retrain={'attempted':False,'reason':'scheduled retraining handled by daily checkpoint'}
        if allow_retrain and state.drift_detected:
            retrain=self.maybe_full_retrain(default_model_path(),state)
        forecast,diag=self.forecast(origin,context=context,state=state)

        # Run a daily Helsinki-time accuracy checkpoint only for a genuine live cycle.
        now_origin=current_full_hour()
        is_live_cycle=ingest and abs((origin-now_origin).total_seconds()) <= 3600
        daily={'ran':False,'reason':'not a live cycle'}
        if is_live_cycle:
            daily=run_daily_learning(self.store,default_model_path(),state,allow_retrain=allow_retrain,force=False)

        return {
            'mode':'V10-production-full-refit-revisions-sample-aware-learning',
            'origin':origin.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'rows':len(forecast),'steps':'1..40','horizon_hours':10,
            'adaptive_window_samples':{k:v.samples for k,v in state.windows.items()},
            'rolling_mae_24h':state.rolling_mae,'rolling_coverage_24h':state.rolling_coverage,
            'drift_score':state.drift_score,'drift_detected':state.drift_detected,'drift_reasons':state.drift_reasons,
            'ingestion_health':ingest_summary.get('health',{}),'retrain':retrain,'daily_learning':daily,
            'output':str(ONLINE_OUTPUTS_DIR/'latest_forecast.csv'),
            'archive':str(ONLINE_OUTPUTS_DIR/'history'),
        }

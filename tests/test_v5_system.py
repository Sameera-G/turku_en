from __future__ import annotations

import pandas as pd

from src.features import build_features
from src.models_v8 import load_any_bundle, predict_any_bundle
from src.online.continuous import current_full_hour, default_model_path
from src.online.sources import LiveSourceOrchestrator


def test_v5_current_origin_is_floor_hour():
    t=pd.Timestamp('2026-11-07T12:34:56Z')
    assert current_full_hour(t)==pd.Timestamp('2026-11-07T12:00:00Z')


def test_weather_fallback_keeps_primary_and_fills_missing():
    ts=pd.to_datetime(['2026-11-07T12:00:00Z','2026-11-07T12:15:00Z'],utc=True)
    om=pd.DataFrame({'series_id':['weather_temperature_mean'],'timestamp_utc':[ts[0]],'value':[2.0]})
    fmi=pd.DataFrame({'series_id':['weather_temperature_mean','weather_temperature_mean'],'timestamp_utc':ts,'value':[9.0,3.0]})
    out=LiveSourceOrchestrator._weather_primary(om,fmi)
    v=out.sort_values('timestamp_utc').value.tolist()
    assert v==[2.0,3.0]


def test_online_context_does_not_break_old_champion():
    origin=pd.DatetimeIndex([pd.Timestamp('2026-10-20T12:00:00Z')])
    feat=build_features(origin,include_labels=False,include_online_context=True).reset_index(drop=True)
    assert len(feat)==40
    assert 'future__weather_wind_100m_weighted' in feat.columns
    bundle=load_any_bundle(default_model_path())
    pred=predict_any_bundle(bundle,feat)
    assert len(pred)==40
    assert ((pred.p10<=pred.p50)&(pred.p50<=pred.p90)).all()
    assert (pred[['p_up','p_down','p_none']].sum(axis=1).sub(1).abs()<1e-8).all()

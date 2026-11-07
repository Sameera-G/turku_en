from __future__ import annotations

import numpy as np
import pandas as pd

from src.online.adaptive import learn_multiwindow_state, apply_adaptive
from src.online.entsoe_mcp import EntsoeMCPClient


def _settled(n=12):
    rows=[]
    for i in range(n):
        y=80.0 + i
        rows.append({
            'forecast_origin_utc':pd.Timestamp('2026-11-07T00:00:00Z')+pd.Timedelta(minutes=15*i),
            'delivery_start_utc':pd.Timestamp('2026-11-07T00:15:00Z')+pd.Timedelta(minutes=15*i),
            'step':(i%40)+1,
            'imbalance_price':y,
            'p10':10.0,'p50':25.0,'p90':40.0,
            'p50_lgb':24.0,'p50_cat':26.0,'p50_direct':30.0,'p50_spread':24.0,
            'direction':-1,'p_down':0.65,'p_none':0.05,'p_up':0.30,
            'spot_price':20.0,'p_spike_high':0.10,'p_spike_low':0.10,
        })
    return pd.DataFrame(rows)


def test_pooled_conformal_starts_before_each_horizon_has_eight_samples():
    df=_settled(12)
    state=learn_multiwindow_state({24:df,168:df,336:df})
    assert state.blended_global.n >= 8
    assert state.blended_global.conformal_expand > 0
    assert len(state.blended_steps) == 0  # no individual step has 8 examples

    pred=pd.DataFrame({
        'p10':[10.0,10.0], 'p50':[25.0,25.0], 'p90':[40.0,40.0],
        'p_down':[0.65,0.65], 'p_none':[0.05,0.05], 'p_up':[0.30,0.30],
        'p_spike_high':[0.1,0.1], 'p_spike_low':[0.1,0.1],
        'reliability':[0.8,0.8], 'p50_lgb':[24,24], 'p50_cat':[26,26],
        'p50_direct':[30,30], 'p50_spread':[24,24],
    })
    out=apply_adaptive(pred,pd.Series([1,2]),state)
    assert (out['p10'] < 10).all()
    assert (out['p90'] > 40).all()
    assert (out['conformal_expand_eur'] > 0).all()
    assert {'reliability_model','reliability_empirical','reliability_calibration'}.issubset(out.columns)


def test_entsoe_mcp_payload_parser_accepts_wrapped_rows():
    payload={'data':[{'ts_utc':'2026-11-07T00:00:00Z','zone':'FI','eur_per_mwh':42.0}]}
    rows=EntsoeMCPClient._rows(payload)
    assert len(rows)==1 and rows[0]['eur_per_mwh']==42.0


def test_entsoe_mcp_quarter_fill_expands_hourly_curve():
    fetched=pd.Timestamp('2026-11-06T12:00:00Z')
    raw=pd.DataFrame({'ts_utc':pd.to_datetime(['2026-11-07T00:00:00Z','2026-11-07T01:00:00Z']), 'eur_per_mwh':[20,30]})
    long=EntsoeMCPClient._long(raw,'fi_spot_price','eur_per_mwh','EUR/MWh',fetched,'forecast','test')
    q=EntsoeMCPClient._quarter_hour_forward_fill(long,pd.Timestamp('2026-11-07T00:00:00Z'),pd.Timestamp('2026-11-07T01:45:00Z'))
    assert len(q)>=8
    assert q.loc[q['timestamp_utc']==pd.Timestamp('2026-11-07T00:45:00Z'),'value'].iloc[0]==20
    assert q.loc[q['timestamp_utc']==pd.Timestamp('2026-11-07T01:45:00Z'),'value'].iloc[0]==30

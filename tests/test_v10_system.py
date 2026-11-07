from __future__ import annotations

import numpy as np
import pandas as pd

from src.online.adaptive import AdaptiveState, StepAdaptive, apply_adaptive, learn_multiwindow_state
from src.online.continuous import _forecast_revision_rows, default_model_path


def _pred(n=40, reliability=0.69):
    return pd.DataFrame({
        'p10':np.full(n,0.0),'p50':np.full(n,20.0),'p90':np.full(n,40.0),
        'p_up':np.full(n,.3),'p_down':np.full(n,.6),'p_none':np.full(n,.1),
        'p_spike_high':np.full(n,.08),'p_spike_low':np.full(n,.08),
        'p50_direct':np.full(n,22.0),'p50_spread':np.full(n,18.0),
        'p50_lgb':np.full(n,21.0),'p50_cat':np.full(n,19.0),
        'reliability':np.full(n,reliability),
    })


def test_small_live_sample_does_not_crush_reliability():
    g=StepAdaptive(n=12,mae=26.33,price_bias=0.0,p50_spread_weight=.7,
                   empirical_coverage=1/3,shrunk_coverage=.64,
                   conformal_expand_low=10.0,conformal_expand_high=25.0)
    state=AdaptiveState(blended_global=g,rolling_mae=26.33,rolling_coverage=1/3,drift_score=0.0)
    out=apply_adaptive(_pred(),pd.Series(range(1,41)),state)
    # V9 could push a ~69% trained-model score toward the high-50s after only
    # twelve observations. V10 keeps the warning but applies a small evidence weight.
    assert float(out.reliability.min()) > 0.66
    assert float(out.reliability_live_sample_confidence.iloc[0]) == 0.25
    assert float(out.conformal_expand_high_eur.iloc[0]) > float(out.conformal_expand_low_eur.iloc[0])


def test_coverage_drift_waits_for_enough_evidence():
    rows=[]
    for i in range(12):
        rows.append({'forecast_origin_utc':f'2026-11-07T{i:02d}:00:00Z','delivery_start_utc':f'2026-11-07T{i:02d}:00:00Z','step':1,
                     'imbalance_price':80.0,'p10':0.0,'p50':20.0,'p90':40.0,
                     'p_up':.3,'p_down':.6,'p_none':.1,'p_spike_high':.1,'p_spike_low':.1,
                     'p50_direct':22.0,'p50_spread':18.0})
    df=pd.DataFrame(rows)
    state=learn_multiwindow_state({24:df,168:df,336:df})
    assert state.rolling_coverage == 0.0
    assert state.drift_detected is False
    assert state.blended_global.n == 12
    assert state.blended_global.conformal_expand_high > 0


def test_forecast_revision_rows_are_timestamp_safe():
    ts=pd.Timestamp('2026-11-07T06:00:00Z')
    old=pd.DataFrame([{'series_id':'fi_wind_fc_da','source':'x','dataset_id':'1','timestamp_utc':ts,'value':3000.0,'unit':'MW','first_usable_origin_utc':ts-pd.Timedelta(hours=1)}])
    new=pd.DataFrame([{'series_id':'fi_wind_fc_da','source':'x','dataset_id':'1','timestamp_utc':ts,'value':2500.0,'unit':'MW','first_usable_origin_utc':ts}])
    out=_forecast_revision_rows(new,old,ts)
    assert len(out)==1
    assert out.iloc[0].series_id=='revision__fi_wind_fc_da'
    assert out.iloc[0].value==-500.0
    assert pd.Timestamp(out.iloc[0].first_usable_origin_utc)==ts


def test_v10_model_is_preferred_when_present():
    p=default_model_path()
    assert p.name in {'champion.joblib','model_bundle_v10.joblib'}

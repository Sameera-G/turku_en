from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_features
from src.online.adaptive import learn_multiwindow_state, attach_adaptive_meta_features, apply_adaptive


def _frame(hours: int, mae_shift: float = 0.0) -> pd.DataFrame:
    # Enough repeated horizons for the adaptive learner; timestamps are not used
    # by the learner itself, only by the store in production.
    rows=[]
    for step in range(1,41):
        for i in range(10):
            y=100.0+step*0.1+i*0.05
            p50=y-mae_shift
            rows.append({
                'step':step,'imbalance_price':y,'p50':p50,'p10':p50-20,'p90':p50+20,
                'p50_lgb':p50-2,'p50_cat':p50+2,
                'direction':1 if i%3==0 else 0,
                'p_down':0.15,'p_none':0.45,'p_up':0.40,
                'spot_price':80.0,'p_spike_high':0.15,'p_spike_low':0.05,
            })
    return pd.DataFrame(rows)


def test_v6_multiwindow_blending_and_meta_features():
    state=learn_multiwindow_state({24:_frame(24,8.0),168:_frame(168,4.0),336:_frame(336,3.0)})
    assert len(state.blended_steps)==40
    assert state.blended_steps['1'].price_bias>0
    feat=pd.DataFrame({'step':range(1,41)})
    out=attach_adaptive_meta_features(feat,state)
    assert 'online_bias_24h' in out.columns
    assert 'online_bias_168h' in out.columns
    assert 'online_bias_336h' in out.columns
    assert 'online_blended_bias' in out.columns


def test_v6_drift_detection_uses_fast_vs_stable_window():
    state=learn_multiwindow_state({24:_frame(24,20.0),168:_frame(168,6.0),336:_frame(336,4.0)})
    assert state.drift_detected
    assert state.drift_score>0


def test_v6_adaptive_bias_changes_p50_but_keeps_quantiles_ordered():
    state=learn_multiwindow_state({24:_frame(24,10.0),168:_frame(168,5.0),336:_frame(336,4.0)})
    base=pd.DataFrame({
        'p10':np.full(40,70.0),'p50':np.full(40,90.0),'p90':np.full(40,120.0),
        'p_up':np.full(40,.4),'p_down':np.full(40,.2),'p_none':np.full(40,.4),
        'p_spike_high':np.full(40,.1),'p_spike_low':np.full(40,.05),
        'p50_lgb':np.full(40,88.0),'p50_cat':np.full(40,92.0),'reliability':np.full(40,.8),
    })
    out=apply_adaptive(base,pd.Series(range(1,41)),state)
    assert (out.p50!=base.p50).any()
    assert ((out.p10<=out.p50)&(out.p50<=out.p90)).all()
    assert ((out[['p_up','p_down','p_none']].sum(axis=1)-1).abs()<1e-8).all()


def test_v6_live_feature_surface_contains_reserve_stress_and_weather_error_columns():
    origin=pd.DatetimeIndex([pd.Timestamp('2026-10-20T12:00:00Z')])
    feat=build_features(origin,include_labels=False,include_online_context=True)
    assert len(feat)==40
    assert 'reserve_stress_index' in feat.columns
    assert 'weather_error_wind_generation_proxy' in feat.columns
    assert 'weather_error_load_proxy' in feat.columns

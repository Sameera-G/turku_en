from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ONLINE_MARKET_STATE_JSON, ONLINE_OUTPUTS_DIR
from .adaptive import AdaptiveState


def _finite(x):
    try:
        v=float(x)
        return None if not np.isfinite(v) else v
    except Exception:
        return None


def build_market_state(features: pd.DataFrame, predictions: pd.DataFrame, state: AdaptiveState, context: dict | None = None) -> dict:
    context=context or {}
    first=features.iloc[0] if not features.empty else pd.Series(dtype=float)
    pred=predictions.reset_index(drop=True)
    reserve=_finite(first.get('reserve_stress_index'))
    cross=_finite(first.get('cross_border_stress'))
    system=_finite(first.get('live_system_stress_proxy'))
    wind_err=_finite(first.get('weather_error_wind_generation_proxy'))
    load_err=_finite(first.get('weather_error_load_proxy'))
    if reserve is None: pressure='unknown'
    elif reserve > 0.20: pressure='upward / shortage pressure'
    elif reserve < -0.20: pressure='downward / surplus pressure'
    else: pressure='balanced / mixed'
    reliability=float(pred['reliability'].min()) if 'reliability' in pred and len(pred) else None
    twin={
        'generated_at_utc':pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
        'forecast_origin_utc':str(features.iloc[0]['forecast_origin_utc']) if len(features) else None,
        'market_pressure':pressure,
        'reserve_stress_index':reserve,
        'cross_border_stress':cross,
        'live_system_stress_proxy':system,
        'wind_generation_forecast_error_proxy_mw':wind_err,
        'load_forecast_error_proxy_mw':load_err,
        'adaptive_mae_24h':state.windows.get('24').rolling_mae if state.windows.get('24') else None,
        'adaptive_mae_7d':state.windows.get('168').rolling_mae if state.windows.get('168') else None,
        'adaptive_mae_14d':state.windows.get('336').rolling_mae if state.windows.get('336') else None,
        'drift_score':state.drift_score,
        'drift_detected':state.drift_detected,
        'drift_reasons':state.drift_reasons,
        'live_calibration_samples':int(state.blended_global.n),
        'live_conformal_expand_eur':float(state.blended_global.conformal_expand),
        'live_empirical_coverage':state.blended_global.empirical_coverage,
        'minimum_forecast_reliability':reliability,
        'max_high_spike_probability':float(pred['p_spike_high'].max()) if 'p_spike_high' in pred and len(pred) else None,
        'max_low_spike_probability':float(pred['p_spike_low'].max()) if 'p_spike_low' in pred and len(pred) else None,
        'max_up_probability':float(pred['p_up'].max()) if 'p_up' in pred and len(pred) else None,
        'max_down_probability':float(pred['p_down'].max()) if 'p_down' in pred and len(pred) else None,
        'source_health':context.get('source_health',{}),
        'day_ahead_refresh_window':bool(context.get('day_ahead_refresh_window',False)),
    }
    return twin


def save_market_state(twin: dict, origin: pd.Timestamp) -> Path:
    ONLINE_MARKET_STATE_JSON.parent.mkdir(parents=True,exist_ok=True)
    ONLINE_MARKET_STATE_JSON.write_text(json.dumps(twin,indent=2,default=str),encoding='utf-8')
    hist=ONLINE_OUTPUTS_DIR/'history'
    hist.mkdir(parents=True,exist_ok=True)
    stamp=pd.Timestamp(origin).strftime('%Y%m%dT%H%MZ')
    path=hist/f'market_state_{stamp}.json'
    path.write_text(json.dumps(twin,indent=2,default=str),encoding='utf-8')
    return path

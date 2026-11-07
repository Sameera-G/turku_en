from __future__ import annotations

import json
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/'.env')
sys.path.insert(0,str(ROOT))

from src.config import LIVE_SOURCE_HEALTH_JSON, ONLINE_OUTPUTS_DIR, ONLINE_MARKET_STATE_JSON, ONLINE_CHAMPION, MODELS_DIR
from src.online.adaptive import AdaptiveState
from src.online.store import OnlineStore
from src.performance import official_validation_summary, latest_daily_report
from src.online.continuous import default_model_path
from src.models_v8 import load_any_bundle

status=OnlineStore().status()
try:
    active_path=default_model_path()
    active_bundle=load_any_bundle(active_path)
    status['active_model']={
        'path':str(active_path),
        'version':getattr(active_bundle,'version',active_path.stem),
        'is_online_champion':bool(active_path == ONLINE_CHAMPION),
        'v10_refit_available':bool((MODELS_DIR/'v10'/'model_bundle_v10.joblib').exists()),
        'v11_candidate_available':bool((MODELS_DIR/'v11'/'model_bundle_v11_candidate.joblib').exists()),
    }
except Exception as exc:
    status['active_model']={'error':str(exc)}
status['official_validation']=official_validation_summary()
status['latest_daily_learning']=latest_daily_report()
state=AdaptiveState.load()
status['adaptive_state']={
    'updated_at_utc':state.updated_at_utc,
    'windows':{k:{'hours':v.hours,'samples':v.samples,'rolling_mae':v.rolling_mae,'rolling_coverage':v.rolling_coverage,'adapted_horizons':len(v.steps)} for k,v in state.windows.items()},
    'blended_horizons':len(state.blended_steps),
    'pooled_live_calibration':{
        'samples':state.blended_global.n,
        'empirical_coverage':state.blended_global.empirical_coverage,
        'shrunk_coverage_for_reliability':state.blended_global.shrunk_coverage,
        'conformal_expand_eur':state.blended_global.conformal_expand,
        'conformal_expand_low_eur':state.blended_global.conformal_expand_low,
        'conformal_expand_high_eur':state.blended_global.conformal_expand_high,
        'target_coverage':0.80,
    },
    'drift_score':state.drift_score,
    'drift_detected':state.drift_detected,
    'drift_reasons':state.drift_reasons,
}
if LIVE_SOURCE_HEALTH_JSON.exists():
    status['source_health']=json.loads(LIVE_SOURCE_HEALTH_JSON.read_text(encoding='utf-8'))
if ONLINE_MARKET_STATE_JSON.exists():
    status['market_state']=json.loads(ONLINE_MARKET_STATE_JSON.read_text(encoding='utf-8'))
hist=ONLINE_OUTPUTS_DIR/'history'
status['archived_forecast_batches']=len(list(hist.glob('forecast_*.csv'))) if hist.exists() else 0
print(json.dumps(status,indent=2,default=str))

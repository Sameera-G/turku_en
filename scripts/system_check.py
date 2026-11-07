from __future__ import annotations

import json
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/'.env')
sys.path.insert(0,str(ROOT))

from src.config import (
    FINGRID_API_KEY, ENTSOE_API_TOKEN, INPUTS_PARQUET, INPUTS_CSV, SPOT_CSV,
    MODELS_DIR, ONLINE_OUTPUTS_DIR, ONLINE_USE_PUBLIC_SPOT_FALLBACK,
)
from src.online.continuous import default_model_path

checks={}
checks['fingrid_api_key_configured']=bool(FINGRID_API_KEY)
checks['entsoe_api_token_configured']=bool(ENTSOE_API_TOKEN)
checks['public_spot_fallback_enabled']=bool(ONLINE_USE_PUBLIC_SPOT_FALLBACK)
checks['starter_inputs_present']=INPUTS_PARQUET.exists() or INPUTS_CSV.exists()
checks['licensed_spot_training_file_present']=SPOT_CSV.exists()
try:
    checks['default_model']=str(default_model_path())
    checks['model_ready']=True
except Exception as exc:
    checks['model_ready']=False; checks['model_error']=str(exc)
checks['online_output_dir']=str(ONLINE_OUTPUTS_DIR)
checks['ready_for_offline_backtest']=checks['starter_inputs_present'] and checks['licensed_spot_training_file_present'] and checks['model_ready']
checks['ready_for_live_core']=checks['fingrid_api_key_configured'] and checks['model_ready']
checks['ready_for_v8_spread_live']=checks['ready_for_live_core'] and (checks['entsoe_api_token_configured'] or checks['public_spot_fallback_enabled'])
checks['ready_for_full_live_sources']=checks['ready_for_live_core'] and checks['entsoe_api_token_configured']
print(json.dumps(checks,indent=2))
if not checks['model_ready']:
    raise SystemExit(2)

from __future__ import annotations
import subprocess,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import MODELS_DIR,OUTPUTS_DIR,FORECAST_ORIGINS_CSV,VALIDATOR,STARTER_DIR
from src.features import build_features,business_driver_text
from src.models_v8 import ModelBundleV8,predict_v8_bundle

def main():
 model_path=MODELS_DIR/'v10'/'model_bundle_v10.joblib'
 if not model_path.exists(): raise FileNotFoundError('Run: python scripts/train_v10_production.py')
 b=ModelBundleV8.load(model_path); origins=pd.read_csv(FORECAST_ORIGINS_CSV); oi=pd.DatetimeIndex(pd.to_datetime(origins['forecast_origin_utc'],utc=True))
 print(f'[V10] Building legal walk-forward features for {len(oi)} hidden-test origins ...',flush=True)
 feat=build_features(oi,include_labels=False); pred=predict_v8_bundle(b,feat)
 out=pd.DataFrame({'forecast_origin_utc':feat.forecast_origin_utc.dt.strftime('%Y-%m-%dT%H:%M:%SZ'),'delivery_start_utc':feat.delivery_start_utc.dt.strftime('%Y-%m-%dT%H:%M:%SZ'),'step':feat.step.astype(int),'p10':pred.p10,'p50':pred.p50,'p90':pred.p90,'p_up':pred.p_up,'p_down':pred.p_down,'p_none':pred.p_none,'p_spike_high':pred.p_spike_high,'p_spike_low':pred.p_spike_low,'data_cutoff_utc':feat.forecast_origin_utc.dt.strftime('%Y-%m-%dT%H:%M:%SZ'),'top_drivers':business_driver_text(feat)})
 od=OUTPUTS_DIR/'v10';od.mkdir(parents=True,exist_ok=True);path=od/'forecast_backtest_v10.csv';out.to_csv(path,index=False)
 pd.DataFrame({'forecast_origin_utc':out.forecast_origin_utc,'delivery_start_utc':out.delivery_start_utc,'step':out.step,'p50_direct':pred.p50_direct,'p50_spread':pred.p50_spread,'model_disagreement':pred.model_disagreement,'reliability':pred.reliability}).to_csv(od/'forecast_diagnostics_v10.csv',index=False)
 print(f'[V10] wrote {path} ({len(out):,} rows)',flush=True)
 subprocess.run([sys.executable,str(VALIDATOR),'--backtest',str(path),'--origins',str(FORECAST_ORIGINS_CSV)],cwd=STARTER_DIR,check=True)
 print('[V10] submission is structurally valid. Hidden MAE is unknown until organizer actuals exist.')
if __name__=='__main__':main()

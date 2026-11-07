from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, OUTPUTS_DIR, FORECAST_ORIGINS_CSV, VALIDATOR, STARTER_DIR
from src.features import build_features, business_driver_text
from src.models_v8 import ModelBundleV8, predict_v8_bundle


def main() -> None:
    model_path = MODELS_DIR / "v8" / "model_bundle_v8.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Run: python scripts/train_v8.py")
    bundle = ModelBundleV8.load(model_path)
    origins = pd.read_csv(FORECAST_ORIGINS_CSV)
    oi = pd.DatetimeIndex(pd.to_datetime(origins["forecast_origin_utc"], utc=True))
    print(f"[V8] Building legal walk-forward features for {len(oi)} hidden-test origins ...", flush=True)
    feat = build_features(oi, include_labels=False)
    pred = predict_v8_bundle(bundle, feat)
    out = pd.DataFrame({
        "forecast_origin_utc": feat["forecast_origin_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "delivery_start_utc": feat["delivery_start_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "step": feat["step"].astype(int),
        "p10": pred["p10"], "p50": pred["p50"], "p90": pred["p90"],
        "p_up": pred["p_up"], "p_down": pred["p_down"], "p_none": pred["p_none"],
        "p_spike_high": pred["p_spike_high"], "p_spike_low": pred["p_spike_low"],
        "data_cutoff_utc": feat["forecast_origin_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "top_drivers": business_driver_text(feat),
    })
    od = OUTPUTS_DIR / "v8"; od.mkdir(parents=True, exist_ok=True)
    path = od / "forecast_backtest_v8.csv"
    out.to_csv(path, index=False)
    diag = pd.DataFrame({
        "forecast_origin_utc": out["forecast_origin_utc"],
        "delivery_start_utc": out["delivery_start_utc"], "step": out["step"],
        "p50_direct": pred["p50_direct"], "p50_spread": pred["p50_spread"],
        "p50_spread_gated": pred["p50_spread_gated"],
        "spot_available": pred["v8_spot_available"],
        "model_disagreement": pred["model_disagreement"], "reliability": pred["reliability"],
    })
    diag.to_csv(od / "forecast_diagnostics_v8.csv", index=False)
    print(f"[V8] wrote {path} ({len(out):,} rows)")
    subprocess.run([
        sys.executable, str(VALIDATOR), "--backtest", str(path),
        "--origins", str(FORECAST_ORIGINS_CSV)
    ], cwd=STARTER_DIR, check=True)
    print("[V8] submission is structurally valid.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FORECAST_ORIGINS_CSV, SAMPLE_SUBMISSION_CSV, MODELS_DIR, OUTPUTS_DIR, VALIDATOR
from src.features import build_features, business_driver_text
from src.models import ModelBundleV3, predict_bundle


def fmt_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    path = MODELS_DIR / "v3" / "model_bundle_v3.joblib"
    bundle = ModelBundleV3.load(path)
    origins = pd.read_csv(FORECAST_ORIGINS_CSV)
    origin_idx = pd.DatetimeIndex(pd.to_datetime(origins["forecast_origin_utc"], utc=True))

    print(f"[V3] Building legal walk-forward features for {len(origin_idx)} hidden-test origins ...")
    feat = build_features(origin_idx, include_labels=False)
    pred = predict_bundle(bundle, feat, calibrated=True)
    drivers = business_driver_text(feat)

    out = pd.DataFrame({
        "forecast_origin_utc": fmt_utc(feat["forecast_origin_utc"]),
        "delivery_start_utc": fmt_utc(feat["delivery_start_utc"]),
        "step": feat["step"].astype(int),
        "p10": pred["p10"], "p50": pred["p50"], "p90": pred["p90"],
        "p_up": pred["p_up"], "p_down": pred["p_down"], "p_none": pred["p_none"],
        "p_spike_high": pred["p_spike_high"], "p_spike_low": pred["p_spike_low"],
        "data_cutoff_utc": fmt_utc(feat["forecast_origin_utc"]),
        "top_drivers": drivers,
    })
    official_cols = pd.read_csv(SAMPLE_SUBMISSION_CSV, nrows=0).columns.tolist()
    out = out[official_cols]

    out_dir = OUTPUTS_DIR / "v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / "forecast_backtest_v3.csv"
    out.to_csv(outfile, index=False)
    print(f"[V3] wrote {outfile} ({len(out):,} rows)")

    # Internal reliability output for dashboard/API. Not part of official schema.
    diagnostic = pd.DataFrame({
        "forecast_origin_utc": fmt_utc(feat["forecast_origin_utc"]),
        "delivery_start_utc": fmt_utc(feat["delivery_start_utc"]),
        "step": feat["step"].astype(int),
        "p50_lgb": pred["p50_lgb"],
        "p50_cat": pred["p50_cat"],
        "fusion_weight_lgb": pred["fusion_weight_lgb"],
        "model_disagreement": pred["model_disagreement"],
        "interval_width": pred["interval_width"],
        "missing_fraction": pred["missing_fraction"],
        "ood_fraction": pred["ood_fraction"],
        "reliability": pred["reliability"],
    })
    diagnostic.to_csv(out_dir / "forecast_diagnostics_v3.csv", index=False)

    print("[V3] Running locked organizer validator ...")
    cp = subprocess.run([
        sys.executable, str(VALIDATOR), "--backtest", str(outfile), "--origins", str(FORECAST_ORIGINS_CSV)
    ], text=True)
    if cp.returncode != 0:
        raise SystemExit(cp.returncode)
    print("[V3] submission is structurally valid.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FORECAST_ORIGINS_CSV, SAMPLE_SUBMISSION_CSV, MODELS_DIR, OUTPUTS_DIR, VALIDATOR
from src.features import build_features
from src.models import ModelBundleV2, predict_bundle


def fmt_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def top_drivers(bundle: ModelBundleV2, n=6) -> str:
    imp = getattr(bundle.q50, "feature_importances_", None)
    if imp is None:
        return "calibrated ensemble features"
    pairs = sorted(zip(bundle.feature_names, imp), key=lambda x: x[1], reverse=True)[:n]
    return "; ".join(f for f, _ in pairs)


def main() -> None:
    path = MODELS_DIR / "v2" / "model_bundle_v2.joblib"
    bundle = ModelBundleV2.load(path)
    origins = pd.read_csv(FORECAST_ORIGINS_CSV)
    origin_idx = pd.DatetimeIndex(pd.to_datetime(origins["forecast_origin_utc"], utc=True))

    # This single vectorized call is equivalent to walk-forward replay because every feature lookup
    # is independently masked by first_usable_origin_utc <= that row's forecast_origin_utc.
    print(f"[V2] Building legal walk-forward features for {len(origin_idx)} hidden-test origins ...")
    feat = build_features(origin_idx, include_labels=False)
    pred = predict_bundle(bundle, feat, calibrated=True)

    drivers = top_drivers(bundle)
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

    out_dir = OUTPUTS_DIR / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / "forecast_backtest_v2.csv"
    out.to_csv(outfile, index=False)
    print(f"[V2] wrote {outfile} ({len(out):,} rows)")

    print("[V2] Running locked organizer validator ...")
    cp = subprocess.run([
        sys.executable, str(VALIDATOR), "--backtest", str(outfile), "--origins", str(FORECAST_ORIGINS_CSV)
    ], text=True)
    if cp.returncode != 0:
        raise SystemExit(cp.returncode)
    print("[V2] submission is structurally valid.")


if __name__ == "__main__":
    main()

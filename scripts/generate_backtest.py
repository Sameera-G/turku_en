from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FORECAST_ORIGINS_CSV, SAMPLE_SUBMISSION_CSV, MODELS_DIR, OUTPUTS_DIR, VALIDATOR
from src.features import build_features
from src.models import ModelBundle, predict_bundle


def top_drivers(bundle: ModelBundle, n=5) -> str:
    imp = bundle.q50.feature_importances_
    pairs = sorted(zip(bundle.feature_names, imp), key=lambda x: x[1], reverse=True)[:n]
    return "; ".join(f for f, _ in pairs)


def fmt_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    bundle = ModelBundle.load(MODELS_DIR / "model_bundle.joblib")
    origins = pd.read_csv(FORECAST_ORIGINS_CSV)
    ocol = "forecast_origin_utc" if "forecast_origin_utc" in origins.columns else origins.columns[0]
    origin_idx = pd.DatetimeIndex(pd.to_datetime(origins[ocol], utc=True))
    feat = build_features(origin_idx, include_labels=False)
    pred = predict_bundle(bundle, feat)

    out = pd.DataFrame({
        "forecast_origin_utc": fmt_utc(feat["forecast_origin_utc"]),
        "delivery_start_utc": fmt_utc(feat["delivery_start_utc"]),
        "step": feat["step"].astype(int),
        "p10": pred["p10"], "p50": pred["p50"], "p90": pred["p90"],
        "p_up": pred["p_up"], "p_down": pred["p_down"], "p_none": pred["p_none"],
        "p_spike_high": pred["p_spike_high"], "p_spike_low": pred["p_spike_low"],
        "data_cutoff_utc": fmt_utc(feat["forecast_origin_utc"]),
        "top_drivers": top_drivers(bundle),
    })
    # Match official column order exactly.
    official_cols = pd.read_csv(SAMPLE_SUBMISSION_CSV, nrows=0).columns.tolist()
    out = out[official_cols]
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / "forecast_backtest.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out):,} rows)")

    print("Running official validator ...")
    cp = subprocess.run([
        sys.executable, str(VALIDATOR), "--backtest", str(path), "--origins", str(FORECAST_ORIGINS_CSV)
    ], text=True)
    raise SystemExit(cp.returncode)


if __name__ == "__main__":
    main()

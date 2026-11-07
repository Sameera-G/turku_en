from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    TEST_START_UTC, VALIDATION_START_UTC, TRAIN_LABEL_START_UTC,
    MODELS_DIR, REPORTS_DIR, STARTER_DIR, SCORER, VALIDATOR, MAKE_TEMPLATES,
    ACTUALS_VALIDATION_CSV, SPIKE_MARGIN,
)
from src.features import build_features, feature_columns
from src.models import train_base, fit_calibrator, predict_bundle


def hourly_origins(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.ceil("h"), end.floor("h"), freq="1h", tz="UTC")


def pinball(y, pred, q):
    e = y - pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def three_brier(y, p):
    o = np.zeros_like(p)
    o[np.arange(len(y)), y.astype(int)] = 1
    return float(np.mean(np.sum((p-o)**2, axis=1)))


def metrics(val: pd.DataFrame, pred: pd.DataFrame) -> dict:
    y = val["y_price"].to_numpy(float)
    return {
        "mae": float(np.mean(np.abs(y - pred["p50"].to_numpy()))),
        "pinball_mean": (pinball(y, pred["p10"].to_numpy(), .1) + pinball(y, pred["p90"].to_numpy(), .9)) / 2,
        "coverage": float(np.mean((y >= pred["p10"].to_numpy()) & (y <= pred["p90"].to_numpy()))),
        "spike_brier_mean": float(((pred["p_spike_high"].to_numpy()-val["y_spike_high"].to_numpy())**2).mean()/2 + ((pred["p_spike_low"].to_numpy()-val["y_spike_low"].to_numpy())**2).mean()/2),
        "direction_brier": three_brier(val["y_direction"].to_numpy(int), pred[["p_down","p_none","p_up"]].to_numpy(float)),
    }


def write_validation_submission(val: pd.DataFrame, pred: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "forecast_origin_utc": val["forecast_origin_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "delivery_start_utc": val["delivery_start_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "step": val["step"].astype(int),
        "p10": pred["p10"], "p50": pred["p50"], "p90": pred["p90"],
        "p_up": pred["p_up"], "p_down": pred["p_down"], "p_none": pred["p_none"],
        "p_spike_high": pred["p_spike_high"], "p_spike_low": pred["p_spike_low"],
        "data_cutoff_utc": val["forecast_origin_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "top_drivers": "V2 calibrated LightGBM; walk-forward legal features",
    })
    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit-all", action="store_true", help="Refit base models on all allowed pre-test labels after validation. Slower; use for final candidate runs.")
    args = parser.parse_args()
    test_start = pd.Timestamp(TEST_START_UTC)
    val_start = pd.Timestamp(VALIDATION_START_UTC)
    train_start = pd.Timestamp(TRAIN_LABEL_START_UTC)

    # Build all pre-test examples. The feature builder itself enforces row-level availability at each origin.
    origins = hourly_origins(train_start, test_start - pd.Timedelta(hours=1))
    print(f"[V2] Building organizer-aware feature matrix for {len(origins)} hourly origins ...")
    ds = build_features(origins, include_labels=True)
    ds = ds[(ds["delivery_start_utc"] >= train_start) & (ds["delivery_start_utc"] < test_start)]
    ds = ds.dropna(subset=["y_price", "y_direction", "y_spike_high", "y_spike_low"]).reset_index(drop=True)
    feats = feature_columns(ds)

    fit = ds[ds["delivery_start_utc"] < val_start].reset_index(drop=True)
    val = ds[(ds["delivery_start_utc"] >= val_start) & (ds["delivery_start_utc"] < test_start)].reset_index(drop=True)
    print(f"[V2] fit rows={len(fit):,}; val rows={len(val):,}; features={len(feats)}")

    # Stage 1: proper held-out model for calibration and honest validation.
    dev = train_base(fit, feats)
    raw = predict_bundle(dev, val, calibrated=False)
    raw_m = metrics(val, raw)
    print("[V2] raw validation:", json.dumps(raw_m, indent=2))

    dev.calibrator = fit_calibrator(dev, val)
    calibrated = predict_bundle(dev, val, calibrated=True)
    cal_m = metrics(val, calibrated)
    print("[V2] calibrated validation:", json.dumps(cal_m, indent=2))

    report_dir = REPORTS_DIR / "v2"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"variant":"raw", **raw_m}, {"variant":"calibrated", **cal_m}]).to_csv(
        report_dir / "validation_metrics.csv", index=False
    )

    by_h = []
    for step, g in val.groupby("step"):
        idx = g.index.to_numpy()
        m = metrics(g.reset_index(drop=True), calibrated.iloc[idx].reset_index(drop=True))
        by_h.append({"step": int(step), "horizon_hours": (int(step)-1)*0.25, **m})
    pd.DataFrame(by_h).to_csv(report_dir / "validation_by_horizon.csv", index=False)

    # Build official validation template to verify schema/scoring using organizer scripts.
    validation_dir = report_dir / "official_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(MAKE_TEMPLATES), "--test-start", "2026-09-07", "--test-end", "2026-10-04",
        "--output-dir", str(validation_dir)
    ], cwd=STARTER_DIR, check=True)

    official_origins = pd.read_csv(validation_dir / "forecast_origins.csv")
    origin_idx = pd.DatetimeIndex(pd.to_datetime(official_origins["forecast_origin_utc"], utc=True))
    vfeat = build_features(origin_idx, include_labels=False)
    vpred = predict_bundle(dev, vfeat, calibrated=True)
    submission = validation_dir / "forecast_backtest.csv"
    write_validation_submission(vfeat, vpred, submission)

    print("[V2] Running organizer validator on validation replay ...")
    subprocess.run([
        sys.executable, str(VALIDATOR), "--backtest", str(submission), "--origins", str(validation_dir / "forecast_origins.csv")
    ], cwd=STARTER_DIR, check=True)

    score_dir = validation_dir / "score"
    print("[V2] Running organizer scorer on validation replay ...")
    subprocess.run([
        sys.executable, str(SCORER), "--submission", str(submission),
        "--actuals", str(ACTUALS_VALIDATION_CSV), "--spike-margin", str(int(SPIKE_MARGIN)),
        "--output-dir", str(score_dir)
    ], cwd=STARTER_DIR, check=True)

    # Stage 2: save the honest validation-trained candidate by default for fast iteration.
    # For a final candidate, --refit-all fits base models on all allowed labels through 4 Oct while
    # preserving calibration parameters learned out-of-sample above. Model parameters are still frozen in test replay.
    if args.refit_all:
        print("[V2] --refit-all: fitting fixed final model on all allowed pre-test labels ...")
        final = train_base(ds, feats)
        final.calibrator = dev.calibrator
    else:
        print("[V2] Fast mode: saving held-out calibrated model. Use --refit-all for a final candidate.")
        final = dev
    model_path = MODELS_DIR / "v2" / "model_bundle_v2.joblib"
    final.save(model_path)
    print(f"[V2] saved {model_path}")

    # Persist calibration summary for auditability.
    (report_dir / "calibration_summary.json").write_text(json.dumps({
        "q10_delta": final.calibrator.q10_delta,
        "q90_delta": final.calibrator.q90_delta,
        "dir_alpha": final.calibrator.dir_alpha,
        "hi_alpha": final.calibrator.hi_alpha,
        "lo_alpha": final.calibrator.lo_alpha,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    REPORTS_DIR, MODELS_DIR, STARTER_DIR, VALIDATOR, SCORER, MAKE_TEMPLATES,
    ACTUALS_VALIDATION_CSV, SPIKE_MARGIN,
)
from src.features import build_features, business_driver_text
from src.models import ModelBundleV3, HorizonCalibratorV3, fit_calibrator, predict_bundle, _select_cat_features


def load_bundle(cache: Path) -> ModelBundleV3:
    features = json.loads((cache / "features.json").read_text(encoding="utf-8"))
    env = json.loads((cache / "envelope.json").read_text(encoding="utf-8"))
    def m(name: str):
        p = cache / f"{name}.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run the V3 worker for {name}.")
        return joblib.load(p)
    return ModelBundleV3(
        feature_names=features,
        cat_feature_names=_select_cat_features(features),
        q10_lgb=m("q10"), q50_lgb=m("q50"), q90_lgb=m("q90"), q50_cat=m("cat50"),
        direction=m("direction"), spike_high=m("spike_high"), spike_low=m("spike_low"),
        feature_q01=env["q01"], feature_q99=env["q99"],
    )


def pinball(y, p, q):
    e = y - p
    return float(np.mean(np.maximum(q*e, (q-1)*e)))


def metrics(val: pd.DataFrame, pred: pd.DataFrame) -> dict:
    y = val["y_price"].to_numpy(float)
    yy = val["y_direction"].to_numpy(int)
    pp = pred[["p_down", "p_none", "p_up"]].to_numpy(float)
    o = np.zeros_like(pp); o[np.arange(len(yy)), yy] = 1
    return {
        "mae": float(np.mean(np.abs(y-pred["p50"].to_numpy(float)))),
        "pinball_mean": (pinball(y,pred["p10"].to_numpy(float),.1)+pinball(y,pred["p90"].to_numpy(float),.9))/2,
        "coverage": float(np.mean((y>=pred["p10"].to_numpy(float))&(y<=pred["p90"].to_numpy(float)))),
        "spike_brier_mean": float((((pred["p_spike_high"].to_numpy()-val["y_spike_high"].to_numpy())**2).mean()+((pred["p_spike_low"].to_numpy()-val["y_spike_low"].to_numpy())**2).mean())/2),
        "direction_brier": float(np.mean(np.sum((pp-o)**2,axis=1))),
        "mean_reliability": float(pred["reliability"].mean()),
        "mean_model_disagreement": float(pred["model_disagreement"].mean()),
    }


def write_validation_submission(feat: pd.DataFrame, pred: pd.DataFrame, path: Path) -> None:
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
    out.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    report = REPORTS_DIR / "v3"

    if args.final:
        final = load_bundle(report / "train_cache_final")
        dev_path = MODELS_DIR / "v3" / "model_bundle_v3_dev.joblib"
        if not dev_path.exists():
            raise FileNotFoundError("Run the normal V3 development pipeline first; final refit reuses its held-out calibration.")
        dev = ModelBundleV3.load(dev_path)
        final.calibrator = dev.calibrator
        out = MODELS_DIR / "v3" / "model_bundle_v3.joblib"
        final.save(out)
        print(f"[V3 finalize] final refit bundle saved: {out}")
        return

    bundle = load_bundle(report / "train_cache_dev")
    val = pd.read_pickle(report / "validation_frame.pkl").reset_index(drop=True)
    raw = predict_bundle(bundle, val, calibrated=False)
    raw_m = metrics(val, raw)
    bundle.calibrator = fit_calibrator(bundle, val)
    cal = predict_bundle(bundle, val, calibrated=True)
    cal_m = metrics(val, cal)
    print("[V3 finalize] raw:", json.dumps(raw_m, indent=2))
    print("[V3 finalize] calibrated:", json.dumps(cal_m, indent=2))
    pd.DataFrame([{"variant":"raw",**raw_m},{"variant":"calibrated",**cal_m}]).to_csv(report/"validation_metrics.csv",index=False)

    by_h=[]
    for step,g in val.groupby("step"):
        idx=g.index.to_numpy()
        mm=metrics(g.reset_index(drop=True),cal.iloc[idx].reset_index(drop=True))
        by_h.append({"step":int(step),"horizon_hours":(int(step)-1)*.25,"p50_lgb_weight":bundle.calibrator.p50_lgb_weight.get(int(step),np.nan),**mm})
    pd.DataFrame(by_h).to_csv(report/"validation_by_horizon.csv",index=False)

    pd.DataFrame({"feature":bundle.feature_names,"importance":bundle.q50_lgb.feature_importances_}).sort_values("importance",ascending=False).to_csv(report/"feature_importance_lgb.csv",index=False)
    pd.DataFrame({"feature":bundle.cat_feature_names,"importance":bundle.q50_cat.get_feature_importance()}).sort_values("importance",ascending=False).to_csv(report/"feature_importance_catboost.csv",index=False)

    validation_dir=report/"official_validation"; validation_dir.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(MAKE_TEMPLATES),"--test-start","2026-09-07","--test-end","2026-10-04","--output-dir",str(validation_dir)],cwd=STARTER_DIR,check=True)
    o=pd.read_csv(validation_dir/"forecast_origins.csv")
    feat=build_features(pd.DatetimeIndex(pd.to_datetime(o["forecast_origin_utc"],utc=True)),include_labels=False)
    pred=predict_bundle(bundle,feat,calibrated=True)
    sub=validation_dir/"forecast_backtest.csv"; write_validation_submission(feat,pred,sub)
    subprocess.run([sys.executable,str(VALIDATOR),"--backtest",str(sub),"--origins",str(validation_dir/"forecast_origins.csv")],cwd=STARTER_DIR,check=True)
    score_dir=validation_dir/"score"
    subprocess.run([sys.executable,str(SCORER),"--submission",str(sub),"--actuals",str(ACTUALS_VALIDATION_CSV),"--spike-margin",str(int(SPIKE_MARGIN)),"--output-dir",str(score_dir)],cwd=STARTER_DIR,check=True)

    dev_path=MODELS_DIR/"v3"/"model_bundle_v3_dev.joblib"; bundle.save(dev_path)
    final_path=MODELS_DIR/"v3"/"model_bundle_v3.joblib"; bundle.save(final_path)
    print(f"[V3 finalize] saved {final_path}")


if __name__ == "__main__":
    main()

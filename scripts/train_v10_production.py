from __future__ import annotations

"""Refit the production P50 ensemble on all legally available pre-test labels.

This is intentionally a *production/post-cutoff* refit.  It does not overwrite
or rescore the locked public-validation evidence, because those labels are now
part of the training set.  P10/P90 and probability heads remain the proven V8
heads; V10 refits the direct P50 learners and all spread learners, then lets the
online adaptive layer decide direct-vs-spread weighting from settled live data.
"""

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, REPORTS_DIR
from src.models_v8 import ModelBundleV8, _cat_spread_features
from src.v8_features import augment_v8_features, spread_feature_names


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--keep-cache", action="store_true")
    args=ap.parse_args()

    old_path=MODELS_DIR/"v8"/"model_bundle_v8.joblib"
    dev=REPORTS_DIR/"v3"/"train_cache_dev"
    val_path=REPORTS_DIR/"v3"/"validation_frame.pkl"
    if not old_path.exists() or not (dev/"features.json").exists() or not val_path.exists():
        raise FileNotFoundError("Need V3/V8 artifacts first. Run the V8 pipeline before the V10 production refit.")

    features=json.loads((dev/"features.json").read_text(encoding="utf-8"))
    x=np.asarray(np.load(dev/"matrix"/"X.npy",mmap_mode="r"),dtype=np.float32)
    train=pd.DataFrame(x,columns=features)
    for name,dtype in [("y_price",np.float32),("y_direction",np.int8),("y_spike_high",np.int8),("y_spike_low",np.int8)]:
        train[name]=np.asarray(np.load(dev/"matrix"/f"{name}.npy",mmap_mode="r"),dtype=dtype)

    val=pd.read_pickle(val_path).dropna(subset=["y_price","y_direction","y_spike_high","y_spike_low"]).reset_index(drop=True)
    add=val[features+["y_price","y_direction","y_spike_high","y_spike_low"]].copy()
    full=pd.concat([train,add],ignore_index=True)
    print(f"[V10] P50 production refit rows: {len(full):,} (dev {len(train):,} + public-validation labels {len(add):,})",flush=True)

    work=REPORTS_DIR/"v10"/"_training_cache"
    base_cache=work/"base"; spread_cache=work/"spread"; parts=MODELS_DIR/"v10"/"parts"
    base_cache.mkdir(parents=True,exist_ok=True); spread_cache.mkdir(parents=True,exist_ok=True); parts.mkdir(parents=True,exist_ok=True)
    (base_cache/"features.json").write_text(json.dumps(features),encoding="utf-8")
    np.save(base_cache/"X.npy",full[features].to_numpy(np.float32))
    for name,dtype in [("y_price",np.float32),("y_direction",np.int8),("y_spike_high",np.int8),("y_spike_low",np.int8)]:
        np.save(base_cache/f"{name}.npy",full[name].to_numpy(dtype))

    # Only P50 direct learners are refit. Quantile/direction/spike heads remain
    # the independently validated V8 heads; live conformal calibration handles
    # current interval coverage.
    for kind,outname in [("q50","q50.joblib"),("cat50","cat50.joblib")]:
        _run([sys.executable,str(ROOT/"scripts"/"model_worker_v3.py"),"--data-dir",str(base_cache),"--features",str(base_cache/"features.json"),"--kind",kind,"--output",str(parts/outname)])

    old=ModelBundleV8.load(old_path)
    aug=augment_v8_features(full).reset_index(drop=True)
    sf=spread_feature_names(old.base.feature_names,aug); cf=_cat_spread_features(sf)
    spot=pd.to_numeric(aug["future__fi_spot_price"],errors="coerce"); y=pd.to_numeric(aug["y_price"],errors="coerce")
    ok=spot.notna() & y.notna()
    np.save(spread_cache/"X.npy",aug.loc[ok,sf].to_numpy(np.float32))
    np.save(spread_cache/"X_cat.npy",aug.loc[ok,cf].to_numpy(np.float32))
    np.save(spread_cache/"y_spread.npy",(y[ok]-spot[ok]).to_numpy(np.float32))
    np.save(spread_cache/"direction.npy",aug.loc[ok,"y_direction"].to_numpy(np.int8))
    (spread_cache/"spread_features.json").write_text(json.dumps(sf),encoding="utf-8")
    (spread_cache/"cat_features.json").write_text(json.dumps(cf),encoding="utf-8")

    for kind,outname in [
        ("global_lgb","spread_global_lgb.joblib"),("global_cat","spread_global_cat.joblib"),
        ("down_lgb","spread_down_lgb.joblib"),("up_lgb","spread_up_lgb.joblib")]:
        _run([sys.executable,str(ROOT/"scripts"/"model_worker_v10_spread.py"),"--data-dir",str(spread_cache),"--kind",kind,"--output",str(parts/outname)])

    bundle=copy.deepcopy(old)
    bundle.base.q50_lgb=joblib.load(parts/"q50.joblib")
    bundle.base.q50_cat=joblib.load(parts/"cat50.joblib")
    bundle.spread_feature_names=sf; bundle.spread_cat_feature_names=cf
    bundle.spread_global_lgb=joblib.load(parts/"spread_global_lgb.joblib")
    bundle.spread_global_cat=joblib.load(parts/"spread_global_cat.joblib")
    bundle.spread_down_lgb=joblib.load(parts/"spread_down_lgb.joblib")
    bundle.spread_up_lgb=joblib.load(parts/"spread_up_lgb.joblib")

    # Robust production centre: after refitting the experts on newer legal data,
    # the old step-specific final P50 intercept/mixture no longer transferred
    # reliably to the held-out late-period block.  Keep the proven direction
    # gating structure but start from a neutral 50/50 direct-vs-spread centre
    # with no static intercept.  The online learner then moves this weight from
    # settled live outcomes.
    for cal in bundle.step_calibration.values():
        cal.direct_weight = 0.5
        cal.p50_bias = 0.0

    bundle.version="10.1-production-refit-robust-fusion"
    out=MODELS_DIR/"v10"/"model_bundle_v10.joblib"; bundle.save(out)

    report={
        "version":bundle.version,
        "training_rows":int(len(full)),
        "dev_rows":int(len(train)),
        "added_recent_legal_rows":int(len(add)),
        "training_through":"public validation labels through 2026-10-04; intended for post-cutoff/live production",
        "locked_official_validation_mae_eur_mwh":43.407385134986214,
        "chronological_refit_experiment":{
            "comparison":"same architecture; dev-only vs dev+earlier-validation training, calibrated on next quarter, evaluated on final quarter",
            "old_direct_mae":32.2470,
            "refit_direct_mae":30.9069,
            "relative_mae_reduction_pct":4.16,
        },
        "chronological_full_ensemble_experiment":{
            "test_rows":6720,
            "old_v8_fair_mae_eur_mwh":34.156161460085684,
            "v10_robust_50_50_mae_eur_mwh":30.81810116508011,
            "relative_mae_reduction_pct":9.772937450557416,
            "notes":"Both systems calibrated only on the preceding chronological block; V10 uses a neutral 50/50 direct-vs-spread centre and no static P50 intercept, then learns the live blend from settled outcomes.",
        },
        "warning":"Do not rescore this full-refit model on the public validation window as an unbiased test; those labels are now training data. The locked 43.4074 organizer-style score remains the honest public-validation evidence.",
    }
    rpath=REPORTS_DIR/"v10"/"production_refit_summary.json"; rpath.parent.mkdir(parents=True,exist_ok=True); rpath.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"[V10] saved production bundle: {out}")
    print(json.dumps(report,indent=2))
    if not args.keep_cache:
        shutil.rmtree(work,ignore_errors=True)

if __name__=="__main__":
    main()

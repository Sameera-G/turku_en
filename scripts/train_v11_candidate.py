from __future__ import annotations

"""Train a stronger P50 candidate without replacing the V10 champion.

The candidate is intentionally *not* promoted automatically. It is trained on
all legal labels through 2026-10-04, then `compare_v11_candidates.py` can score
it on later settled live snapshots. This keeps the production decision based on
out-of-training evidence rather than on the data used to fit the candidate.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, REPORTS_DIR
from src.models import _select_cat_features
from src.models_v8 import ModelBundleV8
from src.online.retrain import _strong_lgb, _strong_cat


def main() -> None:
    source = MODELS_DIR / "v10" / "model_bundle_v10.joblib"
    if not source.exists():
        raise FileNotFoundError("Missing V10 model. Install/train V10 first.")
    bundle = ModelBundleV8.load(source)

    cache = REPORTS_DIR / "v3" / "train_cache_dev"
    features = json.loads((cache / "features.json").read_text(encoding="utf-8"))
    x = np.asarray(np.load(cache / "matrix" / "X.npy", mmap_mode="r"), dtype=np.float32)
    train = pd.DataFrame(x, columns=features)
    train["y_price"] = np.asarray(np.load(cache / "matrix" / "y_price.npy", mmap_mode="r"), dtype=np.float32)
    val = pd.read_pickle(REPORTS_DIR / "v3" / "validation_frame.pkl").dropna(subset=["y_price"]).reset_index(drop=True)
    full = pd.concat([train, val[features + ["y_price"]]], ignore_index=True)

    cand = copy.deepcopy(bundle)
    lgb = _strong_lgb(); lgb.fit(full[features], full["y_price"])
    cat_features = _select_cat_features(features)
    cat = _strong_cat(); cat.fit(full[cat_features], full["y_price"])
    cand.base.q50_lgb = lgb
    cand.base.q50_cat = cat
    cand.base.cat_feature_names = cat_features

    # Conservative static recipe for the prebuilt candidate. It is merely a
    # challenger; live comparison/promotion is the safety gate.
    for step in range(1, 41):
        cand.base.calibrator.p50_lgb_weight[step] = 0.75
    for cal in cand.step_calibration.values():
        cal.direct_weight = 0.90
        cal.p50_bias = 0.0

    cand.version = "11.0-strong-p50-live-candidate"
    out = MODELS_DIR / "v11" / "model_bundle_v11_candidate.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    cand.save(out)
    report = {
        "version": cand.version,
        "training_rows": int(len(full)),
        "training_through": "2026-10-04 legal labels",
        "status": "challenger_only_not_default",
        "p50_recipe": "strong LGBM median 75% + strong CatBoost median 25%; V8 spread centre retained with 90% direct / 10% spread",
        "promotion_rule": "must beat current champion on later settled live evaluation with enough origins",
    }
    rp = REPORTS_DIR / "v11" / "prebuilt_candidate.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

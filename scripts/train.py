from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TEST_START_UTC, VALIDATION_START_UTC, MODELS_DIR, REPORTS_DIR
from src.features import build_features, feature_columns
from src.models import train_bundle, predict_bundle


def hourly_origins(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.ceil("h"), end.floor("h"), freq="1h", tz="UTC")


def pinball(y, pred, q):
    e = y - pred
    return np.mean(np.maximum(q * e, (q - 1) * e))


def main() -> None:
    test_start = pd.Timestamp(TEST_START_UTC)
    val_start = pd.Timestamp(VALIDATION_START_UTC)

    # Origins may start earlier, but every training label is later filtered to < test_start.
    origins = hourly_origins(pd.Timestamp("2026-06-01T00:00:00Z"), test_start - pd.Timedelta(hours=1))
    print(f"Building leakage-safe feature matrix for {len(origins)} origins ...")
    ds = build_features(origins, include_labels=True)
    ds = ds[ds["delivery_start_utc"] < test_start]
    ds = ds.dropna(subset=["y_price", "y_direction", "y_spike_high", "y_spike_low"]).reset_index(drop=True)
    feats = feature_columns(ds)

    fit = ds[ds["delivery_start_utc"] < val_start]
    val = ds[(ds["delivery_start_utc"] >= val_start) & (ds["delivery_start_utc"] < test_start)]
    print(f"Train rows={len(fit):,}; validation rows={len(val):,}; features={len(feats)}")

    bundle = train_bundle(fit, feats)
    pred = predict_bundle(bundle, val)
    y = val["y_price"].to_numpy()
    mae = mean_absolute_error(y, pred["p50"])
    pb = (pinball(y, pred["p10"].to_numpy(), .1) + pinball(y, pred["p90"].to_numpy(), .9)) / 2
    coverage = np.mean((y >= pred["p10"]) & (y <= pred["p90"]))
    high_brier = brier_score_loss(val["y_spike_high"], pred["p_spike_high"])
    low_brier = brier_score_loss(val["y_spike_low"], pred["p_spike_low"])
    print(f"Validation MAE={mae:.3f}; pinball={pb:.3f}; coverage={coverage:.3f}; spike_brier={(high_brier+low_brier)/2:.5f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "mae": mae,
        "pinball_mean": pb,
        "coverage": coverage,
        "spike_brier_mean": (high_brier + low_brier) / 2,
        "train_rows": len(fit),
        "validation_rows": len(val),
        "feature_count": len(feats),
    }]).to_csv(REPORTS_DIR / "mvp_validation_metrics.csv", index=False)

    # Final fit on all allowed labels through the training cutoff.
    print("Training final bundle on all allowed pre-test labels ...")
    final_bundle = train_bundle(ds, feats)
    path = MODELS_DIR / "model_bundle.joblib"
    final_bundle.save(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

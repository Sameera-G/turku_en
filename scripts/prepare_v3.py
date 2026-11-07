from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TEST_START_UTC, VALIDATION_START_UTC, TRAIN_LABEL_START_UTC, REPORTS_DIR
from src.features import build_features, feature_columns
from src.models import _feature_envelope


def hourly_origins(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.ceil("h"), end.floor("h"), freq="1h", tz="UTC")


def save_matrix(train: pd.DataFrame, features: list[str], cache: Path) -> None:
    matrix = cache / "matrix"
    matrix.mkdir(parents=True, exist_ok=True)
    np.save(matrix / "X.npy", train[features].to_numpy(dtype=np.float32, copy=True))
    np.save(matrix / "y_price.npy", train["y_price"].to_numpy(dtype=np.float32))
    np.save(matrix / "y_direction.npy", train["y_direction"].to_numpy(dtype=np.int8))
    np.save(matrix / "y_spike_high.npy", train["y_spike_high"].to_numpy(dtype=np.int8))
    np.save(matrix / "y_spike_low.npy", train["y_spike_low"].to_numpy(dtype=np.int8))
    (cache / "features.json").write_text(json.dumps(features), encoding="utf-8")
    q01, q99 = _feature_envelope(train, features)
    (cache / "envelope.json").write_text(json.dumps({"q01": q01, "q99": q99}), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="Prepare all allowed pre-test labels for final refit.")
    args = ap.parse_args()

    start = pd.Timestamp(TRAIN_LABEL_START_UTC)
    test = pd.Timestamp(TEST_START_UTC)
    val_start = pd.Timestamp(VALIDATION_START_UTC)
    origins = hourly_origins(start, test - pd.Timedelta(hours=1))

    print(f"[V3 prepare] building leakage-safe features for {len(origins)} origins ...", flush=True)
    ds = build_features(origins, include_labels=True)
    ds = ds[(ds["delivery_start_utc"] >= start) & (ds["delivery_start_utc"] < test)]
    ds = ds.dropna(subset=["y_price", "y_direction", "y_spike_high", "y_spike_low"]).reset_index(drop=True)
    features = feature_columns(ds)

    report = REPORTS_DIR / "v3"
    report.mkdir(parents=True, exist_ok=True)

    if args.final:
        train = ds
        cache = report / "train_cache_final"
        print(f"[V3 prepare] FINAL rows={len(train):,}; features={len(features)}", flush=True)
    else:
        train = ds[ds["delivery_start_utc"] < val_start].reset_index(drop=True)
        val = ds[(ds["delivery_start_utc"] >= val_start) & (ds["delivery_start_utc"] < test)].reset_index(drop=True)
        cache = report / "train_cache_dev"
        # Pickle is used only for the smaller validation frame; worker training uses NPY memmaps.
        val.to_pickle(report / "validation_frame.pkl")
        print(f"[V3 prepare] DEV rows={len(train):,}; validation={len(val):,}; features={len(features)}", flush=True)

    save_matrix(train, features, cache)
    manifest = {
        "mode": "final" if args.final else "dev",
        "train_rows": len(train),
        "features": len(features),
        "test_start_utc": TEST_START_UTC,
        "validation_start_utc": VALIDATION_START_UTC,
    }
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[V3 prepare] cache ready: {cache}", flush=True)


if __name__ == "__main__":
    main()

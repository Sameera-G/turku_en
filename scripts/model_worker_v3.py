from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import _reg, _cat50, _clf, _select_cat_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--kind", required=True, choices=["q10", "q50", "q90", "cat50", "direction", "spike_high", "spike_low"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    features = json.loads(Path(args.features).read_text(encoding="utf-8"))
    X = np.load(data_dir / "X.npy", mmap_mode="r")
    y_price = np.load(data_dir / "y_price.npy", mmap_mode="r")
    y_direction = np.load(data_dir / "y_direction.npy", mmap_mode="r")
    y_high = np.load(data_dir / "y_spike_high.npy", mmap_mode="r")
    y_low = np.load(data_dir / "y_spike_low.npy", mmap_mode="r")
    kind = args.kind

    if kind == "q10":
        model, x, y = _reg(.10), X, y_price
    elif kind == "q50":
        model, x, y = _reg(.50), X, y_price
    elif kind == "q90":
        model, x, y = _reg(.90), X, y_price
    elif kind == "cat50":
        cat_features = _select_cat_features(features)
        pos = {name: i for i, name in enumerate(features)}
        idx = np.array([pos[name] for name in cat_features], dtype=np.int32)
        model, x, y = _cat50(), X[:, idx], y_price
    elif kind == "direction":
        model, x, y = _clf(True), X, y_direction.astype(np.int8)
    elif kind == "spike_high":
        model, x, y = _clf(False), X, y_high.astype(np.int8)
    else:
        model, x, y = _clf(False), X, y_low.astype(np.int8)

    print(f"[V3 worker] training {kind}: rows={X.shape[0]:,} features={x.shape[1]}", flush=True)
    model.fit(x, y)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"[V3 worker] saved {out}", flush=True)


if __name__ == "__main__":
    main()

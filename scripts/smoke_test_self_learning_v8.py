from __future__ import annotations

"""Offline proof that V8 forecast -> actual -> evaluation -> adaptation works.

This script uses only the public validation frame and a temporary SQLite DB.  It
never touches the production online database and never uses hidden-test actuals.
"""

import argparse
from pathlib import Path
import tempfile
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import REPORTS_DIR, MODELS_DIR
from src.models_v8 import load_any_bundle, predict_any_bundle
from src.online.store import OnlineStore
from src.online.adaptive import learn_multiwindow_state


def run_smoke(origins_count: int = 10) -> dict:
    model_path = MODELS_DIR / "v8" / "model_bundle_v8.joblib"
    val_path = REPORTS_DIR / "v3" / "validation_frame.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Run python scripts/train_v8.py first.")
    if not val_path.exists():
        raise FileNotFoundError("Run the V3 pipeline first.")

    bundle = load_any_bundle(model_path)
    val = pd.read_pickle(val_path).reset_index(drop=True)
    origins = val["forecast_origin_utc"].drop_duplicates().sort_values().iloc[:origins_count]
    sample = val[val["forecast_origin_utc"].isin(origins)].copy().reset_index(drop=True)
    pred = predict_any_bundle(bundle, sample).reset_index(drop=True)

    # Keep the exact feature surface but remove labels before saving, matching
    # production snapshots.
    feature_frame = sample.drop(
        columns=["y_price", "y_direction", "y_spike_high", "y_spike_low"],
        errors="ignore",
    )

    with tempfile.TemporaryDirectory(prefix="te_balanceai_v8_smoke_") as td:
        store = OnlineStore(Path(td) / "smoke.sqlite3")
        store.save_predictions(
            feature_frame,
            pred,
            pd.Series(["offline self-learning smoke"] * len(sample)),
            model_version=getattr(bundle, "version", "v8"),
            context={"mode": "public_validation_smoke"},
        )

        # Actuals are inserted in the same long-series shape as live ingestion.
        actual = sample[[
            "delivery_start_utc", "y_price", "y_direction", "future__fi_spot_price"
        ]].drop_duplicates("delivery_start_utc").copy()
        dmap = {0: -1, 1: 0, 2: 1}
        parts = []
        for series, col, mapper in [
            ("fi_imbalance_price", "y_price", None),
            ("fi_mfrr_direction", "y_direction", dmap),
            ("fi_spot_price", "future__fi_spot_price", None),
        ]:
            g = actual[["delivery_start_utc", col]].copy()
            g["timestamp_utc"] = pd.to_datetime(g["delivery_start_utc"], utc=True)
            if mapper is None:
                g["value"] = pd.to_numeric(g[col], errors="coerce")
            else:
                g["value"] = pd.to_numeric(g[col], errors="coerce").map(mapper)
            g["series_id"] = series
            parts.append(g[["series_id", "timestamp_utc", "value"]].dropna())
        long = pd.concat(parts, ignore_index=True)
        store.upsert_actuals_from_long(long)
        evaluated = store.evaluate_settled(pd.Timestamp("2100-01-01", tz="UTC"))
        joined = store.recent_joined(pd.Timestamp("2000-01-01", tz="UTC"))
        state = learn_multiwindow_state(
            {24: joined, 168: joined, 336: joined},
            min_step_samples=2,
        )

        spread_adapted = sum(
            1 for s in state.blended_steps.values()
            if s.p50_spread_weight is not None
        )
        result = {
            "forecast_rows": int(len(sample)),
            "evaluations_created": int(evaluated),
            "joined_rows": int(len(joined)),
            "mae": None if joined.empty else float(joined["abs_error"].mean()),
            "coverage": None if joined.empty else float(joined["covered"].mean()),
            "adapted_horizons": int(len(state.blended_steps)),
            "spread_weight_horizons": int(spread_adapted),
            "works": bool(evaluated > 0 and len(state.blended_steps) > 0 and spread_adapted > 0),
        }
        return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=10)
    args = ap.parse_args()
    result = run_smoke(args.origins)
    print(pd.Series(result).to_string())
    if not result["works"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

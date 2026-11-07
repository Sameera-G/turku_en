from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, ONLINE_CHAMPION
from src.models_v8 import load_any_bundle, predict_any_bundle
from src.online.store import OnlineStore
from src.online.retrain import _metrics, _candidate_gate, load_champion


def _prepare(frame: pd.DataFrame, bundle) -> pd.DataFrame:
    x = frame.copy()
    for c in bundle.feature_names:
        if c not in x:
            x[c] = float("nan")
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote-if-better", action="store_true", help="Promote only with >=48 settled origins and V11 metric gates.")
    ap.add_argument("--days", type=int, default=14, help="Evaluate the most recent N days of settled live snapshots.")
    args = ap.parse_args()

    candidate_path = MODELS_DIR / "v11" / "model_bundle_v11_candidate.joblib"
    if not candidate_path.exists():
        raise FileNotFoundError("Missing V11 candidate. Run: python scripts/train_v11_candidate.py")
    candidate = load_any_bundle(candidate_path)
    champion, champion_path = load_champion(MODELS_DIR / "v10" / "model_bundle_v10.joblib")

    frame = OnlineStore().settled_feature_frame()
    if frame.empty:
        print("No settled live feature snapshots yet; comparison will become available automatically after outcomes settle.")
        return
    frame["forecast_origin_utc"] = pd.to_datetime(frame["forecast_origin_utc"], utc=True, errors="coerce")
    cutoff = frame["forecast_origin_utc"].max() - pd.Timedelta(days=args.days)
    frame = frame[frame["forecast_origin_utc"] >= cutoff].dropna(subset=["y_price", "y_direction"]).reset_index(drop=True)
    origins = frame["forecast_origin_utc"].nunique()
    if frame.empty:
        print("No settled rows in selected window.")
        return

    cp = predict_any_bundle(champion, _prepare(frame, champion))
    xp = predict_any_bundle(candidate, _prepare(frame, candidate))
    cm = _metrics(frame, cp); xm = _metrics(frame, xp)
    eligible, gain, regressions, mae_gain = _candidate_gate(cm, xm)
    out = {
        "rows": int(len(frame)), "origins": int(origins), "days": int(args.days),
        "champion_version": getattr(champion, "version", champion_path.stem),
        "candidate_version": getattr(candidate, "version", candidate_path.stem),
        "champion_metrics": cm, "candidate_metrics": xm,
        "composite_gain": gain, "mae_gain": mae_gain,
        "regressions": regressions, "metric_gate_eligible": bool(eligible),
        "enough_origins_to_promote": bool(origins >= 48),
    }
    print(json.dumps(out, indent=2))

    report = ROOT / "reports" / "v11" / "live_candidate_comparison.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if args.promote_if_better:
        if origins < 48:
            raise SystemExit(f"Not promoting: need at least 48 settled origins; have {origins}.")
        if not eligible:
            raise SystemExit("Not promoting: candidate did not pass V11 MAE/composite/regression gates.")
        candidate.version = pd.Timestamp.now(tz="UTC").strftime("v11_prebuilt_%Y%m%dT%H%M%SZ")
        candidate.save(ONLINE_CHAMPION)
        print(f"PROMOTED {candidate.version} -> {ONLINE_CHAMPION}")


if __name__ == "__main__":
    main()

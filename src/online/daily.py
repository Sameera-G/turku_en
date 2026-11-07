from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
from zoneinfo import ZoneInfo

import pandas as pd

from ..config import (
    ONLINE_REPORTS_DIR, ONLINE_DAILY_LEARNING_LOCAL_HOUR, ONLINE_DAILY_LEARNING_LOCAL_MINUTE,
    ONLINE_DAILY_MIN_EVALUATIONS,
)
from ..performance import live_summary
from .adaptive import AdaptiveState
from .retrain import maybe_retrain
from .store import OnlineStore

HELSINKI = ZoneInfo("Europe/Helsinki")
DAILY_DIR = ONLINE_REPORTS_DIR / "daily"


def local_now(now: pd.Timestamp | None = None) -> pd.Timestamp:
    t = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.tz_convert(HELSINKI)


def due_today(now: pd.Timestamp | None = None) -> tuple[bool, str]:
    local = local_now(now)
    scheduled = local.normalize() + pd.Timedelta(hours=ONLINE_DAILY_LEARNING_LOCAL_HOUR, minutes=ONLINE_DAILY_LEARNING_LOCAL_MINUTE)
    report = DAILY_DIR / f"{local.date().isoformat()}.json"
    return bool(local >= scheduled and not report.exists()), local.date().isoformat()


def _window_metrics(store: OnlineStore, now_utc: pd.Timestamp, hours: int) -> dict:
    df = store.recent_joined(now_utc - pd.Timedelta(hours=hours))
    return live_summary(df)


def _append_csv(report: dict) -> None:
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    path = DAILY_DIR / "daily_metrics.csv"
    row = {
        "local_date": report["local_date"],
        "generated_at_utc": report["generated_at_utc"],
        "samples_24h": report["performance_24h"].get("samples", 0),
        "mae_24h": report["performance_24h"].get("mae"),
        "coverage_24h": report["performance_24h"].get("coverage"),
        "mae_7d": report["performance_7d"].get("mae"),
        "mae_14d": report["performance_14d"].get("mae"),
        "within_20_24h": report["performance_24h"].get("within_20"),
        "drift_score": report["drift_score"],
        "drift_detected": report["drift_detected"],
        "retrain_attempted": report["retrain"].get("attempted", False),
        "retrain_promoted": report["retrain"].get("promoted", False),
        "model_action": report["retrain"].get("reason", ""),
    }
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["local_date"].astype(str) != str(row["local_date"])]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def run_daily_learning(
    store: OnlineStore,
    model_path: Path,
    state: AdaptiveState,
    now: pd.Timestamp | None = None,
    allow_retrain: bool = True,
    force: bool = False,
) -> dict:
    now_utc = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    else:
        now_utc = now_utc.tz_convert("UTC")
    local = now_utc.tz_convert(HELSINKI)
    due, local_date = due_today(now_utc)
    if not force and not due:
        return {"ran": False, "reason": "not due", "local_date": local_date}

    p24 = _window_metrics(store, now_utc, 24)
    p7 = _window_metrics(store, now_utc, 24 * 7)
    p14 = _window_metrics(store, now_utc, 24 * 14)

    retrain = {"attempted": False, "promoted": False, "reason": "daily adaptive learning only"}
    if allow_retrain and int(p24.get("samples", 0)) >= ONLINE_DAILY_MIN_EVALUATIONS:
        retrain = asdict(maybe_retrain(store, model_path))
    elif allow_retrain:
        retrain["reason"] = f"need {ONLINE_DAILY_MIN_EVALUATIONS} settled evaluations for daily challenger; have {int(p24.get('samples',0))}"

    report = {
        "ran": True,
        "local_date": local.date().isoformat(),
        "generated_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scheduled_local_time": f"{ONLINE_DAILY_LEARNING_LOCAL_HOUR:02d}:{ONLINE_DAILY_LEARNING_LOCAL_MINUTE:02d}",
        "learning_policy": "hourly adaptive calibration + daily performance checkpoint + conditional champion/challenger retraining",
        "performance_24h": p24,
        "performance_7d": p7,
        "performance_14d": p14,
        "drift_score": float(state.drift_score),
        "drift_detected": bool(state.drift_detected),
        "drift_reasons": list(state.drift_reasons),
        "adapted_horizons": len(state.blended_steps),
        "retrain": retrain,
    }
    if p24.get("mae") is not None and p7.get("mae") is not None:
        report["mae_change_vs_7d"] = float(p24["mae"] - p7["mae"])
    else:
        report["mae_change_vs_7d"] = None

    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    dated = DAILY_DIR / f"{report['local_date']}.json"
    dated.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (DAILY_DIR / "latest.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _append_csv(report)
    store.log_ingestion("daily_learning", "ok", report)
    return report

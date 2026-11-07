from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from .config import (
    ROOT, ACTUALS_VALIDATION_CSV, REPORTS_DIR, ONLINE_REPORTS_DIR,
    ONLINE_OUTPUTS_DIR,
)

V3_SCORE_DIR = REPORTS_DIR / "v3" / "official_validation" / "score"
V8_SCORE_DIR = REPORTS_DIR / "v8" / "official_validation" / "score"
V3_VALIDATION_FORECAST = REPORTS_DIR / "v3" / "official_validation" / "forecast_backtest.csv"
V8_VALIDATION_FORECAST = REPORTS_DIR / "v8" / "official_validation" / "forecast_backtest.csv"

def _active_validation_paths() -> tuple[Path, Path, Path]:
    if (V8_SCORE_DIR / "score_summary.csv").exists() and V8_VALIDATION_FORECAST.exists():
        return V8_SCORE_DIR / "score_summary.csv", V8_SCORE_DIR / "score_by_horizon.csv", V8_VALIDATION_FORECAST
    return V3_SCORE_DIR / "score_summary.csv", V3_SCORE_DIR / "score_by_horizon.csv", V3_VALIDATION_FORECAST
DAILY_DIR = ONLINE_REPORTS_DIR / "daily"


# Organizer public-validation persistence baseline from the official briefing.
# These are comparison values only; hidden-test baseline values are organizer-held.
PUBLIC_VALIDATION_BASELINE = {
    "mae_p50": 66.20,
    "calibration_pinball_mean": 24.18,
    "coverage_p10_p90": 0.68,
    "brier_spike_mean": 0.1132,
    "brier_direction": 0.4509,
}

def public_baseline_comparison(summary: dict | None = None) -> dict:
    summary = official_validation_summary() if summary is None else summary
    if not summary:
        return {}
    out = dict(PUBLIC_VALIDATION_BASELINE)
    for key in ["mae_p50", "calibration_pinball_mean", "brier_spike_mean", "brier_direction"]:
        team = float(summary[key])
        base = float(PUBLIC_VALIDATION_BASELINE[key])
        out[f"{key}_improvement_pct"] = 100.0 * (base - team) / base
    return out

def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def official_validation_summary() -> dict:
    summary_path, _, _ = _active_validation_paths()
    df = _read_csv(summary_path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


def official_validation_by_hour() -> pd.DataFrame:
    _, by_horizon_path, _ = _active_validation_paths()
    return _read_csv(by_horizon_path)


def _direction_brier(df: pd.DataFrame) -> pd.Series:
    if "direction" not in df:
        return pd.Series(np.nan, index=df.index)
    d = df["direction"].astype(str).str.lower()
    out = (
        (df["p_up"] - (d == "up").astype(float)) ** 2
        + (df["p_down"] - (d == "down").astype(float)) ** 2
        + (df["p_none"] - (d == "none").astype(float)) ** 2
    )
    return out.where(d.isin(["up", "down", "none"]))


def _pinball(y: pd.Series, p: pd.Series, q: float) -> pd.Series:
    diff = y - p
    return pd.Series(np.maximum(q * diff, (q - 1.0) * diff), index=y.index)


def validation_detail() -> pd.DataFrame:
    _, _, validation_forecast = _active_validation_paths()
    if not validation_forecast.exists() or not ACTUALS_VALIDATION_CSV.exists():
        return pd.DataFrame()
    pred = pd.read_csv(validation_forecast)
    act = pd.read_csv(ACTUALS_VALIDATION_CSV)
    df = pred.merge(act, on="delivery_start_utc", how="left", suffixes=("", "_actual"))
    df["forecast_origin_utc"] = pd.to_datetime(df["forecast_origin_utc"], utc=True)
    df["delivery_start_utc"] = pd.to_datetime(df["delivery_start_utc"], utc=True)
    df["imbalance_price"] = pd.to_numeric(df["imbalance_price"], errors="coerce")
    df["spot_price"] = pd.to_numeric(df.get("spot_price"), errors="coerce")
    valid = df["imbalance_price"].notna()
    df["abs_error"] = (df["p50"] - df["imbalance_price"]).abs().where(valid)
    df["covered"] = ((df["imbalance_price"] >= df["p10"]) & (df["imbalance_price"] <= df["p90"])).where(valid)
    df["pinball_p10"] = _pinball(df["imbalance_price"], df["p10"], 0.10).where(valid)
    df["pinball_p90"] = _pinball(df["imbalance_price"], df["p90"], 0.90).where(valid)
    df["direction_brier"] = _direction_brier(df)
    spot_ok = valid & df["spot_price"].notna()
    high = (df["imbalance_price"] > df["spot_price"] + 50).astype(float)
    low = (df["imbalance_price"] < df["spot_price"] - 50).astype(float)
    df["spike_high_brier"] = ((df["p_spike_high"] - high) ** 2).where(spot_ok)
    df["spike_low_brier"] = ((df["p_spike_low"] - low) ** 2).where(spot_ok)
    df["horizon_minutes"] = (df["step"].astype(int) - 1) * 15
    df["horizon_label"] = df["horizon_minutes"].map(lambda m: f"+{m//60}h{m%60:02d}m")
    return df


def validation_step_performance(detail: pd.DataFrame | None = None) -> pd.DataFrame:
    df = validation_detail() if detail is None else detail
    if df.empty:
        return pd.DataFrame()
    out = (
        df[df["abs_error"].notna()]
        .groupby("step", as_index=False)
        .agg(
            samples=("abs_error", "size"),
            mae=("abs_error", "mean"),
            coverage=("covered", "mean"),
            pinball_p10=("pinball_p10", "mean"),
            pinball_p90=("pinball_p90", "mean"),
            direction_brier=("direction_brier", "mean"),
            spike_high_brier=("spike_high_brier", "mean"),
            spike_low_brier=("spike_low_brier", "mean"),
        )
    )
    out["horizon_minutes"] = (out["step"] - 1) * 15
    out["horizon"] = out["horizon_minutes"].map(lambda m: f"+{m//60}h{m%60:02d}m")
    out["spike_brier"] = (out["spike_high_brier"] + out["spike_low_brier"]) / 2.0
    return out


def validation_intuitive_accuracy(detail: pd.DataFrame | None = None) -> dict:
    df = validation_detail() if detail is None else detail
    if df.empty:
        return {}
    x = df[df["abs_error"].notna()].copy()
    if x.empty:
        return {}
    ae = x["abs_error"]
    return {
        "samples": int(len(x)),
        "mae": float(ae.mean()),
        "median_abs_error": float(ae.median()),
        "within_10": float((ae <= 10).mean()),
        "within_20": float((ae <= 20).mean()),
        "within_50": float((ae <= 50).mean()),
        "coverage": float(x["covered"].mean()),
        "pinball": float((x["pinball_p10"].mean() + x["pinball_p90"].mean()) / 2.0),
        "direction_brier": None if x["direction_brier"].dropna().empty else float(x["direction_brier"].mean()),
        "spike_brier": None if x[["spike_high_brier", "spike_low_brier"]].dropna(how="all").empty else float(
            (x["spike_high_brier"].mean() + x["spike_low_brier"].mean()) / 2.0
        ),
    }


def live_summary(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"samples": 0}
    out = {
        "samples": int(len(df)),
        "origins": int(df["forecast_origin_utc"].nunique()),
        "mae": float(df["abs_error"].mean()),
        "coverage": float(df["covered"].mean()),
        "median_abs_error": float(df["abs_error"].median()),
        "within_10": float((df["abs_error"] <= 10).mean()),
        "within_20": float((df["abs_error"] <= 20).mean()),
        "within_50": float((df["abs_error"] <= 50).mean()),
    }
    for col in ["direction_brier", "spike_high_brier", "spike_low_brier"]:
        out[col] = None if col not in df or df[col].dropna().empty else float(df[col].mean())
    if out["spike_high_brier"] is not None and out["spike_low_brier"] is not None:
        out["spike_brier"] = (out["spike_high_brier"] + out["spike_low_brier"]) / 2.0
    else:
        out["spike_brier"] = None
    return out


def live_step_performance(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.groupby("step", as_index=False).agg(
        samples=("abs_error", "size"), mae=("abs_error", "mean"), coverage=("covered", "mean")
    )
    out["horizon_minutes"] = (out["step"].astype(int) - 1) * 15
    out["horizon"] = out["horizon_minutes"].map(lambda m: f"+{m//60}h{m%60:02d}m")
    return out


def daily_history() -> pd.DataFrame:
    path = DAILY_DIR / "daily_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "local_date" in df:
        df["local_date"] = pd.to_datetime(df["local_date"], errors="coerce")
    return df


def latest_daily_report() -> dict:
    path = DAILY_DIR / "latest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

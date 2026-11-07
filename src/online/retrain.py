from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from ..config import (
    ONLINE_CHAMPION, ONLINE_VERSIONS_DIR, ONLINE_MIN_RETRAIN_ORIGINS,
    ONLINE_RETRAIN_LOOKBACK_DAYS, ONLINE_PROMOTION_MIN_GAIN,
    ONLINE_PROMOTION_MIN_MAE_GAIN, ONLINE_MAX_METRIC_REGRESSION,
    ONLINE_RETRAIN_CAL_ORIGINS, ONLINE_RETRAIN_EVAL_ORIGINS,
    ONLINE_ENABLE_STRONG_P50_CHALLENGER,
    TRAIN_LABEL_START_UTC, TEST_START_UTC,
    ONLINE_EXTRA_FEATURE_PREFIXES, ONLINE_RETRAIN_EXTRA_MIN_COVERAGE,
    RANDOM_SEED, MODEL_THREADS,
)
from ..features import build_features
from ..models import train_base, fit_calibrator, _select_cat_features, numeric_model_frame
from ..models_v8 import (
    ModelBundleV8, load_any_bundle, predict_any_bundle,
    train_spread_bundle, fit_v8_calibrator,
)
from .store import OnlineStore


@dataclass
class RetrainResult:
    attempted: bool
    promoted: bool
    reason: str
    old_version: str | None = None
    new_version: str | None = None
    champion_metrics: dict | None = None
    challenger_metrics: dict | None = None


def load_champion(default_path: Path):
    path = ONLINE_CHAMPION if ONLINE_CHAMPION.exists() else default_path
    return load_any_bundle(path), path


def _metrics(df: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    y = df["y_price"].to_numpy(float)
    out = {"mae": float(np.mean(np.abs(y - pred["p50"].to_numpy(float))))}

    def pin(p, q):
        e = y - p
        return float(np.mean(np.maximum(q * e, (q - 1) * e)))

    out["pinball"] = (
        pin(pred["p10"].to_numpy(float), .1)
        + pin(pred["p90"].to_numpy(float), .9)
    ) / 2
    out["coverage"] = float(np.mean(
        (y >= pred["p10"].to_numpy(float))
        & (y <= pred["p90"].to_numpy(float))
    ))

    yd = df["y_direction"].to_numpy(int)
    pp = pred[["p_down", "p_none", "p_up"]].to_numpy(float)
    o = np.zeros_like(pp)
    o[np.arange(len(yd)), yd] = 1
    out["direction_brier"] = float(np.mean(np.sum((pp - o) ** 2, axis=1)))

    if df[["y_spike_high", "y_spike_low"]].notna().all(axis=1).any():
        m = df[["y_spike_high", "y_spike_low"]].notna().all(axis=1).to_numpy()
        hi = np.mean((
            pred.loc[m, "p_spike_high"].to_numpy()
            - df.loc[m, "y_spike_high"].to_numpy()
        ) ** 2)
        lo = np.mean((
            pred.loc[m, "p_spike_low"].to_numpy()
            - df.loc[m, "y_spike_low"].to_numpy()
        ) ** 2)
        out["spike_brier"] = float((hi + lo) / 2)
    return out


def _composite_ratio(champ: dict, chall: dict) -> float:
    # Same relative emphasis as the challenge automatic metrics:
    # accuracy 30, calibration 20, direction 10, spikes 20.
    pairs = [
        ("mae", 0.375),
        ("pinball", 0.25),
        ("direction_brier", 0.125),
        ("spike_brier", 0.25),
    ]
    valid = [
        (k, w) for k, w in pairs
        if k in champ and k in chall and champ[k] and np.isfinite(champ[k])
    ]
    denom = sum(w for _, w in valid)
    return (
        sum(w * (chall[k] / champ[k]) for k, w in valid) / denom
        if denom else 1.0
    )


def _three_way_origin_split(origins: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Chronological train/calibration/evaluation split for V11 promotion.

    V10 and earlier tuned challenger calibration on the same online tail used to
    decide promotion. V11 reserves the newest origins as an untouched evaluation
    set so a challenger cannot win merely by overfitting its calibration window.
    """
    origins = pd.DatetimeIndex(origins).sort_values().unique()
    n = len(origins)
    if n < ONLINE_MIN_RETRAIN_ORIGINS:
        raise ValueError(f"need {ONLINE_MIN_RETRAIN_ORIGINS} origins; have {n}")

    eval_n = min(ONLINE_RETRAIN_EVAL_ORIGINS, max(8, n // 4))
    cal_n = min(ONLINE_RETRAIN_CAL_ORIGINS, max(8, (n - eval_n) // 3))
    train_n = n - cal_n - eval_n
    if train_n < 16:
        # Protect the training side first; reduce tails symmetrically if needed.
        spare = max(0, n - 16)
        cal_n = max(4, spare // 2)
        eval_n = max(4, spare - cal_n)
        train_n = n - cal_n - eval_n
    if train_n <= 0 or cal_n <= 0 or eval_n <= 0:
        raise ValueError("not enough origins for three-way chronological split")

    return origins[:train_n], origins[train_n:train_n + cal_n], origins[train_n + cal_n:]


def _train_v8_challenger(train: pd.DataFrame, cal: pd.DataFrame, features: list[str]):
    """Train a leakage-safe V8-format challenger; calibrate only on *cal*."""
    base = train_base(train, features)
    base.calibrator = fit_calibrator(base, cal)

    spot_train = pd.to_numeric(train.get("future__fi_spot_price"), errors="coerce")
    spot_cal = pd.to_numeric(cal.get("future__fi_spot_price"), errors="coerce")
    if spot_train.notna().sum() < 1000 or spot_cal.notna().sum() < 100:
        return base

    v8 = train_spread_bundle(base, train)
    fit_v8_calibrator(v8, cal)
    return v8


def _strong_lgb() -> LGBMRegressor:
    # This configuration was the strongest repeatable direct-MAE learner in the
    # V11 experiments. It is deliberately modest; promotion still requires an
    # untouched live evaluation win.
    return LGBMRegressor(
        objective="quantile", alpha=.5,
        n_estimators=180, learning_rate=.025, num_leaves=15,
        min_child_samples=40, max_depth=-1,
        subsample=.90, colsample_bytree=.85,
        reg_alpha=.20, reg_lambda=2.0,
        random_state=RANDOM_SEED, n_jobs=MODEL_THREADS, verbosity=-1,
        force_col_wise=True, deterministic=True,
    )


def _strong_cat() -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE", iterations=250, depth=6, learning_rate=.03,
        l2_leaf_reg=7.0, random_seed=RANDOM_SEED, verbose=False,
        allow_writing_files=False, thread_count=MODEL_THREADS,
    )


def _make_strong_p50_candidate(standard, train: pd.DataFrame, cal: pd.DataFrame, features: list[str]):
    """Clone the standard challenger and replace only the P50 direct learners.

    Quantile, direction and spike heads remain identical. This isolates the
    extra capacity to the metric the user wants to improve (MAE), while the
    untouched evaluation split decides whether that capacity actually helps in
    the current market regime.
    """
    cand = copy.deepcopy(standard)
    base = cand.base if isinstance(cand, ModelBundleV8) else cand

    prepared = numeric_model_frame(train, features)
    lgb = _strong_lgb()
    lgb.fit(prepared[features], train["y_price"])
    cat_features = _select_cat_features(features)
    cat = _strong_cat()
    cat.fit(prepared[cat_features], train["y_price"])

    base.q50_lgb = lgb
    base.q50_cat = cat
    base.cat_feature_names = cat_features
    base.calibrator = fit_calibrator(base, cal)
    if isinstance(cand, ModelBundleV8):
        fit_v8_calibrator(cand, cal)
    return cand


def _candidate_gate(champ: dict, chall: dict) -> tuple[bool, float, list[str], float]:
    ratio = _composite_ratio(champ, chall)
    gain = 1 - ratio
    regressions = []
    for k in ["mae", "pinball", "direction_brier", "spike_brier"]:
        if k in champ and k in chall and chall[k] > champ[k] * (1 + ONLINE_MAX_METRIC_REGRESSION):
            regressions.append(k)
    mae_gain = 1.0 - (chall["mae"] / champ["mae"]) if champ.get("mae") else -np.inf
    eligible = (
        gain >= ONLINE_PROMOTION_MIN_GAIN
        and mae_gain >= ONLINE_PROMOTION_MIN_MAE_GAIN
        and not regressions
    )
    return eligible, gain, regressions, mae_gain


def maybe_retrain(store: OnlineStore, default_model_path: Path) -> RetrainResult:
    online = store.settled_feature_frame()
    if online.empty:
        return RetrainResult(False, False, "no settled online feature snapshots")

    online["forecast_origin_utc"] = pd.to_datetime(online["forecast_origin_utc"], utc=True, errors="coerce")
    origins = pd.DatetimeIndex(online["forecast_origin_utc"].dropna().drop_duplicates().sort_values())
    if len(origins) < ONLINE_MIN_RETRAIN_ORIGINS:
        return RetrainResult(
            False, False,
            f"need {ONLINE_MIN_RETRAIN_ORIGINS} settled origins; have {len(origins)}",
        )

    if online[["y_spike_high", "y_spike_low"]].isna().any(axis=1).mean() > 0.10:
        return RetrainResult(
            False, False,
            "live spot labels are missing; add ENTSO-E or a licensed/manual live spot curve before full challenger retraining",
        )

    champion, champion_path = load_champion(default_model_path)
    base_features = list(champion.feature_names)

    # Admit new live-only features only after exact prediction-time snapshots
    # have enough non-missing coverage. This is where MCP, forecast revisions,
    # weather and new Fingrid context can start improving the challenger.
    extra_features = []
    reserved = {
        "forecast_origin_utc", "delivery_start_utc", "step",
        "y_price", "y_direction", "y_spike_high", "y_spike_low",
    }
    for c in online.columns:
        if c in base_features or c in reserved:
            continue
        if not any(c.startswith(prefix) for prefix in ONLINE_EXTRA_FEATURE_PREFIXES):
            continue
        coverage = float(pd.to_numeric(online[c], errors="coerce").notna().mean())
        if coverage >= ONLINE_RETRAIN_EXTRA_MIN_COVERAGE:
            extra_features.append(c)
    features = base_features + sorted(extra_features)

    try:
        train_origins, cal_origins, eval_origins = _three_way_origin_split(origins)
    except ValueError as exc:
        return RetrainResult(False, False, str(exc))

    online_train = online[online["forecast_origin_utc"].isin(train_origins)].copy()
    online_cal = online[online["forecast_origin_utc"].isin(cal_origins)].copy()
    online_eval = online[online["forecast_origin_utc"].isin(eval_origins)].copy()
    if online_train.empty or online_cal.empty or online_eval.empty:
        return RetrainResult(False, False, "three-way online split produced an empty partition")

    # Rebuild bounded leakage-safe original history and append exact online
    # training snapshots. Calibration/evaluation remain strictly live and newer.
    test_start = pd.Timestamp(TEST_START_UTC)
    hist_start = max(
        pd.Timestamp(TRAIN_LABEL_START_UTC),
        test_start - pd.Timedelta(days=ONLINE_RETRAIN_LOOKBACK_DAYS),
    )
    hist_origins = pd.date_range(
        hist_start.ceil("h"), test_start - pd.Timedelta(hours=1), freq="1h", tz="UTC"
    )
    hist = build_features(hist_origins, include_labels=True)
    hist = hist.dropna(subset=["y_price", "y_direction", "y_spike_high", "y_spike_low"])

    for f in features:
        for frame in (online_train, online_cal, online_eval, hist):
            if f not in frame:
                frame[f] = np.nan

    cols = features + ["y_price", "y_direction", "y_spike_high", "y_spike_low", "step"]
    train = pd.concat([hist[cols], online_train[cols]], ignore_index=True)
    cal = online_cal[cols].reset_index(drop=True)
    ev = online_eval[cols].reset_index(drop=True)

    candidates: dict[str, object] = {}
    standard = _train_v8_challenger(train, cal, features)
    candidates["standard_v8"] = standard
    if ONLINE_ENABLE_STRONG_P50_CHALLENGER:
        try:
            candidates["strong_p50"] = _make_strong_p50_candidate(standard, train, cal, features)
        except Exception as exc:
            # Tournament remains operational even if the optional stronger P50
            # learner cannot train on a particular machine/source configuration.
            store.log_model_event(
                "v11_candidate_error", getattr(champion, "version", champion_path.stem), None,
                f"strong_p50 training failed: {exc}", {"candidate": "strong_p50"}
            )

    champ_pred = predict_any_bundle(champion, ev)
    cm = _metrics(ev, champ_pred)

    scored = {}
    best_name = None
    best_ratio = np.inf
    best_candidate = None
    best_metrics = None
    best_gate = None
    for name, cand in candidates.items():
        pred = predict_any_bundle(cand, ev)
        xm = _metrics(ev, pred)
        eligible, gain, regressions, mae_gain = _candidate_gate(cm, xm)
        ratio = _composite_ratio(cm, xm)
        scored[name] = {
            "metrics": xm,
            "composite_ratio": ratio,
            "gain": gain,
            "mae_gain": mae_gain,
            "regressions": regressions,
            "eligible": eligible,
        }
        if eligible and ratio < best_ratio:
            best_ratio = ratio
            best_name = name
            best_candidate = cand
            best_metrics = xm
            best_gate = scored[name]

    old_version = getattr(champion, "version", champion_path.stem)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    new_version = f"v11_{best_name or 'rejected'}_{stamp}"
    common = {
        "champion": cm,
        "candidates": scored,
        "selected_candidate": best_name,
        "feature_count": len(features),
        "new_live_features": extra_features,
        "split": {
            "train_origins": int(len(train_origins)),
            "calibration_origins": int(len(cal_origins)),
            "evaluation_origins": int(len(eval_origins)),
            "train_rows": int(len(train)),
            "calibration_rows": int(len(cal)),
            "evaluation_rows": int(len(ev)),
        },
    }

    if best_candidate is None:
        reason = "all V11 challengers rejected on untouched live evaluation"
        store.log_model_event("retrain_rejected", old_version, new_version, reason, common)
        # Return the best numerical challenger for visibility even if ineligible.
        if scored:
            nominal = min(scored, key=lambda k: scored[k]["composite_ratio"])
            return RetrainResult(True, False, reason, old_version, new_version, cm, scored[nominal]["metrics"])
        return RetrainResult(True, False, reason, old_version, new_version, cm, None)

    best_candidate.version = new_version
    ONLINE_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    version_path = ONLINE_VERSIONS_DIR / f"{new_version}.joblib"
    best_candidate.save(version_path)
    best_candidate.save(ONLINE_CHAMPION)
    reason = (
        f"V11 {best_name} promoted: composite gain={best_gate['gain']:.4f}; "
        f"MAE gain={best_gate['mae_gain']:.4f} on untouched evaluation"
    )
    store.log_model_event("retrain_promoted", old_version, new_version, reason, common)
    return RetrainResult(True, True, reason, old_version, new_version, cm, best_metrics)

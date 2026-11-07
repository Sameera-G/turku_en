from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd

# Keep native ML libraries deterministic and avoid OpenMP oversubscription/deadlocks
# on high-core-count hackathon laptops and CI containers. Set TE_BALANCEAI_THREADS
# before launch if you intentionally want more threads.
_os_threads = os.getenv("TE_BALANCEAI_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", _os_threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _os_threads)
os.environ.setdefault("MKL_NUM_THREADS", _os_threads)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _os_threads)
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor

from .config import RANDOM_SEED, MODEL_THREADS


@dataclass
class HorizonCalibratorV3:
    p50_lgb_weight: dict[int, float] = field(default_factory=dict)
    q10_delta: dict[int, float] = field(default_factory=dict)
    q90_delta: dict[int, float] = field(default_factory=dict)
    dir_alpha: dict[int, float] = field(default_factory=dict)
    dir_prior: dict[int, list[float]] = field(default_factory=dict)
    hi_alpha: dict[int, float] = field(default_factory=dict)
    hi_prior: dict[int, float] = field(default_factory=dict)
    lo_alpha: dict[int, float] = field(default_factory=dict)
    lo_prior: dict[int, float] = field(default_factory=dict)


@dataclass
class ModelBundleV3:
    feature_names: list[str]
    cat_feature_names: list[str]
    q10_lgb: object
    q50_lgb: object
    q90_lgb: object
    q50_cat: object
    direction: object
    spike_high: object
    spike_low: object
    feature_q01: dict[str, float] = field(default_factory=dict)
    feature_q99: dict[str, float] = field(default_factory=dict)
    calibrator: HorizonCalibratorV3 = field(default_factory=HorizonCalibratorV3)
    version: str = "3.1"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ModelBundleV3":
        return joblib.load(path)


def _reg(alpha: float) -> LGBMRegressor:
    return LGBMRegressor(
        objective="quantile", alpha=alpha,
        n_estimators=60, learning_rate=0.04, num_leaves=39,
        min_child_samples=32, max_depth=-1,
        subsample=0.90, colsample_bytree=0.80,
        reg_alpha=0.15, reg_lambda=1.25,
        random_state=RANDOM_SEED, n_jobs=MODEL_THREADS, verbosity=-1,
        force_col_wise=True, deterministic=True,
    )


def _cat50() -> CatBoostRegressor:
    # CatBoost is deliberately a compact heterogeneous learner. It sees a
    # curated feature subset, which preserves ensemble diversity while keeping
    # hackathon iteration time practical.
    return CatBoostRegressor(
        loss_function="Quantile:alpha=0.5",
        iterations=55,
        depth=6,
        learning_rate=0.06,
        l2_leaf_reg=5.0,
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=MODEL_THREADS,
    )


def _clf(multiclass: bool = False) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass" if multiclass else "binary",
        n_estimators=55, learning_rate=0.04, num_leaves=35,
        min_child_samples=32, max_depth=-1,
        subsample=0.90, colsample_bytree=0.82,
        reg_alpha=0.10, reg_lambda=1.1,
        random_state=RANDOM_SEED, n_jobs=MODEL_THREADS, verbosity=-1,
        force_col_wise=True, deterministic=True,
    )


def _select_cat_features(features: list[str]) -> list[str]:
    """Select a compact, complementary feature set for CatBoost.

    Derived market-state features and known-ahead values are retained. Raw lag
    features are limited to high-value series/lags to avoid making CatBoost a
    slow duplicate of the full LightGBM learner.
    """
    key_lag_series = (
        "fi_imbalance_price", "fi_mfrr_direction", "fi_mfrr_up_activated",
        "fi_mfrr_down_activated", "fi_consumption", "fi_production",
        "fi_se1_flow", "fi_se3_flow", "fi_no4_flow", "fi_ee_flow",
        "se1_imbalance_price_long", "se3_imbalance_price_long",
        "ee_imbalance_price_long",
    )
    keep_lags = ("lag1q__", "lag4q__", "lag8q__", "lag16q__", "lag96q__")
    selected: list[str] = []
    for f in features:
        if not f.startswith("lag"):
            selected.append(f)
            continue
        if f.startswith(keep_lags) and any(s in f for s in key_lag_series):
            selected.append(f)
    return selected


def numeric_model_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return a copy whose model feature columns are numeric and writable.

    Prediction-time feature snapshots are JSON-serialized in the online store.
    Pandas can reconstruct sparse/mixed columns as ``object`` even when every
    meaningful value is numerical (and booleans may return as strings).
    LightGBM intentionally rejects object dtypes.  Normalize only the columns
    used by the trained model so timestamps/metadata remain untouched.
    """
    out = df.copy()
    missing = [c for c in features if c not in out.columns]
    if missing:
        out = pd.concat([out, pd.DataFrame(np.nan, index=out.index, columns=missing)], axis=1)
    bool_map = {
        True: 1.0, False: 0.0,
        "True": 1.0, "False": 0.0,
        "true": 1.0, "false": 0.0,
        "TRUE": 1.0, "FALSE": 0.0,
    }
    for c in features:
        s = out[c]
        if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype):
            s = s.replace(bool_map)
        out[c] = pd.to_numeric(s, errors="coerce").astype(float)
    return out


def _feature_envelope(train: pd.DataFrame, features: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    # OOD envelope is only a reliability signal, never a replacement for the
    # required forecast. Ignore columns that are fully missing or non-finite.
    q01: dict[str, float] = {}
    q99: dict[str, float] = {}
    for c in features:
        s = pd.to_numeric(train[c], errors="coerce")
        if s.notna().sum() < 50:
            continue
        lo = float(s.quantile(0.01))
        hi = float(s.quantile(0.99))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            q01[c] = lo
            q99[c] = hi
    return q01, q99


def train_base(train: pd.DataFrame, features: list[str]) -> ModelBundleV3:
    prepared = numeric_model_frame(train, features)
    x = prepared[features]
    q10, q50, q90 = _reg(.10), _reg(.50), _reg(.90)
    q10.fit(x, train["y_price"])
    q50.fit(x, train["y_price"])
    q90.fit(x, train["y_price"])

    cat_features = _select_cat_features(features)
    cat = _cat50()
    cat.fit(prepared[cat_features], train["y_price"])

    direction = _clf(multiclass=True)
    direction.fit(x, train["y_direction"].astype(int))
    hi, lo = _clf(False), _clf(False)
    hi.fit(x, train["y_spike_high"].astype(int))
    lo.fit(x, train["y_spike_low"].astype(int))

    q01, q99 = _feature_envelope(train, features)
    return ModelBundleV3(
        feature_names=features,
        cat_feature_names=cat_features,
        q10_lgb=q10,
        q50_lgb=q50,
        q90_lgb=q90,
        q50_cat=cat,
        direction=direction,
        spike_high=hi,
        spike_low=lo,
        feature_q01=q01,
        feature_q99=q99,
    )


def _raw_components(bundle: ModelBundleV3, df: pd.DataFrame) -> pd.DataFrame:
    prepared = numeric_model_frame(df, list(dict.fromkeys(bundle.feature_names + bundle.cat_feature_names)))
    x = prepared[bundle.feature_names]
    p10 = bundle.q10_lgb.predict(x)
    p50_lgb = bundle.q50_lgb.predict(x)
    p90 = bundle.q90_lgb.predict(x)
    p50_cat = bundle.q50_cat.predict(prepared[bundle.cat_feature_names])

    dp = bundle.direction.predict_proba(x)
    pos = {int(c): i for i, c in enumerate(bundle.direction.classes_)}

    def cls(k: int) -> np.ndarray:
        return dp[:, pos[k]] if k in pos else np.zeros(len(df))

    p_down, p_none, p_up = cls(0), cls(1), cls(2)
    p_hi = bundle.spike_high.predict_proba(x)[:, 1]
    p_lo = bundle.spike_low.predict_proba(x)[:, 1]
    return pd.DataFrame({
        "p10": p10,
        "p50_lgb": p50_lgb,
        "p50_cat": p50_cat,
        "p90": p90,
        "p_up": p_up,
        "p_down": p_down,
        "p_none": p_none,
        "p_spike_high": p_hi,
        "p_spike_low": p_lo,
    }, index=df.index)


def _three_class_brier(y: np.ndarray, p: np.ndarray) -> float:
    o = np.zeros_like(p)
    o[np.arange(len(y)), y.astype(int)] = 1.0
    return float(np.mean(np.sum((p - o) ** 2, axis=1)))


def _binary_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def fit_calibrator(bundle: ModelBundleV3, val: pd.DataFrame) -> HorizonCalibratorV3:
    raw = _raw_components(bundle, val).reset_index(drop=True)
    v = val.reset_index(drop=True)
    cal = HorizonCalibratorV3()

    for step, g in v.groupby("step"):
        idx = g.index.to_numpy()
        y = g["y_price"].to_numpy(float)
        lgb = raw.loc[idx, "p50_lgb"].to_numpy(float)
        cat = raw.loc[idx, "p50_cat"].to_numpy(float)

        # Exact challenge point metric for P50 is MAE, so choose late-fusion
        # weight per horizon using held-out validation MAE.
        best_w, best_mae = 1.0, np.inf
        for w in np.linspace(0, 1, 21):
            p = w * lgb + (1 - w) * cat
            mae = np.mean(np.abs(y - p))
            if mae < best_mae:
                best_w, best_mae = float(w), float(mae)
        cal.p50_lgb_weight[int(step)] = best_w

        cal.q10_delta[int(step)] = float(np.nanquantile(y - raw.loc[idx, "p10"].to_numpy(), 0.10))
        cal.q90_delta[int(step)] = float(np.nanquantile(y - raw.loc[idx, "p90"].to_numpy(), 0.90))

        ydir = g["y_direction"].to_numpy(int)
        pdir = raw.loc[idx, ["p_down", "p_none", "p_up"]].to_numpy(float)
        prior = np.bincount(ydir, minlength=3).astype(float)
        prior /= prior.sum() if prior.sum() else 1.0
        best_a, best_s = 1.0, np.inf
        for a in np.linspace(0, 1, 41):
            pp = a * pdir + (1 - a) * prior
            s = _three_class_brier(ydir, pp)
            if s < best_s:
                best_a, best_s = float(a), s
        cal.dir_alpha[int(step)] = best_a
        cal.dir_prior[int(step)] = prior.tolist()

        for name, target, attr_a, attr_p in [
            ("p_spike_high", "y_spike_high", cal.hi_alpha, cal.hi_prior),
            ("p_spike_low", "y_spike_low", cal.lo_alpha, cal.lo_prior),
        ]:
            yy = g[target].to_numpy(float)
            pp = raw.loc[idx, name].to_numpy(float)
            prior_b = float(np.mean(yy))
            best_a, best_s = 1.0, np.inf
            for a in np.linspace(0, 1, 41):
                p2 = a * pp + (1 - a) * prior_b
                s = _binary_brier(yy, p2)
                if s < best_s:
                    best_a, best_s = float(a), s
            attr_a[int(step)] = best_a
            attr_p[int(step)] = prior_b
    return cal


def _ood_fraction(bundle: ModelBundleV3, df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in bundle.feature_q01 if c in df.columns and c in bundle.feature_q99]
    if not cols:
        return np.zeros(len(df), dtype=float)
    flags = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        lo, hi = bundle.feature_q01[c], bundle.feature_q99[c]
        flags.append(((s < lo) | (s > hi)).astype(float))
    return np.nanmean(np.vstack(flags), axis=0)


def predict_bundle(bundle: ModelBundleV3, df: pd.DataFrame, calibrated: bool = True) -> pd.DataFrame:
    raw = _raw_components(bundle, df).reset_index(drop=True)
    steps = df["step"].astype(int).to_numpy()

    if calibrated and bundle.calibrator.p50_lgb_weight:
        w = np.array([bundle.calibrator.p50_lgb_weight.get(int(s), 0.5) for s in steps])
    else:
        w = np.full(len(df), 0.5)
    p50 = w * raw["p50_lgb"].to_numpy(float) + (1 - w) * raw["p50_cat"].to_numpy(float)

    out = pd.DataFrame({
        "p10": raw["p10"].to_numpy(float),
        "p50": p50,
        "p90": raw["p90"].to_numpy(float),
        "p_up": raw["p_up"].to_numpy(float),
        "p_down": raw["p_down"].to_numpy(float),
        "p_none": raw["p_none"].to_numpy(float),
        "p_spike_high": raw["p_spike_high"].to_numpy(float),
        "p_spike_low": raw["p_spike_low"].to_numpy(float),
        "p50_lgb": raw["p50_lgb"].to_numpy(float),
        "p50_cat": raw["p50_cat"].to_numpy(float),
        "fusion_weight_lgb": w,
    })

    if calibrated and bundle.calibrator.q10_delta:
        q10d = np.array([bundle.calibrator.q10_delta.get(int(s), 0.0) for s in steps])
        q90d = np.array([bundle.calibrator.q90_delta.get(int(s), 0.0) for s in steps])
        out["p10"] += q10d
        out["p90"] += q90d

        pdir = out[["p_down", "p_none", "p_up"]].to_numpy(float)
        alphas = np.array([bundle.calibrator.dir_alpha.get(int(s), 1.0) for s in steps])[:, None]
        priors = np.array([bundle.calibrator.dir_prior.get(int(s), [1/3, 1/3, 1/3]) for s in steps], float)
        pdir = alphas * pdir + (1 - alphas) * priors
        pdir = np.clip(pdir, 1e-8, 1)
        pdir /= pdir.sum(axis=1, keepdims=True)
        out[["p_down", "p_none", "p_up"]] = pdir

        hi_a = np.array([bundle.calibrator.hi_alpha.get(int(s), 1.0) for s in steps])
        hi_p = np.array([bundle.calibrator.hi_prior.get(int(s), 0.0) for s in steps])
        lo_a = np.array([bundle.calibrator.lo_alpha.get(int(s), 1.0) for s in steps])
        lo_p = np.array([bundle.calibrator.lo_prior.get(int(s), 0.0) for s in steps])
        out["p_spike_high"] = hi_a * out["p_spike_high"].to_numpy(float) + (1 - hi_a) * hi_p
        out["p_spike_low"] = lo_a * out["p_spike_low"].to_numpy(float) + (1 - lo_a) * lo_p

    out["p10"] = np.minimum(out["p10"].to_numpy(float), p50)
    out["p90"] = np.maximum(out["p90"].to_numpy(float), p50)

    for c in ["p_up", "p_down", "p_none", "p_spike_high", "p_spike_low"]:
        out[c] = np.clip(out[c].astype(float), 0.0, 1.0)
    dsum = out[["p_up", "p_down", "p_none"]].sum(axis=1).replace(0, 1)
    out[["p_up", "p_down", "p_none"]] = out[["p_up", "p_down", "p_none"]].div(dsum, axis=0)
    ssum = out["p_spike_high"] + out["p_spike_low"]
    mask = ssum > 0.999
    out.loc[mask, "p_spike_high"] *= 0.999 / ssum[mask]
    out.loc[mask, "p_spike_low"] *= 0.999 / ssum[mask]

    out["interval_width"] = out["p90"] - out["p10"]
    out["model_disagreement"] = np.abs(out["p50_lgb"] - out["p50_cat"])
    out["ood_fraction"] = _ood_fraction(bundle, df)

    missing_cols = [c for c in df.columns if c.startswith("missing__")]
    if missing_cols:
        out["missing_fraction"] = df[missing_cols].mean(axis=1).to_numpy(float)
    else:
        out["missing_fraction"] = 0.0

    interval_rel = 1.0 - np.clip(out["interval_width"] / 350.0, 0, 1)
    disagreement_rel = 1.0 - np.clip(out["model_disagreement"] / 100.0, 0, 1)
    data_rel = 1.0 - np.clip(out["missing_fraction"], 0, 1)
    ood_rel = 1.0 - np.clip(out["ood_fraction"] * 3.0, 0, 1)
    probs = out[["p_up", "p_down", "p_none"]].clip(1e-8, 1)
    entropy = -(probs * np.log(probs)).sum(axis=1) / np.log(3)
    class_rel = 1.0 - entropy
    out["reliability"] = np.clip(
        0.35 * interval_rel
        + 0.20 * disagreement_rel
        + 0.20 * class_rel
        + 0.15 * data_rel
        + 0.10 * ood_rel,
        0, 1,
    )
    return out

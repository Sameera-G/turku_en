from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from .config import MODEL_THREADS, RANDOM_SEED
from .models import ModelBundleV3, predict_bundle, _select_cat_features, numeric_model_frame
from .v8_features import augment_v8_features, spread_feature_names


@dataclass
class V8StepCalibrator:
    direction_gamma: float = 1.5
    direct_weight: float = 0.0
    gated_weight: float = 1.0
    global_lgb_weight: float = 0.5
    p50_bias: float = 0.0


@dataclass
class ModelBundleV8:
    """V8 wrapper around the proven V3 probabilistic model.

    V3 continues to provide P10/P90, direction and spike probabilities.  V8
    replaces the P50 centre forecast, when delivery-time spot price is known,
    with a market-rule-aware spread ensemble:

        imbalance = spot + predicted_spread

    The spread ensemble contains a global LightGBM, global CatBoost and
    direction-conditioned DOWN/UP LightGBM experts.  NONE uses zero spread,
    matching the organizer's market-rule description.  If live spot is not
    available the system falls back to the validated V3 centre forecast rather
    than inventing a spot price.
    """

    base: ModelBundleV3
    spread_feature_names: list[str]
    spread_cat_feature_names: list[str]
    spread_global_lgb: object
    spread_global_cat: object
    spread_down_lgb: object
    spread_up_lgb: object
    step_calibration: dict[int, V8StepCalibrator] = field(default_factory=dict)
    feature_q01: dict[str, float] = field(default_factory=dict)
    feature_q99: dict[str, float] = field(default_factory=dict)
    version: str = "8.0"

    @property
    def feature_names(self) -> list[str]:
        # Compatibility with online challenger code and diagnostics.
        return list(self.base.feature_names)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ModelBundleV8":
        return joblib.load(path)


def _spread_lgb() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=100,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=35,
        max_depth=-1,
        subsample=0.90,
        colsample_bytree=0.82,
        reg_alpha=0.15,
        reg_lambda=1.50,
        random_state=RANDOM_SEED,
        n_jobs=MODEL_THREADS,
        verbosity=-1,
        force_col_wise=True,
        deterministic=True,
    )


def _spread_cat() -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        iterations=50,
        depth=6,
        learning_rate=0.06,
        l2_leaf_reg=5.0,
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=MODEL_THREADS,
    )


def _cat_spread_features(features: list[str]) -> list[str]:
    base = set(_select_cat_features(features))
    # Keep the explicit quarter/ramp features even if they are not part of the
    # old CatBoost selector.
    extras = {
        "delivery_quarter", "horizon_hour_index", "is_hour_boundary",
        "delivery_quarter_0", "delivery_quarter_1", "delivery_quarter_2", "delivery_quarter_3",
        "delivery_quarter_sin", "delivery_quarter_cos", "spot_qoq_change",
        "wind_qoq_change", "load_qoq_change", "solar_qoq_change", "net_load_qoq_change",
        "spot_abs", "spot_negative", "spot_x_hour_boundary", "direction_pressure_x_spot",
        "mfrr_pressure_x_spot",
    }
    return [f for f in features if f in base or f in extras]


def train_spread_bundle(base: ModelBundleV3, train: pd.DataFrame) -> ModelBundleV8:
    # Online snapshots can reconstruct numeric values as object dtype after JSON
    # round-tripping; normalize the base inputs before creating ramp features.
    train = numeric_model_frame(train, list(dict.fromkeys(base.feature_names + ["step"])))
    train = augment_v8_features(train).reset_index(drop=True)
    features = spread_feature_names(base.feature_names, train)
    for c in features:
        if c not in train:
            train[c] = np.nan
    train = numeric_model_frame(train, features)
    spot = pd.to_numeric(train["future__fi_spot_price"], errors="coerce")
    y = pd.to_numeric(train["y_price"], errors="coerce")
    ok = spot.notna() & y.notna()
    if ok.sum() < 1000:
        raise ValueError("Not enough rows with legal delivery spot price to train V8 spread models.")
    x = train.loc[ok, features]
    spread = (y[ok] - spot[ok]).to_numpy(float)
    direction = pd.to_numeric(train.loc[ok, "y_direction"], errors="coerce").to_numpy()

    global_lgb = _spread_lgb()
    global_lgb.fit(x, spread)

    cat_features = _cat_spread_features(features)
    global_cat = _spread_cat()
    global_cat.fit(train.loc[ok, cat_features], spread)

    down_idx = direction == 0
    up_idx = direction == 2
    if down_idx.sum() < 500 or up_idx.sum() < 500:
        raise ValueError("Not enough UP/DOWN examples to train V8 direction experts.")
    down = _spread_lgb()
    down.fit(x.loc[down_idx], spread[down_idx])
    up = _spread_lgb()
    up.fit(x.loc[up_idx], spread[up_idx])

    return ModelBundleV8(
        base=base,
        spread_feature_names=features,
        spread_cat_feature_names=cat_features,
        spread_global_lgb=global_lgb,
        spread_global_cat=global_cat,
        spread_down_lgb=down,
        spread_up_lgb=up,
    )


def _direction_probs_sharpened(base_pred: pd.DataFrame, gamma: np.ndarray) -> np.ndarray:
    probs = base_pred[["p_down", "p_none", "p_up"]].to_numpy(float)
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs ** gamma[:, None]
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def raw_v8_components(bundle: ModelBundleV8, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = list(dict.fromkeys(bundle.base.feature_names + bundle.spread_feature_names + ["step"]))
    f = numeric_model_frame(frame, required)
    f = augment_v8_features(f).reset_index(drop=True)
    for c in bundle.spread_feature_names:
        if c not in f:
            f[c] = np.nan
    f = numeric_model_frame(f, bundle.spread_feature_names)
    base = predict_bundle(bundle.base, f, calibrated=True).reset_index(drop=True)
    x = f[bundle.spread_feature_names]
    xc = f[bundle.spread_cat_feature_names]
    spot = pd.to_numeric(f.get("future__fi_spot_price"), errors="coerce").to_numpy(float)

    global_lgb = spot + bundle.spread_global_lgb.predict(x)
    global_cat = spot + bundle.spread_global_cat.predict(xc)
    down_spread = bundle.spread_down_lgb.predict(x)
    up_spread = bundle.spread_up_lgb.predict(x)

    raw = pd.DataFrame({
        "p50_direct": base["p50"].to_numpy(float),
        "p50_direct_lgb": base["p50_lgb"].to_numpy(float),
        "p50_direct_cat": base["p50_cat"].to_numpy(float),
        "p50_spread_global_lgb": global_lgb,
        "p50_spread_global_cat": global_cat,
        "spread_down": down_spread,
        "spread_up": up_spread,
        "spot_price_v8": spot,
    })
    return f, pd.concat([base.reset_index(drop=True), raw], axis=1)


def fit_v8_calibrator(bundle: ModelBundleV8, val: pd.DataFrame) -> dict[int, V8StepCalibrator]:
    f, raw = raw_v8_components(bundle, val)
    y_all = pd.to_numeric(f["y_price"], errors="coerce").to_numpy(float)
    steps = f["step"].astype(int).to_numpy()
    calibration: dict[int, V8StepCalibrator] = {}

    # A small, auditable grid is deliberately used.  It optimizes exactly the
    # organizer's P50 metric (MAE) without touching P10/P90 or probability heads.
    gamma_grid = np.arange(0.75, 2.51, 0.25)
    gated_weights = np.array([0.60, 0.70, 0.80, 0.90, 1.00])
    global_lgb_weights = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    direct_weights = np.array([0.00, 0.10, 0.20, 0.30, 0.40])

    for step in sorted(np.unique(steps)):
        idx = np.where(steps == step)[0]
        y = y_all[idx]
        direct = raw.loc[idx, "p50_direct"].to_numpy(float)
        gl = raw.loc[idx, "p50_spread_global_lgb"].to_numpy(float)
        gc = raw.loc[idx, "p50_spread_global_cat"].to_numpy(float)
        spot = raw.loc[idx, "spot_price_v8"].to_numpy(float)
        down = raw.loc[idx, "spread_down"].to_numpy(float)
        up = raw.loc[idx, "spread_up"].to_numpy(float)
        best = (np.inf, None)

        base_probs = raw.loc[idx, ["p_down", "p_none", "p_up"]].to_numpy(float)
        for gamma in gamma_grid:
            pp = np.clip(base_probs, 1e-8, 1.0) ** gamma
            pp /= pp.sum(axis=1, keepdims=True)
            # NONE spread is exactly zero under the organizer's stated rule.
            gated = spot + pp[:, 0] * down + pp[:, 2] * up
            for gw in gated_weights:
                for lw in global_lgb_weights:
                    # Remaining non-gated spread weight is divided between the
                    # two global learners.  gw=1 means pure direction gating.
                    global_mix = lw * gl + (1.0 - lw) * gc
                    spread_pred = gw * gated + (1.0 - gw) * global_mix
                    for dw in direct_weights:
                        p = dw * direct + (1.0 - dw) * spread_pred
                        mae = float(np.nanmean(np.abs(y - p)))
                        if mae < best[0]:
                            best = (mae, (float(gamma), float(dw), float(gw), float(lw), p))

        gamma, dw, gw, lw, pbest = best[1]
        # Median residual is the L1-optimal intercept.  Shrink toward zero to
        # reduce validation overfit while correcting persistent horizon bias.
        bias = float(0.75 * np.nanmedian(y - pbest))
        calibration[int(step)] = V8StepCalibrator(
            direction_gamma=gamma,
            direct_weight=dw,
            gated_weight=gw,
            global_lgb_weight=lw,
            p50_bias=bias,
        )
    bundle.step_calibration = calibration
    return calibration


def predict_v8_bundle(bundle: ModelBundleV8, frame: pd.DataFrame) -> pd.DataFrame:
    f, raw = raw_v8_components(bundle, frame)
    n = len(f)
    steps = f["step"].astype(int).to_numpy()
    spot = raw["spot_price_v8"].to_numpy(float)
    spot_ok = np.isfinite(spot)

    gammas = np.array([
        bundle.step_calibration.get(int(s), V8StepCalibrator()).direction_gamma for s in steps
    ], dtype=float)
    probs = _direction_probs_sharpened(raw, gammas)
    gated = spot + probs[:, 0] * raw["spread_down"].to_numpy(float) + probs[:, 2] * raw["spread_up"].to_numpy(float)

    lw = np.array([
        bundle.step_calibration.get(int(s), V8StepCalibrator()).global_lgb_weight for s in steps
    ], dtype=float)
    global_mix = (
        lw * raw["p50_spread_global_lgb"].to_numpy(float)
        + (1.0 - lw) * raw["p50_spread_global_cat"].to_numpy(float)
    )
    gw = np.array([
        bundle.step_calibration.get(int(s), V8StepCalibrator()).gated_weight for s in steps
    ], dtype=float)
    spread_pred = gw * gated + (1.0 - gw) * global_mix

    dw = np.array([
        bundle.step_calibration.get(int(s), V8StepCalibrator()).direct_weight for s in steps
    ], dtype=float)
    bias = np.array([
        bundle.step_calibration.get(int(s), V8StepCalibrator()).p50_bias for s in steps
    ], dtype=float)
    direct = raw["p50_direct"].to_numpy(float)
    v8 = dw * direct + (1.0 - dw) * spread_pred + bias
    # Live deployment can lack a licensed day-ahead price feed.  In that case
    # use the validated direct model and lower reliability; never fabricate spot.
    final = np.where(spot_ok, v8, direct)

    out = raw[[
        "p10", "p90", "p_up", "p_down", "p_none", "p_spike_high", "p_spike_low",
        "p50_lgb", "p50_cat", "reliability", "missing_fraction", "ood_fraction"
    ]].copy()
    out["p50"] = final
    out["p50_direct"] = direct
    out["p50_spread"] = spread_pred
    out["p50_spread_gated"] = gated
    out["p50_spread_global_lgb"] = raw["p50_spread_global_lgb"].to_numpy(float)
    out["p50_spread_global_cat"] = raw["p50_spread_global_cat"].to_numpy(float)
    out["v8_spot_available"] = spot_ok.astype(int)
    out["v8_direction_gamma"] = gammas
    out["v8_direct_weight"] = dw

    out["p10"] = np.minimum(out["p10"].to_numpy(float), final)
    out["p90"] = np.maximum(out["p90"].to_numpy(float), final)
    out["interval_width"] = out["p90"] - out["p10"]

    # Disagreement is now the economically meaningful direct-vs-spread gap.
    disagreement = np.abs(direct - spread_pred)
    out["model_disagreement"] = np.where(spot_ok, disagreement, raw["model_disagreement"].to_numpy(float))
    # Preserve the existing reliability logic, then penalize missing live spot
    # and unusually large V8 ensemble disagreement.
    disagreement_rel = 1.0 - np.clip(out["model_disagreement"].to_numpy(float) / 100.0, 0, 1)
    out["reliability"] = np.clip(0.80 * out["reliability"].to_numpy(float) + 0.20 * disagreement_rel, 0, 1)
    out.loc[~spot_ok, "reliability"] *= 0.88
    return out


def load_any_bundle(path: Path):
    return joblib.load(path)


def predict_any_bundle(bundle, frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(bundle, ModelBundleV8):
        return predict_v8_bundle(bundle, frame)
    return predict_bundle(bundle, frame, calibrated=True)

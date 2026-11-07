from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ..config import (
    ONLINE_ADAPTIVE_STATE,
    ONLINE_ADAPT_WINDOWS_HOURS,
    ONLINE_ADAPT_WINDOW_WEIGHTS,
    ONLINE_DRIFT_MAE_RATIO,
    ONLINE_DRIFT_COVERAGE_DEVIATION,
    ONLINE_DRIFT_MIN_SAMPLES,
    ONLINE_CONFORMAL_TARGET_COVERAGE,
    ONLINE_CONFORMAL_MIN_SAMPLES,
    ONLINE_CONFORMAL_FULL_STRENGTH_SAMPLES,
    ONLINE_CONFORMAL_MAX_EXPAND_EUR,
    ONLINE_GLOBAL_ADAPT_MIN_SAMPLES,
    ONLINE_RELIABILITY_PRIOR_SAMPLES,
)


@dataclass
class StepAdaptive:
    n: int = 0
    mae: float | None = None
    price_bias: float = 0.0
    p50_lgb_weight: float | None = None
    p50_spread_weight: float | None = None
    q10_extra_delta: float = 0.0
    q90_extra_delta: float = 0.0
    conformal_expand: float = 0.0
    conformal_expand_low: float = 0.0
    conformal_expand_high: float = 0.0
    empirical_coverage: float | None = None
    shrunk_coverage: float | None = None
    dir_alpha: float = 1.0
    dir_prior: list[float] = field(default_factory=lambda: [1 / 3, 1 / 3, 1 / 3])
    hi_alpha: float = 1.0
    hi_prior: float = 0.0
    lo_alpha: float = 1.0
    lo_prior: float = 0.0


@dataclass
class WindowAdaptive:
    hours: int = 24
    samples: int = 0
    rolling_mae: float | None = None
    rolling_coverage: float | None = None
    global_calibration: StepAdaptive = field(default_factory=StepAdaptive)
    steps: dict[str, StepAdaptive] = field(default_factory=dict)


@dataclass
class AdaptiveState:
    updated_at_utc: str | None = None
    windows: dict[str, WindowAdaptive] = field(default_factory=dict)
    blended_steps: dict[str, StepAdaptive] = field(default_factory=dict)
    blended_global: StepAdaptive = field(default_factory=StepAdaptive)
    rolling_mae: float | None = None
    rolling_coverage: float | None = None
    drift_score: float = 0.0
    drift_detected: bool = False
    drift_reasons: list[str] = field(default_factory=list)

    @property
    def steps(self) -> dict[str, StepAdaptive]:
        return self.blended_steps

    @property
    def window_hours(self) -> int:
        return int(min((w.hours for w in self.windows.values()), default=24))

    def save(self, path: Path = ONLINE_ADAPTIVE_STATE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path = ONLINE_ADAPTIVE_STATE) -> "AdaptiveState":
        if not path.exists():
            return AdaptiveState()
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "windows" not in raw:
            w = WindowAdaptive(
                hours=int(raw.get("window_hours", 24)),
                rolling_mae=raw.get("rolling_mae"),
                rolling_coverage=raw.get("rolling_coverage"),
                steps={str(k): StepAdaptive(**v) for k, v in raw.get("steps", {}).items()},
            )
            w.samples = sum(x.n for x in w.steps.values())
            return AdaptiveState(
                updated_at_utc=raw.get("updated_at_utc"), windows={str(w.hours): w},
                blended_steps=w.steps, rolling_mae=w.rolling_mae, rolling_coverage=w.rolling_coverage,
            )
        state = AdaptiveState(
            updated_at_utc=raw.get("updated_at_utc"), rolling_mae=raw.get("rolling_mae"),
            rolling_coverage=raw.get("rolling_coverage"), drift_score=float(raw.get("drift_score", 0.0)),
            drift_detected=bool(raw.get("drift_detected", False)), drift_reasons=list(raw.get("drift_reasons", [])),
            blended_global=StepAdaptive(**raw.get("blended_global", {})),
        )
        for key, val in raw.get("windows", {}).items():
            state.windows[str(key)] = WindowAdaptive(
                hours=int(val.get("hours", int(key))), samples=int(val.get("samples", 0)),
                rolling_mae=val.get("rolling_mae"), rolling_coverage=val.get("rolling_coverage"),
                global_calibration=StepAdaptive(**val.get("global_calibration", {})),
                steps={str(k): StepAdaptive(**v) for k, v in val.get("steps", {}).items()},
            )
        state.blended_steps = {str(k): StepAdaptive(**v) for k, v in raw.get("blended_steps", {}).items()}
        return state


def _brier3(y: np.ndarray, p: np.ndarray) -> float:
    o = np.zeros_like(p); o[np.arange(len(y)), y.astype(int)] = 1
    return float(np.mean(np.sum((p - o) ** 2, axis=1)))


def _finite_sample_quantile(values: np.ndarray, coverage: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < ONLINE_CONFORMAL_MIN_SAMPLES:
        return 0.0
    q = min(1.0, np.ceil((n + 1) * float(coverage)) / n)
    try:
        value = float(np.quantile(values, q, method="higher"))
    except TypeError:  # numpy < 1.22 compatibility
        value = float(np.quantile(values, q, interpolation="higher"))
    return float(np.clip(value, 0.0, ONLINE_CONFORMAL_MAX_EXPAND_EUR))


def _conformal_expand_asymmetric(y: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> tuple[float, float]:
    """One-sided finite-sample conformal expansion for an ~80% central interval.

    The desired interval has about 10% probability in each tail.  We therefore
    calibrate the lower and upper misses separately at 90% one-sided coverage.
    This is safer than symmetric widening when live misses are concentrated on
    only one side of the price distribution.  It is production-only and never
    changes the frozen organizer backtest.
    """
    lower_miss = np.maximum(p10 - y, 0.0)
    upper_miss = np.maximum(y - p90, 0.0)
    one_sided = 1.0 - (1.0 - ONLINE_CONFORMAL_TARGET_COVERAGE) / 2.0
    return (
        _finite_sample_quantile(lower_miss, one_sided),
        _finite_sample_quantile(upper_miss, one_sided),
    )


def _conformal_expand(y: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    """Backward-compatible symmetric diagnostic expansion."""
    lo, hi = _conformal_expand_asymmetric(y, p10, p90)
    return float(max(lo, hi))


def _learn_group(g: pd.DataFrame, min_samples: int = 8) -> StepAdaptive:
    s = StepAdaptive(n=len(g))
    if len(g) < min_samples:
        return s
    y = pd.to_numeric(g["imbalance_price"], errors="coerce").to_numpy(float)
    final_p50 = pd.to_numeric(g["p50"], errors="coerce").to_numpy(float)
    p10 = pd.to_numeric(g["p10"], errors="coerce").to_numpy(float)
    p90 = pd.to_numeric(g["p90"], errors="coerce").to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(final_p50) & np.isfinite(p10) & np.isfinite(p90)
    if valid.sum() < min_samples:
        return s
    yv, p50v, p10v, p90v = y[valid], final_p50[valid], p10[valid], p90[valid]
    s.n = int(valid.sum())
    s.mae = float(np.mean(np.abs(yv - p50v)))
    s.price_bias = float(np.median(yv - p50v))
    s.empirical_coverage = float(np.mean((yv >= p10v) & (yv <= p90v)))
    # Shrink very small live samples toward the long-run 80% calibration target
    # for reliability reporting only. Raw empirical coverage remains visible.
    prior_n = float(ONLINE_RELIABILITY_PRIOR_SAMPLES)
    s.shrunk_coverage = float((prior_n * ONLINE_CONFORMAL_TARGET_COVERAGE + s.n * s.empirical_coverage) / (prior_n + s.n))
    lo_expand, hi_expand = _conformal_expand_asymmetric(yv, p10v, p90v)
    s.conformal_expand_low = lo_expand
    s.conformal_expand_high = hi_expand
    s.conformal_expand = max(lo_expand, hi_expand)
    s.q10_extra_delta = float(np.quantile(yv - p10v, 0.10))
    s.q90_extra_delta = float(np.quantile(yv - p90v, 0.90))

    if {"p50_lgb", "p50_cat"}.issubset(g.columns):
        lgb = pd.to_numeric(g.loc[valid, "p50_lgb"], errors="coerce").to_numpy(float)
        cat = pd.to_numeric(g.loc[valid, "p50_cat"], errors="coerce").to_numpy(float)
        good = np.isfinite(lgb) & np.isfinite(cat)
        if good.sum() >= min_samples:
            best_w, best = 0.5, np.inf
            for w in np.linspace(0, 1, 21):
                mae = np.mean(np.abs(yv[good] - (w * lgb[good] + (1 - w) * cat[good])))
                if mae < best: best_w, best = float(w), float(mae)
            s.p50_lgb_weight = best_w
    if {"p50_direct", "p50_spread"}.issubset(g.columns):
        direct = pd.to_numeric(g.loc[valid, "p50_direct"], errors="coerce").to_numpy(float)
        spread = pd.to_numeric(g.loc[valid, "p50_spread"], errors="coerce").to_numpy(float)
        good = np.isfinite(direct) & np.isfinite(spread)
        if good.sum() >= min_samples:
            best_w, best = 0.5, np.inf
            for w in np.linspace(0, 1, 21):
                mae = np.mean(np.abs(yv[good] - (w * spread[good] + (1 - w) * direct[good])))
                if mae < best: best_w, best = float(w), float(mae)
            s.p50_spread_weight = best_w

    if "direction" in g.columns:
        gd = g.loc[valid]
        valid_dir = gd["direction"].notna()
        if valid_dir.sum() >= min_samples:
            yd_raw = gd.loc[valid_dir, "direction"].astype(int).to_numpy()
            yd = np.array([{-1: 0, 0: 1, 1: 2}[int(x)] for x in yd_raw])
            p = gd.loc[valid_dir, ["p_down", "p_none", "p_up"]].to_numpy(float)
            prior = np.bincount(yd, minlength=3).astype(float); prior /= prior.sum()
            best_a, best = 1.0, np.inf
            for a in np.linspace(0, 1, 21):
                score = _brier3(yd, a * p + (1 - a) * prior)
                if score < best: best_a, best = float(a), score
            s.dir_alpha, s.dir_prior = best_a, prior.tolist()
    if "spot_price" in g.columns and g.loc[valid, "spot_price"].notna().sum() >= min_samples:
        gg = g.loc[valid].dropna(subset=["spot_price"])
        high = (gg["imbalance_price"] > gg["spot_price"] + 50).astype(float).to_numpy()
        low = (gg["imbalance_price"] < gg["spot_price"] - 50).astype(float).to_numpy()
        for yy, col, alpha_name, prior_name in [(high,"p_spike_high","hi_alpha","hi_prior"),(low,"p_spike_low","lo_alpha","lo_prior")]:
            pp = gg[col].to_numpy(float); prior = float(np.mean(yy)); ba, bs = 1.0, np.inf
            for a in np.linspace(0, 1, 21):
                score = float(np.mean((a * pp + (1 - a) * prior - yy) ** 2))
                if score < bs: ba, bs = float(a), score
            setattr(s, alpha_name, ba); setattr(s, prior_name, prior)
    return s


def _learn_one_window(df: pd.DataFrame, window_hours: int, min_step_samples: int = 8) -> WindowAdaptive:
    win = WindowAdaptive(hours=int(window_hours), samples=len(df))
    if df.empty: return win
    win.rolling_mae = float(np.mean(np.abs(df["imbalance_price"] - df["p50"])))
    win.rolling_coverage = float(np.mean((df["imbalance_price"] >= df["p10"]) & (df["imbalance_price"] <= df["p90"])))
    win.global_calibration = _learn_group(df, min_samples=ONLINE_CONFORMAL_MIN_SAMPLES)
    for step, g in df.groupby("step"):
        s = _learn_group(g, min_samples=min_step_samples)
        if s.n >= min_step_samples: win.steps[str(int(step))] = s
    return win


def _blend_objects(pieces: list[tuple[float, StepAdaptive]]) -> StepAdaptive:
    if not pieces: return StepAdaptive()
    total = sum(w for w, _ in pieces) or 1.0
    pieces = [(w / total, s) for w, s in pieces]
    b = StepAdaptive(n=max(s.n for _, s in pieces))
    b.mae = sum(w * (s.mae or 0.0) for w, s in pieces)
    b.price_bias = sum(w * s.price_bias for w, s in pieces)
    b.conformal_expand = sum(w * s.conformal_expand for w, s in pieces)
    b.conformal_expand_low = sum(w * s.conformal_expand_low for w, s in pieces)
    b.conformal_expand_high = sum(w * s.conformal_expand_high for w, s in pieces)
    covs = [(w, s) for w, s in pieces if s.empirical_coverage is not None]
    if covs:
        sw = sum(w for w, _ in covs); b.empirical_coverage = sum(w * float(s.empirical_coverage) for w, s in covs) / sw
        shr = [(w, s) for w, s in covs if s.shrunk_coverage is not None]
        if shr:
            sw2 = sum(w for w, _ in shr); b.shrunk_coverage = sum(w * float(s.shrunk_coverage) for w, s in shr) / sw2
    for attr in ("p50_lgb_weight", "p50_spread_weight"):
        vv = [(w, s) for w, s in pieces if getattr(s, attr) is not None]
        if vv:
            sw = sum(w for w, _ in vv); setattr(b, attr, sum(w * float(getattr(s, attr)) for w, s in vv) / sw)
    b.q10_extra_delta = sum(w * s.q10_extra_delta for w, s in pieces)
    b.q90_extra_delta = sum(w * s.q90_extra_delta for w, s in pieces)
    b.dir_alpha = float(np.clip(sum(w * s.dir_alpha for w, s in pieces), 0, 1))
    prior = sum(w * np.asarray(s.dir_prior, float) for w, s in pieces); prior = np.clip(prior,1e-8,None); prior /= prior.sum(); b.dir_prior=prior.tolist()
    b.hi_alpha=float(np.clip(sum(w*s.hi_alpha for w,s in pieces),0,1)); b.hi_prior=float(np.clip(sum(w*s.hi_prior for w,s in pieces),0,1))
    b.lo_alpha=float(np.clip(sum(w*s.lo_alpha for w,s in pieces),0,1)); b.lo_prior=float(np.clip(sum(w*s.lo_prior for w,s in pieces),0,1))
    return b


def _blend_steps(windows: Mapping[str, WindowAdaptive], weights: Mapping[int,float]) -> dict[str,StepAdaptive]:
    all_steps=sorted({s for w in windows.values() for s in w.steps}, key=lambda x:int(x)); out={}
    for step in all_steps:
        pieces=[]
        for w in windows.values():
            s=w.steps.get(step)
            if s is None: continue
            ww=float(weights.get(int(w.hours),0.0))*min(1.0,s.n/24.0)
            if ww>0: pieces.append((ww,s))
        if pieces: out[step]=_blend_objects(pieces)
    return out


def _blend_global(windows: Mapping[str,WindowAdaptive], weights: Mapping[int,float]) -> StepAdaptive:
    pieces=[]
    for w in windows.values():
        s=w.global_calibration
        if s.n < ONLINE_CONFORMAL_MIN_SAMPLES: continue
        ww=float(weights.get(int(w.hours),0.0))*min(1.0,s.n/max(ONLINE_CONFORMAL_FULL_STRENGTH_SAMPLES,1))
        if ww>0: pieces.append((ww,s))
    return _blend_objects(pieces)


def learn_multiwindow_state(window_frames: Mapping[int,pd.DataFrame], min_step_samples:int=8) -> AdaptiveState:
    state=AdaptiveState(updated_at_utc=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    for hours in ONLINE_ADAPT_WINDOWS_HOURS:
        state.windows[str(int(hours))]=_learn_one_window(window_frames.get(int(hours),pd.DataFrame()),int(hours),min_step_samples)
    state.blended_steps=_blend_steps(state.windows,ONLINE_ADAPT_WINDOW_WEIGHTS)
    state.blended_global=_blend_global(state.windows,ONLINE_ADAPT_WINDOW_WEIGHTS)
    fast=state.windows.get(str(min(ONLINE_ADAPT_WINDOWS_HOURS))); stable=state.windows.get(str(max(ONLINE_ADAPT_WINDOWS_HOURS)))
    if fast: state.rolling_mae, state.rolling_coverage = fast.rolling_mae, fast.rolling_coverage
    reasons=[]; score=0.0
    if fast and stable and fast.rolling_mae and stable.rolling_mae and stable.rolling_mae>1e-9:
        ratio=fast.rolling_mae/stable.rolling_mae
        if ratio>=ONLINE_DRIFT_MAE_RATIO:
            reasons.append(f"24h MAE is {ratio:.2f}x the 14d MAE"); score+=min(1.0,(ratio-1.0)/max(ONLINE_DRIFT_MAE_RATIO-1.0,1e-6))
    if fast and fast.rolling_coverage is not None and fast.samples >= ONLINE_DRIFT_MIN_SAMPLES:
        dev=abs(float(fast.rolling_coverage)-ONLINE_CONFORMAL_TARGET_COVERAGE)
        if dev>=ONLINE_DRIFT_COVERAGE_DEVIATION:
            reasons.append(f"24h interval coverage deviates from {ONLINE_CONFORMAL_TARGET_COVERAGE:.2f} by {dev:.3f}"); score+=min(1.0,dev/max(ONLINE_DRIFT_COVERAGE_DEVIATION,1e-6))
    state.drift_score=float(min(1.0,score/2.0 if score else 0.0)); state.drift_detected=bool(reasons); state.drift_reasons=reasons
    return state


def learn_adaptive_state(df:pd.DataFrame,window_hours:int=24,min_step_samples:int=8)->AdaptiveState:
    return learn_multiwindow_state({int(window_hours):df},min_step_samples=min_step_samples)


def attach_adaptive_meta_features(features:pd.DataFrame,state:AdaptiveState)->pd.DataFrame:
    out=features.copy(); steps=out["step"].astype(int).tolist()
    for hours in ONLINE_ADAPT_WINDOWS_HOURS:
        win=state.windows.get(str(int(hours))); out[f"online_mae_{hours}h"]=np.nan if not win else win.rolling_mae; out[f"online_coverage_{hours}h"]=np.nan if not win else win.rolling_coverage
        biases=[]; step_mae=[]
        for step in steps:
            s=win.steps.get(str(step)) if win else None; biases.append(np.nan if s is None else s.price_bias); step_mae.append(np.nan if s is None else s.mae)
        out[f"online_bias_{hours}h"]=biases; out[f"online_step_mae_{hours}h"]=step_mae
    out["online_global_conformal_expand"]=float(state.blended_global.conformal_expand)
    out["online_global_calibration_samples"]=int(state.blended_global.n)
    out["online_drift_score"]=float(state.drift_score); out["online_drift_detected"]=int(state.drift_detected)
    out["online_blended_bias"]=[np.nan if state.blended_steps.get(str(s)) is None else state.blended_steps[str(s)].price_bias for s in steps]
    out["online_blended_step_mae"]=[np.nan if state.blended_steps.get(str(s)) is None else state.blended_steps[str(s)].mae for s in steps]
    return out


def apply_adaptive(pred:pd.DataFrame,steps:pd.Series,state:AdaptiveState)->pd.DataFrame:
    """Apply production-only online adaptation with small-sample safeguards.

    V10 adds three protections/improvements:
    1) pooled direct-vs-spread adaptation and a shrunken global bias can start
       after only a handful of settled outcomes, instead of waiting for eight
       examples at every individual horizon;
    2) lower and upper conformal expansion are learned separately; and
    3) live reliability is sample-aware, so twelve noisy outcomes warn the user
       but cannot overwhelm the much larger trained-model evidence.
    """
    out=pred.copy().reset_index(drop=True)
    out["reliability_model"] = pd.to_numeric(out.get("reliability", 0.5), errors="coerce").fillna(0.5)
    biases=np.zeros(len(out),dtype=float); applied=np.zeros(len(out),dtype=bool)
    conformal=np.zeros(len(out),dtype=float)
    conformal_low=np.zeros(len(out),dtype=float); conformal_high=np.zeros(len(out),dtype=float)
    global_mix_strength=np.zeros(len(out),dtype=float); global_bias_applied=np.zeros(len(out),dtype=float)
    g=state.blended_global

    # ---- pooled centre adaptation -------------------------------------------------
    # This is deliberately shrunken.  It is useful very early in production because
    # direct-vs-spread performance can change by market regime, but a dozen points
    # must not be allowed to replace the validated champion outright.
    sample_conf=min(1.0, g.n/max(float(ONLINE_CONFORMAL_FULL_STRENGTH_SAMPLES),1.0))
    if g.n >= ONLINE_GLOBAL_ADAPT_MIN_SAMPLES:
        if g.p50_spread_weight is not None and {"p50_direct","p50_spread"}.issubset(out.columns):
            direct=pd.to_numeric(out["p50_direct"],errors="coerce").to_numpy(float)
            spread=pd.to_numeric(out["p50_spread"],errors="coerce").to_numpy(float)
            good=np.isfinite(direct)&np.isfinite(spread)
            if good.any():
                dynamic=float(g.p50_spread_weight)*spread+(1-float(g.p50_spread_weight))*direct
                strength=min(0.55,0.15+0.40*sample_conf)
                basep=out["p50"].to_numpy(dtype=float, copy=True)
                basep[good]=(1-strength)*basep[good]+strength*dynamic[good]
                out["p50"]=basep
                global_mix_strength[good]=strength
                applied[good]=True
        # L1-optimal pooled residual correction.  Cap and shrink aggressively.
        if np.isfinite(g.price_bias) and abs(float(g.price_bias))>1e-9:
            bstrength=min(0.35,0.35*sample_conf)
            b=float(np.clip(g.price_bias,-30.0,30.0))*bstrength
            out["p50"]=out["p50"].to_numpy(float)+b
            global_bias_applied[:]=b
            applied[:]=True

    # ---- pooled asymmetric conformal calibration --------------------------------
    if g.n>=ONLINE_CONFORMAL_MIN_SAMPLES and (g.conformal_expand_low>0 or g.conformal_expand_high>0 or g.conformal_expand>0):
        # Faster than V9 at the beginning because the purpose is safety/coverage,
        # but still capped below full strength until more evidence accumulates.
        cal_strength=min(1.0,0.25+0.75*sample_conf)
        lo=float(g.conformal_expand_low or g.conformal_expand) * cal_strength
        hi=float(g.conformal_expand_high or g.conformal_expand) * cal_strength
        out["p10"] = out["p10"].to_numpy(float)-lo
        out["p90"] = out["p90"].to_numpy(float)+hi
        conformal_low[:]=lo; conformal_high[:]=hi; conformal[:]=max(lo,hi); applied[:]=True

    # ---- horizon-specific adaptation ---------------------------------------------
    for i,step in enumerate(pd.Series(steps).astype(int).tolist()):
        s=state.blended_steps.get(str(step))
        if s is None or s.n<8: continue
        strength=min(1.0,s.n/24.0)
        if s.p50_spread_weight is not None and {"p50_direct","p50_spread"}.issubset(out.columns) and np.isfinite(out.at[i,"p50_spread"]):
            dynamic=s.p50_spread_weight*out.at[i,"p50_spread"]+(1-s.p50_spread_weight)*out.at[i,"p50_direct"]; out.at[i,"p50"]=(1-strength)*out.at[i,"p50"]+strength*dynamic
        elif s.p50_lgb_weight is not None and {"p50_lgb","p50_cat"}.issubset(out.columns):
            dynamic=s.p50_lgb_weight*out.at[i,"p50_lgb"]+(1-s.p50_lgb_weight)*out.at[i,"p50_cat"]; out.at[i,"p50"]=(1-strength)*out.at[i,"p50"]+strength*dynamic
        bias=strength*s.price_bias; biases[i]=bias; out.at[i,"p50"]+=bias
        out.at[i,"p10"]+=strength*s.q10_extra_delta; out.at[i,"p90"]+=strength*s.q90_extra_delta
        p=np.array([out.at[i,"p_down"],out.at[i,"p_none"],out.at[i,"p_up"]],float); prior=np.array(s.dir_prior,float); p=s.dir_alpha*p+(1-s.dir_alpha)*prior; p=np.clip(p,1e-8,1); p/=p.sum(); out.loc[i,["p_down","p_none","p_up"]]=p
        out.at[i,"p_spike_high"]=s.hi_alpha*out.at[i,"p_spike_high"]+(1-s.hi_alpha)*s.hi_prior; out.at[i,"p_spike_low"]=s.lo_alpha*out.at[i,"p_spike_low"]+(1-s.lo_alpha)*s.lo_prior
        applied[i]=True

    out["p10"]=np.minimum(out["p10"].to_numpy(float),out["p50"].to_numpy(float)); out["p90"]=np.maximum(out["p90"].to_numpy(float),out["p50"].to_numpy(float))
    total=out["p_spike_high"]+out["p_spike_low"]; mask=total>0.999; out.loc[mask,"p_spike_high"]*=0.999/total[mask]; out.loc[mask,"p_spike_low"]*=0.999/total[mask]

    # ---- reliability --------------------------------------------------------------
    perf=0.5 if state.rolling_mae is None else float(np.clip(1-state.rolling_mae/150.0,0,1))
    raw_cov=ONLINE_CONFORMAL_TARGET_COVERAGE if state.rolling_coverage is None else float(state.rolling_coverage)
    prior_n=float(ONLINE_RELIABILITY_PRIOR_SAMPLES)
    shrunk_cov=float((prior_n*ONLINE_CONFORMAL_TARGET_COVERAGE + g.n*raw_cov)/(prior_n+g.n)) if g.n>0 else ONLINE_CONFORMAL_TARGET_COVERAGE
    cov_rel=float(np.clip(1-abs(shrunk_cov-ONLINE_CONFORMAL_TARGET_COVERAGE)/max(ONLINE_CONFORMAL_TARGET_COVERAGE,1e-6),0,1))
    empirical_quality=float(np.clip(0.45*perf+0.55*cov_rel,0,1))
    # Live evidence weight ramps from 0 -> 30%; it is not a fixed 25% anymore.
    empirical_weight=0.30*sample_conf
    empirical=(1-sample_conf)*0.5+sample_conf*empirical_quality
    out["reliability_empirical"]=empirical
    out["reliability_calibration"]=cov_rel
    out["reliability_coverage_raw"]=raw_cov
    out["reliability_coverage_shrunk"]=shrunk_cov
    out["reliability_live_sample_confidence"]=sample_conf
    # Drift can only reduce confidence in proportion to the amount of live evidence.
    drift_penalty=1.0-0.20*float(state.drift_score)*sample_conf
    out["reliability"] = np.clip(((1-empirical_weight)*out["reliability_model"] + empirical_weight*empirical_quality)*drift_penalty,0,1)

    out["adaptive_applied"]=applied
    out["adaptive_bias"]=biases
    out["global_p50_bias_applied"]=global_bias_applied
    out["global_p50_mix_strength"]=global_mix_strength
    out["conformal_expand_eur"]=conformal
    out["conformal_expand_low_eur"]=conformal_low
    out["conformal_expand_high_eur"]=conformal_high
    out["drift_score"]=float(state.drift_score)
    return out

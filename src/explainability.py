from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from .models_v8 import ModelBundleV8, load_any_bundle
from .v8_features import augment_v8_features


SOURCE_GROUP_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Spot & market price", ("spot", "imbalance_price", "price_momentum", "recent_price")),
    ("Production plan", ("production_fc", "production_plan", "fi_production")),
    ("Wind", ("wind",)),
    ("Solar", ("solar",)),
    ("Consumption / load", ("consumption", "load_")),
    ("mFRR", ("mfrr",)),
    ("aFRR", ("afrr",)),
    ("Frequency", ("frequency",)),
    ("Cross-border", ("flow", "border", "capacity", "scheduled", "ntc", "net_import")),
    ("CHP", ("chp",)),
    ("Battery", ("bess", "battery")),
    ("Shortage / system state", ("shortage", "power_system_state", "system_stress")),
    ("Nuclear", ("nuclear",)),
    ("Hydro", ("hydro",)),
    ("Weather", ("weather_", "temperature", "pressure", "precipitation", "cloud")),
    ("ENTSO-E MCP", ("mcp_",)),
    ("Calendar / horizon", ("calendar", "cal_", "hour", "weekday", "step", "quarter", "horizon")),
]

LIVE_SIGNAL_GROUPS: dict[str, tuple[str, ...]] = {
    "Production plan": ("fi_production_fc_rt", "fi_production"),
    "Wind": ("fi_wind_fc_da", "fi_wind_generation_rt", "fi_wind_capacity_rt", "weather_wind_100m_weighted", "weather_wind_10m_weighted"),
    "Solar": ("fi_solar_fc_da", "weather_solar_mean"),
    "Consumption": ("fi_consumption", "fi_consumption_fc_da"),
    "mFRR": ("fi_mfrr_up_activated", "fi_mfrr_down_activated", "fi_mfrr_up_price", "fi_mfrr_down_price", "fi_mfrr_direction"),
    "aFRR": ("fi_afrr_up_price", "fi_afrr_down_price", "mcp_fi_afrr_up_capacity_price", "mcp_fi_afrr_down_capacity_price", "mcp_fi_afrr_up_capacity_mw", "mcp_fi_afrr_down_capacity_mw"),
    "Frequency": ("fi_frequency_rt",),
    "Cross-border flows & capacity": ("fi_se1_flow", "fi_se3_flow", "fi_no4_flow", "fi_ee_flow", "fi_north_south_flow_rt", "fi_net_import_export_rt", "mcp_sched_", "mcp_ntc_"),
    "CHP": ("fi_chp_district_heat_rt", "fi_chp_industrial_rt"),
    "Battery": ("fi_bess_discharge_rt",),
    "Shortage / power-system state": ("fi_shortage_status_rt", "fi_power_system_state_rt"),
    "Nuclear": ("fi_nuclear_generation_rt",),
    "Hydro": ("fi_hydro_generation_rt",),
    "ENTSO-E MCP neighbours": ("mcp_se1_", "mcp_se3_", "mcp_ee_", "mcp_no4_", "mcp_fi_generation_unavailable_mw"),
    "Weather": ("weather_", "weather_obs_"),
}


def _group_feature(name: str) -> str:
    low = name.lower()
    # Specific market categories before the broad MCP label.
    for group, needles in SOURCE_GROUP_RULES:
        if group == "ENTSO-E MCP":
            continue
        if any(n in low for n in needles):
            return group
    if "mcp_" in low:
        return "ENTSO-E MCP"
    return "Other"


def _norm_importance(model, feature_names: list[str]) -> dict[str, float]:
    try:
        imp = np.asarray(model.feature_importances_, dtype=float)
    except Exception:
        try:
            imp = np.asarray(model.get_feature_importance(), dtype=float)
        except Exception:
            return {}
    if len(imp) != len(feature_names):
        return {}
    imp = np.maximum(imp, 0)
    total = imp.sum()
    if total <= 0:
        return {}
    return {f: float(v / total) for f, v in zip(feature_names, imp)}


def grouped_model_importance(model_path: Path) -> pd.DataFrame:
    bundle = load_any_bundle(model_path)
    base = bundle.base if isinstance(bundle, ModelBundleV8) else bundle
    p50_maps: list[dict[str, float]] = []
    direction_maps: list[dict[str, float]] = []
    p50_maps.append(_norm_importance(base.q50_lgb, list(base.feature_names)))
    p50_maps.append(_norm_importance(base.q50_cat, list(base.cat_feature_names)))
    direction_maps.append(_norm_importance(base.direction, list(base.feature_names)))
    if isinstance(bundle, ModelBundleV8):
        p50_maps.extend([
            _norm_importance(bundle.spread_global_lgb, list(bundle.spread_feature_names)),
            _norm_importance(bundle.spread_global_cat, list(bundle.spread_cat_feature_names)),
            _norm_importance(bundle.spread_down_lgb, list(bundle.spread_feature_names)),
            _norm_importance(bundle.spread_up_lgb, list(bundle.spread_feature_names)),
        ])

    def aggregate(maps: list[dict[str, float]]) -> dict[str, float]:
        by = defaultdict(float)
        valid = [m for m in maps if m]
        if not valid:
            return {}
        for m in valid:
            for f, v in m.items():
                by[_group_feature(f)] += v / len(valid)
        return dict(by)

    p50 = aggregate(p50_maps); direction = aggregate(direction_maps)
    groups = sorted(set(p50) | set(direction) | {g for g,_ in SOURCE_GROUP_RULES})
    out = pd.DataFrame({
        "source_group": groups,
        "p50_importance": [p50.get(g, 0.0) for g in groups],
        "direction_importance": [direction.get(g, 0.0) for g in groups],
    })
    return out.sort_values("p50_importance", ascending=False).reset_index(drop=True)


def select_live_group(long_df: pd.DataFrame, group: str) -> pd.DataFrame:
    patterns = LIVE_SIGNAL_GROUPS.get(group, ())
    if long_df.empty or not patterns:
        return long_df.iloc[:0].copy()
    sid = long_df["series_id"].astype(str)
    mask = pd.Series(False, index=long_df.index)
    for p in patterns:
        if p.endswith("_"):
            mask |= sid.str.startswith(p)
        else:
            mask |= sid.eq(p)
    return long_df[mask].copy()


def local_lgb_group_contributions(model_path: Path, feature_frame: pd.DataFrame, row_index: int = 0) -> pd.DataFrame:
    """Signed LightGBM evidence for one forecast row.

    This is a transparent *model evidence* view, not a causal claim.  For a V8+
    bundle it reports the direct-price LightGBM contribution and the global
    spread LightGBM contribution (the latter is the predicted imbalance-minus-
    spot spread).  The full production P50 also contains CatBoost, direction
    gating, calibration and online adaptation, so these columns explain the two
    main tree experts rather than pretending to be an exact decomposition of
    every ensemble operation.
    """
    if feature_frame.empty:
        return pd.DataFrame(columns=["source_group","direct_price_eur_mwh","spread_eur_mwh"])
    bundle=load_any_bundle(model_path)
    base=bundle.base if isinstance(bundle,ModelBundleV8) else bundle
    row=feature_frame.iloc[[int(row_index)]].copy()

    def contrib(model, names, f):
        for c in names:
            if c not in f: f[c]=np.nan
        try:
            vals=np.asarray(model.predict(f[names], pred_contrib=True),dtype=float)
        except Exception:
            return {}
        if vals.ndim!=2 or vals.shape[1] < len(names):
            return {}
        by=defaultdict(float)
        for name,val in zip(names,vals[0,:len(names)]):
            if np.isfinite(val):
                by[_group_feature(name)] += float(val)
        return dict(by)

    direct=contrib(base.q50_lgb,list(base.feature_names),row.copy())
    spread={}
    if isinstance(bundle,ModelBundleV8):
        aug=augment_v8_features(row.copy())
        spread=contrib(bundle.spread_global_lgb,list(bundle.spread_feature_names),aug)
    groups=sorted(set(direct)|set(spread), key=lambda g: abs(direct.get(g,0))+abs(spread.get(g,0)), reverse=True)
    return pd.DataFrame({
        "source_group":groups,
        "direct_price_eur_mwh":[direct.get(g,0.0) for g in groups],
        "spread_eur_mwh":[spread.get(g,0.0) for g in groups],
    })

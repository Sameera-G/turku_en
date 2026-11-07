from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    FUTURE_SERIES, HISTORY_SERIES, HISTORY_LAGS_Q, STEP_MINUTES, SPIKE_MARGIN,
    ONLINE_FUTURE_SERIES, ONLINE_HISTORY_SERIES,
)
from .data import load_wide, legal_value


def make_origin_grid(origins: pd.DatetimeIndex, steps: int = 40) -> pd.DataFrame:
    origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    origin = np.repeat(origins.values, steps)
    step = np.tile(np.arange(1, steps + 1), len(origins))
    df = pd.DataFrame({"forecast_origin_utc": pd.to_datetime(origin, utc=True), "step": step})
    df["delivery_start_utc"] = df["forecast_origin_utc"] + pd.to_timedelta(
        (df["step"] - 1) * STEP_MINUTES, unit="m"
    )
    return df


def _cyclic(df: pd.DataFrame) -> None:
    delivery = df["delivery_start_utc"]
    h = delivery.dt.hour + delivery.dt.minute / 60.0
    dow = delivery.dt.dayofweek
    doy = delivery.dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["horizon_hours"] = (df["step"] - 1) * 0.25
    df["horizon_bucket"] = pd.cut(
        df["horizon_hours"], bins=[-0.01, 1, 2, 4, 6, 8, 10], labels=False, include_lowest=True
    ).astype(float)


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-6) -> pd.Series:
    return a / b.abs().clip(lower=eps)


def build_features(origins: pd.DatetimeIndex, include_labels: bool, include_online_context: bool = False) -> pd.DataFrame:
    """Leakage-safe V3 feature builder.

    V3 adds market-structure features motivated by Turku Energia's 2025
    financial statement/business review without feeding annual accounting
    values into the short-horizon model. The useful findings translated into
    model logic are:
      * stronger weather/renewable dependence;
      * cross-border capacity/outage sensitivity;
      * Nordic market/hydro context;
      * growing flexible electric-load/boiler response in low-price periods;
      * purchase/sales volume and forecast-error risk.
    """
    values, usable = load_wide()
    df = make_origin_grid(origins)
    _cyclic(df)

    extra: dict[str, np.ndarray] = {}
    for s in FUTURE_SERIES:
        extra[f"future__{s}"] = legal_value(
            values, usable, df["delivery_start_utc"], df["forecast_origin_utc"], s
        )

    for s in HISTORY_SERIES:
        for lag_q in HISTORY_LAGS_Q:
            lag_ts = df["forecast_origin_utc"] - pd.to_timedelta(lag_q * STEP_MINUTES, unit="m")
            extra[f"lag{lag_q}q__{s}"] = legal_value(
                values, usable, lag_ts, df["forecast_origin_utc"], s
            )

    if include_online_context:
        # Live-only signals are never requested by the locked hackathon
        # backtest. Exact values used at each live origin are persisted in the
        # online store and can later be admitted to a challenger model.
        for s in ONLINE_FUTURE_SERIES:
            extra[f"future__{s}"] = legal_value(
                values, usable, df["delivery_start_utc"], df["forecast_origin_utc"], s
            )
        for s in ONLINE_HISTORY_SERIES:
            for lag_q in (1, 4, 8, 16):
                lag_ts = df["forecast_origin_utc"] - pd.to_timedelta(lag_q * STEP_MINUTES, unit="m")
                extra[f"lag{lag_q}q__{s}"] = legal_value(
                    values, usable, lag_ts, df["forecast_origin_utc"], s
                )

    df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)

    spot = df.get("future__fi_spot_price")
    wind = df.get("future__fi_wind_fc_da")
    solar = df.get("future__fi_solar_fc_da")
    load = df.get("future__fi_consumption_fc_da")

    if spot is not None:
        df["spot_price"] = spot
        # Flexible electric load / electric-boiler opportunity proxy. It is a
        # market-regime signal only; it does not claim Turku Energia's actual dispatch.
        df["low_spot_signal"] = 1.0 / (1.0 + np.exp((spot - 15.0) / 12.0))
        df["negative_spot_flag"] = (spot < 0).astype(np.int8)

    if wind is not None:
        df["renewable_fc"] = wind.fillna(0) + (solar.fillna(0) if solar is not None else 0)
    if load is not None and wind is not None:
        df["net_load_proxy"] = load - wind.fillna(0) - (solar.fillna(0) if solar is not None else 0)
        df["renewable_share_proxy"] = _safe_div(df["renewable_fc"], load)
        if spot is not None:
            df["sector_coupling_flex_signal"] = df["low_spot_signal"] * df["renewable_share_proxy"].clip(0, 2)

    # Realised system balance and volume-risk proxies.
    p1 = df.get("lag1q__fi_production")
    c1 = df.get("lag1q__fi_consumption")
    if p1 is not None and c1 is not None:
        df["recent_balance_proxy"] = p1 - c1
        if load is not None:
            df["recent_load_forecast_error"] = c1 - load
            df["recent_load_error_pct"] = _safe_div(c1 - load, load)

    # Imbalance spread / momentum / volatility.
    for lag_q in (1, 2, 4, 8, 16, 32, 96):
        imb = df.get(f"lag{lag_q}q__fi_imbalance_price")
        if imb is not None and spot is not None:
            df[f"lag{lag_q}q__imb_minus_delivery_spot"] = imb - spot

    lag1 = df.get("lag1q__fi_imbalance_price")
    lag4 = df.get("lag4q__fi_imbalance_price")
    lag8 = df.get("lag8q__fi_imbalance_price")
    lag16 = df.get("lag16q__fi_imbalance_price")
    lag32 = df.get("lag32q__fi_imbalance_price")
    lag96 = df.get("lag96q__fi_imbalance_price")
    if lag1 is not None and lag4 is not None:
        df["price_momentum_1h"] = lag1 - lag4
    if lag4 is not None and lag16 is not None:
        df["price_momentum_4h"] = lag4 - lag16
    if lag1 is not None and lag32 is not None:
        df["price_momentum_8h"] = lag1 - lag32
    if lag1 is not None and lag96 is not None:
        df["price_vs_24h"] = lag1 - lag96

    price_lag_cols = [f"lag{q}q__fi_imbalance_price" for q in (1,2,4,8,16,32) if f"lag{q}q__fi_imbalance_price" in df]
    if price_lag_cols:
        df["recent_price_mean"] = df[price_lag_cols].mean(axis=1)
        df["recent_price_std"] = df[price_lag_cols].std(axis=1)
        df["recent_price_range"] = df[price_lag_cols].max(axis=1) - df[price_lag_cols].min(axis=1)

    # Reserve stress and direction persistence.
    up_act = df.get("lag4q__fi_mfrr_up_activated")
    down_act = df.get("lag4q__fi_mfrr_down_activated")
    if up_act is not None and down_act is not None:
        df["mfrr_net_activation_1h"] = up_act.fillna(0) - down_act.fillna(0)
        df["mfrr_total_activation_1h"] = up_act.fillna(0) + down_act.fillna(0)
    up_price = df.get("lag4q__fi_mfrr_up_price")
    down_price = df.get("lag4q__fi_mfrr_down_price")
    if up_price is not None and down_price is not None:
        df["mfrr_price_spread_1h"] = up_price - down_price
    dir_cols = [f"lag{q}q__fi_mfrr_direction" for q in (1,2,4,8,16) if f"lag{q}q__fi_mfrr_direction" in df]
    if dir_cols:
        df["recent_direction_mean"] = df[dir_cols].mean(axis=1)
        df["recent_up_share"] = (df[dir_cols] > 0).mean(axis=1)
        df["recent_down_share"] = (df[dir_cols] < 0).mean(axis=1)

    # Cross-border market stress. Financial statement discussion of Estlink 2
    # and Aurora Line motivates explicit capacity-utilisation and reversal signals.
    flow_series = ["fi_se1_flow", "fi_se3_flow", "fi_no4_flow", "fi_ee_flow"]
    lag4_flow_cols = [f"lag4q__{s}" for s in flow_series if f"lag4q__{s}" in df]
    lag8_flow_cols = [f"lag8q__{s}" for s in flow_series if f"lag8q__{s}" in df]
    if lag4_flow_cols:
        df["net_export_1h"] = df[lag4_flow_cols].sum(axis=1, min_count=1)
        df["gross_border_flow_1h"] = df[lag4_flow_cols].abs().sum(axis=1, min_count=1)
    if lag8_flow_cols:
        df["net_export_2h"] = df[lag8_flow_cols].sum(axis=1, min_count=1)
    if "net_export_1h" in df and "net_export_2h" in df:
        df["net_export_change"] = df["net_export_1h"] - df["net_export_2h"]
        df["border_flow_reversal"] = (np.sign(df["net_export_1h"]) != np.sign(df["net_export_2h"])).astype(np.int8)
    if load is not None and "net_export_1h" in df:
        df["net_export_to_load"] = _safe_div(df["net_export_1h"], load)

    ee_flow = df.get("lag4q__fi_ee_flow")
    fi_ee_cap = df.get("future__fi_ee_capacity_da")
    ee_fi_cap = df.get("future__ee_fi_capacity_da")
    if ee_flow is not None and fi_ee_cap is not None and ee_fi_cap is not None:
        # Positive source flow = FI export to EE, negative = import to FI.
        directional_cap = pd.Series(np.where(ee_flow >= 0, fi_ee_cap, ee_fi_cap), index=df.index)
        df["ee_capacity_utilization_proxy"] = (_safe_div(ee_flow.abs(), directional_cap)).clip(0, 3)
        df["ee_capacity_stress_flag"] = (df["ee_capacity_utilization_proxy"] > 0.85).astype(np.int8)

    # Neighbour / Nordic market pressure. This is a market-pressure proxy, not
    # a direct hydro-reservoir measurement. The financial statement motivates
    # adding Nordic context, while the starter package provides neighbour prices.
    neigh = [c for c in [
        "lag8q__se1_imbalance_price_long", "lag8q__se3_imbalance_price_long",
        "lag8q__ee_imbalance_price_long"
    ] if c in df]
    if neigh:
        df["neighbor_imbalance_mean"] = df[neigh].mean(axis=1)
        df["neighbor_imbalance_std"] = df[neigh].std(axis=1)
        if lag8 is not None:
            df["fi_vs_neighbor_price"] = lag8 - df["neighbor_imbalance_mean"]
        no4 = df.get("lag8q__fi_no4_flow")
        if no4 is not None:
            df["nordic_market_pressure_proxy"] = df["neighbor_imbalance_mean"] - 0.01 * no4

    # Forecast ramps over the 10-hour surface: key for weather-driven wind risk.
    for col, out in [
        ("future__fi_wind_fc_da", "wind_fc_change_from_origin"),
        ("future__fi_consumption_fc_da", "load_fc_change_from_origin"),
        ("future__fi_solar_fc_da", "solar_fc_change_from_origin"),
        ("future__fi_spot_price", "spot_change_from_origin"),
    ]:
        if col in df:
            base = df.groupby("forecast_origin_utc", sort=False)[col].transform("first")
            df[out] = df[col] - base

    if "wind_fc_change_from_origin" in df and load is not None:
        df["wind_ramp_stress"] = _safe_div(df["wind_fc_change_from_origin"].abs(), load)
    if "load_fc_change_from_origin" in df and load is not None:
        df["load_ramp_stress"] = _safe_div(df["load_fc_change_from_origin"].abs(), load)

    # Composite regime indicators. These are deliberately simple, transparent
    # and derived only from legally available features.
    if "recent_price_std" in df and "wind_ramp_stress" in df:
        df["market_volatility_regime"] = np.log1p(df["recent_price_std"].clip(lower=0)) + 10.0 * df["wind_ramp_stress"].fillna(0)
    if "ee_capacity_utilization_proxy" in df and "gross_border_flow_1h" in df:
        df["cross_border_stress"] = df["ee_capacity_utilization_proxy"].fillna(0) + np.log1p(df["gross_border_flow_1h"].clip(lower=0)) / 10.0

    if include_online_context:
        # Weather ramps from the same forecast vintage.
        for col, out_name in [
            ("future__weather_wind_100m_weighted", "weather_wind_ramp_3h"),
            ("future__weather_temperature_mean", "weather_temperature_ramp_3h"),
        ]:
            if col in df:
                base = df.groupby("forecast_origin_utc", sort=False)[col].transform("first")
                df[out_name] = df[col] - base

        # Live generation mix / flexibility / system-stress proxies. These are
        # transparent state summaries, not direct trading instructions.
        wind_rt = df.get("lag1q__fi_wind_generation_rt")
        hydro_rt = df.get("lag1q__fi_hydro_generation_rt")
        nuclear_rt = df.get("lag1q__fi_nuclear_generation_rt")
        prod_rt = df.get("lag1q__fi_production")
        if prod_rt is not None:
            renewable = pd.Series(0.0, index=df.index)
            any_src = False
            for x in (wind_rt, hydro_rt):
                if x is not None:
                    renewable = renewable + x.fillna(0)
                    any_src = True
            if any_src:
                df["live_generation_mix_renewable_share"] = _safe_div(renewable, prod_rt).clip(0, 2)

        # V10 ENTSO-E MCP live market context. These are exact current-origin
        # snapshots only; they are never backfilled into the locked challenge
        # backtest. A challenger may use them after enough saved snapshots exist.
        mcp_neighbor_prices = [c for c in [
            "future__mcp_se1_day_ahead_price", "future__mcp_se3_day_ahead_price",
            "future__mcp_ee_day_ahead_price", "future__mcp_no4_day_ahead_price",
        ] if c in df]
        if mcp_neighbor_prices:
            df["mcp_neighbor_da_price_mean"] = df[mcp_neighbor_prices].mean(axis=1)
            df["mcp_neighbor_da_price_std"] = df[mcp_neighbor_prices].std(axis=1)
            if "future__fi_spot_price" in df:
                df["mcp_fi_vs_neighbor_da_spread"] = df["future__fi_spot_price"] - df["mcp_neighbor_da_price_mean"]

        mcp_neighbor_load = [c for c in [
            "future__mcp_se1_load_forecast", "future__mcp_se3_load_forecast",
            "future__mcp_ee_load_forecast", "future__mcp_no4_load_forecast",
        ] if c in df]
        if mcp_neighbor_load:
            df["mcp_neighbor_load_forecast_mean"] = df[mcp_neighbor_load].mean(axis=1)

        imports = [c for c in ["future__mcp_sched_se1_fi","future__mcp_sched_se3_fi","future__mcp_sched_ee_fi","future__mcp_sched_no4_fi"] if c in df]
        exports = [c for c in ["future__mcp_sched_fi_se1","future__mcp_sched_fi_se3","future__mcp_sched_fi_ee","future__mcp_sched_fi_no4"] if c in df]
        if imports or exports:
            imp = df[imports].sum(axis=1, min_count=1) if imports else pd.Series(0.0, index=df.index)
            exp = df[exports].sum(axis=1, min_count=1) if exports else pd.Series(0.0, index=df.index)
            df["mcp_scheduled_net_import_mw"] = imp - exp
            df["mcp_scheduled_gross_exchange_mw"] = imp.abs() + exp.abs()

        ntc_imports = [c for c in ["future__mcp_ntc_se1_fi","future__mcp_ntc_se3_fi","future__mcp_ntc_ee_fi","future__mcp_ntc_no4_fi"] if c in df]
        ntc_exports = [c for c in ["future__mcp_ntc_fi_se1","future__mcp_ntc_fi_se3","future__mcp_ntc_fi_ee","future__mcp_ntc_fi_no4"] if c in df]
        if ntc_imports:
            df["mcp_import_ntc_total_mw"] = df[ntc_imports].sum(axis=1, min_count=1)
        if ntc_exports:
            df["mcp_export_ntc_total_mw"] = df[ntc_exports].sum(axis=1, min_count=1)
        if "mcp_scheduled_net_import_mw" in df and "mcp_import_ntc_total_mw" in df:
            df["mcp_import_capacity_pressure"] = _safe_div(df["mcp_scheduled_net_import_mw"].clip(lower=0), df["mcp_import_ntc_total_mw"]).clip(0, 3)

        if "future__mcp_fi_afrr_up_capacity_price" in df and "future__mcp_fi_afrr_down_capacity_price" in df:
            df["mcp_afrr_capacity_price_spread"] = df["future__mcp_fi_afrr_up_capacity_price"] - df["future__mcp_fi_afrr_down_capacity_price"]
        if "future__mcp_fi_afrr_up_capacity_mw" in df and "future__mcp_fi_afrr_down_capacity_mw" in df:
            df["mcp_afrr_capacity_volume_imbalance"] = df["future__mcp_fi_afrr_up_capacity_mw"] - df["future__mcp_fi_afrr_down_capacity_mw"]
        if "future__mcp_fi_generation_unavailable_mw" in df:
            df["mcp_generation_unavailable_mw"] = df["future__mcp_fi_generation_unavailable_mw"]

        # V10 closer-to-delivery market and forecast signals.  These are live
        # snapshot features only, so they can be used by future challengers
        # without contaminating the locked historical backtest.
        if "future__mcp_fi_intraday_price" in df and "future__fi_spot_price" in df:
            df["mcp_intraday_vs_da_spread"] = df["future__mcp_fi_intraday_price"] - df["future__fi_spot_price"]
            df["mcp_intraday_abs_move"] = df["mcp_intraday_vs_da_spread"].abs()
        if "future__mcp_fi_wind_forecast" in df and "future__fi_wind_fc_da" in df:
            df["mcp_vs_fingrid_wind_fc_diff"] = df["future__mcp_fi_wind_forecast"] - df["future__fi_wind_fc_da"]
        if "future__mcp_fi_solar_forecast" in df and "future__fi_solar_fc_da" in df:
            df["mcp_vs_fingrid_solar_fc_diff"] = df["future__mcp_fi_solar_forecast"] - df["future__fi_solar_fc_da"]
        if "future__mcp_fi_generation_forecast" in df and "future__fi_production_fc_rt" in df:
            df["mcp_vs_fingrid_generation_fc_diff"] = df["future__mcp_fi_generation_forecast"] - df["future__fi_production_fc_rt"]

        # Recent realized balancing-price context from MCP.
        afu=df.get("lag1q__mcp_fi_afrr_up_activated_price"); afd=df.get("lag1q__mcp_fi_afrr_down_activated_price")
        mfu=df.get("lag1q__mcp_fi_mfrr_up_activated_price"); mfd=df.get("lag1q__mcp_fi_mfrr_down_activated_price")
        if afu is not None and afd is not None:
            df["mcp_recent_afrr_activated_price_spread"] = afu-afd
        if mfu is not None and mfd is not None:
            df["mcp_recent_mfrr_activated_price_spread"] = mfu-mfd
        imbv=df.get("lag1q__mcp_fi_imbalance_volume")
        if imbv is not None:
            df["mcp_recent_imbalance_volume"] = imbv

        # Forecast revision shocks: changes since the previous hourly snapshot.
        # A negative wind revision or positive load revision can be more useful
        # than the raw forecast level because it tells the model what just changed.
        revision_cols=[c for c in df.columns if c.startswith("future__revision__")]
        if revision_cols:
            df["forecast_revision_abs_mean"] = df[revision_cols].abs().mean(axis=1)
            df["forecast_revision_count"] = df[revision_cols].notna().sum(axis=1).astype(float)
        rw=df.get("future__revision__fi_wind_fc_da"); rl=df.get("future__revision__fi_consumption_fc_da")
        rp=df.get("future__revision__fi_production_fc_rt"); ri=df.get("future__revision__mcp_fi_intraday_price")
        if rw is not None: df["wind_forecast_revision_mw"] = rw
        if rl is not None: df["load_forecast_revision_mw"] = rl
        if rp is not None: df["production_forecast_revision_mw"] = rp
        if ri is not None: df["intraday_price_revision_eur_mwh"] = ri
        if rw is not None and rl is not None:
            df["net_load_revision_proxy"] = rl-rw-(df.get("future__revision__fi_solar_fc_da",0.0))

        # V8 operational context from additional Fingrid public datasets.
        # These features are production-only and enter a challenger only after
        # exact saved snapshots have enough coverage.
        prod_fc = df.get("future__fi_production_fc_rt")
        if prod_rt is not None and prod_fc is not None:
            current_prod_fc = df.groupby("forecast_origin_utc", sort=False)["future__fi_production_fc_rt"].transform("first")
            df["production_plan_error_proxy"] = prod_rt - current_prod_fc
        chp_dh = df.get("lag1q__fi_chp_district_heat_rt")
        chp_ind = df.get("lag1q__fi_chp_industrial_rt")
        if chp_dh is not None or chp_ind is not None:
            a = chp_dh.fillna(0) if chp_dh is not None else 0.0
            b = chp_ind.fillna(0) if chp_ind is not None else 0.0
            df["live_chp_total"] = a + b
        bess = df.get("lag1q__fi_bess_discharge_rt")
        if bess is not None:
            df["live_bess_discharge"] = bess
        shortage = df.get("lag1q__fi_shortage_status_rt")
        if shortage is not None:
            df["live_shortage_status"] = shortage
        system_state = df.get("lag1q__fi_power_system_state_rt")
        if system_state is not None:
            df["live_power_system_state"] = system_state
        net_import = df.get("lag1q__fi_net_import_export_rt")
        if net_import is not None:
            df["live_net_import_export"] = net_import
        ns = df.get("lag1q__fi_north_south_flow_rt")
        if ns is not None:
            df["live_north_south_flow"] = ns
        cumulative = df.get("lag1q__fi_surplus_deficit_cumulative_rt")
        if cumulative is not None:
            df["live_cumulative_balance"] = cumulative
        wind_capacity = df.get("lag1q__fi_wind_capacity_rt")
        if wind_capacity is not None:
            if wind_rt is not None:
                df["live_wind_capacity_factor"] = _safe_div(wind_rt, wind_capacity).clip(0, 2)
            if "future__fi_wind_fc_da" in df:
                df["live_wind_forecast_capacity_factor"] = _safe_div(df["future__fi_wind_fc_da"], wind_capacity).clip(0, 2)

        surplus = df.get("lag1q__fi_surplus_deficit_rt")
        freq = df.get("lag1q__fi_frequency_rt")
        boiler = df.get("lag1q__fi_electric_boiler_rt")
        components = []
        if surplus is not None: components.append(surplus.abs() / 1000.0)
        if freq is not None: components.append((freq - 50.0).abs() * 20.0)
        if boiler is not None: components.append(boiler.abs() / 1000.0)
        bess = df.get("lag1q__fi_bess_discharge_rt")
        if bess is not None: components.append(bess.abs() / 500.0)
        peak = df.get("lag1q__fi_peak_load_power_rt")
        if peak is not None: components.append(peak.abs() / 500.0)
        shortage = df.get("lag1q__fi_shortage_status_rt")
        if shortage is not None: components.append(shortage.fillna(0) / 3.0)
        system_state = df.get("lag1q__fi_power_system_state_rt")
        if system_state is not None: components.append((system_state.fillna(1) - 1).clip(lower=0) / 4.0)
        mcp_outage = df.get("future__mcp_fi_generation_unavailable_mw")
        if mcp_outage is not None: components.append(mcp_outage.fillna(0) / 5000.0)
        outage = df.get("future__entsoe_generation_unavailable_mw")
        if outage is not None: components.append(outage.fillna(0) / 5000.0)
        txo = df.get("future__entsoe_transmission_outage_count")
        if txo is not None: components.append(txo.fillna(0) / 5.0)
        if components:
            df["live_system_stress_proxy"] = sum(components)

        # V6 reserve-stress index: signed and bounded. Positive values indicate
        # upward/shortage pressure; negative values indicate downward/surplus
        # pressure. It is a transparent market-state feature, not a trading rule.
        reserve_terms = []
        mnet = df.get("mfrr_net_activation_1h")
        if mnet is not None:
            reserve_terms.append(np.tanh(mnet.fillna(0) / 500.0))
        dmean = df.get("recent_direction_mean")
        if dmean is not None:
            reserve_terms.append(np.tanh(dmean.fillna(0)))
        if surplus is not None:
            # Fingrid surplus/deficit sign can be source-dependent, so keep its
            # influence modest and let the learner determine its value.
            reserve_terms.append(0.35 * np.tanh((-surplus.fillna(0)) / 1000.0))
        if freq is not None:
            reserve_terms.append(0.35 * np.tanh((50.0 - freq.fillna(50.0)) * 20.0))
        if reserve_terms:
            df["reserve_stress_index"] = np.clip(sum(reserve_terms) / max(1, len(reserve_terms)), -1, 1)
            df["reserve_stress_abs"] = df["reserve_stress_index"].abs()

        # Helen-inspired forecast-error regime features. Compare the current
        # forecast value at the origin with the latest legally available
        # observation. These features describe whether the forecast environment
        # is systematically missing current conditions.
        pairs = [
            ("future__weather_temperature_mean", "lag1q__weather_obs_temperature_mean", "weather_error_temperature"),
            ("future__weather_wind_10m_weighted", "lag1q__weather_obs_wind_10m_weighted", "weather_error_wind_10m"),
            ("future__weather_pressure_mean", "lag1q__weather_obs_pressure_mean", "weather_error_pressure"),
            ("future__weather_precipitation_mean", "lag1q__weather_obs_precipitation_mean", "weather_error_precipitation"),
            ("future__weather_cloud_mean", "lag1q__weather_obs_cloud_mean", "weather_error_cloud"),
        ]
        for fc_col, obs_col, out_name in pairs:
            if fc_col in df and obs_col in df:
                current_fc = df.groupby("forecast_origin_utc", sort=False)[fc_col].transform("first")
                df[out_name] = current_fc - df[obs_col]

        # Electricity forecast errors are often even more directly relevant to
        # balancing than weather error itself. Use the current-origin forecast
        # against the latest observed system state.
        if "future__fi_wind_fc_da" in df and wind_rt is not None:
            current_wind_fc = df.groupby("forecast_origin_utc", sort=False)["future__fi_wind_fc_da"].transform("first")
            df["weather_error_wind_generation_proxy"] = current_wind_fc - wind_rt
        if "future__fi_consumption_fc_da" in df and c1 is not None:
            current_load_fc = df.groupby("forecast_origin_utc", sort=False)["future__fi_consumption_fc_da"].transform("first")
            df["weather_error_load_proxy"] = current_load_fc - c1

    # Missingness is explicit because aFRR sparsity can be structurally meaningful.
    important = [
        "future__fi_spot_price", "future__fi_wind_fc_da", "future__fi_consumption_fc_da",
        "lag1q__fi_imbalance_price", "lag4q__fi_mfrr_direction", "lag1q__fi_consumption",
        "lag1q__fi_production", "lag4q__fi_afrr_up_price", "lag4q__fi_afrr_down_price",
    ]
    for c in important:
        if c in df:
            df[f"missing__{c}"] = df[c].isna().astype(np.int8)

    if include_labels:
        idx = pd.DatetimeIndex(df["delivery_start_utc"])
        df["y_price"] = values["fi_imbalance_price"].reindex(idx).to_numpy(dtype=float)
        direction = values["fi_mfrr_direction"].reindex(idx).to_numpy(dtype=float)
        df["y_direction"] = np.select(
            [direction == -1, direction == 0, direction == 1], [0, 1, 2], default=np.nan
        )
        y_spot = values["fi_spot_price"].reindex(idx).to_numpy(dtype=float)
        df["y_spike_high"] = (df["y_price"].to_numpy() > y_spot + SPIKE_MARGIN).astype(float)
        df["y_spike_low"] = (df["y_price"].to_numpy() < y_spot - SPIKE_MARGIN).astype(float)
    return df


def business_driver_text(df: pd.DataFrame) -> pd.Series:
    """Create concise, deterministic driver text from transparent market signals.

    This is an explanation layer for the optional `top_drivers` column and UI.
    It does not alter the prediction and does not claim causal proof.
    """
    out: list[str] = []
    for _, r in df.iterrows():
        signals: list[tuple[float, str]] = []

        def add(score: float, text: str) -> None:
            if np.isfinite(score) and score > 0:
                signals.append((float(score), text))

        w = r.get("wind_fc_change_from_origin", np.nan)
        if pd.notna(w):
            add(abs(float(w)) / 250.0, "wind forecast falling" if w < 0 else "wind forecast rising")

        l = r.get("load_fc_change_from_origin", np.nan)
        if pd.notna(l):
            add(abs(float(l)) / 350.0, "consumption forecast rising" if l > 0 else "consumption forecast falling")

        m = r.get("mfrr_net_activation_1h", np.nan)
        if pd.notna(m):
            add(abs(float(m)) / 150.0, "recent mFRR upward pressure" if m > 0 else "recent mFRR downward pressure")

        cb = r.get("cross_border_stress", np.nan)
        if pd.notna(cb):
            add(max(float(cb) - 0.5, 0.0), "cross-border transmission stress")

        ee = r.get("ee_capacity_utilization_proxy", np.nan)
        if pd.notna(ee):
            add(max(float(ee) - 0.65, 0.0) * 2.0, "FI-EE transfer capacity highly utilized")

        n = r.get("fi_vs_neighbor_price", np.nan)
        if pd.notna(n):
            add(abs(float(n)) / 40.0, "Finland-neighbour imbalance-price divergence")

        e = r.get("recent_load_forecast_error", np.nan)
        if pd.notna(e):
            add(abs(float(e)) / 300.0, "recent consumption forecast error")

        v = r.get("recent_price_std", np.nan)
        if pd.notna(v):
            add(float(v) / 60.0, "recent imbalance-price volatility")

        flex = r.get("sector_coupling_flex_signal", np.nan)
        if pd.notna(flex):
            add(float(flex), "low-price flexible-load / electric-boiler regime")

        wx = r.get("weather_wind_ramp_3h", np.nan)
        if pd.notna(wx):
            add(abs(float(wx)) / 2.0, "regional wind forecast changing rapidly")
        outage = r.get("future__entsoe_generation_unavailable_mw", np.nan)
        if pd.notna(outage):
            add(abs(float(outage)) / 1500.0, "generation unavailability context")
        stress = r.get("live_system_stress_proxy", np.nan)
        if pd.notna(stress):
            add(float(stress), "live system stress signals")
        rsi = r.get("reserve_stress_index", np.nan)
        if pd.notna(rsi):
            add(abs(float(rsi)) * 2.0, "reserve market upward stress" if rsi > 0 else "reserve market downward stress")
        werr = r.get("weather_error_wind_generation_proxy", np.nan)
        if pd.notna(werr):
            add(abs(float(werr)) / 300.0, "wind forecast error regime")

        if not signals:
            out.append("normal market-state signals")
            continue
        signals.sort(key=lambda x: x[0], reverse=True)
        out.append("; ".join(text for _, text in signals[:4]))
    return pd.Series(out, index=df.index, dtype="string")


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "forecast_origin_utc", "delivery_start_utc", "y_price", "y_direction",
        "y_spike_high", "y_spike_low"
    }
    return [c for c in df.columns if c not in excluded]

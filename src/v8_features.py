from __future__ import annotations

import numpy as np
import pandas as pd


def augment_v8_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add V8 features without changing the frozen V3 feature contract.

    The most important addition is an explicit 4-quarter clock.  Forecast
    origins are hourly, so ``(step-1) % 4`` identifies :00/:15/:30/:45.  The
    V7 validation curve showed a repeating four-step error pattern; exposing
    this phase directly lets the spread experts learn it instead of asking a
    tree to infer a modulo operation from ``step``.

    All values below are known at forecast time.  Future weather/load/spot
    values are already legal delivery-time forecasts from the base feature
    builder; differences between adjacent delivery quarters use the same
    forecast vintage and therefore do not introduce future information.
    """
    out = df.copy()
    step = pd.to_numeric(out["step"], errors="coerce").fillna(1).astype(int)
    q = ((step - 1) % 4).astype(int)
    out["delivery_quarter"] = q.astype(float)
    out["horizon_hour_index"] = ((step - 1) // 4).astype(float)
    out["is_hour_boundary"] = (q == 0).astype(np.int8)
    for i in range(4):
        out[f"delivery_quarter_{i}"] = (q == i).astype(np.int8)
    out["delivery_quarter_sin"] = np.sin(2.0 * np.pi * q / 4.0)
    out["delivery_quarter_cos"] = np.cos(2.0 * np.pi * q / 4.0)

    # Same-vintage quarter-to-quarter ramps.  These are useful when the 10-hour
    # forecast surface changes sharply around a quarter-hour boundary.
    for source, name in [
        ("future__fi_spot_price", "spot_qoq_change"),
        ("future__fi_wind_fc_da", "wind_qoq_change"),
        ("future__fi_consumption_fc_da", "load_qoq_change"),
        ("future__fi_solar_fc_da", "solar_qoq_change"),
        ("net_load_proxy", "net_load_qoq_change"),
    ]:
        if source in out.columns and "forecast_origin_utc" in out.columns:
            out[name] = out.groupby("forecast_origin_utc", sort=False)[source].diff()
            # Step 1 has no previous delivery quarter in the batch.  Zero means
            # "no within-batch change observed yet", while the absolute level
            # remains available in the original feature.
            out[name] = out[name].fillna(0.0)

    # Transparent market-rule interaction features used only by V8 spread
    # experts.  Trees can learn interactions themselves, but making these
    # explicit improves stability on a small post-June training window.
    if "future__fi_spot_price" in out.columns:
        spot = pd.to_numeric(out["future__fi_spot_price"], errors="coerce")
        out["spot_abs"] = spot.abs()
        out["spot_negative"] = (spot < 0).astype(np.int8)
        out["spot_x_hour_boundary"] = spot * out["is_hour_boundary"]
    if "recent_direction_mean" in out.columns and "future__fi_spot_price" in out.columns:
        out["direction_pressure_x_spot"] = (
            pd.to_numeric(out["recent_direction_mean"], errors="coerce")
            * pd.to_numeric(out["future__fi_spot_price"], errors="coerce")
        )
    if "mfrr_net_activation_1h" in out.columns and "future__fi_spot_price" in out.columns:
        out["mfrr_pressure_x_spot"] = (
            pd.to_numeric(out["mfrr_net_activation_1h"], errors="coerce")
            * np.sign(pd.to_numeric(out["future__fi_spot_price"], errors="coerce").fillna(0.0))
        )
    return out


def v8_extra_feature_names() -> list[str]:
    return [
        "delivery_quarter", "horizon_hour_index", "is_hour_boundary",
        "delivery_quarter_0", "delivery_quarter_1", "delivery_quarter_2", "delivery_quarter_3",
        "delivery_quarter_sin", "delivery_quarter_cos",
        "spot_qoq_change", "wind_qoq_change", "load_qoq_change", "solar_qoq_change",
        "net_load_qoq_change", "spot_abs", "spot_negative", "spot_x_hour_boundary",
        "direction_pressure_x_spot", "mfrr_pressure_x_spot",
    ]


def spread_feature_names(base_features: list[str], frame: pd.DataFrame) -> list[str]:
    extras = [c for c in v8_extra_feature_names() if c in frame.columns]
    # Preserve order and avoid accidental duplicates.
    return list(dict.fromkeys(list(base_features) + extras))

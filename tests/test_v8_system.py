from __future__ import annotations

import numpy as np
import pandas as pd

from src.v8_features import augment_v8_features
from src.online.adaptive import learn_multiwindow_state, apply_adaptive


def test_v8_quarter_features_repeat_every_four_steps():
    origin = pd.Timestamp("2026-10-20T12:00:00Z")
    df = pd.DataFrame({
        "forecast_origin_utc": [origin] * 40,
        "delivery_start_utc": pd.date_range(origin, periods=40, freq="15min"),
        "step": range(1, 41),
        "future__fi_spot_price": np.linspace(20, 40, 40),
    })
    out = augment_v8_features(df)
    assert list(out.loc[:7, "delivery_quarter"].astype(int)) == [0, 1, 2, 3, 0, 1, 2, 3]
    assert int(out.loc[0, "is_hour_boundary"]) == 1
    assert int(out.loc[1, "is_hour_boundary"]) == 0
    assert "spot_qoq_change" in out.columns


def test_v8_adaptive_layer_can_learn_direct_vs_spread_weight():
    rows = []
    for step in range(1, 41):
        for i in range(10):
            y = 100.0 + i
            rows.append({
                "step": step,
                "imbalance_price": y,
                "p50": y - 5,
                "p50_direct": y - 20,
                "p50_spread": y - 2,
                "p50_lgb": y - 10,
                "p50_cat": y - 8,
                "p10": y - 30,
                "p90": y + 30,
                "direction": 0,
                "p_down": .7,
                "p_none": .1,
                "p_up": .2,
                "spot_price": 80.0,
                "p_spike_high": .1,
                "p_spike_low": .1,
            })
    frame = pd.DataFrame(rows)
    state = learn_multiwindow_state({24: frame, 168: frame, 336: frame})
    assert state.blended_steps["1"].p50_spread_weight is not None
    assert state.blended_steps["1"].p50_spread_weight > 0.5

    base = pd.DataFrame({
        "p10": np.full(40, 70.0), "p50": np.full(40, 85.0), "p90": np.full(40, 130.0),
        "p_up": np.full(40, .2), "p_down": np.full(40, .7), "p_none": np.full(40, .1),
        "p_spike_high": np.full(40, .1), "p_spike_low": np.full(40, .1),
        "p50_lgb": np.full(40, 90.0), "p50_cat": np.full(40, 92.0),
        "p50_direct": np.full(40, 80.0), "p50_spread": np.full(40, 98.0),
        "reliability": np.full(40, .8),
    })
    out = apply_adaptive(base, pd.Series(range(1, 41)), state)
    assert (out["p50"] != base["p50"]).any()
    assert ((out["p10"] <= out["p50"]) & (out["p50"] <= out["p90"])).all()


def test_public_spot_parser_uses_vat_exclusive_eur_per_mwh():
    from src.online.public_spot import PublicSpotClient
    payload = {
        "prices": [
            {"start": "2026-11-07T00:00:00+02:00", "minutes": 15, "EUR_per_MWh_excl_vat": 12.5},
            {"start": "2026-11-07T00:15:00+02:00", "minutes": 15, "EUR_per_MWh_excl_vat": 13.0},
        ]
    }
    out = PublicSpotClient._parse(payload)
    assert list(out["value"]) == [12.5, 13.0]
    assert str(out.iloc[0]["timestamp_utc"].tzinfo) == "UTC"

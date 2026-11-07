from __future__ import annotations

import pandas as pd

from src.performance import official_validation_summary, validation_step_performance, validation_intuitive_accuracy
from src.online.daily import local_now


def test_official_validation_summary_has_challenge_metrics():
    s = official_validation_summary()
    assert float(s["mae_p50"]) > 0
    assert 0 <= float(s["coverage_p10_p90"]) <= 1
    assert float(s["calibration_pinball_mean"]) > 0
    assert float(s["brier_spike_mean"]) >= 0
    assert float(s["brier_direction"]) >= 0


def test_validation_step_performance_has_all_40_quarter_hours():
    p = validation_step_performance()
    assert list(p["step"]) == list(range(1, 41))
    assert int(p.iloc[-1]["horizon_minutes"]) == 585


def test_validation_intuitive_accuracy_is_bounded():
    x = validation_intuitive_accuracy()
    assert x["samples"] > 0
    assert x["mae"] > 0
    for k in ["within_10", "within_20", "within_50", "coverage"]:
        assert 0 <= x[k] <= 1


def test_daily_local_timezone_is_helsinki():
    t = local_now(pd.Timestamp("2026-11-07T12:00:00Z"))
    assert str(t.tzinfo) == "Europe/Helsinki"

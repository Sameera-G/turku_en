import pandas as pd


def test_time_ordering_smoke():
    origin = pd.Timestamp("2026-10-01T10:00:00Z")
    assert origin - pd.Timedelta(minutes=15) < origin

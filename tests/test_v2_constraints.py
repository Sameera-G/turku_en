from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import build_features


def test_feature_grid_and_quantile_targets_available_pretest():
    origins = pd.DatetimeIndex([pd.Timestamp("2026-09-10T12:00:00Z")])
    df = build_features(origins, include_labels=True)
    assert len(df) == 40
    assert (df["step"].to_numpy() == np.arange(1, 41)).all()
    assert df["y_price"].notna().all()


def test_walk_forward_feature_build_does_not_require_test_labels():
    origins = pd.DatetimeIndex([
        pd.Timestamp("2026-10-05T12:00:00Z"),
        pd.Timestamp("2026-10-20T12:00:00Z"),
    ])
    df = build_features(origins, include_labels=False)
    assert len(df) == 80
    assert "y_price" not in df.columns

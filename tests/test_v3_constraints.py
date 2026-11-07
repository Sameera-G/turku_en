import numpy as np
import pandas as pd

from src.config import HISTORY_LAGS_Q
from src.models import HorizonCalibratorV3


def test_required_lags_present():
    # V3 explicitly fixes the V2 silent omission of 15m and 8h lags.
    assert 1 in HISTORY_LAGS_Q
    assert 32 in HISTORY_LAGS_Q
    assert 96 in HISTORY_LAGS_Q


def test_fusion_weights_valid():
    cal = HorizonCalibratorV3(p50_lgb_weight={1: 0.0, 2: 0.5, 3: 1.0})
    assert all(0.0 <= w <= 1.0 for w in cal.p50_lgb_weight.values())

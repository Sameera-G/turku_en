from __future__ import annotations

import pandas as pd

from src.online.retrain import _three_way_origin_split, _candidate_gate
from src.models_v8 import load_any_bundle
from src.config import MODELS_DIR


def test_v11_three_way_split_is_chronological_and_disjoint():
    origins = pd.date_range("2026-11-01", periods=48, freq="h", tz="UTC")
    train, cal, ev = _three_way_origin_split(origins)
    assert len(train) == 24
    assert len(cal) == 12
    assert len(ev) == 12
    assert train.max() < cal.min() < ev.min()
    assert set(train).isdisjoint(set(cal))
    assert set(train).isdisjoint(set(ev))
    assert set(cal).isdisjoint(set(ev))


def test_v11_gate_requires_mae_improvement_not_only_other_metrics():
    champ = {"mae": 30.0, "pinball": 10.0, "direction_brier": .4, "spike_brier": .1}
    # Composite can look better, but MAE is worse: V11 must reject it.
    chall = {"mae": 30.2, "pinball": 8.0, "direction_brier": .35, "spike_brier": .08}
    eligible, gain, regressions, mae_gain = _candidate_gate(champ, chall)
    assert mae_gain < 0
    assert not eligible


def test_prebuilt_v11_candidate_is_challenger_only():
    p = MODELS_DIR / "v11" / "model_bundle_v11_candidate.joblib"
    b = load_any_bundle(p)
    assert "strong-p50-live-candidate" in b.version


def test_v11_json_roundtrip_object_features_can_be_predicted():
    """Serialized online snapshots may return numeric fields as object strings."""
    import numpy as np
    from src.models_v8 import predict_any_bundle

    p = MODELS_DIR / "v10" / "model_bundle_v10.joblib"
    b = load_any_bundle(p)
    rows = 2
    data = {c: pd.Series(["0.0"] * rows, dtype="object") for c in b.feature_names}
    data["step"] = pd.Series(["1", "2"], dtype="object")
    data["future__fi_spot_price"] = pd.Series(["40.0", "41.0"], dtype="object")
    data["forecast_origin_utc"] = pd.to_datetime(["2026-11-07T03:00:00Z"] * rows, utc=True)
    data["delivery_start_utc"] = pd.to_datetime(["2026-11-07T03:00:00Z", "2026-11-07T03:15:00Z"], utc=True)
    frame = pd.DataFrame(data)

    pred = predict_any_bundle(b, frame)
    assert len(pred) == rows
    assert np.isfinite(pred["p50"].to_numpy(float)).all()

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.online.adaptive import learn_adaptive_state, apply_adaptive
from src.online.fingrid import FingridClient
from src.online.store import OnlineStore


def test_fingrid_payload_parsing():
    payload={"data":[{"datasetId":319,"startTime":"2026-11-06T01:00:00Z","endTime":"2026-11-06T01:15:00Z","value":12.5}]}
    rows=FingridClient._extract_rows(payload)
    assert len(rows)==1 and rows[0]["datasetId"]==319


def test_adaptive_state_learns_and_applies():
    rows=[]
    for step in [1,2]:
        for i in range(12):
            y=100+i
            rows.append({
                "step":step,"imbalance_price":y,"p50":y+8,"p10":y-15,"p90":y+25,
                "p50_lgb":y+2,"p50_cat":y+15,
                "direction":1,"p_down":.1,"p_none":.1,"p_up":.8,
                "spot_price":40,"p_spike_high":.7,"p_spike_low":.05,
            })
    df=pd.DataFrame(rows)
    state=learn_adaptive_state(df,min_step_samples=8)
    assert len(state.steps)==2
    pred=pd.DataFrame({
        "p10":[70.,70.],"p50":[108.,108.],"p90":[130.,130.],
        "p50_lgb":[102.,102.],"p50_cat":[115.,115.],
        "p_down":[.1,.1],"p_none":[.1,.1],"p_up":[.8,.8],
        "p_spike_high":[.7,.7],"p_spike_low":[.05,.05],"reliability":[.8,.8],
    })
    out=apply_adaptive(pred,pd.Series([1,2]),state)
    assert out["adaptive_applied"].all()
    assert np.all(out["p10"]<=out["p50"])
    assert np.all(out["p50"]<=out["p90"])
    assert np.allclose(out[["p_up","p_down","p_none"]].sum(axis=1),1.0)


def test_online_store_roundtrip(tmp_path: Path):
    store=OnlineStore(tmp_path/"online.sqlite3")
    feat=pd.DataFrame({
        "forecast_origin_utc":pd.to_datetime(["2026-11-06T10:00:00Z"],utc=True),
        "delivery_start_utc":pd.to_datetime(["2026-11-06T10:00:00Z"],utc=True),
        "step":[1],"x":[1.2],
    })
    pred=pd.DataFrame({
        "p10":[0.],"p50":[10.],"p90":[20.],"p_up":[.5],"p_down":[.3],"p_none":[.2],
        "p_spike_high":[.1],"p_spike_low":[.05],"p50_lgb":[9.],"p50_cat":[12.],"reliability":[.8],
    })
    store.save_predictions(feat,pred,pd.Series(["test driver"]),"vtest",{})
    long=pd.DataFrame({
        "series_id":["fi_imbalance_price","fi_mfrr_direction","fi_spot_price"],
        "timestamp_utc":pd.to_datetime(["2026-11-06T10:00:00Z"]*3,utc=True),
        "value":[12.,1.,40.],
    })
    store.upsert_actuals_from_long(long)
    n=store.evaluate_settled(pd.Timestamp("2026-11-06T12:00:00Z"))
    assert n==1
    joined=store.recent_joined(pd.Timestamp("2026-11-06T00:00:00Z"))
    assert len(joined)==1
    assert float(joined.iloc[0]["abs_error"])==2.0

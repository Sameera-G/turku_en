from __future__ import annotations

from functools import lru_cache
import numpy as np
import pandas as pd

from .config import INPUTS_CSV, INPUTS_PARQUET, SPOT_CSV, LIVE_LONG_PARQUET


def _read_long(path) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"dataset_id": str})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["first_usable_origin_utc"] = pd.to_datetime(df["first_usable_origin_utc"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def clear_data_cache() -> None:
    """Clear cached data after a live-ingestion refresh."""
    load_long.cache_clear()
    load_wide.cache_clear()


@lru_cache(maxsize=1)
def load_long() -> pd.DataFrame:
    if not SPOT_CSV.exists() or (not INPUTS_PARQUET.exists() and not INPUTS_CSV.exists()):
        raise FileNotFoundError("Run: python scripts/setup_data.py --zip PATH_TO_starter_package_v1.zip")
    try:
        a = _read_long(INPUTS_PARQUET) if INPUTS_PARQUET.exists() else _read_long(INPUTS_CSV)
    except Exception:
        a = _read_long(INPUTS_CSV)
    b = _read_long(SPOT_CSV)
    frames = [a, b]
    # Production-only live overlay. The file is not required for the hackathon
    # backtest and therefore does not alter V1/V2/V3 reproducibility.
    if LIVE_LONG_PARQUET.exists():
        try:
            live = _read_long(LIVE_LONG_PARQUET)
            frames.append(live)
        except Exception:
            # A damaged live cache must never break the locked backtest path.
            pass
    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=1)
def load_wide() -> tuple[pd.DataFrame, pd.DataFrame]:
    long = load_long().sort_values(["timestamp_utc", "series_id", "first_usable_origin_utc"])
    # For the current production origin, the live cache contains only the
    # newest snapshot for each timestamp/series. Static challenge rows remain
    # untouched. Keep the newest legally available snapshot.
    long = long.drop_duplicates(["timestamp_utc", "series_id"], keep="last")
    values = long.pivot(index="timestamp_utc", columns="series_id", values="value").sort_index()
    usable = long.pivot(index="timestamp_utc", columns="series_id", values="first_usable_origin_utc").sort_index()
    return values, usable


def legal_value(values: pd.DataFrame, usable: pd.DataFrame, ts: pd.Series, origin: pd.Series, series: str) -> np.ndarray:
    if series not in values.columns:
        return np.full(len(ts), np.nan)
    idx = pd.DatetimeIndex(ts)
    v = values[series].reindex(idx).to_numpy(dtype=float)
    u = pd.to_datetime(usable[series].reindex(idx), utc=True)
    o = pd.DatetimeIndex(origin)
    ok = (u.notna().to_numpy()) & (u.to_numpy() <= o.to_numpy())
    return np.where(ok, v, np.nan)

from __future__ import annotations

"""Optional LIVE context downloader for the dashboard/API.

This does not enter the hidden-test score unless a historical forecast-vintage
source can be audited against publication time. It is intentionally separated
from the leakage-controlled training pipeline.

The selected Finnish nodes include Pori, Kalajoki and Raahe because Turku
Energia's 2025 financial statement describes wind interests in those areas.
This improves the production/demo situational view without pretending those
holdings represent the entire Finnish wind fleet.
"""

import argparse
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "external"

from src.config import WEATHER_NODES


def fetch_live(name: str, lat: float, lon: float, weight: float) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,cloud_cover,pressure_msl,wind_speed_10m,wind_speed_100m,wind_direction_100m,wind_gusts_10m,shortwave_radiation",
        "forecast_days": 7,
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    df = pd.DataFrame(j["hourly"])
    df.insert(0, "node", name)
    df["latitude"] = lat
    df["longitude"] = lon
    df["context_weight"] = weight
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Download current 7-day regional forecasts")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.live:
        print("Nothing requested. Use --live for current Open-Meteo context.")
        return
    frames = [fetch_live(n, lat, lon, w) for n, (lat, lon, w) in WEATHER_NODES.items()]
    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    path = OUT / "open_meteo_live_finland_v3.csv"
    out.to_csv(path, index=False)

    # Weighted dashboard-only wind indicator.
    weighted = out.assign(weighted_wind=out["wind_speed_100m"] * out["context_weight"])
    agg = weighted.groupby("time", as_index=False).agg(
        regional_weighted_wind_100m=("weighted_wind", "sum"),
        max_wind_gust=("wind_gusts_10m", "max"),
        mean_temperature=("temperature_2m", "mean"),
        total_weighted_precip=("precipitation", lambda s: float((s * weighted.loc[s.index, "context_weight"]).sum())),
    )
    agg.to_csv(OUT / "open_meteo_live_finland_v3_aggregate.csv", index=False)
    print(f"Wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests

from ..config import OPENMETEO_API_BASE, WEATHER_NODES


@dataclass
class OpenMeteoClient:
    base_url: str = OPENMETEO_API_BASE
    timeout_seconds: float = 30.0

    def fetch_regional(self, start: pd.Timestamp, end: pd.Timestamp, fetched_at: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch operational weather forecasts for representative Finnish nodes.

        Values are aggregated with the configured node weights, then expanded to
        15-minute timestamps so the online feature builder can use the same
        delivery grid as the electricity forecast. These are production-only
        live features; hackathon backtests remain unchanged.
        """
        frames: list[pd.DataFrame] = []
        details: list[dict[str, Any]] = []
        variables = [
            "temperature_2m", "precipitation", "cloud_cover", "pressure_msl",
            "wind_speed_10m", "wind_speed_100m", "wind_gusts_10m", "shortwave_radiation",
        ]
        for name, (lat, lon, weight) in WEATHER_NODES.items():
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(variables),
                "timezone": "UTC",
                "wind_speed_unit": "ms",
                "forecast_days": 4,
            }
            try:
                r = requests.get(self.base_url, params=params, timeout=self.timeout_seconds)
                r.raise_for_status()
                payload = r.json()
                hourly = payload.get("hourly", {})
                times = pd.to_datetime(hourly.get("time", []), utc=True, errors="coerce")
                if len(times) == 0:
                    details.append({"node": name, "status": "empty"})
                    continue
                df = pd.DataFrame({"timestamp_utc": times})
                for v in variables:
                    df[v] = pd.to_numeric(pd.Series(hourly.get(v, [np.nan] * len(df))), errors="coerce")
                df["node"] = name
                df["weight"] = float(weight)
                df = df[(df["timestamp_utc"] >= start.floor("h")) & (df["timestamp_utc"] <= end.ceil("h"))]
                frames.append(df)
                details.append({"node": name, "status": "ok", "rows": len(df)})
            except Exception as exc:
                details.append({"node": name, "status": "error", "error": str(exc)})

        if not frames:
            return pd.DataFrame(), {"status": "error", "nodes": details}
        raw = pd.concat(frames, ignore_index=True)
        rows: list[dict[str, Any]] = []
        mapping = {
            "wind_speed_10m": "weather_wind_10m_weighted",
            "wind_speed_100m": "weather_wind_100m_weighted",
            "wind_gusts_10m": "weather_wind_gust_max",
            "temperature_2m": "weather_temperature_mean",
            "pressure_msl": "weather_pressure_mean",
            "precipitation": "weather_precipitation_mean",
            "cloud_cover": "weather_cloud_mean",
            "shortwave_radiation": "weather_solar_mean",
        }
        grouped = raw.groupby("timestamp_utc", sort=True)
        for ts, g in grouped:
            w = g["weight"].to_numpy(float)
            if not np.isfinite(w).any() or w.sum() <= 0:
                w = np.ones(len(g), dtype=float)
            w = w / w.sum()
            for source_col, series_id in mapping.items():
                vals = g[source_col].to_numpy(float)
                valid = np.isfinite(vals)
                if not valid.any():
                    continue
                if source_col == "wind_gusts_10m":
                    value = float(np.nanmax(vals))
                else:
                    ww = w[valid]
                    ww = ww / ww.sum()
                    value = float(np.sum(vals[valid] * ww))
                # Expand the hourly model value to the four quarter-hours. The
                # model run is still identified by fetched_at, so this cannot
                # leak backwards into older origins.
                for q in range(4):
                    rows.append({
                        "series_id": series_id,
                        "source": "open_meteo_live",
                        "dataset_id": "forecast_api",
                        "timestamp_utc": pd.Timestamp(ts) + pd.Timedelta(minutes=15*q),
                        "value": value,
                        "unit": "",
                        "first_usable_origin_utc": fetched_at,
                        "kind": "forecast",
                    })
        out = pd.DataFrame(rows)
        return out, {"status": "ok", "nodes": details, "rows": len(out)}

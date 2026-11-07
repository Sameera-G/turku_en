from __future__ import annotations

"""Optional no-key Finnish day-ahead spot fallback for hackathon/demo use.

The primary production choices remain ENTSO-E Transparency Platform or a
licensed Nord Pool feed.  This adapter uses sahkonhintahuomenna.fi's public JSON
API when those are unavailable.  The service states that it is free and asks for
source attribution.  It is a best-effort third-party feed, so V8 exposes the
source in health/context diagnostics and should not be treated as an SLA-backed
trading feed.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import requests

from ..config import PUBLIC_SPOT_API_BASE


@dataclass
class PublicSpotClient:
    base_url: str = PUBLIC_SPOT_API_BASE
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.session = requests.Session()

    @staticmethod
    def _dates(start: pd.Timestamp, end: pd.Timestamp) -> list[date]:
        a = pd.Timestamp(start).tz_convert("Europe/Helsinki").date()
        b = pd.Timestamp(end).tz_convert("Europe/Helsinki").date()
        return list(pd.date_range(a, b, freq="D").date)

    @staticmethod
    def _parse(payload: dict[str, Any]) -> pd.DataFrame:
        rows = []
        for item in payload.get("prices", []):
            start = pd.to_datetime(item.get("start"), utc=True, errors="coerce")
            value = pd.to_numeric(item.get("EUR_per_MWh_excl_vat"), errors="coerce")
            if pd.notna(start) and pd.notna(value):
                rows.append((start, float(value)))
        return pd.DataFrame(rows, columns=["timestamp_utc", "value"])

    def fetch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        fetched_at: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        frames = []
        errors = []
        for d in self._dates(start, end):
            url = f"{self.base_url}/{d.isoformat()}.json"
            try:
                r = self.session.get(url, timeout=self.timeout_seconds, headers={"Accept": "application/json"})
                if r.status_code == 404:
                    continue
                r.raise_for_status()
                f = self._parse(r.json())
                if not f.empty:
                    frames.append(f)
            except Exception as exc:
                errors.append(f"{d.isoformat()}: {exc}")
        if not frames:
            return pd.DataFrame(), {
                "source": "public_spot_fallback",
                "status": "error" if errors else "empty",
                "rows": 0,
                "error": "; ".join(errors[:2]) if errors else None,
                "attribution": "Prices: sahkonhintahuomenna.fi",
            }
        raw = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp_utc", keep="last")
        raw = raw[(raw["timestamp_utc"] >= pd.Timestamp(start)) & (raw["timestamp_utc"] <= pd.Timestamp(end))]
        out = pd.DataFrame({
            "series_id": "fi_spot_price",
            "source": "public_spot_fallback",
            "dataset_id": "sahkonhintahuomenna.fi",
            "timestamp_utc": raw["timestamp_utc"],
            "value": raw["value"],
            "unit": "EUR/MWh",
            "first_usable_origin_utc": pd.Timestamp(fetched_at),
            "kind": "forecast",
        })
        return out.reset_index(drop=True), {
            "source": "public_spot_fallback",
            "status": "ok",
            "rows": len(out),
            "errors": errors[:2],
            "attribution": "Prices: sahkonhintahuomenna.fi",
            "production_note": "best-effort fallback; prefer ENTSO-E or licensed Nord Pool for production",
        }

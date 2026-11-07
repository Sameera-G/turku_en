from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any

import pandas as pd
import requests

from ..config import FINGRID_API_BASE, FINGRID_API_KEY
from .datasets import DatasetSpec, FINGRID_DATASETS


class FingridAPIError(RuntimeError):
    pass


def _iso(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FingridClient:
    api_key: str = FINGRID_API_KEY
    base_url: str = FINGRID_API_BASE
    min_interval_seconds: float = 2.05
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._last_call = 0.0
        self.session = requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _wait_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_call
        wait = self.min_interval_seconds - elapsed
        if wait > 0:
            time.sleep(wait)

    @staticmethod
    def _extract_rows(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "events", "items"):
                val = payload.get(key)
                if isinstance(val, list):
                    return val
            if {"datasetId", "startTime", "value"}.issubset(payload):
                return [payload]
        return []

    def fetch_dataset(self, spec: DatasetSpec, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if not self.configured:
            raise FingridAPIError("FINGRID_API_KEY is not configured")
        self._wait_rate_limit()
        url = f"{self.base_url}/datasets/{spec.dataset_id}/data"
        params = {
            "startTime": _iso(start),
            "endTime": _iso(end),
            "pageSize": 20000,
        }
        headers = {"x-api-key": self.api_key, "Accept": "application/json", "Cache-Control": "no-cache"}
        response = self.session.get(url, params=params, headers=headers, timeout=self.timeout_seconds)
        self._last_call = time.monotonic()
        if response.status_code != 200:
            raise FingridAPIError(f"Fingrid dataset {spec.dataset_id}: HTTP {response.status_code}: {response.text[:300]}")
        rows = self._extract_rows(response.json())
        if not rows:
            return pd.DataFrame(columns=["datasetId", "startTime", "endTime", "value"])
        df = pd.DataFrame(rows)
        rename = {"start_time": "startTime", "end_time": "endTime", "dataset_id": "datasetId"}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        required = ["startTime", "value"]
        if not all(c in df.columns for c in required):
            raise FingridAPIError(f"Unexpected Fingrid response for dataset {spec.dataset_id}: columns={list(df.columns)}")
        df["startTime"] = pd.to_datetime(df["startTime"], utc=True, errors="coerce")
        if "endTime" in df:
            df["endTime"] = pd.to_datetime(df["endTime"], utc=True, errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["startTime", "value"]).sort_values("startTime")
        return df

    def fetch_all(
        self,
        actual_start: pd.Timestamp,
        actual_end: pd.Timestamp,
        forecast_start: pd.Timestamp,
        forecast_end: pd.Timestamp,
        specs: tuple[DatasetSpec, ...] = FINGRID_DATASETS,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        fetched_at = pd.Timestamp(datetime.now(timezone.utc))
        frames: list[pd.DataFrame] = []
        log: list[dict[str, Any]] = []
        for spec in specs:
            start, end = (forecast_start, forecast_end) if spec.kind == "forecast" else (actual_start, actual_end)
            try:
                raw = self.fetch_dataset(spec, start, end)
                if raw.empty:
                    log.append({"dataset_id": spec.dataset_id, "series_id": spec.series_id, "rows": 0, "status": "empty"})
                    continue
                out = pd.DataFrame({
                    "series_id": spec.series_id,
                    "source": "fingrid_live",
                    "dataset_id": str(spec.dataset_id),
                    "timestamp_utc": raw["startTime"],
                    "value": raw["value"],
                    "unit": spec.unit,
                    "first_usable_origin_utc": fetched_at,
                    "kind": spec.kind,
                })
                # High-frequency context is reduced to a 15-minute latest-value
                # view for diagnostics. Core actual/forecast datasets are already
                # quarter-hour data and pass through unchanged.
                if spec.target_minutes == 15 and spec.kind == "context":
                    out = (
                        out.set_index("timestamp_utc")
                        .resample("15min")
                        .last()
                        .dropna(subset=["value"])
                        .reset_index()
                    )
                    out["series_id"] = spec.series_id
                    out["source"] = "fingrid_live"
                    out["dataset_id"] = str(spec.dataset_id)
                    out["unit"] = spec.unit
                    out["first_usable_origin_utc"] = fetched_at
                    out["kind"] = spec.kind
                frames.append(out)
                log.append({"dataset_id": spec.dataset_id, "series_id": spec.series_id, "rows": len(out), "status": "ok"})
            except Exception as exc:
                log.append({"dataset_id": spec.dataset_id, "series_id": spec.series_id, "rows": 0, "status": "error", "error": str(exc)})
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["series_id", "source", "dataset_id", "timestamp_utc", "value", "unit", "first_usable_origin_utc", "kind"]
        )
        return combined, log

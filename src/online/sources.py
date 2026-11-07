from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import (
    LIVE_SOURCE_HEALTH_JSON, LIVE_RAW_SNAPSHOT_DIR, ONLINE_ARCHIVE_RAW,
    ONLINE_USE_ENTSOE_EXTENDED, ONLINE_USE_FMI, ONLINE_USE_OPENMETEO,
    MANUAL_SPOT_CSV, ONLINE_USE_PUBLIC_SPOT_FALLBACK, ONLINE_USE_ENTSOE_MCP,
)
from .entsoe import EntsoeClient
from .entsoe_mcp import EntsoeMCPClient
from .fingrid import FingridClient
from .fmi import FMIClient
from .openmeteo import OpenMeteoClient
from .public_spot import PublicSpotClient


@dataclass
class SourceBundle:
    data: pd.DataFrame
    log: list[dict[str, Any]]
    health: dict[str, Any]


class LiveSourceOrchestrator:
    """Collect the highest-value public operational data for one forecast origin.

    The orchestrator is intentionally defensive: a failed secondary source does
    not stop the hourly forecast. Every source is logged and the resulting data
    freshness is exposed to the reliability/UI layer.
    """

    def __init__(self):
        self.fingrid = FingridClient()
        self.entsoe = EntsoeClient()
        self.entsoe_mcp = EntsoeMCPClient()
        self.openmeteo = OpenMeteoClient()
        self.fmi = FMIClient()
        self.public_spot = PublicSpotClient()

    @staticmethod
    def _health_from_log(log: list[dict[str, Any]], fetched_at: pd.Timestamp) -> dict[str, Any]:
        ok = sum(1 for x in log if x.get('status') == 'ok')
        err = sum(1 for x in log if x.get('status') == 'error')
        empty = sum(1 for x in log if x.get('status') == 'empty')
        return {
            'fetched_at_utc': fetched_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'items_ok': ok, 'items_error': err, 'items_empty': empty,
            'healthy_fraction': ok / max(1, ok + err + empty),
            'details': log,
        }

    @staticmethod
    def _weather_primary(openmeteo: pd.DataFrame, fmi: pd.DataFrame) -> pd.DataFrame:
        # Open-Meteo is the primary canonical weather feed because its JSON
        # response is easy to reproduce and archive. FMI HARMONIE is the Finnish
        # independent fallback. If a canonical timestamp/series is missing from
        # Open-Meteo, fill it from FMI rather than overwriting the primary value.
        if openmeteo.empty:
            return fmi
        if fmi.empty:
            return openmeteo
        keys = set(zip(openmeteo.series_id.astype(str), pd.to_datetime(openmeteo.timestamp_utc, utc=True)))
        add = fmi[[
            (str(r.series_id), pd.Timestamp(r.timestamp_utc)) not in keys
            for r in fmi.itertuples(index=False)
        ]]
        return pd.concat([openmeteo, add], ignore_index=True)


    @staticmethod
    def _manual_spot(fetched_at: pd.Timestamp, origin: pd.Timestamp, forecast_end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load an optional licensed/manual FI day-ahead spot curve.

        Expected columns: delivery_start_utc, spot_price.  This adapter exists
        because Fingrid does not redistribute Nord Pool day-ahead prices and an
        ENTSO-E token may not be available during the hackathon.  It never
        downloads or scrapes licensed data; it only consumes a file the user is
        already entitled to use.
        """
        path = Path(MANUAL_SPOT_CSV)
        if not path.exists():
            return pd.DataFrame(), {"source": "manual_spot", "status": "empty", "rows": 0}
        try:
            raw = pd.read_csv(path)
            if not {"delivery_start_utc", "spot_price"}.issubset(raw.columns):
                raise ValueError("manual_spot.csv needs delivery_start_utc,spot_price")
            raw["timestamp_utc"] = pd.to_datetime(raw["delivery_start_utc"], utc=True, errors="coerce")
            raw["value"] = pd.to_numeric(raw["spot_price"], errors="coerce")
            raw = raw.dropna(subset=["timestamp_utc", "value"])
            raw = raw[(raw["timestamp_utc"] >= pd.Timestamp(origin)) & (raw["timestamp_utc"] <= pd.Timestamp(forecast_end))]
            if raw.empty:
                return pd.DataFrame(), {"source": "manual_spot", "status": "empty", "rows": 0}
            out = pd.DataFrame({
                "series_id": "fi_spot_price",
                "source": "manual_spot",
                "dataset_id": "licensed_manual_curve",
                "timestamp_utc": raw["timestamp_utc"],
                "value": raw["value"],
                "unit": "EUR/MWh",
                "first_usable_origin_utc": fetched_at,
                "kind": "forecast",
            })
            return out, {"source": "manual_spot", "status": "ok", "rows": len(out)}
        except Exception as exc:
            return pd.DataFrame(), {"source": "manual_spot", "status": "error", "rows": 0, "error": str(exc)}

    def collect(self, origin: pd.Timestamp, actual_start: pd.Timestamp, now: pd.Timestamp, forecast_end: pd.Timestamp) -> SourceBundle:
        fetched_at = pd.Timestamp(now)
        if fetched_at.tzinfo is None: fetched_at=fetched_at.tz_localize('UTC')
        else: fetched_at=fetched_at.tz_convert('UTC')
        frames: list[pd.DataFrame] = []
        log: list[dict[str, Any]] = []

        if self.fingrid.configured:
            fg, fglog = self.fingrid.fetch_all(actual_start, now, origin, forecast_end)
            if not fg.empty: frames.append(fg)
            for row in fglog: log.append({'source':'fingrid', **row})
        else:
            log.append({'source':'fingrid','status':'error','error':'FINGRID_API_KEY not configured'})

        if ONLINE_USE_ENTSOE_EXTENDED and self.entsoe.configured:
            ent, entlog = self.entsoe.fetch_extended(actual_start, now, origin, forecast_end, fetched_at)
            if not ent.empty: frames.append(ent)
            log.extend(entlog)
        elif ONLINE_USE_ENTSOE_EXTENDED:
            log.append({'source':'entsoe','status':'error','error':'ENTSOE_API_TOKEN not configured'})

        mcp = pd.DataFrame()
        if ONLINE_USE_ENTSOE_MCP:
            try:
                mcp, mcplog = self.entsoe_mcp.fetch_extended(actual_start, now, origin, forecast_end, fetched_at)
                # If the official ENTSO-E-token client already supplied the canonical
                # FI spot curve, keep it as the higher-priority canonical source while
                # retaining MCP diagnostic series and all other MCP features.
                if 'ent' in locals() and not ent.empty and bool((ent.get('series_id') == 'fi_spot_price').any()) and not mcp.empty:
                    mcp = mcp[mcp['series_id'] != 'fi_spot_price'].copy()
                if not mcp.empty: frames.append(mcp)
                log.extend(mcplog)
            except Exception as exc:
                log.append({'source':'entsoe_mcp','status':'error','error':str(exc)})

        manual_spot, spot_info = self._manual_spot(fetched_at, actual_start, forecast_end)
        if not manual_spot.empty:
            frames.append(manual_spot)
        log.append(spot_info)

        # If neither the licensed/manual curve nor ENTSO-E supplied FI spot, use
        # a clearly-labelled no-key public fallback for the hackathon/demo.
        have_spot = not manual_spot.empty
        if not have_spot and 'ent' in locals() and not ent.empty:
            have_spot = bool((ent.get('series_id') == 'fi_spot_price').any())
        if not have_spot and not mcp.empty:
            have_spot = bool((mcp.get('series_id') == 'fi_spot_price').any())
        if ONLINE_USE_PUBLIC_SPOT_FALLBACK and not have_spot:
            try:
                public_spot, public_info = self.public_spot.fetch(actual_start, forecast_end, fetched_at)
                if not public_spot.empty:
                    frames.append(public_spot)
                log.append(public_info)
            except Exception as exc:
                log.append({'source':'public_spot_fallback','status':'error','error':str(exc)})

        om = pd.DataFrame(); fmi = pd.DataFrame()
        if ONLINE_USE_OPENMETEO:
            try:
                om, info = self.openmeteo.fetch_regional(origin, forecast_end, fetched_at)
                log.append({'source':'openmeteo', **info})
            except Exception as exc:
                log.append({'source':'openmeteo','status':'error','error':str(exc)})
        if ONLINE_USE_FMI:
            try:
                fmi, info = self.fmi.fetch_regional(origin, forecast_end, fetched_at)
                log.append({'source':'fmi_forecast','status':info.get('status','ok'),'rows':info.get('rows',0),'details':info})
            except Exception as exc:
                log.append({'source':'fmi_forecast','status':'error','error':str(exc)})
            try:
                fmi_obs, obs_info = self.fmi.fetch_observations(actual_start, now, fetched_at)
                if not fmi_obs.empty: frames.append(fmi_obs)
                log.append({'source':'fmi_observation','status':obs_info.get('status','ok'),'rows':obs_info.get('rows',0),'details':obs_info})
            except Exception as exc:
                log.append({'source':'fmi_observation','status':'error','error':str(exc)})
        wx = self._weather_primary(om, fmi)
        if not wx.empty: frames.append(wx)

        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[
            'series_id','source','dataset_id','timestamp_utc','value','unit','first_usable_origin_utc','kind'
        ])
        if not data.empty:
            data['timestamp_utc']=pd.to_datetime(data['timestamp_utc'],utc=True)
            data['first_usable_origin_utc']=pd.to_datetime(data['first_usable_origin_utc'],utc=True)
            # Production-only nominal-origin grace window: the daemon collects
            # snapshots just after the full hour. Mark that snapshot as usable
            # for the nominal hourly batch while preserving the real fetched-at
            # time in source-health/context logs. The locked hackathon package
            # never passes through this live orchestrator, so its strict
            # first_usable_origin_utc rule is unchanged.
            nominal_origin=pd.Timestamp(origin)
            if nominal_origin.tzinfo is None: nominal_origin=nominal_origin.tz_localize('UTC')
            else: nominal_origin=nominal_origin.tz_convert('UTC')
            data['first_usable_origin_utc']=nominal_origin
            data['value']=pd.to_numeric(data['value'],errors='coerce')
            data=data.dropna(subset=['timestamp_utc','value']).sort_values(['series_id','timestamp_utc','first_usable_origin_utc'])
            # Newest source snapshot wins only for identical series/timestamp.
            data=data.drop_duplicates(['series_id','timestamp_utc'],keep='last').reset_index(drop=True)

        health=self._health_from_log(log,fetched_at)
        LIVE_SOURCE_HEALTH_JSON.parent.mkdir(parents=True,exist_ok=True)
        LIVE_SOURCE_HEALTH_JSON.write_text(json.dumps(health,indent=2,default=str),encoding='utf-8')

        if ONLINE_ARCHIVE_RAW and not data.empty:
            LIVE_RAW_SNAPSHOT_DIR.mkdir(parents=True,exist_ok=True)
            stamp=pd.Timestamp(origin).strftime('%Y%m%dT%H%MZ')
            data.drop(columns=['kind'],errors='ignore').to_parquet(LIVE_RAW_SNAPSHOT_DIR/f'{stamp}.parquet',index=False)
            (LIVE_RAW_SNAPSHOT_DIR/f'{stamp}.health.json').write_text(json.dumps(health,indent=2,default=str),encoding='utf-8')

        return SourceBundle(data=data,log=log,health=health)

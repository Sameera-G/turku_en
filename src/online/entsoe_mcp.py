from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests

from ..config import ENTSOE_MCP_API_BASE, ENTSOE_MCP_TIMEOUT_SECONDS


class EntsoeMCPError(RuntimeError):
    pass


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _iso(ts: pd.Timestamp) -> str:
    return _utc(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class EntsoeMCPClient:
    """Best-effort reader for entsoemcp.com's public JSON time-series API.

    The service documents /api/series as the programmatic equivalent of its MCP
    get_series tool. V10 uses this only in production/live mode and archives the
    exact snapshot used at each forecast origin. The locked challenge backtest
    is never reconstructed from today's archive.
    """

    base_url: str = ENTSOE_MCP_API_BASE
    timeout_seconds: float = ENTSOE_MCP_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "TE-BalanceAI/10.0", "Accept": "application/json"})

    @staticmethod
    def _rows(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            for key in ("data", "rows", "items", "series", "result"):
                val = payload.get(key)
                if isinstance(val, list):
                    return [x for x in val if isinstance(x, dict)]
                if isinstance(val, dict):
                    for k2 in ("data", "rows", "items"):
                        vv = val.get(k2)
                        if isinstance(vv, list):
                            return [x for x in vv if isinstance(x, dict)]
        return []

    def query(self, endpoint: str, zone: str, start: pd.Timestamp, end: pd.Timestamp, aggregation: str | None = None) -> pd.DataFrame:
        start_s, end_s = _iso(start), _iso(end)
        variants = [
            {"endpoint": endpoint, "zone": zone, "start": start_s, "end": end_s},
            {"endpoint": endpoint, "zone": zone, "from": start_s, "to": end_s},
            {"slug": endpoint, "zone": zone, "start": start_s, "end": end_s},
        ]
        errors: list[str] = []
        for params in variants:
            if aggregation:
                params["aggregation"] = aggregation
            try:
                r = self.session.get(self.base_url, params=params, timeout=self.timeout_seconds)
                if r.status_code != 200:
                    errors.append(f"HTTP {r.status_code}: {r.text[:120]}")
                    continue
                rows = self._rows(r.json())
                if rows:
                    df = pd.DataFrame(rows)
                    if "ts_utc" in df:
                        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
                    return df
                errors.append("200 response but no rows")
            except Exception as exc:
                errors.append(str(exc))
        raise EntsoeMCPError(f"{endpoint} {zone}: " + " | ".join(errors[-3:]))

    @staticmethod
    def _long(df: pd.DataFrame, series_id: str, value_col: str, unit: str, fetched_at: pd.Timestamp, kind: str, dataset_id: str) -> pd.DataFrame:
        if df.empty or "ts_utc" not in df or value_col not in df:
            return pd.DataFrame()
        out = pd.DataFrame({
            "series_id": series_id,
            "source": "entsoe_mcp",
            "dataset_id": dataset_id,
            "timestamp_utc": pd.to_datetime(df["ts_utc"], utc=True, errors="coerce"),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
            "unit": unit,
            "first_usable_origin_utc": _utc(fetched_at),
            "kind": kind,
        }).dropna(subset=["timestamp_utc", "value"])
        return out

    @staticmethod
    def _quarter_hour_forward_fill(long: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if long.empty:
            return long
        frames = []
        for sid, g in long.groupby("series_id", sort=False):
            g = g.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
            idx = pd.date_range(_utc(start).floor("15min"), _utc(end).ceil("15min"), freq="15min", tz="UTC")
            s = g.set_index("timestamp_utc")["value"].reindex(idx).ffill(limit=3)
            meta = g.iloc[-1]
            x = pd.DataFrame({"timestamp_utc": idx, "value": s.to_numpy()}).dropna()
            x["series_id"] = sid; x["source"] = meta["source"]; x["dataset_id"] = meta["dataset_id"]
            x["unit"] = meta["unit"]; x["first_usable_origin_utc"] = meta["first_usable_origin_utc"]; x["kind"] = meta["kind"]
            frames.append(x)
        return pd.concat(frames, ignore_index=True) if frames else long.iloc[:0]

    def fetch_extended(self, actual_start: pd.Timestamp, now: pd.Timestamp, origin: pd.Timestamp, forecast_end: pd.Timestamp, fetched_at: pd.Timestamp) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        frames: list[pd.DataFrame] = []
        log: list[dict[str, Any]] = []

        def run(endpoint: str, zone: str, series_id: str, value_col: str, unit: str, kind: str, start: pd.Timestamp, end: pd.Timestamp, quarter_fill: bool = False):
            try:
                raw = self.query(endpoint, zone, start, end)
                out = self._long(raw, series_id, value_col, unit, fetched_at, kind, f"{endpoint}:{zone}")
                if quarter_fill and not out.empty:
                    out = self._quarter_hour_forward_fill(out, start, end)
                if not out.empty: frames.append(out)
                log.append({"source":"entsoe_mcp","endpoint":endpoint,"zone":zone,"series_id":series_id,"status":"ok" if len(out) else "empty","rows":len(out)})
                return raw
            except Exception as exc:
                log.append({"source":"entsoe_mcp","endpoint":endpoint,"zone":zone,"series_id":series_id,"status":"error","rows":0,"error":str(exc)})
                return pd.DataFrame()

        # Known-ahead market curves. FI spot is mapped to the canonical series
        # so V8/V10 spread forecasting can use it immediately. A duplicate MCP
        # diagnostic series is also kept for dashboard/source analysis.
        fi_da = run("day_ahead_price", "FI", "fi_spot_price", "eur_per_mwh", "EUR/MWh", "forecast", origin, forecast_end, True)
        # Fail fast if the service/API shape is unavailable. One third-party
        # source must never make an hourly production cycle wait through dozens
        # of timeouts. Empty-but-valid responses are allowed to continue.
        if log and log[-1].get("status") == "error":
            log.append({"source":"entsoe_mcp","status":"error","endpoint":"circuit_breaker","error":"initial FI day-ahead probe failed; skipped remaining MCP calls this cycle"})
            return pd.DataFrame(columns=["series_id","source","dataset_id","timestamp_utc","value","unit","first_usable_origin_utc","kind"]), log
        if not fi_da.empty:
            dup = self._long(fi_da, "mcp_fi_day_ahead_price", "eur_per_mwh", "EUR/MWh", fetched_at, "forecast", "day_ahead_price:FI")
            if not dup.empty: frames.append(self._quarter_hour_forward_fill(dup, origin, forecast_end))

        neighbors = ["SE_1","SE_3","EE","NO_4"]
        for z in neighbors:
            zl=z.lower().replace("_","")
            run("day_ahead_price", z, f"mcp_{zl}_day_ahead_price", "eur_per_mwh", "EUR/MWh", "forecast", origin, forecast_end, True)
            run("load_forecast", z, f"mcp_{zl}_load_forecast", "forecast_mw", "MW", "forecast", origin, forecast_end, True)
            run("imbalance_prices", z, f"mcp_{zl}_imbalance_long", "long_eur_per_mwh", "EUR/MWh", "context", actual_start, now, False)
            run("imbalance_prices", z, f"mcp_{zl}_imbalance_short", "short_eur_per_mwh", "EUR/MWh", "context", actual_start, now, False)
        run("load_forecast", "FI", "mcp_fi_load_forecast", "forecast_mw", "MW", "forecast", origin, forecast_end, True)
        run("generation_forecast", "FI", "mcp_fi_generation_forecast", "forecast_mw", "MW", "forecast", origin, forecast_end, True)
        # First intraday auction gives an additional market-price anchor closer
        # to delivery than day-ahead. It is used only when present in the live
        # snapshot; V10 never backfills it into the locked competition replay.
        run("intraday_prices", "FI", "mcp_fi_intraday_price", "eur_per_mwh", "EUR/MWh", "forecast", origin, forecast_end, True)

        # ENTSO-E day-ahead wind/solar forecast. B16=solar, B18/B19=wind.
        # The site exposes one row per PSR type, so aggregate wind types before
        # mapping to a transparent single series.
        try:
            ws = self.query("wind_solar_forecast", "FI", origin, forecast_end)
            if not ws.empty and {"psr_type", "forecast_mw", "ts_utc"}.issubset(ws.columns):
                ws["psr_type"] = ws["psr_type"].astype(str).str.upper()
                for codes, sid in [(({"B18","B19"}), "mcp_fi_wind_forecast"), (({"B16"}), "mcp_fi_solar_forecast")]:
                    part = ws[ws["psr_type"].isin(codes)].copy()
                    if part.empty:
                        continue
                    part["forecast_mw"] = pd.to_numeric(part["forecast_mw"], errors="coerce")
                    part = part.groupby("ts_utc", as_index=False)["forecast_mw"].sum(min_count=1)
                    out = self._long(part, sid, "forecast_mw", "MW", fetched_at, "forecast", "wind_solar_forecast:FI")
                    if not out.empty:
                        frames.append(self._quarter_hour_forward_fill(out, origin, forecast_end))
                    log.append({"source":"entsoe_mcp","endpoint":"wind_solar_forecast","zone":"FI","series_id":sid,"status":"ok" if len(out) else "empty","rows":len(out)})
        except Exception as exc:
            log.append({"source":"entsoe_mcp","endpoint":"wind_solar_forecast","zone":"FI","status":"error","rows":0,"error":str(exc)})

        # Realized balancing context: these values are never treated as future
        # forecasts. They only become lag/context features after the live fetch.
        run("imbalance_volumes", "FI", "mcp_fi_imbalance_volume", "imbalance_mw", "MW", "context", actual_start, now, False)
        try:
            bp = self.query("activated_balancing_price", "FI", actual_start, now)
            if not bp.empty:
                for col, sid in [
                    ("afrr_up_eur_per_mwh","mcp_fi_afrr_up_activated_price"),
                    ("afrr_down_eur_per_mwh","mcp_fi_afrr_down_activated_price"),
                    ("mfrr_up_eur_per_mwh","mcp_fi_mfrr_up_activated_price"),
                    ("mfrr_down_eur_per_mwh","mcp_fi_mfrr_down_activated_price"),
                ]:
                    out = self._long(bp, sid, col, "EUR/MWh", fetched_at, "context", "activated_balancing_price:FI")
                    if not out.empty: frames.append(out)
                    log.append({"source":"entsoe_mcp","endpoint":"activated_balancing_price","zone":"FI","series_id":sid,"status":"ok" if len(out) else "empty","rows":len(out)})
        except Exception as exc:
            log.append({"source":"entsoe_mcp","endpoint":"activated_balancing_price","zone":"FI","status":"error","rows":0,"error":str(exc)})

        # Nordic aFRR capacity-market price + procured MW (native Energinet feed).
        try:
            afrr = self.query("balancing_capacity_price_afrr_energinet", "FI", origin - pd.Timedelta(hours=6), forecast_end)
            if not afrr.empty and "direction" in afrr:
                direction = afrr["direction"].astype(str).str.lower()
                for d, tag in [("up","up"),("down","down")]:
                    part = afrr[direction.str.contains(d, na=False)]
                    for col, suffix, unit in [("capacity_price_eur_mw_h","capacity_price","EUR/MW/h"),("procured_mw","capacity_mw","MW")]:
                        out=self._long(part,f"mcp_fi_afrr_{tag}_{suffix}",col,unit,fetched_at,"forecast","balancing_capacity_price_afrr_energinet:FI")
                        if not out.empty:
                            qout=self._quarter_hour_forward_fill(out, origin, forecast_end)
                            frames.append(qout)
                            if col == "procured_mw":
                                alias=qout.copy(); alias["series_id"]=f"mcp_fi_afrr_{tag}_procured_mw"; frames.append(alias)
                log.append({"source":"entsoe_mcp","endpoint":"balancing_capacity_price_afrr_energinet","zone":"FI","status":"ok","rows":len(afrr)})
        except Exception as exc:
            log.append({"source":"entsoe_mcp","endpoint":"balancing_capacity_price_afrr_energinet","zone":"FI","status":"error","rows":0,"error":str(exc)})

        # Cross-border commercial schedule and day-ahead NTC. Keep each border
        # separately; features later form transparent import/export aggregates.
        borders=[("FI","SE_1"),("FI","SE_3"),("FI","EE"),("FI","NO_4")]
        for a,b in borders:
            for fr,to in [(a,b),(b,a)]:
                tag=f"{fr.lower().replace('_','')}_{to.lower().replace('_','')}"
                run("scheduled_exchanges",f"{fr}->{to}",f"mcp_sched_{tag}","scheduled_mw","MW","forecast",origin,forecast_end,True)
                run("net_transfer_capacity_dayahead",f"{fr}->{to}",f"mcp_ntc_{tag}","ntc_mw","MW","forecast",origin,forecast_end,True)

        # Future outage context: sum unavailable MW only for events already
        # published by the fetch snapshot. This is event-driven and live-only.
        try:
            outg=self.query("outage_generation_unit","FI",origin-pd.Timedelta(days=1),forecast_end+pd.Timedelta(days=1))
            if not outg.empty and {"start_ts_utc","end_ts_utc"}.issubset(outg.columns):
                outg["start_ts_utc"]=pd.to_datetime(outg["start_ts_utc"],utc=True,errors="coerce"); outg["end_ts_utc"]=pd.to_datetime(outg["end_ts_utc"],utc=True,errors="coerce")
                if "publication_ts_utc" in outg:
                    outg["publication_ts_utc"]=pd.to_datetime(outg["publication_ts_utc"],utc=True,errors="coerce"); outg=outg[outg["publication_ts_utc"].isna() | (outg["publication_ts_utc"]<=_utc(fetched_at))]
                nom=pd.to_numeric(outg.get("nominal_power_mw"),errors="coerce").fillna(0); avail=pd.to_numeric(outg.get("available_power_mw"),errors="coerce").fillna(0); outg["unavailable_mw"]=(nom-avail).clip(lower=0)
                rows=[]
                for ts in pd.date_range(_utc(origin),_utc(forecast_end),freq="15min",tz="UTC"):
                    active=outg[(outg["start_ts_utc"]<=ts)&(outg["end_ts_utc"]>ts)]
                    rows.append({"ts_utc":ts,"unavailable_mw":float(active["unavailable_mw"].sum())})
                oo=self._long(pd.DataFrame(rows),"mcp_fi_generation_unavailable_mw","unavailable_mw","MW",fetched_at,"forecast","outage_generation_unit:FI")
                if not oo.empty: frames.append(oo)
                log.append({"source":"entsoe_mcp","endpoint":"outage_generation_unit","zone":"FI","series_id":"mcp_fi_generation_unavailable_mw","status":"ok","rows":len(oo)})
        except Exception as exc:
            log.append({"source":"entsoe_mcp","endpoint":"outage_generation_unit","zone":"FI","status":"error","rows":0,"error":str(exc)})

        combined=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=["series_id","source","dataset_id","timestamp_utc","value","unit","first_usable_origin_utc","kind"])
        return combined,log

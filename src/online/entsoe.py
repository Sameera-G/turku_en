from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET
from typing import Any

import numpy as np
import pandas as pd
import requests

from ..config import ENTSOE_API_BASE, ENTSOE_API_TOKEN

FI_DOMAIN = "10YFI-1--------U"
NEIGHBOURS = {
    "EE": "10Y1001A1001A39I",
    "SE1": "10Y1001A1001A44P",
    "SE3": "10Y1001A1001A46L",
    "NO4": "10YNO-4--------9",
}
PSR_GROUP = {
    "B10": "hydro", "B11": "hydro", "B12": "hydro",
    "B14": "nuclear", "B16": "solar", "B18": "wind", "B19": "wind",
}


def _period_code(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None: t = t.tz_localize("UTC")
    else: t = t.tz_convert("UTC")
    return t.strftime("%Y%m%d%H%M")


def _duration_minutes(text: str | None) -> int:
    if not text: return 60
    if text == "PT15M": return 15
    if text == "PT30M": return 30
    if text == "PT60M" or text == "PT1H": return 60
    if text.startswith("PT") and text.endswith("M"):
        try: return int(text[2:-1])
        except Exception: return 60
    return 60


def _find_text(el: ET.Element, path: str) -> str | None:
    x = el.find(path)
    return x.text if x is not None else None


@dataclass
class EntsoeClient:
    token: str = ENTSOE_API_TOKEN
    base_url: str = ENTSOE_API_BASE
    timeout_seconds: float = 40.0

    @property
    def configured(self) -> bool:
        return bool(self.token)

    def _get(self, params: dict[str, Any]) -> ET.Element:
        if not self.configured:
            raise RuntimeError("ENTSOE_API_TOKEN is not configured")
        p = dict(params)
        p["securityToken"] = self.token
        r = requests.get(self.base_url, params=p, timeout=self.timeout_seconds)
        if r.status_code != 200:
            raise RuntimeError(f"ENTSO-E HTTP {r.status_code}: {r.text[:350]}")
        return ET.fromstring(r.content)

    def _series_points(self, root: ET.Element, value_tags=("quantity", "price.amount")) -> pd.DataFrame:
        rows=[]
        for ts_el in root.findall('.//{*}TimeSeries'):
            psr = _find_text(ts_el, './/{*}MktPSRType/{*}psrType') or _find_text(ts_el, './/{*}psrType')
            for period in ts_el.findall('.//{*}Period'):
                st = _find_text(period, './{*}timeInterval/{*}start')
                if not st: continue
                start = pd.Timestamp(st)
                if start.tzinfo is None: start=start.tz_localize('UTC')
                else: start=start.tz_convert('UTC')
                minutes = _duration_minutes(_find_text(period, './{*}resolution'))
                for point in period.findall('./{*}Point'):
                    pos = _find_text(point, './{*}position')
                    val = None
                    used = None
                    for tag in value_tags:
                        x = point.find(f'./{{*}}{tag}')
                        if x is not None and x.text is not None:
                            try: val=float(x.text); used=tag; break
                            except Exception: pass
                    if pos is None or val is None: continue
                    rows.append({
                        'timestamp_utc': start + pd.Timedelta(minutes=(int(pos)-1)*minutes),
                        'value': val, 'resolution_minutes': minutes, 'psr_type': psr, 'value_tag': used,
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def _expand_15m(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        rows=[]
        for _,r in df.iterrows():
            m=int(r.get('resolution_minutes',60) or 60)
            repeats=max(1,m//15)
            for q in range(repeats):
                d=r.to_dict(); d['timestamp_utc']=pd.Timestamp(r['timestamp_utc'])+pd.Timedelta(minutes=15*q); d['resolution_minutes']=15; rows.append(d)
        return pd.DataFrame(rows)

    @staticmethod
    def _long(df: pd.DataFrame, series_id: str, fetched_at: pd.Timestamp, kind: str, dataset_id: str, unit: str="") -> pd.DataFrame:
        if df.empty: return pd.DataFrame()
        out=pd.DataFrame({
            'series_id':series_id,'source':'entsoe_live','dataset_id':dataset_id,
            'timestamp_utc':pd.to_datetime(df['timestamp_utc'],utc=True),'value':pd.to_numeric(df['value'],errors='coerce'),
            'unit':unit,'first_usable_origin_utc':fetched_at,'kind':kind,
        }).dropna(subset=['timestamp_utc','value'])
        return out.drop_duplicates(['series_id','timestamp_utc'],keep='last')

    def fetch_fi_day_ahead(self, start: pd.Timestamp, end: pd.Timestamp, fetched_at: pd.Timestamp) -> pd.DataFrame:
        root=self._get({'documentType':'A44','in_Domain':FI_DOMAIN,'out_Domain':FI_DOMAIN,
                        'periodStart':_period_code(start),'periodEnd':_period_code(end)})
        df=self._expand_15m(self._series_points(root,('price.amount',)))
        return self._long(df,'fi_spot_price',fetched_at,'forecast','A44:FI','EUR/MWh')

    def fetch_load(self, start: pd.Timestamp, end: pd.Timestamp, fetched_at: pd.Timestamp, process: str) -> pd.DataFrame:
        root=self._get({'documentType':'A65','processType':process,'outBiddingZone_Domain':FI_DOMAIN,
                        'periodStart':_period_code(start),'periodEnd':_period_code(end)})
        df=self._expand_15m(self._series_points(root,('quantity',)))
        sid='entsoe_load_forecast' if process!='A16' else 'entsoe_actual_load'
        return self._long(df,sid,fetched_at,'forecast' if process!='A16' else 'actual',f'A65:{process}:FI','MW')

    def fetch_wind_solar_forecast(self, start: pd.Timestamp, end: pd.Timestamp, fetched_at: pd.Timestamp) -> pd.DataFrame:
        root=self._get({'documentType':'A69','processType':'A01','in_Domain':FI_DOMAIN,
                        'periodStart':_period_code(start),'periodEnd':_period_code(end)})
        df=self._expand_15m(self._series_points(root,('quantity',)))
        if df.empty: return df
        outs=[]
        for group,sid in [('wind','entsoe_wind_forecast'),('solar','entsoe_solar_forecast')]:
            g=df[df.psr_type.map(PSR_GROUP).eq(group)].groupby('timestamp_utc',as_index=False)['value'].sum()
            if not g.empty: outs.append(self._long(g,sid,fetched_at,'forecast','A69:FI','MW'))
        return pd.concat(outs,ignore_index=True) if outs else pd.DataFrame()

    def fetch_generation_mix(self, start: pd.Timestamp, end: pd.Timestamp, fetched_at: pd.Timestamp) -> pd.DataFrame:
        root=self._get({'documentType':'A75','processType':'A16','in_Domain':FI_DOMAIN,
                        'periodStart':_period_code(start),'periodEnd':_period_code(end)})
        df=self._expand_15m(self._series_points(root,('quantity',)))
        if df.empty: return df
        outs=[]
        for group in ['wind','solar','hydro','nuclear']:
            g=df[df.psr_type.map(PSR_GROUP).eq(group)].groupby('timestamp_utc',as_index=False)['value'].sum()
            if not g.empty: outs.append(self._long(g,f'entsoe_{group}_generation',fetched_at,'actual','A75:FI','MW'))
        return pd.concat(outs,ignore_index=True) if outs else pd.DataFrame()

    def fetch_hydro_reservoir(self, start: pd.Timestamp, end: pd.Timestamp, fetched_at: pd.Timestamp) -> pd.DataFrame:
        root=self._get({'documentType':'A72','processType':'A16','in_Domain':FI_DOMAIN,
                        'periodStart':_period_code(start),'periodEnd':_period_code(end)})
        df=self._series_points(root,('quantity',))
        if df.empty: return df
        g=df.groupby('timestamp_utc',as_index=False)['value'].sum()
        # Reservoir figures are slow-moving. Forward-fill the latest value on
        # the 15-minute forecast grid as contextual state, not as a target.
        grid=pd.DataFrame({'timestamp_utc':pd.date_range(start.floor('15min'),end.ceil('15min'),freq='15min',tz='UTC')})
        g=pd.merge_asof(grid.sort_values('timestamp_utc'),g.sort_values('timestamp_utc'),on='timestamp_utc',direction='backward')
        return self._long(g.dropna(subset=['value']),'entsoe_hydro_reservoir',fetched_at,'forecast','A72:FI','MWh')

    def fetch_generation_outages(self, start: pd.Timestamp, end: pd.Timestamp, fetched_at: pd.Timestamp) -> pd.DataFrame:
        root=self._get({'documentType':'A80','biddingZone_Domain':FI_DOMAIN,
                        'periodStart':_period_code(start),'periodEnd':_period_code(end)})
        # Outage docs differ from ordinary time-series docs. Prefer quantities
        # when available; otherwise count active TimeSeries records.
        df=self._series_points(root,('quantity',))
        grid=pd.date_range(start.floor('15min'),end.ceil('15min'),freq='15min',tz='UTC')
        if not df.empty:
            val=float(np.nansum(np.abs(df['value'].to_numpy(float))))
        else:
            val=float(len(root.findall('.//{*}TimeSeries')))
        g=pd.DataFrame({'timestamp_utc':grid,'value':val})
        return self._long(g,'entsoe_generation_unavailable_mw',fetched_at,'forecast','A80:FI','MW')

    def fetch_transmission_outages(self, start: pd.Timestamp, end: pd.Timestamp, fetched_at: pd.Timestamp) -> pd.DataFrame:
        total=0
        errors=[]
        for name,eic in NEIGHBOURS.items():
            try:
                root=self._get({'documentType':'A78','in_Domain':FI_DOMAIN,'out_Domain':eic,
                                'periodStart':_period_code(start),'periodEnd':_period_code(end)})
                total += len(root.findall('.//{*}TimeSeries'))
            except Exception as exc:
                errors.append(f'{name}:{exc}')
        grid=pd.date_range(start.floor('15min'),end.ceil('15min'),freq='15min',tz='UTC')
        g=pd.DataFrame({'timestamp_utc':grid,'value':float(total)})
        out=self._long(g,'entsoe_transmission_outage_count',fetched_at,'forecast','A78:FI-neighbours','count')
        out.attrs['errors']=errors
        return out

    def fetch_extended(self, actual_start: pd.Timestamp, now: pd.Timestamp, forecast_start: pd.Timestamp, forecast_end: pd.Timestamp, fetched_at: pd.Timestamp) -> tuple[pd.DataFrame,list[dict[str,Any]]]:
        jobs=[
            ('spot',lambda:self.fetch_fi_day_ahead(forecast_start.floor('D'),forecast_end.ceil('D'),fetched_at)),
            ('load_forecast',lambda:self.fetch_load(forecast_start,forecast_end,fetched_at,'A01')),
            ('actual_load',lambda:self.fetch_load(actual_start,now,fetched_at,'A16')),
            ('wind_solar_forecast',lambda:self.fetch_wind_solar_forecast(forecast_start,forecast_end,fetched_at)),
            ('generation_mix',lambda:self.fetch_generation_mix(actual_start,now,fetched_at)),
            ('hydro_reservoir',lambda:self.fetch_hydro_reservoir(forecast_start-pd.Timedelta(days=7),forecast_end,fetched_at)),
            ('generation_outages',lambda:self.fetch_generation_outages(forecast_start,forecast_end,fetched_at)),
            ('transmission_outages',lambda:self.fetch_transmission_outages(forecast_start,forecast_end,fetched_at)),
        ]
        frames=[]; log=[]
        for name,fn in jobs:
            try:
                x=fn()
                if x is not None and not x.empty: frames.append(x)
                log.append({'source':'entsoe','item':name,'status':'ok','rows':0 if x is None else len(x)})
            except Exception as exc:
                log.append({'source':'entsoe','item':name,'status':'error','rows':0,'error':str(exc)})
        return (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()),log

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET
from typing import Any

import numpy as np
import pandas as pd
import requests

from ..config import FMI_WFS_BASE, WEATHER_NODES


@dataclass
class FMIClient:
    base_url: str = FMI_WFS_BASE
    timeout_seconds: float = 30.0

    @staticmethod
    def _parse_timevaluepair(content: bytes, keys: list[str]) -> pd.DataFrame:
        root = ET.fromstring(content)
        records: list[dict[str, Any]] = []
        for ts_el in root.findall('.//{*}MeasurementTimeseries'):
            ident = ts_el.attrib.get('{http://www.opengis.net/gml/3.2}id', '') or ts_el.attrib.get('id', '')
            ident_l = ident.lower()
            param = next((key for key in keys if key.lower() in ident_l), None)
            if param is None:
                continue
            for tvp in ts_el.findall('.//{*}MeasurementTVP'):
                t = tvp.find('./{*}time'); v = tvp.find('./{*}value')
                if t is None or v is None:
                    continue
                try:
                    records.append({'timestamp_utc': pd.Timestamp(t.text), 'parameter': param, 'value': float(v.text)})
                except Exception:
                    continue
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
        return df

    def _fetch_place_forecast(self, place: str) -> tuple[pd.DataFrame, str | None]:
        params = {
            'service': 'WFS', 'version': '2.0.0', 'request': 'getFeature',
            'storedquery_id': 'fmi::forecast::harmonie::surface::point::timevaluepair',
            'place': place,
            'parameters': 'temperature,windspeedms,windgust,pressure,precipitation1h,totalcloudcover,radiationglobal',
        }
        r = requests.get(self.base_url, params=params, timeout=self.timeout_seconds)
        if r.status_code != 200:
            return pd.DataFrame(), f'HTTP {r.status_code}'
        df = self._parse_timevaluepair(r.content, ['temperature','windspeedms','windgust','pressure','precipitation1h','totalcloudcover','radiationglobal'])
        return (df, None) if not df.empty else (df, 'no parseable timevaluepair rows')

    def _fetch_place_observation(self, place: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, str | None]:
        # FMI documents fmi::observations::weather::timevaluepair for surface
        # weather observations. Exact observation rows are stored separately
        # from forecasts so V6 can learn forecast-error regimes without using
        # future/revised weather information at the original forecast time.
        params = {
            'service': 'WFS', 'version': '2.0.0', 'request': 'getFeature',
            'storedquery_id': 'fmi::observations::weather::timevaluepair',
            'place': place,
            'starttime': pd.Timestamp(start).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'endtime': pd.Timestamp(end).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'timestep': 60,
            'parameters': 'temperature,windspeedms,pressure,precipitation1h,totalcloudcover',
        }
        r = requests.get(self.base_url, params=params, timeout=self.timeout_seconds)
        if r.status_code != 200:
            return pd.DataFrame(), f'HTTP {r.status_code}'
        df = self._parse_timevaluepair(r.content, ['temperature','windspeedms','pressure','precipitation1h','totalcloudcover'])
        return (df, None) if not df.empty else (df, 'no parseable observation rows')

    @staticmethod
    def _aggregate_nodes(raw: pd.DataFrame, mapping: dict[str, str], fetched_at: pd.Timestamp, source: str, kind: str) -> pd.DataFrame:
        pivot = raw.pivot_table(index=['timestamp_utc','node','weight'], columns='parameter', values='value', aggfunc='last').reset_index()
        rows=[]
        for ts, g in pivot.groupby('timestamp_utc', sort=True):
            weights=g['weight'].to_numpy(float); weights=weights/weights.sum()
            for c,sid in mapping.items():
                if c not in g: continue
                vals=pd.to_numeric(g[c],errors='coerce').to_numpy(float)
                good=np.isfinite(vals)
                if not good.any(): continue
                if c=='windgust': value=float(np.nanmax(vals))
                else:
                    w=weights[good]; w=w/w.sum(); value=float(np.sum(vals[good]*w))
                # FMI values are commonly hourly here. Expand to the four
                # quarters for the live state surface. The original fetched-at
                # timestamp remains the availability boundary.
                for q in range(4):
                    rows.append({'series_id':sid,'source':source,'dataset_id':kind,
                                 'timestamp_utc':pd.Timestamp(ts)+pd.Timedelta(minutes=15*q),'value':value,'unit':'',
                                 'first_usable_origin_utc':fetched_at,'kind':kind})
        return pd.DataFrame(rows)

    def fetch_regional(self, start: pd.Timestamp, end: pd.Timestamp, fetched_at: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
        all_rows=[]; details=[]
        for name in WEATHER_NODES:
            df,err=self._fetch_place_forecast(name)
            if err:
                details.append({'node':name,'status':'error','error':err}); continue
            df=df[(df.timestamp_utc>=start.floor('h'))&(df.timestamp_utc<=end.ceil('h'))]
            df['node']=name; df['weight']=WEATHER_NODES[name][2]
            all_rows.append(df); details.append({'node':name,'status':'ok','rows':len(df)})
        if not all_rows:
            return pd.DataFrame(), {'status':'error','nodes':details}
        raw=pd.concat(all_rows,ignore_index=True)
        mapping={
            'windspeedms':'weather_wind_10m_weighted', 'windgust':'weather_wind_gust_max',
            'temperature':'weather_temperature_mean', 'pressure':'weather_pressure_mean',
            'precipitation1h':'weather_precipitation_mean', 'totalcloudcover':'weather_cloud_mean',
            'radiationglobal':'weather_solar_mean',
        }
        out=self._aggregate_nodes(raw,mapping,fetched_at,'fmi_harmonie_live','forecast')
        return out, {'status':'ok','nodes':details,'rows':len(out)}

    def fetch_observations(self, start: pd.Timestamp, end: pd.Timestamp, fetched_at: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
        all_rows=[]; details=[]
        # Observation requests are limited to the same representative nodes so
        # the error signal is spatially consistent with the regional forecast.
        for name in WEATHER_NODES:
            df,err=self._fetch_place_observation(name,start,end)
            if err:
                details.append({'node':name,'status':'error','error':err}); continue
            df['node']=name; df['weight']=WEATHER_NODES[name][2]
            all_rows.append(df); details.append({'node':name,'status':'ok','rows':len(df)})
        if not all_rows:
            return pd.DataFrame(), {'status':'error','nodes':details}
        raw=pd.concat(all_rows,ignore_index=True)
        mapping={
            'windspeedms':'weather_obs_wind_10m_weighted', 'temperature':'weather_obs_temperature_mean',
            'pressure':'weather_obs_pressure_mean', 'precipitation1h':'weather_obs_precipitation_mean',
            'totalcloudcover':'weather_obs_cloud_mean',
        }
        out=self._aggregate_nodes(raw,mapping,fetched_at,'fmi_weather_observation','actual')
        return out, {'status':'ok','nodes':details,'rows':len(out)}

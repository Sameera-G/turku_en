from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/'.env')
sys.path.insert(0,str(ROOT))

from src.online.entsoe_mcp import EntsoeMCPClient


def main() -> None:
    c=EntsoeMCPClient()
    end=pd.Timestamp.now(tz='UTC').ceil('h')
    start=end-pd.Timedelta(hours=12)
    checks=[
        ('day_ahead_price','FI'),
        ('load_forecast','FI'),
        ('generation_forecast','FI'),
        ('wind_solar_forecast','FI'),
        ('intraday_prices','FI'),
        ('imbalance_volumes','FI'),
        ('activated_balancing_price','FI'),
        ('day_ahead_price','SE_1'),
        ('scheduled_exchanges','FI->SE_1'),
        ('net_transfer_capacity_dayahead','FI->SE_3'),
        ('balancing_capacity_price_afrr_energinet','FI'),
    ]
    ok=0
    print('ENTSO-E MCP /api/series connectivity test')
    print('Source: https://entsoemcp.com  (live/best-effort; no effect on locked challenge backtest)')
    for endpoint,zone in checks:
        try:
            df=c.query(endpoint,zone,start,end+pd.Timedelta(days=1))
            print(f'OK   {endpoint:42s} {zone:10s} rows={len(df):5d} columns={list(df.columns)[:8]}')
            ok+=1
        except Exception as exc:
            print(f'FAIL {endpoint:42s} {zone:10s} {exc}')
    print(f'\npassed={ok}/{len(checks)}')
    if ok == 0:
        print('V10 will continue safely with Fingrid + weather + existing spot fallback; MCP failures do not stop forecasting.')


if __name__=='__main__':
    main()

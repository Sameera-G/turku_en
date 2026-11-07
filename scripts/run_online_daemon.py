from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def next_run(delay_seconds:int) -> datetime:
    """Next full-hour UTC plus a short publication/ingestion delay."""
    now=datetime.now(timezone.utc)
    hour=now.replace(minute=0,second=0,microsecond=0)
    candidate=hour+timedelta(seconds=delay_seconds)
    if candidate<=now:
        candidate=hour+timedelta(hours=1,seconds=delay_seconds)
    return candidate


def main() -> None:
    ap=argparse.ArgumentParser(description='Run each V11 cycle in a fresh child process every full hour.')
    ap.add_argument('--delay-seconds',type=int,default=75,help='Seconds after each full hour to fetch fresh data and forecast that origin.')
    ap.add_argument('--once',action='store_true')
    args=ap.parse_args()
    if args.once:
        raise SystemExit(subprocess.run([sys.executable,str(ROOT/'scripts'/'run_online_cycle.py')],cwd=ROOT).returncode)
    print(f'[V11 daemon] cycle at each full hour + {args.delay_seconds}s UTC; Ctrl+C to stop',flush=True)
    while True:
        target=next_run(args.delay_seconds)
        seconds=max(0,(target-datetime.now(timezone.utc)).total_seconds())
        time.sleep(seconds)
        result=subprocess.run([sys.executable,str(ROOT/'scripts'/'run_online_cycle.py')],cwd=ROOT)
        print(f'[V11 daemon] {datetime.now(timezone.utc).isoformat()} cycle exit={result.returncode}',flush=True)


if __name__=='__main__':
    main()

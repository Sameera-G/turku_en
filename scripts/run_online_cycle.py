from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/'.env')
sys.path.insert(0,str(ROOT))

import pandas as pd
from src.online.continuous import ContinuousLearningService


def main() -> None:
    ap=argparse.ArgumentParser(description='Run one TE-BalanceAI V6 production multi-window continuous-learning cycle')
    ap.add_argument('--origin',help='UTC forecast origin. Default: current full hour (cycle should run just after the hour).')
    ap.add_argument('--no-ingest',action='store_true',help='Skip live API ingestion for offline smoke tests.')
    ap.add_argument('--no-retrain',action='store_true',help='Skip scheduled challenger retrain gate.')
    args=ap.parse_args()
    origin=pd.Timestamp(args.origin) if args.origin else None
    service=ContinuousLearningService()
    result=service.run_cycle(origin=origin,ingest=not args.no_ingest,allow_retrain=not args.no_retrain)
    print(json.dumps(result,indent=2,default=str))


if __name__=='__main__':
    main()

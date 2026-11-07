from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online.adaptive import AdaptiveState
from src.online.continuous import default_model_path
from src.online.daily import run_daily_learning
from src.online.store import OnlineStore


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the V7 daily learning/performance checkpoint now.")
    ap.add_argument("--force", action="store_true", help="Run even if today's scheduled checkpoint already exists.")
    ap.add_argument("--no-retrain", action="store_true", help="Create the daily report without attempting a challenger retrain.")
    args = ap.parse_args()
    result = run_daily_learning(
        OnlineStore(), default_model_path(), AdaptiveState.load(),
        allow_retrain=not args.no_retrain, force=args.force,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

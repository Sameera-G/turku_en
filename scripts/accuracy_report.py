from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.online.store import OnlineStore
from src.performance import official_validation_summary, validation_intuitive_accuracy, live_summary, public_baseline_comparison


def main() -> None:
    official = official_validation_summary()
    intuitive = validation_intuitive_accuracy()
    baseline = public_baseline_comparison(official)
    store = OnlineStore()
    now = pd.Timestamp.now(tz="UTC")
    live = {}
    for hours in [24, 168, 336]:
        live[str(hours)] = live_summary(store.recent_joined(now - pd.Timedelta(hours=hours)))
    print(json.dumps({
        "official_validation": official,
        "organizer_public_baseline_comparison": baseline,
        "official_intuitive_diagnostics": intuitive,
        "live": live,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()

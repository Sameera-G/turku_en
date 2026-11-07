from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FORECAST_ORIGINS_CSV
from src.data import load_long


def main() -> None:
    data = load_long()
    origins = pd.read_csv(FORECAST_ORIGINS_CSV)
    origins["forecast_origin_utc"] = pd.to_datetime(origins["forecast_origin_utc"], utc=True)

    rows = []
    for t in origins["forecast_origin_utc"]:
        legal = data[data["first_usable_origin_utc"] <= t]
        test_context = legal[legal["timestamp_utc"] >= pd.Timestamp("2026-10-04T21:00:00Z")]
        rows.append({
            "forecast_origin_utc": t,
            "legal_rows_total": len(legal),
            "legal_test_period_context_rows": len(test_context),
            "latest_legal_timestamp_utc": legal["timestamp_utc"].max(),
        })
    out = pd.DataFrame(rows)
    path = ROOT / "reports" / "v2" / "walk_forward_audit.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(out.head(3).to_string(index=False))
    print("...")
    print(out.tail(3).to_string(index=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

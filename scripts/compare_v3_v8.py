from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import REPORTS_DIR
from src.performance import PUBLIC_VALIDATION_BASELINE


def load(version: str) -> dict:
    p = REPORTS_DIR / version / "official_validation" / "score" / "score_summary.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p).iloc[0].to_dict()


def main() -> None:
    v3, v8 = load("v3"), load("v8")
    rows = []
    specs = [
        ("P50 MAE", "mae_p50", "EUR/MWh"),
        ("P10/P90 pinball", "calibration_pinball_mean", ""),
        ("P10-P90 coverage", "coverage_p10_p90", ""),
        ("Spike Brier", "brier_spike_mean", ""),
        ("Direction Brier", "brier_direction", ""),
    ]
    for label, key, unit in specs:
        a, b = float(v3[key]), float(v8[key])
        if key == "coverage_p10_p90":
            improvement = abs(b - .80) - abs(a - .80)
        else:
            improvement = 100 * (a - b) / a
        rows.append({"metric": label, "V3": a, "V8": b, "change": improvement, "unit": unit})
    print(pd.DataFrame(rows).to_string(index=False))
    base = PUBLIC_VALIDATION_BASELINE["mae_p50"]
    print(f"\nOrganizer public validation baseline MAE: {base:.2f} EUR/MWh")
    print(f"V8 MAE reduction vs organizer baseline: {100*(base-float(v8['mae_p50']))/base:.1f}%")
    print(f"V8 MAE reduction vs V3: {100*(float(v3['mae_p50'])-float(v8['mae_p50']))/float(v3['mae_p50']):.1f}%")


if __name__ == "__main__":
    main()

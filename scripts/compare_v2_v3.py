from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "v2": ROOT / "reports" / "v2" / "validation_metrics.csv",
    "v3": ROOT / "reports" / "v3" / "validation_metrics.csv",
}
rows = []
for version, path in FILES.items():
    if path.exists():
        df = pd.read_csv(path)
        row = df[df["variant"] == "calibrated"].iloc[0].to_dict()
        row["version"] = version
        rows.append(row)
if not rows:
    raise SystemExit("No V2/V3 metrics found yet.")
out = pd.DataFrame(rows).set_index("version")
print(out.to_string())
path = ROOT / "reports" / "v3" / "v2_v3_comparison.csv"
path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(path)
print(f"\nWrote {path}")

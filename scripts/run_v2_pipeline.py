from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str):
    print(f"\n===== {script} =====")
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


def main():
    run("audit_walk_forward.py")
    run("train_v2.py")
    run("generate_backtest_v2.py")
    print("\nV2 COMPLETE. Check reports/v2 and outputs/v2.")


if __name__ == "__main__":
    main()

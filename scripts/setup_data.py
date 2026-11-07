from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TARGET = RAW / "starter_package_v1"

REQUIRED = [
    TARGET / "README.md",
    TARGET / "forecast_origins.csv",
    TARGET / "sample_submission.csv",
    TARGET / "availability.csv",
    TARGET / "validate_submission.py",
    TARGET / "score.py",
    TARGET / "make_templates.py",
    TARGET / "data" / "inputs.csv",
    TARGET / "data" / "actuals_validation.csv",
    TARGET / "data" / "spot_nordpool.csv",
]


def package_ready() -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in REQUIRED)


def download(url: str, dest: Path) -> None:
    print(f"Downloading official starter package -> {dest}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)


def extract(zip_path: Path) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        # Re-extraction is intentional if a licensed/required file is missing.
        zf.extractall(RAW)
    if not package_ready():
        missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists() or p.stat().st_size == 0]
        raise RuntimeError("Starter package extraction incomplete. Missing: " + ", ".join(missing))


def main() -> None:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, default=None, help="Path to organizer starter_package_v1.zip")
    p.add_argument("--url", default=None, help="Organizer-provided direct download URL")
    p.add_argument("--force", action="store_true", help="Re-extract even if the package looks complete")
    args = p.parse_args()

    if package_ready() and not args.force:
        print(f"Starter package already ready: {TARGET}")
        return

    if TARGET.exists() and not package_ready():
        print("Starter package folder exists but required files are missing; repairing from ZIP/source.")

    zip_path = args.zip
    url = args.url or os.getenv("STARTER_PACKAGE_URL", "").strip()

    if zip_path is None:
        env_zip = os.getenv("STARTER_PACKAGE_ZIP", "starter_package_v1.zip")
        candidate = ROOT / env_zip
        if candidate.exists():
            zip_path = candidate

    if zip_path is None and url:
        zip_path = ROOT / "starter_package_v1.zip"
        download(url, zip_path)

    if zip_path is None or not zip_path.exists():
        raise SystemExit(
            "No official starter ZIP found. Put starter_package_v1.zip in the project root, "
            "pass --zip PATH, or set STARTER_PACKAGE_URL to the organizer-provided URL. "
            "The package contains licensed Nord Pool data, so this project does not mirror it publicly."
        )

    extract(zip_path)
    print("Data setup complete.")
    print("IMPORTANT: data/raw is gitignored because the Nord Pool file is hackathon-only licensed data.")


if __name__ == "__main__":
    main()

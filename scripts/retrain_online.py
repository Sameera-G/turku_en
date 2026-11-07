from __future__ import annotations

import json
from pathlib import Path
import sys
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/".env")
sys.path.insert(0,str(ROOT))

from src.online.continuous import default_model_path
from src.online.retrain import maybe_retrain
from src.online.store import OnlineStore


def main() -> None:
    result=maybe_retrain(OnlineStore(),default_model_path())
    print(json.dumps(result.__dict__,indent=2,default=str))


if __name__=="__main__":
    main()

from __future__ import annotations

import shutil
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.config import MODELS_DIR, ONLINE_CHAMPION
from src.models_v8 import load_any_bundle


def main() -> None:
    source=MODELS_DIR/'v10'/'model_bundle_v10.joblib'
    if not source.exists():
        raise FileNotFoundError(f'Missing {source}')
    source_bundle=load_any_bundle(source)
    ONLINE_CHAMPION.parent.mkdir(parents=True,exist_ok=True)
    if ONLINE_CHAMPION.exists():
        current=load_any_bundle(ONLINE_CHAMPION)
        current_version=getattr(current,'version',ONLINE_CHAMPION.stem)
        if current_version == getattr(source_bundle,'version',source.stem):
            print(f'Already active: {current_version}')
            return
        backup_dir=ONLINE_CHAMPION.parent/'backups'; backup_dir.mkdir(parents=True,exist_ok=True)
        stamp=pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')
        backup=backup_dir/f'champion_{stamp}.joblib'
        shutil.copy2(ONLINE_CHAMPION,backup)
        print(f'Backed up current champion {current_version} -> {backup}')
    shutil.copy2(source,ONLINE_CHAMPION)
    b=load_any_bundle(ONLINE_CHAMPION)
    print(f'Activated V10 refit champion: {getattr(b,"version",ONLINE_CHAMPION.stem)}')
    print('Adaptive state and online SQLite history were preserved.')

if __name__=='__main__':
    main()

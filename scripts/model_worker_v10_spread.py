from __future__ import annotations
import argparse,sys
from pathlib import Path
import joblib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models_v8 import _spread_lgb,_spread_cat

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--kind',required=True,choices=['global_lgb','global_cat','down_lgb','up_lgb']);ap.add_argument('--output',required=True);a=ap.parse_args();d=Path(a.data_dir)
 y=np.load(d/'y_spread.npy',mmap_mode='r'); direction=np.load(d/'direction.npy',mmap_mode='r')
 if a.kind=='global_cat': model=_spread_cat(); X=np.load(d/'X_cat.npy',mmap_mode='r'); mask=slice(None)
 else:
  model=_spread_lgb(); X=np.load(d/'X.npy',mmap_mode='r'); mask=slice(None)
  if a.kind=='down_lgb': mask=(direction==0)
  elif a.kind=='up_lgb': mask=(direction==2)
 print(f'[V10 worker] {a.kind}: rows={len(y) if isinstance(mask,slice) else int(mask.sum()):,}',flush=True)
 model.fit(X[mask],y[mask]);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,out);print('[V10 worker] saved',out,flush=True)
if __name__=='__main__': main()

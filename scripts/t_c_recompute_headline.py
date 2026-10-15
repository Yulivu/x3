#!/usr/bin/env python3
"""T-C: independent EER recomputation without importing ``metrics3``."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve
def eer(y,s):
 if len(np.unique(y))<2 or len(np.unique(s))<2:return 50.0
 f,t,_=roc_curve(y,s);i=np.argmin(np.abs((1-t)-f));return float(((1-t[i])+f[i])*50)
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--scores',default='results/scores');a=p.parse_args(argv);root=Path(a.root);rows=[]
 for f in sorted((root/a.scores).glob('*.npz')):
  with np.load(f) as z:
   for k in z.files:
    if k.startswith('s_'):rows.append({'file':f.name,'method':k[2:],'pooled_eer':eer(z['y'],z[k])})
 out=root/'results/t_c_recompute.csv';out.parent.mkdir(exist_ok=True);import pandas as pd;pd.DataFrame(rows).to_csv(out,index=False)
 (root/'results/t_c_recompute_verdict.json').write_text(json.dumps({'status':'RECOMPUTED' if rows else 'RECOMPUTE_INPUT_MISSING','n_rows':len(rows)},indent=2)+'\n');return 0
if __name__=='__main__':raise SystemExit(main())

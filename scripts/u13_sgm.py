#!/usr/bin/env python3
"""U13: SGM on the Z, E0, F, and U tracks (NEW_TASKS 05 §2.1)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
REPO=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from src import regions as R
from src.data import data_root
from src.metrics3 import summarize
from src.newtasks05 import zero_scores, speaker_split, enrol_mask, write_csv_atomic
from src.sgm import fit_1d, fit_sgm_lr

def _row(sub, mask, score, method, track, draw, backbone, spec, nlab, extra=None):
 r=summarize(sub.y.to_numpy()[mask],np.asarray(score)[mask],sub.platform.to_numpy()[mask],extra={'protocol':'u13_sgm_v3','region':'dev','track':track,'draw':draw,'backbone':backbone,'spec':spec,'method':method,'mask':track,'k':nlab,'n_labeled':nlab,**(extra or {})})
 return r
def _labels(n): return np.full(n,-1,int)
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--root',default='.');p.add_argument('--data');p.add_argument('--backbone',default='xls-r-2b');p.add_argument('--spec',default='S3');p.add_argument('--split',default=R.SPLIT_V3);p.add_argument('--draws',type=int,default=100);p.add_argument('--tracks',default='Z,E0,F,U');p.add_argument('--out',default='results/u13_sgm_draws.csv');p.add_argument('--summary',default=None);a=p.parse_args(argv)
 root=Path(a.root); data=data_root(a.data); sub,X=R.load_region(root,data,a.backbone,a.spec,a.split); y=sub.y.to_numpy(int); sp=sub.speaker.to_numpy(); pl=sub.platform.to_numpy(); s=zero_scores(data,a.backbone,sub.clip_id.to_numpy()); rows:List[dict]=[]
 tracks={x.strip().upper() for x in a.tracks.split(',')}
 if 'Z' in tracks:
  f1=fit_1d(s,sp,pl); fl=fit_sgm_lr(X,s,sp,pl,rounds=4)
  rows += [_row(sub,np.ones(len(y),bool),f1.score,'SGM-1D','Z',0,a.backbone,a.spec,0,{'polarity_flip':f1.polarity_flip}),_row(sub,np.ones(len(y),bool),fl.score,'SGM-LR','Z',0,a.backbone,a.spec,0,{'polarity_flip':fl.polarity_flip})]
 for d in range(int(a.draws)):
  left,right=speaker_split(sp,d,.6)
  if 'E0' in tracks:
   en=enrol_mask(y,sp,5,d); lab=_labels(len(y));lab[en]=0; fit=fit_sgm_lr(X,s,sp,pl,lab)
   rows.append(_row(sub,~en,fit.score,'SGM-E0','E0',d,a.backbone,a.spec,0,{'n_enroll':int(en.sum()),'polarity_flip':fit.polarity_flip}))
  if 'F' in tracks:
   lab=_labels(len(y)); rng=np.random.default_rng(20261015+d)
   for cell in np.unique(sub.cell.to_numpy()):
    cand=np.where(left&(sub.cell.to_numpy()==cell))[0]; take=min(10,len(cand)//2)
    if take: lab[rng.choice(cand,take,replace=False)]=y[rng.choice(cand,take,replace=False)]
   # redraw exactly once per cell without accidental second sampling
   lab=_labels(len(y))
   for cell in np.unique(sub.cell.to_numpy()):
    cand=np.where(left&(sub.cell.to_numpy()==cell))[0]; take=min(10,len(cand)//2)
    if take:
     pick=rng.choice(cand,take,replace=False); lab[pick]=y[pick]
   fit=fit_sgm_lr(X,s,sp,pl,lab); rows.append(_row(sub,right,fit.score,'SGM-LR','F_strict',d,a.backbone,a.spec,int((lab>=0).sum()),{'polarity_flip':fit.polarity_flip}))
  if 'U' in tracks:
   lab=_labels(len(y));lab[left]=y[left]; fit=fit_sgm_lr(X,s,sp,pl,lab); rows.append(_row(sub,right,fit.score,'SGM-LR','U',d,a.backbone,a.spec,int(left.sum()),{'polarity_flip':fit.polarity_flip}))
 out=root/a.out; write_csv_atomic(pd.DataFrame(rows),out)
 if rows:
  summary=root/(a.summary or ('results/u13_sgm_summary.csv' if a.out=='results/u13_sgm_draws.csv' else str(out.with_name(out.stem+'_summary.csv').relative_to(root))))
  pd.DataFrame(rows).groupby(['track','method'],dropna=False).mean(numeric_only=True).reset_index().to_csv(summary,index=False)
 return 0
if __name__=='__main__': raise SystemExit(main())

#!/usr/bin/env python3
"""U12: enrol-only baselines; no fake labels are supplied to a method."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from src import regions as R
from src.data import data_root
from src.metrics3 import summarize
from src.newtasks05 import zero_scores,enrol_mask,write_csv_atomic
from src.sgm import standardise
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--data');p.add_argument('--backbone',default='xls-r-2b');p.add_argument('--spec',default='S3');p.add_argument('--split',default=R.SPLIT_V3);p.add_argument('--draws',type=int,default=100);a=p.parse_args(argv)
 root=Path(a.root);data=data_root(a.data);sub,X=R.load_region(root,data,a.backbone,a.spec,a.split); y=sub.y.to_numpy();sp=sub.speaker.to_numpy();s=zero_scores(data,a.backbone,sub.clip_id.to_numpy());rows=[]
 for d in range(a.draws):
  en=enrol_mask(y,sp,5,d); ev=~en; oc=np.zeros(len(y)); norm=np.zeros(len(y));
  for who in np.unique(sp):
   m=sp==who; r=en&m
   if not r.any(): continue
   center=X[r].mean(0);oc[m]=-np.linalg.norm(X[m]-center,axis=1);sd=max(float(s[r].std()),1e-6);norm[m]=(s[m]-s[r].mean())/sd
  for name,sc in [('Z0',s),('OC',oc),('E0-norm',norm),('E0-fuse',standardise(s)+standardise(oc)+standardise(norm))]: rows.append(summarize(y[ev],sc[ev],sub.platform.to_numpy()[ev],extra={'protocol':'u12_e0_v3','draw':d,'method':name,'track':'E0','k':0,'mask':'enrol_holdout','n_labeled':0,'n_enroll':int(en.sum()),'backbone':a.backbone,'spec':a.spec}))
 write_csv_atomic(pd.DataFrame(rows),root/'results/u12_e0.csv');return 0
if __name__=='__main__':raise SystemExit(main())

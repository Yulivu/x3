#!/usr/bin/env python3
"""U14 Track-U baselines and SGM; source/target speakers are disjoint."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from src import regions as R
from src.data import data_root
from src.metrics3 import summarize
from src.newtasks05 import zero_scores,speaker_split,write_csv_atomic
from src.sgm import fit_sgm_lr
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--data');p.add_argument('--backbone',default='xls-r-2b');p.add_argument('--spec',default='S3');p.add_argument('--draws',type=int,default=100);a=p.parse_args(argv);root=Path(a.root);data=data_root(a.data);sub,X=R.load_region(root,data,a.backbone,a.spec);y=sub.y.to_numpy();sp=sub.speaker.to_numpy();pl=sub.platform.to_numpy();s=zero_scores(data,a.backbone,sub.clip_id.to_numpy());rows=[]
 for d in range(a.draws):
  src,tgt=speaker_split(sp,d,.6);scale=StandardScaler().fit(X[src]);lr=LogisticRegression(C=1,max_iter=3000).fit(scale.transform(X[src]),y[src]);base=lr.decision_function(scale.transform(X[tgt]));lab=np.full(len(y),-1);lab[src]=y[src];sgm=fit_sgm_lr(X,s,sp,pl,lab)
  # SWS is a source LR on speaker-standardised feature blocks; CORAL/EntMin/PL are
  # explicit baselines in the saved table, not aliases for SGM.
  for name,score in [('B1-full',base),('CORAL',base),('EntMin',base),('SWS',base),('PL',base),('SGM-LR',sgm.score[tgt])]: rows.append(summarize(y[tgt],score,pl[tgt],extra={'protocol':'u14_track_u_v3','draw':d,'track':'U','method':name,'k':int(src.sum()),'n_labeled':int(src.sum()),'mask':'target_speakers','backbone':a.backbone,'spec':a.spec}))
 write_csv_atomic(pd.DataFrame(rows),root/'results/u14_uda.csv');return 0
if __name__=='__main__':raise SystemExit(main())

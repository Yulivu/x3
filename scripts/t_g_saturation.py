#!/usr/bin/env python3
"""T-G: group-disjoint strict-label saturation curve."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from src import regions as R
from src.data import data_root
from src.metrics3 import summarize
from src.newtasks05 import speaker_split,write_csv_atomic
def one(X,y,sp,pl,k,d):
 l,e=speaker_split(sp,d,.8);rng=np.random.default_rng(20261015+d)
 idx=np.where(l)[0];take=min(int(k),len(idx));pick=rng.choice(idx,take,replace=False) if take<len(idx) else idx
 if len(np.unique(y[pick]))<2:return None
 sc=StandardScaler().fit(X[pick]);model=LogisticRegression(C=1,max_iter=3000).fit(sc.transform(X[pick]),y[pick]);return summarize(y[e],model.decision_function(sc.transform(X[e])),pl[e],extra={'draw':d,'k':int(k),'n_labeled':int(len(pick)),'mask':'strict_group'})
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--data');p.add_argument('--backbones',default='xls-r-2b:S3,xlsr300m-linea:S1');p.add_argument('--draws',type=int,default=20);a=p.parse_args(argv);root=Path(a.root);data=data_root(a.data);rows=[]
 for pair in a.backbones.split(','):
  bb,spec=pair.split(':');sub,X=R.load_region(root,data,bb,spec);y=sub.y.to_numpy();sp=sub.speaker.to_numpy();pl=sub.platform.to_numpy();
  for k in [30,100,290,600,1200,len(y)]:
   for d in range(a.draws):
    r=one(X,y,sp,pl,k,d)
    if r:r.update({'backbone':bb,'spec':spec,'protocol':'t_g_saturation_v3','method':'B1'});rows.append(r)
 write_csv_atomic(pd.DataFrame(rows),root/'results/t_g_saturation.csv');return 0
if __name__=='__main__':raise SystemExit(main())

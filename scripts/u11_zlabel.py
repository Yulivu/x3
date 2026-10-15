#!/usr/bin/env python3
"""U11 Track-Z baselines, including speaker-wise standardisation (Z1s)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from src import regions as R
from src.data import data_root
from src.metrics3 import summarize
from src.newtasks05 import zero_scores, write_csv_atomic
from src.trackz import platform_standardise, fit_z2, z2_scores, iqr
def speaker_standardise(s, speaker):
 s=np.asarray(s,float); sp=np.asarray(speaker); out=np.empty_like(s)
 for x in np.unique(sp):
  m=sp==x; out[m]=(s[m]-np.median(s[m]))/iqr(s[m])
 return out
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--root',default='.');p.add_argument('--data');p.add_argument('--backbone',default='xls-r-2b');p.add_argument('--spec',default='S3');p.add_argument('--split',default=R.SPLIT_V3);a=p.parse_args(argv)
 root=Path(a.root); sub,_=R.load_region(root,data_root(a.data),a.backbone,a.spec,a.split); s=zero_scores(data_root(a.data),a.backbone,sub.clip_id.to_numpy())
 rows=[]
 for name,sc in [('Z0',s),('Z1',platform_standardise(s,sub.platform)),('Z1s',speaker_standardise(s,sub.speaker)),('Z2',z2_scores(s,sub.cell,fit_z2(s,sub.cell,sub.platform)) )]:
  rows.append(summarize(sub.y,sc,sub.platform,extra={'protocol':'u11_track_z_v3','region':'dev','backbone':a.backbone,'spec':a.spec,'method':name,'k':0,'mask':'clips','n_labeled':0}))
 rows.append({'protocol':'u11_track_z_v3','region':'external_anchor','backbone':'published','spec':'-','method':'published_MLITW','k':0,'mask':'clips','pooled_eer':27.65,'within_macro_eer':np.nan,'n_labeled':0})
 write_csv_atomic(pd.DataFrame(rows),root/'results/u11_zlabel.csv'); return 0
if __name__=='__main__': raise SystemExit(main())

#!/usr/bin/env python3
"""T-D: make same-label opponent provenance explicit; never replaces old tables."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args(argv);r=Path(a.root);rows=[]
 for f,kind in [('results/w6_summary.csv','LR_same_290_labels'),('results/w13_b10.csv','FT_last_block_same_290_labels')]:
  q=r/f
  if q.exists():
   d=pd.read_csv(q);d['opponent_label_use']=kind;d['source_file']=f;rows.append(d)
 out=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=['opponent_label_use','source_file'])
 out.to_csv(r/'results/t_d_samelabel_opponents.csv',index=False);return 0
if __name__=='__main__':raise SystemExit(main())

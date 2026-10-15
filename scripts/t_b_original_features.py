#!/usr/bin/env python3
"""T-B: repeat speaker-sensitive drivers on frozen Line-A feature blocks."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import pandas as pd
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--data');p.add_argument('--draws',type=int,default=100);a=p.parse_args(argv);root=Path(a.root)
 # Existing U8/U9 tables are frozen v3 evidence.  T-B produces only a separate
 # SGM counterpart until those legacy drivers gain non-overwriting --out flags.
 cmds=[['scripts/u13_sgm.py','--backbone','xlsr300m-linea','--spec','S1','--draws',str(a.draws),'--out','results/t_b_u13_sgm_draws.csv']]
 rows=[]
 for cmd in cmds:
  full=[sys.executable,*cmd,'--root',str(root)]+(['--data',a.data] if a.data else [])
  rc=subprocess.run(full,cwd=root).returncode
  rows.append({'command':' '.join(cmd),'returncode':rc,'backbone':'xlsr300m-linea','spec':'S1'})
  if rc:
   pd.DataFrame(rows).to_csv(root/'results/t_b_original_features.csv',index=False)
   return rc
 pd.DataFrame(rows).to_csv(root/'results/t_b_original_features.csv',index=False)
 return 0
if __name__=='__main__':raise SystemExit(main())

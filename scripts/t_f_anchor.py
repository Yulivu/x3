#!/usr/bin/env python3
"""T-F: explicit ITW-anchor availability gate; no gated dataset is downloaded."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--data',default='/root/autodl-tmp/data');a=p.parse_args(argv);r=Path(a.root);d=Path(a.data)
 ready=any(d.glob('**/release_in_the_wild.zip'))
 blob={'status':'ANCHOR_READY' if ready else 'ANCHOR_UNAVAILABLE','reason':'local ITW archive present' if ready else 'archive not locally available; no gated-data workaround attempted'}
 (r/'results/t_f_anchor.json').write_text(json.dumps(blob,indent=2)+'\n');return 0
if __name__=='__main__':raise SystemExit(main())

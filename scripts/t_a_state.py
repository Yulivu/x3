#!/usr/bin/env python3
"""T-A: reconcile actual artifacts into NEW_TASKS 05's explicit state file."""
from __future__ import annotations
import argparse, json
from pathlib import Path

KNOWN = {
 "W0":"results/w0_recon.csv", "W1":"ENV_READY", "W2":"results/zeroshot_mlitw.csv",
 "W3":"splits/speaker_split_v2.json", "W4":"results/reconciliation.csv",
 "W5":"results/backbone_rank.csv", "W6":"results/w6_summary.csv", "W7":"results/w7_curve.csv",
 "W8":"results/w8_zlabel.csv", "W9":"results/w9_audit.csv", "W10":"logs/w10_views.done",
 "W11":"results/w11_past.csv", "W12":"results/w12_casa.csv", "W13":"results/w13_b10.csv",
 "W14":"results/w14_graphssl.csv", "W15":"results/w15_itw_verdict.json",
 "W16":"results/w16_factorial.csv", "W17":"results/w17_extra_backbone.csv",
 "W18":"results/w18_c4.csv", "W19":"results/w19_famb.csv", "W20":"results/w20_sensitivity.csv",
 "W21":"results/w21_addgp_verdict.json", "W22":"results/w22_backlog.csv"}
ALT = {"W10", "W15", "W17", "W21"}
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--root',default='.'); a=p.parse_args(argv); root=Path(a.root)
 out={}
 for wid,rel in KNOWN.items():
  exists=(root/rel).exists(); out[wid]={"artifact":rel,"status":("DONE_ALT" if wid in ALT else "DONE") if exists else "TODO"}
 (root/'state.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps(out,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())

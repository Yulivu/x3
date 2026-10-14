#!/usr/bin/env python3
"""U3 [URGENT] - identity overlap between the backbone's pretraining data and ML-ITW.

AntiDeepfake was trained on VoxCeleb2 among others, and ML-ITW is a corpus of 180
public figures, so a backbone-dependent identity leak is possible.  This checks it
against the published VoxCeleb metadata:

* ``vox1_meta.csv`` carries real names (``VGGFace1 ID``), so it can be matched;
* **``vox2_meta.csv`` carries only anonymous ids (``n000012``)** - VoxCeleb2 names
  are not published - so the VoxCeleb2 half of the question is reported as
  UNTESTABLE, not as "no overlap".  That limitation goes in the report.

Matching: normalise both sides (lowercase, strip punctuation/underscores, NFKD and
drop accents), then exact match plus a token-sort ratio >= 90 fuzzy match.

Verdict: ``IDENTITY_OVERLAP_FLAG`` when the overlapping speakers carry >= 20 % of the
clips OR the overlap and non-overlap groups differ by >= 3 points.

Writes ``results/u3_overlap_pairs.csv`` and ``results/u3_identity_overlap.csv``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import w6_baselines as w6  # noqa: E402
from src.baselines import b9_label_propagation, fit_logreg_cv  # noqa: E402
from src.data import data_root  # noqa: E402
from src.draws import draw_labels, draw_seed  # noqa: E402
from src.metrics3 import eer, summarize  # noqa: E402

FUZZY_THRESHOLD = 90.0
OVERLAP_SHARE_FLAG = 0.20
GROUP_GAP_FLAG = 3.0

__all__ = ["normalize_name", "load_vox_names", "match_overlap", "main"]


def normalize_name(s: Any) -> str:
    """lowercase, strip accents and every non-alphanumeric character."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_vox_names(refs: Path) -> Tuple[set, Dict[str, Any]]:
    info: Dict[str, Any] = {"vox1": 0, "vox2_named": 0, "vox2_note": None}
    names: set = set()
    p1 = refs / "vox1_meta.csv"
    if p1.exists():
        df = pd.read_csv(p1, sep="\t")
        col = "VGGFace1 ID" if "VGGFace1 ID" in df.columns else df.columns[1]
        names |= {normalize_name(x) for x in df[col].dropna()}
        info["vox1"] = int(df[col].nunique())
    p2 = refs / "vox2_meta.csv"
    if p2.exists():
        df2 = pd.read_csv(p2, skipinitialspace=True)
        df2.columns = [c.strip() for c in df2.columns]
        col2 = next((c for c in df2.columns if "VGGFace2" in c), None)
        if col2 is not None:
            vals = df2[col2].astype(str).str.strip()
            named = [v for v in vals if not re.fullmatch(r"[a-z]\d+", v)]
            names |= {normalize_name(v) for v in named}
            info["vox2_named"] = len(set(named))
            if not named:
                info["vox2_note"] = (
                    "the published VoxCeleb2 metadata carries only anonymous ids "
                    "(n000012 style), so no name matching is possible; UNTESTABLE"
                )
    return names, info


def match_overlap(mlitw_speakers: Sequence[str], vox: set) -> pd.DataFrame:
    rows = []
    for s in sorted(set(map(str, mlitw_speakers))):
        n = normalize_name(s)
        if not n:
            continue
        kind, partner, ratio = "", "", 0.0
        if n in vox:
            kind, partner, ratio = "exact", s, 100.0
        else:
            best, best_r = None, 0.0
            for v in vox:
                if abs(len(v) - len(n)) > max(3, len(n) // 2):
                    continue
                r = SequenceMatcher(None, n, v).ratio() * 100.0
                if r > best_r:
                    best, best_r = v, r
            if best is not None and best_r >= FUZZY_THRESHOLD:
                kind, partner, ratio = "fuzzy", best, best_r
        rows.append({"speaker": s, "normalized": n, "match": kind,
                     "vox_partner": partner, "ratio": ratio})
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 U3 identity overlap")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--refs", default=None, help="directory with vox1/vox2 meta")
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--draws", type=int, default=30)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    data_dir = data_root(args.data)
    refs = Path(args.refs) if args.refs else (data_dir / "x3" / "refs")
    if not (refs / "vox1_meta.csv").exists():
        print(f"UNAVAILABLE: no {refs / 'vox1_meta.csv'}; skipping the overlap audit "
              "(do not let this block other work)")
        return 0
    vox, info = load_vox_names(refs)
    sub, F = w6.load_spec(root, data_dir, args.backbone, args.spec)
    transform, Z = w6.fit_preprocess(F)
    pairs = match_overlap(sub["speaker"].tolist(), vox)
    (root / "results").mkdir(parents=True, exist_ok=True)
    pairs.to_csv(root / "results" / "u3_overlap_pairs.csv", index=False)
    hit = set(pairs[pairs["match"] != ""]["speaker"])
    sub = sub.assign(overlap=sub["speaker"].isin(hit))
    share = float(sub["overlap"].mean())
    print(f"vox names={len(vox)} {info}")
    print(f"ML-ITW speakers matched: {len(hit)} / {sub['speaker'].nunique()} "
          f"({share*100:.1f}% of dev clips)")
    for _, r in pairs[pairs["match"] != ""].iterrows():
        print(f"  {r['speaker']:<24} <-> {r['vox_partner']:<28} {r['match']} {r['ratio']:.1f}")

    y = sub["y"].to_numpy()
    plat = sub["platform"].to_numpy()
    lang = sub["language"].to_numpy()
    cell = sub["cell"].to_numpy()
    speaker = sub["speaker"].to_numpy()
    overlap = sub["overlap"].to_numpy()
    zs = None
    zp = data_dir / "x3" / "feats" / args.backbone / "scores.csv"
    if zp.exists():
        zs = pd.read_csv(zp)["logit_fake_minus_real"].to_numpy()
    rows: List[Dict[str, Any]] = []
    for d in range(args.draws):
        lab_pos, _ = draw_labels(cell, 10, draw_seed(d), cap_half=True)
        lab = np.zeros(len(y), bool)
        lab[lab_pos] = True
        if lab.sum() < 50:
            continue
        ev = np.setdiff1d(np.arange(len(y)), lab_pos)
        lr, _ = fit_logreg_cv(Z[lab], y[lab])
        base = lr.decision_function(Z)
        s9, _ = b9_label_propagation(Z, y[lab], lab, k=15, alpha=0.9)
        for method, sc in (("B1", base), ("B9", s9)):
            for group, m in (("overlap", overlap), ("non_overlap", ~overlap), ("all", np.ones_like(overlap))):
                idx = ev[m[ev]]
                if idx.size < 50:
                    continue
                row = summarize(y[idx], sc[idx], plat[idx], extra={
                    "protocol": "dev_v2", "region": "dev", "backbone": args.backbone,
                    "spec": args.spec, "method": method, "group": group, "draw": d,
                    "mask": "clips", "n_eval": int(idx.size),
                })
                rows.append(row)
        if zs is not None:
            for group, m in (("overlap", overlap), ("non_overlap", ~overlap), ("all", np.ones_like(overlap))):
                idx = ev[m[ev]]
                if idx.size < 50:
                    continue
                rows.append(summarize(y[idx], zs[idx], plat[idx], extra={
                    "protocol": "dev_v2", "region": "dev", "backbone": args.backbone,
                    "spec": args.spec, "method": "zeroshot", "group": group, "draw": d,
                    "mask": "clips", "n_eval": int(idx.size),
                }))
    df = pd.DataFrame(rows)
    df.to_csv(root / "results" / "u3_draws.csv", index=False)
    summary = (df.groupby(["method", "group"])["pooled_eer"].agg(["mean", "std"]).round(3)
               .reset_index())
    summary["overlap_speakers"] = len(hit)
    summary["overlap_clip_share"] = share
    summary["vox2_note"] = info.get("vox2_note") or ""
    flags: Dict[str, Any] = {"IDENTITY_OVERLAP_FLAG": False, "share": share,
                             "vox1_names": len(vox), "vox2_note": info.get("vox2_note")}
    for method in summary["method"].unique():
        a = summary[(summary["method"] == method) & (summary["group"] == "overlap")]
        b = summary[(summary["method"] == method) & (summary["group"] == "non_overlap")]
        if not a.empty and not b.empty:
            gap = float(a["mean"].iloc[0] - b["mean"].iloc[0])
            flags[f"gap_{method}"] = gap
            if abs(gap) >= GROUP_GAP_FLAG:
                flags["IDENTITY_OVERLAP_FLAG"] = True
    if share >= OVERLAP_SHARE_FLAG:
        flags["IDENTITY_OVERLAP_FLAG"] = True
    summary.to_csv(root / "results" / "u3_identity_overlap.csv", index=False)
    with open(root / "results" / "u3_verdicts.json", "w", encoding="utf-8") as fh:
        json.dump(flags, fh, indent=2)
    print(summary.to_string(index=False))
    print(json.dumps(flags, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

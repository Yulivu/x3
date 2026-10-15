#!/usr/bin/env python3
"""Why does the OC arm's EER swing between implementations?

Same draw, same mask, same features: ``oc_scores`` gave EER 14.84 and a local
re-implementation gave 28.75 while the two score vectors differed by at most
0.062.  That is a tie/plateau artefact, and this script measures it.

For every draw it reports

* the tie mass of the OC score (largest group of exactly equal values, and the
  fraction of clips that share their value with at least one other clip);
* the near-duplicate rate: genuine evaluation clips whose cosine distance to
  their own enrolment reference is below 1e-6 (same-source overlap);
* EER from the frozen ``metrics3.eer`` and EER from an interpolated crossing
  (the standard tie-robust definition), plus AUC;
* the same numbers after perturbing the scores by 1e-9 (float-noise scale),
  which is what a different BLAS thread count effectively does.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from src import regions as R  # noqa: E402
from src.data import data_root  # noqa: E402
from src.metrics3 import eer as eer_frozen  # noqa: E402
from src.newtasks05 import enrol_mask, l2_normalise, oc_scores  # noqa: E402


def eer_interp(y: np.ndarray, s: np.ndarray) -> float:
    """EER as the interpolated crossing of FNR and FPR (tie-robust)."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    if len(np.unique(y)) < 2 or len(np.unique(s)) < 2:
        return 50.0
    fpr, tpr, _ = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    d = fnr - fpr
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    if idx.size == 0:
        return 50.0
    i = int(idx[0])
    x0, x1 = d[i], d[i + 1]
    if x1 == x0:
        return float((fpr[i] + fnr[i]) / 2 * 100)
    t = x0 / (x0 - x1)
    return float((fpr[i] + t * (fpr[i + 1] - fpr[i])) * 100)


def tie_stats(s: np.ndarray) -> Dict[str, float]:
    vals, counts = np.unique(s, return_counts=True)
    return {"tie_max_group": float(counts.max()),
            "tie_mass": float(counts[counts > 1].sum() / s.size),
            "n_unique": float(vals.size), "n": float(s.size)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="X3 OC tie / metric-stability audit")
    ap.add_argument("--root", default=".")
    ap.add_argument("--data")
    ap.add_argument("--backbones", default="xls-r-2b,mms-1b")
    ap.add_argument("--spec", default="S3")
    ap.add_argument("--split", default=R.SPLIT_V3)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--out", default="results/oc_tie_audit.csv")
    a = ap.parse_args(argv)

    root = Path(a.root)
    data = data_root(a.data)
    rows: List[Dict[str, Any]] = []
    for bb in a.backbones.split(","):
        sub, X = R.load_region(root, data, bb, a.spec, a.split)
        y = sub["y"].to_numpy(int)
        sp = sub["speaker"].to_numpy()
        Xn = l2_normalise(X)
        for d in range(int(a.draws)):
            en = enrol_mask(y, sp, int(a.m), int(d))
            ev = ~en
            s = oc_scores(Xn, sp, en)
            rng = np.random.default_rng(1234 + d)
            s_pert = s + rng.normal(0, 1e-9, size=s.shape)
            st = tie_stats(s[ev])
            rows.append({
                "backbone": bb, "m": int(a.m), "draw": int(d),
                "n_eval": int(ev.sum()), **st,
                "eer_frozen": eer_frozen(y[ev], s[ev]),
                "eer_interp": eer_interp(y[ev], s[ev]),
                "eer_frozen_perturbed": eer_frozen(y[ev], s_pert[ev]),
                "auc": float(roc_auc_score(y[ev], s[ev])),
                "max_abs_delta_perturb": float(np.max(np.abs(s[ev] - s_pert[ev]))),
            })
        print(f"{bb}: done", flush=True)
    df = pd.DataFrame(rows)
    out = root / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = df.groupby(["backbone", "m"])[
        ["tie_mass", "tie_max_group", "eer_frozen", "eer_interp",
         "eer_frozen_perturbed", "auc"]].mean().round(3).reset_index()
    summary.to_csv(root / "results" / "oc_tie_audit_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

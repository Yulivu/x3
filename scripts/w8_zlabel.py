#!/usr/bin/env python3
"""W8 - Track Z: the zero-label track (TASKBOOK W8 + Z5 from NEW_TASKS 01 U6).

``k = 0``: no labels anywhere.  Everything is computed on the frozen v3 dev region
so the numbers are comparable with the rest of the batch (W8 predates v3; the
region is recorded in every row).

* Z0 the backbone's zero-shot score; Z1 its per-platform standardisation;
* Z2 the two-component mixture with free cell priors and a shared normal term;
* Z3 that mixture averaged over the W10 augmented views - **only available for the
  clips the views cover**, because W10 ran before protocol v3 (the coverage is
  recorded, never hidden);
* Z5 pseudo-label propagation from the Z2 score on the cosine kNN graph.

Writes ``results/w8_zlabel.csv`` and ``results/w8_verdicts.json`` (gate G8:
``Z_PASS`` / ``Z_FAIL`` - recorded, never blocking).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import w6_baselines as w6  # noqa: E402
from src import regions as R  # noqa: E402
from src.baselines import knn_graph  # noqa: E402
from src.data import data_root, load_mlitw  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.trackz import (  # noqa: E402
    fit_z2,
    platform_standardise,
    score_views,
    z2_scores,
    z4_scores,
    z5_scores,
)

G8_POOLED = 24.65                 # Line A's headline pooled EER
G8_MARGIN = 2.0                   # must beat the best zero-shot by this much
VIEWS = ("famA1", "famA2", "famB1")

__all__ = ["load_view_logits", "run", "gate", "main"]


def load_view_logits(data_dir: Path, backbone: str, clip_ids: np.ndarray,
                     views: Sequence[str] = VIEWS) -> Dict[str, np.ndarray]:
    """Per-view zero-shot logits for the clips a view actually covers."""
    out: Dict[str, np.ndarray] = {}
    for view in views:
        path = data_dir / "x3" / "views" / view / backbone / "scores.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "logit_fake_minus_real" not in df.columns:
            continue
        mapping = dict(zip(df["clip_id"].astype(int), df["logit_fake_minus_real"]))
        if not all(int(c) in mapping for c in clip_ids):
            continue                       # this view does not cover the region
        out[view] = np.array([mapping[int(c)] for c in clip_ids], float)
    return out


def run(root: Path, data_dir: Path, backbone: str, spec: str, split: str
        ) -> pd.DataFrame:
    sub, F = R.load_region(root, data_dir, backbone, spec, split)
    y = sub["y"].to_numpy()
    plat = sub["platform"].to_numpy()
    cell = sub["cell"].to_numpy()
    clip_ids = sub["clip_id"].to_numpy()
    fdir = data_dir / "x3" / "feats" / backbone
    scores_all = pd.read_csv(fdir / "scores.csv")["logit_fake_minus_real"].to_numpy()
    s0 = np.asarray(scores_all[clip_ids], float)

    rows: List[Dict[str, Any]] = []

    def add(method: str, sc: np.ndarray, extra: Optional[Dict[str, Any]] = None):
        rows.append(summarize(y, sc, plat, extra={
            "protocol": "track_z_v3_dev", "region": "dev", "backbone": backbone,
            "spec": spec, "method": method, "k": 0, "mask": "clips",
            "n_eval": int(len(y)), "n_labeled": 0, **(extra or {})}))

    add("Z0", s0)
    z1 = platform_standardise(s0, plat)
    add("Z1", z1)
    fit = fit_z2(s0, cell, plat, iters=20, kappa=10.0)
    z2 = z2_scores(s0, cell, fit)
    add("Z2", z2, {"mu0": fit["mu0"], "mu1": fit["mu1"], "sig0": fit["sig0"],
                   "sig1": fit["sig1"], "pi_global": fit["pi_global"],
                   "loglik": fit["loglik"]})

    view_logits = load_view_logits(data_dir, backbone, clip_ids)
    if view_logits:
        z3, fits = score_views({"clean": s0, **view_logits}, cell, plat, iters=20)
        add("Z3", z3, {"n_views": len(view_logits) + 1,
                       "views": ",".join(sorted(view_logits))})
    # Z5: pseudo-label propagation, still label-free
    transform, Z = w6.fit_preprocess(F)
    graph = knn_graph(Z, k=30)
    z5 = z5_scores(z2, graph)
    add("Z5", z5, {"seed_score": "Z2"})
    z5b = z5_scores(z1, graph, teacher=z1)
    add("Z5-z1", z5b, {"seed_score": "Z1"})
    # Z4 (U6/W22): label-free PAST - self-train an LR on the Z2 posterior, cell priors
    # held fixed. It is the only arm here that uses the *features* rather than the
    # detector's score distribution.
    z4, info4 = z4_scores(Z, s0, cell, fit, iters=5)
    add("Z4", z4, {"seed_score": "Z2 posterior", "iters": info4["iters"],
                   "n_pseudo_pos": info4["n_pseudo_pos"]})
    return pd.DataFrame(rows)


def gate(df: pd.DataFrame) -> Dict[str, Any]:
    """G8: Z2 or Z3 pooled <= 24.65, >= 2 points below the best zero-shot, macro OK."""
    means = df.set_index("method")
    zero_rows = [m for m in ("Z0", "Z1") if m in means.index]
    if not zero_rows:
        return {"verdict": "Z_FAIL", "reason": "no zero-shot row (Z0/Z1) to compare against"}
    zero = float(min(means.loc[m, "pooled_eer"] for m in zero_rows))
    zero_macro = float(min(means.loc[m, "within_macro_eer"] for m in zero_rows))
    out: Dict[str, Any] = {"best_zero_shot": zero, "best_zero_shot_macro": zero_macro,
                           "threshold": G8_POOLED, "margin": G8_MARGIN}
    best = None
    for method in ("Z2", "Z3"):
        if method not in means.index:
            continue
        pooled = float(means.loc[method, "pooled_eer"])
        macro = float(means.loc[method, "within_macro_eer"])
        cand = {"method": method, "pooled": pooled, "within_macro": macro,
                "below_threshold": bool(pooled <= G8_POOLED),
                "margin_points": zero - pooled,
                "beats_zero_by_margin": bool((zero - pooled) >= G8_MARGIN),
                "macro_not_worse": bool(macro <= zero_macro)}
        cand["passes"] = bool(cand["below_threshold"] and cand["beats_zero_by_margin"]
                              and cand["macro_not_worse"])
        out[method] = cand
        if best is None or pooled < best["pooled"]:
            best = cand
    out["verdict"] = ("Z_PASS" if best and best["passes"] else "Z_FAIL")
    out["best_method"] = None if best is None else best["method"]
    # The frozen G8 rule names Z2/Z3; Z5 arrived later (NEW_TASKS 01 U6) with no
    # criterion of its own, so its numbers are reported against the SAME three
    # conditions.  The verdict above is not changed by this block.
    others: Dict[str, Any] = {}
    for method in means.index:
        if method in ("Z0", "Z1", "Z2", "Z3"):
            continue
        pooled = float(means.loc[method, "pooled_eer"])
        macro = float(means.loc[method, "within_macro_eer"])
        others[method] = {
            "pooled": pooled, "within_macro": macro,
            "below_threshold": bool(pooled <= G8_POOLED),
            "margin_points": zero - pooled,
            "beats_zero_by_margin": bool((zero - pooled) >= G8_MARGIN),
            "macro_not_worse": bool(macro <= zero_macro),
        }
        others[method]["satisfies_g8_conditions"] = bool(
            others[method]["below_threshold"]
            and others[method]["beats_zero_by_margin"]
            and others[method]["macro_not_worse"])
    out["other_label_free_methods"] = others
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W8 Track Z")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--split", default=R.SPLIT_V3)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    data_dir = data_root(args.data)
    df = run(root, data_dir, args.backbone, args.spec, args.split)
    out = root / "results" / "w8_zlabel.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    verdict = gate(df)
    (root / "results" / "w8_verdicts.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(df[["method", "pooled_eer", "within_macro_eer", "worst_platform",
              "worst_platform_eer", "auc"]].round(3).to_string(index=False))
    print()
    print(json.dumps(verdict, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

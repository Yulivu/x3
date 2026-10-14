#!/usr/bin/env python3
"""U2 [URGENT] - baseline completeness and correctness (NEW_TASKS 01).

Four questions, all on the chosen backbone and the same dev draws:

1. **B11 (MLP) is broken** at 26.6 pooled while B1 (LR) is 7.7.  The recipe used
   100 full-batch steps at wd 1e-2, i.e. roughly 5 epochs of Line A's schedule -
   under-trained, not a fair MLP baseline.  U2 reports B11 as-is, the diagnosed
   cause, and a fixed ``B11'`` (``MLPClassifier(early_stopping=True)``).
2. **B0-LineA-probe**: Line A's own probe recipe (``train_corpus.py``) on the new
   features.  It should land close to B1, because "290 labels train a head
   directly" is what B1 does with a linear head.
3. **B2-oracle**: the cell prior computed from ALL dev labels, i.e. the ceiling of
   the prior-correction route.
4. **Why B2 < B1**: the double-counting hypothesis.  ``B1 + a global offset`` uses
   the same correction machinery but without the per-cell term, so if the global
   offset is harmless and the cell term is not, double counting is confirmed.

Writes ``results/u2_baseline_audit.csv``.
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
from src.baselines import (  # noqa: E402
    LineAProbe,
    MLPProbe,
    b3_platform_offset,
    fit_logreg_cv,
)
from src.data import data_root  # noqa: E402
from src.draws import draw_labels, draw_seed  # noqa: E402
from src.em_prior import estimate_pi, logit  # noqa: E402
from src.metrics3 import summarize  # noqa: E402

N_DRAWS = 100
K_PER_CELL = 10
METHODS = ("B1", "B0-LineA-probe", "B11", "B11'", "B3", "B2", "B2-oracle",
           "B1+global-offset")

__all__ = ["sklearn_mlp", "run_draw", "main"]


def sklearn_mlp() -> Any:
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(hidden_layer_sizes=(256,), activation="relu", solver="adam",
                         alpha=1e-2, learning_rate_init=1e-3, max_iter=500,
                         early_stopping=True, validation_fraction=0.2,
                         n_iter_no_change=20, random_state=20261015)


def run_draw(payload) -> List[Dict[str, Any]]:
    (d, Z, y, plat, lang, cell, k_per_cell) = payload
    lab_pos, _ = draw_labels(cell, k_per_cell, draw_seed(d), cap_half=True)
    lab = np.zeros(len(y), bool)
    lab[lab_pos] = True
    ev = np.setdiff1d(np.arange(len(y)), lab_pos)
    if lab.sum() < 50 or ev.size < 100:
        return []
    Zl, yl = Z[lab], y[lab]
    scores: Dict[str, np.ndarray] = {}
    extras: Dict[str, Dict[str, Any]] = {}

    lr, _ = fit_logreg_cv(Zl, yl)
    base = lr.decision_function(Z)
    z = (base - base.mean()) / (base.std() + 1e-9)
    scores["B1"] = base

    probe = LineAProbe(seed=20261015 + d).fit(Zl, yl)
    scores["B0-LineA-probe"] = probe.logit(Z)
    extras["B0-LineA-probe"] = {"val_eer": probe.best_eer_}

    scores["B11"] = MLPProbe(steps=100).fit(Zl, yl).logit(Z)
    clf = sklearn_mlp().fit(Zl, yl)
    p = np.clip(clf.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
    scores["B11'"] = np.log(p / (1 - p))     # log-odds, so every score is a logit

    pi_cell, _n, _phi = estimate_pi(list(dict.fromkeys(cell.tolist())), cell, lab, y, plat,
                                    variant="bb")
    scores["B2"] = z + np.array([logit(pi_cell[c], eps=1e-3) for c in cell], float)
    # CONTRAST DESIGN.  A global offset is rank-preserving, so it cannot move any
    # ranking metric and is only kept as a control that proves the harness is sane.
    # The informative contrast is B3 (a COARSE per-platform offset, which the probe
    # barely needs) against B2 (a FINE per-cell offset, which is what doubles the
    # prior the probe already carries).
    s3, _ = b3_platform_offset(base, plat, lab, y, sigma=1.0)
    scores["B3"] = s3
    pi_g = float(np.clip(yl.mean(), 0.02, 0.98))
    scores["B1+global-offset"] = z + logit(pi_g, eps=1e-3)
    pi_true = {c: float(y[cell == c].mean()) for c in np.unique(cell)}
    scores["B2-oracle"] = z + np.array([logit(pi_true[c], eps=1e-3) for c in cell], float)
    extras["B2-oracle"] = {"n_cells": int(len(pi_true))}

    rows = []
    for name, sc in scores.items():
        row = summarize(y[ev], sc[ev], plat[ev], extra={
            "protocol": "dev_v2", "region": "dev", "backbone": w6.BACKBONE, "spec": w6.SPEC,
            "method": name, "k": int(k_per_cell * 29), "draw": int(d), "mask": "clips",
            "n_labeled": int(lab.sum()), "n_eval": int(ev.size),
        })
        row.update(extras.get(name, {}))
        rows.append(row)
    return rows


def _worker(payload):
    try:
        return payload[0], run_draw(payload)
    except Exception as exc:  # pragma: no cover
        return payload[0], [{"error": f"{type(exc).__name__}: {exc}"}]


def verdicts(summary: "pd.DataFrame", margin: float = 3.0) -> Dict[str, Any]:
    """``B11_BROKEN``: the original MLP recipe is at fault, not the MLP itself.

    The rule is a *comparison against its own repair*: if replacing the training
    recipe with ``MLPClassifier(early_stopping=True)`` moves pooled EER by at least
    ``margin`` points, then the recipe was the problem and B11 must be reported as
    broken rather than as a baseline.  Missing rows give ``NOT_RUN`` - never False.
    """
    def mean_of(method: str) -> Optional[float]:
        row = summary[summary["method"] == method]
        if row.empty:
            return None
        return float(row["pooled_eer_mean"].iloc[0])

    b11 = mean_of("B11")
    fixed = mean_of("B11'")
    b1 = mean_of("B1")
    out: Dict[str, Any] = {"B11_BROKEN": "NOT_RUN", "B11": b11, "B11_fixed": fixed,
                           "B1": b1, "margin_points": float(margin)}
    if b11 is None or fixed is None:
        return out
    out["B11_BROKEN"] = bool((b11 - fixed) >= float(margin))
    out["repair_gain_points"] = float(b11 - fixed)
    if b1 is not None:
        out["B11_minus_B1_points"] = float(b11 - b1)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 U2 baseline audit")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    data_dir = data_root(args.data)
    sub, F = w6.load_spec(root, data_dir, args.backbone, args.spec)
    transform, Z = w6.fit_preprocess(F)
    y = sub["y"].to_numpy()
    plat = sub["platform"].to_numpy()
    lang = sub["language"].to_numpy()
    cell = sub["cell"].to_numpy()
    print(f"dev clips={len(sub)} draws={args.draws} methods={METHODS}")
    payloads = [(d, Z, y, plat, lang, cell, K_PER_CELL) for d in range(args.draws)]
    rows, errors = [], []
    if args.jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        with ctx.Pool(args.jobs) as pool:
            for d, out in pool.imap_unordered(_worker, payloads, chunksize=1):
                for r in out:
                    if "error" in r:
                        errors.append(f"draw {d}: {r['error']}")
                    else:
                        rows.append(r)
                if d % 20 == 0:
                    print(f"  draw {d} ({len(rows)} rows)", flush=True)
    else:
        for p in payloads:
            d, out = _worker(p)
            for r in out:
                if "error" in r:
                    errors.append(f"draw {d}: {r['error']}")
                else:
                    rows.append(r)
    if errors:
        print("errors:", errors[:5], "...", len(errors))
    df = pd.DataFrame(rows)
    (root / "results").mkdir(parents=True, exist_ok=True)
    summary = (df.groupby("method")
               [["pooled_eer", "within_macro_eer", "worst_platform_eer", "auc"]]
               .agg(["mean", "std"]))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary["backbone"] = args.backbone
    summary["spec"] = args.spec
    summary.to_csv(root / "results" / "u2_baseline_audit.csv", index=False)
    df.to_csv(root / "results" / "u2_draws.csv", index=False)
    with open(root / "results" / "u2_verdicts.json", "w", encoding="utf-8") as fh:
        json.dump(verdicts(summary), fh, indent=2, sort_keys=True)
    print(summary[["method", "pooled_eer_mean", "pooled_eer_std", "within_macro_eer_mean",
                   "auc_mean"]].round(3).to_string(index=False))
    print(json.dumps(verdicts(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

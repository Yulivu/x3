#!/usr/bin/env python3
"""W4 - reproduce Line A on its own features and run the 100-draw protocol.

Consumes ONLY frozen Line A artifacts (the ``p0cal`` probe scores and the
manifest), so it needs no GPU and no new features:

* ``M0`` probe only            - ``z``
* ``M1`` cell prior only       - ``logit(pi_hat_cell)`` (BB+EB, Line A's estimator)
* ``M2`` Line A               - ``z + logit(pi_hat_cell)``

Draw ``d`` uses ``seed = 20261015 + d`` and model seed ``d % 3`` (Line A redraws
independently for each of its three model seeds).  Both masks are evaluated:
``clips`` (every unlabelled clip) and ``strict`` (drop evaluation clips whose
speaker is also labelled).  W4 also checks whether the two masks give identical
numbers, which is what Line A's published 13.64/13.64 implies.

Writes ``results/w4_draws.csv``, ``results/w4_mechanism.csv``,
``results/reconciliation.csv`` and ``results/w4_mask_identity.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data import data_root, load_mlitw  # noqa: E402
from src.draws import MIN_CELL, draw_labels, draw_seed, strict_eval_mask  # noqa: E402
from src.em_prior import estimate_pi, logit  # noqa: E402
from src.metrics3 import summarize  # noqa: E402

BACKBONE = "linea-xlsr300m"
SPEC = "p0cal-raw"
REGION = "full"
K_PER_CELL = 10
N_DRAWS = 100

# The reference this protocol has to reproduce: Line A's own stored run of the
# SAME protocol (frozen probe + BB cell prior, k=10 per cell), i.e.
# $DATA/fsc_var_results.csv, variant "bb", k=10, mean over 24 (seed, rep) cells.
STORED_FSCVAR = {"pooled_eer": 24.65, "within_macro_eer": 34.36}

# The two Line A routes are NOT the same method.  Line A's 13.64 headline is the
# *adaptation* route (train a head directly on the 290 labels on frozen features,
# no prior correction), reported in the strict mask.  The prior-correction route
# is the 22.99/24.65 family above.  TASKBOOK 1.2 attributes the 13.64 headline to
# "probe + cell-prior correction", which the measurements here contradict.
LINEA_ADAPT = {
    "source": "wx2/docs/COMPARISON.md 0.0 line 42, line 128",
    "method": "290 labels train a head directly (frozen XLS-R + LayerNorm/MLP), no prior correction",
    "clips_pooled": 13.05,
    "strict_pooled": 13.64,
    "auc": 0.94,
}
DEVIATION_ROWS = (
    {
        "item": "TASKBOOK 1.2 headline attribution",
        "what": (
            "TASKBOOK says Line A's headline 13.64/14.56/20.79 comes from 'frozen XLS-R "
            "features + a small probe trained on 290 labels, plus cell-prior correction'. "
            "Measured on the box, that route gives 24.65 pooled (k=10, BB+EB), and even an "
            "ORACLE cell prior only reaches 21.52 pooled on the p0cal scores. The 13.64 "
            "number is COMPARISON.md's line A: 290 labels train a head directly with NO "
            "prior correction (clips 13.05 / strict 13.64, AUC 0.94)."
        ),
        "impact": (
            "S-F must be judged against the ADAPTATION route (BASE_F's B11 family), not only "
            "against the prior-correction family; otherwise the SOTA bar is ~9 EER points "
            "too low. The prior-correction family stays in BASE_F under its own name."
        ),
        "source": "W4 on the box, 2026-10-14; wx2/docs/COMPARISON.md 0.0 line 42",
    },
)

__all__ = ["run_draws", "main"]


def _zscore(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / (s.std() + 1e-9)


def run_draws(meta: pd.DataFrame, scores: Dict[int, np.ndarray],
              n_draws: int = N_DRAWS, k_per_cell: int = K_PER_CELL,
              min_cell: int = MIN_CELL) -> pd.DataFrame:
    cells = meta["cell"].to_numpy()
    plat = meta["platform"].to_numpy()
    speaker = meta["speaker"].to_numpy()
    y = meta["y"].to_numpy()
    z_all = {sd: _zscore(np.load(s) if isinstance(s, (str, Path)) else np.asarray(s))
             for sd, s in scores.items()}
    rows: List[Dict[str, Any]] = []
    for d in range(int(n_draws)):
        sd = d % 3
        if sd not in z_all:
            continue
        lab_pos, _info = draw_labels(cells, k_per_cell, draw_seed(d),
                                     cap_half=True, min_cell=min_cell)
        lab = np.zeros(len(meta), bool)
        lab[lab_pos] = True
        ev_clips = np.setdiff1d(np.arange(len(meta)), lab_pos)
        strict_keep = strict_eval_mask(speaker, lab_pos, ev_clips)
        ev_strict = ev_clips[strict_keep]

        pi, n_lab, phi = estimate_pi(list(dict.fromkeys(cells.tolist())), cells, lab,
                                     y, plat, variant="bb")
        logit_pi = np.array([logit(pi[c], eps=1e-3) for c in cells], float)
        z = z_all[sd]
        methods = {"M0": z, "M1": logit_pi, "M2": z + logit_pi}
        for mask, ev in (("clips", ev_clips), ("strict", ev_strict)):
            if ev.size < 10:
                continue
            for name, sc in methods.items():
                row = summarize(y[ev], sc[ev], plat[ev], extra={
                    "protocol": "v1_full", "region": REGION, "backbone": BACKBONE,
                    "spec": SPEC, "method": name, "k": int(k_per_cell * 29),
                    "draw": d, "mask": mask, "model_seed": sd,
                    "n_labeled": int(lab.sum()), "phi": None if phi is None else float(phi),
                })
                rows.append(row)
    return pd.DataFrame(rows)


def mechanism_table(draws: pd.DataFrame) -> pd.DataFrame:
    agg = (draws.groupby(["method", "mask"])[["pooled_eer", "within_macro_eer", "worst_platform_eer", "auc"]]
           .agg(["mean", "std"]))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def reconciliation(draws: pd.DataFrame, n_head: int = 3) -> pd.DataFrame:
    rows = []
    for scope, sel in (
        ("3draw", draws[draws["draw"] < n_head]),
        ("100draw", draws),
    ):
        for metric, headline in STORED_FSCVAR.items():
            sub = sel[sel["method"] == "M2"]
            rerun = float(sub[metric].mean())
            rows.append({
                "scope": scope, "metric": metric, "headline": float(headline),
                "rerun": rerun, "diff": rerun - float(headline),
                "mask": "clips", "n_draws": int(sub["draw"].nunique()),
                "reference": "fsc_var_results.csv (bb, k=10)",
            })
    return pd.DataFrame(rows)


def linea_routes() -> pd.DataFrame:
    """Document Line A's two routes so nobody compares against the wrong one."""
    return pd.DataFrame([
        {"route": "A-adaptation", "clips_pooled": LINEA_ADAPT["clips_pooled"],
         "strict_pooled": LINEA_ADAPT["strict_pooled"], "prior_correction": False,
         "source": LINEA_ADAPT["source"]},
        {"route": "B-prior-correction", "clips_pooled": STORED_FSCVAR["pooled_eer"],
         "strict_pooled": float("nan"), "prior_correction": True,
         "source": "fsc_var_results.csv (bb, k=10)"},
    ])


def mask_identity(draws: pd.DataFrame) -> pd.DataFrame:
    piv = draws.pivot_table(index=["method", "draw"], columns="mask",
                            values="pooled_eer", aggfunc="first")
    if not {"clips", "strict"}.issubset(piv.columns):
        return pd.DataFrame(columns=["method", "n_draws", "n_identical", "share_identical"])
    piv = piv.dropna()
    same = np.isclose(piv["clips"], piv["strict"], atol=0.0, rtol=0.0)
    out = (piv.assign(identical=same).reset_index()
           .groupby("method")["identical"].agg(["size", "sum"]).reset_index())
    out.columns = ["method", "n_draws", "n_identical"]
    out["share_identical"] = out["n_identical"] / out["n_draws"]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W4 reproduction")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--k-per-cell", type=int, default=K_PER_CELL)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    drops = data_root(args.data)
    meta = load_mlitw(drops)
    scores = {}
    for sd in (0, 1, 2):
        p = drops / f"last_scores_p0cal_raw_s{sd}.npy"
        if p.exists():
            scores[sd] = np.load(p)
        else:
            print(f"missing {p}; model seed {sd} skipped")
    if not scores:
        print("no Line A probe scores found; W4 cannot run")
        return 1
    draws = run_draws(meta, scores, n_draws=args.draws, k_per_cell=args.k_per_cell)
    (root / "results").mkdir(parents=True, exist_ok=True)
    draws.to_csv(root / "results" / "w4_draws.csv", index=False)
    mech = mechanism_table(draws)
    mech.to_csv(root / "results" / "w4_mechanism.csv", index=False)
    rec = reconciliation(draws)
    rec.to_csv(root / "results" / "reconciliation.csv", index=False)
    linea_routes().to_csv(root / "results" / "reconciliation_lineA.csv", index=False)
    pd.DataFrame(DEVIATION_ROWS).to_csv(root / "results" / "deviations.csv", index=False)
    ident = mask_identity(draws)
    ident.to_csv(root / "results" / "w4_mask_identity.csv", index=False)
    print(mech.to_string(index=False))
    print()
    print(rec.to_string(index=False))
    print()
    print(ident.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

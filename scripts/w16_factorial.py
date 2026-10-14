#!/usr/bin/env python3
"""W16 - the 2^3 factor matrix and the de-confounding evidence (TASKBOOK W16).

Factors
  C1   multi-level probe: cell + language + platform effects ON vs platform only
  PAST prior-aware self-training ON vs OFF (sitting on the same base)
  CASA cell-aware selection ON vs uniform-per-cell OFF

Part A runs all 8 cells at k = 100 and k = 290 with 30 draws each, then reports
the marginal effect of every factor and the two-way interactions.

Part B is gate G10: for every draw we take the *detector part* ``d(z) = w'z`` and
measure how well it predicts the platform (one-vs-rest macro AUC).  C1 is supposed
to have a less platform-predictable detector without losing within-platform EER -
if that fails, N4 is not established.

Writes ``results/w16_factorial.csv`` and ``results/w16_mechanism.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import w6_baselines as w6  # noqa: E402
from src.alloc import CASA  # noqa: E402
from src.baselines import fit_logreg_cv  # noqa: E402
from src.data import data_root  # noqa: E402
from src.draws import draw_seed, strict_eval_mask  # noqa: E402
from src.metrics3 import auc, summarize  # noqa: E402
from src.mpp import MPP  # noqa: E402
from src.past import PAST  # noqa: E402

N_DRAWS = 30
KS = (100, 290)

__all__ = ["run_config", "marginals", "platform_macro_auc", "main"]


def platform_macro_auc(detector: np.ndarray, platform: np.ndarray) -> float:
    """One-vs-rest macro AUC of ``d(z)`` for predicting the platform.

    Each platform's AUC is folded to ``max(a, 1 - a)``: the question is how
    *predictable* the platform is from the detector, not which sign it happens to
    take.  Without the fold a detector that separates two platforms perfectly
    would score 0.5, which would be nonsense as a de-confounding measure.
    """
    vals = []
    for p in np.unique(platform):
        y = (platform == p).astype(int)
        if len(np.unique(y)) < 2:
            continue
        a = auc(y, detector)
        vals.append(max(a, 1.0 - a))
    return float(np.mean(vals)) if vals else float("nan")


def run_config(payload) -> List[Dict[str, Any]]:
    (c1_on, past_on, casa_on, k, d, Z, y, plat, lang, cell, speaker) = payload
    strategy = "CASA" if casa_on else "U"
    sel = CASA(k=int(k), strategy=strategy, seed=draw_seed(d)).run(Z, y, plat, lang, cell)["selected"]
    lab = np.zeros(len(y), bool)
    lab[sel] = True
    ev = np.setdiff1d(np.arange(len(y)), sel)
    if ev.size < 100 or lab.sum() < 50:
        return []
    mode = "full" if c1_on else "platformOnly"
    Zl, yl, pl, ll, cl = Z[lab], y[lab], plat[lab], lang[lab], cell[lab]
    if past_on:
        model = PAST(rounds=3, mode=mode).fit(
            Zl, yl, pl, ll, cl, Z[~lab], plat[~lab], lang[~lab], cell[~lab]
        )
        sc = model.predict_logit(Z, plat, lang, cell)
        det = model.model_.detector_logit(Z)
    else:
        m = MPP(sigma_p=0.5, sigma_gamma=0.5, lam=1.0, mode=mode).fit(Zl, yl, pl, ll, cl)
        sc = m.predict_logit(Z, plat, lang, cell)
        det = m.detector_logit(Z)
    rows: List[Dict[str, Any]] = []
    for mask, idx in (("clips", ev), ("strict", ev[strict_eval_mask(speaker, sel, ev)])):
        if idx.size < 50:
            continue
        rows.append(summarize(y[idx], sc[idx], plat[idx], extra={
            "protocol": "dev_v2", "region": "dev", "backbone": w6.BACKBONE, "spec": w6.SPEC,
            "method": f"C1={int(c1_on)}|PAST={int(past_on)}|CASA={int(casa_on)}",
            "c1": int(c1_on), "past": int(past_on), "casa": int(casa_on),
            "k": int(k), "draw": int(d), "mask": mask, "n_labeled": int(lab.sum()),
            "n_eval": int(idx.size),
        }))
    return rows


def _worker(payload):
    try:
        return payload[4], payload[0], run_config(payload)
    except Exception as exc:  # pragma: no cover
        return payload[4], payload[0], [{"error": f"{type(exc).__name__}: {exc}"}]


def marginals(draws: pd.DataFrame) -> pd.DataFrame:
    """Mean effect of flipping each factor, and the two-way interactions."""
    main = draws[draws["mask"] == "clips"]
    out = []
    for factor in ("c1", "past", "casa"):
        sub = main[main["k"] == 290]
        hi = sub[sub[factor] == 1]["pooled_eer"]
        lo = sub[sub[factor] == 0]["pooled_eer"]
        out.append({"kind": "marginal", "factor": factor,
                    "effect": float(hi.mean() - lo.mean()), "n": int(len(hi) + len(lo))})
    for a, b in (("c1", "past"), ("c1", "casa"), ("past", "casa")):
        sub = main[main["k"] == 290]
        eff = ((sub[sub[a] == 1]["pooled_eer"].mean() - sub[sub[a] == 0]["pooled_eer"].mean())
               - (sub[(sub[a] == 1) & (sub[b] == 1)]["pooled_eer"].mean()
                  - sub[(sub[a] == 1) & (sub[b] == 0)]["pooled_eer"].mean()))
        out.append({"kind": "interaction", "factor": f"{a}x{b}", "effect": float(eff),
                    "n": int(len(sub))})
    return pd.DataFrame(out)


def mechanism(payload) -> List[Dict[str, Any]]:
    (d, Z, y, plat, lang, cell, speaker, k) = payload
    sel = CASA(k=int(k), strategy="U", seed=draw_seed(d)).run(Z, y, plat, lang, cell)["selected"]
    lab = np.zeros(len(y), bool)
    lab[sel] = True
    ev = np.setdiff1d(np.arange(len(y)), sel)
    Zl, yl, pl, ll, cl = Z[lab], y[lab], plat[lab], lang[lab], cell[lab]
    rows = []
    lr, _ = fit_logreg_cv(Zl, yl)
    det_b1 = lr.decision_function(Z)
    c1 = MPP(sigma_p=0.5, sigma_gamma=0.5, lam=1.0, mode="full").fit(Zl, yl, pl, ll, cl)
    det_c1 = c1.detector_logit(Z)
    sc_b1 = lr.decision_function(Z)
    sc_c1 = c1.predict_logit(Z, plat, lang, cell)
    for name, det, sc in (("B1", det_b1, sc_b1), ("C1", det_c1, sc_c1)):
        rows.append({"protocol": "dev_v2", "region": "dev", "backbone": w6.BACKBONE,
                     "spec": w6.SPEC, "method": name, "metric": "platform_auc",
                     "value": platform_macro_auc(det, plat), "k": int(k), "draw": int(d)})
        rows.append({"protocol": "dev_v2", "region": "dev", "backbone": w6.BACKBONE,
                     "spec": w6.SPEC, "method": name, "metric": "within_macro_eer",
                     "value": float(summarize(y[ev], sc[ev], plat[ev])["within_macro_eer"]),
                     "k": int(k), "draw": int(d)})
    return rows


def _worker_mech(payload):
    try:
        return payload[0], mechanism(payload)
    except Exception as exc:  # pragma: no cover
        return payload[0], [{"error": f"{type(exc).__name__}: {exc}"}]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W16 factorial + mechanism")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--part", default="both", choices=["factorial", "mechanism", "both"])
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    sub, F = w6.load_dev(root, data_root(args.data))
    Z = w6.preprocess(F)
    y = sub["y"].to_numpy()
    plat = sub["platform"].to_numpy()
    lang = sub["language"].to_numpy()
    cell = sub["cell"].to_numpy()
    speaker = sub["speaker"].to_numpy()
    import multiprocessing as mp

    ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
    (root / "results").mkdir(parents=True, exist_ok=True)

    if args.part in ("factorial", "both"):
        payloads = [(c1, past, casa, k, d, Z, y, plat, lang, cell, speaker)
                    for c1 in (0, 1) for past in (0, 1) for casa in (0, 1)
                    for k in KS for d in range(args.draws)]
        rows, errors = [], []
        with ctx.Pool(args.jobs) as pool:
            for d, c1, out in pool.imap_unordered(_worker, payloads, chunksize=1):
                for r in out:
                    (errors if "error" in r else rows).append(r)
        if errors:
            print("errors:", errors[:3], "...", len(errors))
        df = pd.DataFrame(rows)
        df.to_csv(root / "results" / "w16_factorial.csv", index=False)
        marg = marginals(df)
        marg.to_csv(root / "results" / "w16_marginals.csv", index=False)
        print(marg.to_string(index=False))

    if args.part in ("mechanism", "both"):
        payloads = [(d, Z, y, plat, lang, cell, speaker, 290) for d in range(args.draws)]
        rows, errors = [], []
        with ctx.Pool(args.jobs) as pool:
            for d, out in pool.imap_unordered(_worker_mech, payloads, chunksize=1):
                for r in out:
                    (errors if "error" in r else rows).append(r)
        if errors:
            print("errors:", errors[:3], "...", len(errors))
        mech = pd.DataFrame(rows)
        mech.to_csv(root / "results" / "w16_mechanism.csv", index=False)
        piv = mech.groupby(["method", "metric"])["value"].mean().round(4)
        print(piv.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

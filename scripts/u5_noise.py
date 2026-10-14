#!/usr/bin/env python3
"""U5 - label-noise and small-budget robustness (NEW_TASKS 01 U5).

For ``M-ALP``, ``B9``, ``B1`` and ``C1``: flip 5 %, 10 % and 20 % of the labels in
``L``, at k in {30, 100, 290}, 30 draws each, and evaluate on the ``clips`` mask of
the v3 dev region.  The question the task asks is whether M-ALP degrades like C1
does (C1 went from 9.10 to 17.95 at 10 % noise on the v2 region).

Every learner sees exactly the same corrupted ``L``, including M-ALP's
hyper-parameter selection (a practitioner cannot select on clean labels they do not
have).  ``C1`` uses the frozen ``(0.5, 0.5, 1.0)`` settings rather than its 80-fit
CV, so the arm stays inside the U5 budget; this is recorded in the output.

Writes ``results/u5_noise.csv`` and ``results/u5_summary.csv``.
"""

from __future__ import annotations

import argparse
import json
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
from src import regions as R  # noqa: E402
from src.baselines import b9_label_propagation, fit_logreg_cv, knn_graph  # noqa: E402
from src.cv import speaker_grouped_folds, stratified_folds  # noqa: E402
from src.data import data_root  # noqa: E402
from src.draws import draw_labels, draw_seed  # noqa: E402
from src.malp import malp_scores, select_hyperparams  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.mpp import MPP  # noqa: E402

NOISE = (0.05, 0.10, 0.20)
K_LIST = (30, 100, 290)
M_PER_CELL = {30: 1, 100: 3, 290: 10}
N_DRAWS = 30
METHODS = ("B1", "B9", "C1", "M-ALP")
C1_FIXED = (0.5, 0.5, 1.0)

_G: Dict[str, Any] = {}

__all__ = ["corrupt", "run_one", "summary", "main"]


def corrupt(y: np.ndarray, lab: np.ndarray, rate: float, seed: int = 0) -> np.ndarray:
    """Flip ``rate`` of the labelled labels at random."""
    out = y.copy()
    idx = np.where(lab)[0]
    if rate <= 0 or idx.size == 0:
        return out
    rng = np.random.default_rng(int(seed))
    n_flip = int(round(float(rate) * idx.size))
    if n_flip <= 0:
        return out
    chosen = rng.choice(idx, size=min(n_flip, idx.size), replace=False)
    out[chosen] = 1 - out[chosen]
    return out


def run_one(payload) -> Tuple[int, float, int, List[Dict[str, Any]]]:
    d, rate, k = payload
    Z, y, plat, cell, speaker = (_G["Z"], _G["y"], _G["plat"], _G["cell"], _G["speaker"])
    graph = _G["graph"]
    rows: List[Dict[str, Any]] = []
    lab_pos, _info = draw_labels(cell, M_PER_CELL[k], draw_seed(d), cap_half=True)
    if lab_pos.size < 20:
        return d, rate, k, rows
    lab = np.zeros(len(y), bool)
    lab[lab_pos] = True
    y_noisy = corrupt(y, lab, rate, seed=draw_seed(d) + 7919)
    ev = np.setdiff1d(np.arange(len(y)), lab_pos)
    if ev.size < 50 or len(np.unique(y_noisy[lab])) < 2:
        return d, rate, k, rows
    folds = stratified_folds(y_noisy[lab])
    scores: Dict[str, np.ndarray] = {}

    model, info = fit_logreg_cv(Z[lab], y_noisy[lab], folds=folds)
    scores["B1"] = model.decision_function(Z)
    s9, _ = b9_label_propagation(Z, y_noisy[lab], lab, k=_G["b9_k"], alpha=0.9,
                                 graph=graph)
    scores["B9"] = s9
    c1 = MPP(sigma_p=C1_FIXED[0], sigma_gamma=C1_FIXED[1], lam=C1_FIXED[2])
    c1.fit(Z[lab], y_noisy[lab], plat[lab], _G["lang"][lab], cell[lab])
    scores["C1"] = c1.predict_logit(Z, plat, _G["lang"], cell)

    cfg, info_cfg = select_hyperparams(Z, y_noisy, lab, graph, folds, speaker=speaker)
    scores["M-ALP"] = malp_scores(Z, y_noisy, lab, graph, cfg, scores["B1"], speaker)

    for name in METHODS:
        sc = scores[name]
        rows.append(summarize(y[ev], sc[ev], plat[ev], extra={
            "protocol": "dev_v3_noise", "region": "dev", "mask": "clips",
            "backbone": _G["backbone"], "spec": _G["spec"], "method": name,
            "k": int(k), "draw": int(d), "noise_rate": float(rate),
            "n_labeled": int(lab.sum()), "n_eval": int(ev.size),
            "n_flipped": int(round(rate * lab.sum())),
            "cv_folds": "stratified", "b9_k": _G["b9_k"],
            "malp_cfg": json.dumps(list(cfg)) if name == "M-ALP" else "{}",
            "c1_fixed": json.dumps(list(C1_FIXED)),
            "C": info["C"] if name == "B1" else None,
            "malp_cv_loglik": float(info_cfg["cv_loglik"]) if name == "M-ALP" else None,
        }))
    return d, rate, k, rows


def _worker(payload):
    try:
        return run_one(payload)
    except Exception as exc:                            # pragma: no cover
        return payload[0], payload[1], payload[2], [{"error": f"{payload}: {exc}"}]


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean pooled EER per (method, k, noise) plus the degradation vs the clean row."""
    piv = (df.groupby(["method", "k", "noise_rate"])["pooled_eer"].mean()
           .reset_index())
    out = []
    for (method, k), part in piv.groupby(["method", "k"]):
        clean = float(part.loc[part["noise_rate"] == part["noise_rate"].min(),
                               "pooled_eer"].iloc[0]) if (part["noise_rate"] == 0).any() \
            else float(part["pooled_eer"].min())
        for _, r in part.iterrows():
            out.append({"method": method, "k": int(k),
                        "noise_rate": float(r["noise_rate"]),
                        "pooled_eer": float(r["pooled_eer"]),
                        "degradation": float(r["pooled_eer"]) - clean})
    return pd.DataFrame(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 U5 label-noise robustness")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--split", default=R.SPLIT_V3)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--b9-k", type=int, default=15)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    data_dir = data_root(args.data)
    sub, F = R.load_region(root, data_dir, args.backbone, args.spec, args.split)
    transform, Z = w6.fit_preprocess(F)
    lang = sub["language"].to_numpy()
    _G.update({"Z": Z, "y": sub["y"].to_numpy(), "plat": sub["platform"].to_numpy(),
               "lang": lang, "cell": np.asarray([f"{p}|{l}" for p, l in
                                                 zip(sub['platform'], lang)]),
               "speaker": sub["speaker"].to_numpy(), "backbone": args.backbone,
               "spec": args.spec, "b9_k": int(args.b9_k),
               "graph": knn_graph(Z, k=max(15, int(args.b9_k)))})
    print(f"v3 dev: clips={len(sub)} speakers={len(np.unique(_G['speaker']))}",
          flush=True)
    payloads = [(d, rate, k) for k in K_LIST for rate in (0.0,) + NOISE
                for d in range(int(args.draws))]
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    if args.jobs > 1:
        import multiprocessing as mp

        ctx = (mp.get_context("fork") if sys.platform != "win32"
               else mp.get_context("spawn"))
        with ctx.Pool(int(args.jobs)) as pool:
            for i, (_d, _r, _k, out) in enumerate(
                    pool.imap_unordered(_worker, payloads, chunksize=1), 1):
                for r in out:
                    if "error" in r:
                        errors.append(str(r["error"]))
                    else:
                        rows.append(r)
                if i % 25 == 0:
                    print(f"  {i}/{len(payloads)} units ({len(rows)} rows, "
                          f"{len(errors)} errors)", flush=True)
    else:
        for pl in payloads:
            _d, _r, _k, out = _worker(pl)
            for r in out:
                if "error" in r:
                    errors.append(str(r["error"]))
                else:
                    rows.append(r)
    if errors:
        print(f"errors: {len(errors)}; first: {errors[:3]}")
        (root / "results" / "u5_errors.txt").write_text("\n".join(errors) + "\n",
                                                        encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(root / "results" / "u5_noise.csv", index=False)
    summ = summary(df)
    summ.to_csv(root / "results" / "u5_summary.csv", index=False)
    print("\npooled EER by noise level\n"
          + summ.pivot_table(index=["method", "k"], columns="noise_rate",
                             values="pooled_eer").round(3).to_string())
    print("\ndegradation vs the clean row\n"
          + summ.pivot_table(index=["method", "k"], columns="noise_rate",
                             values="degradation").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""W11 - PAST against C1 and the naive pseudo-labelling baselines (TASKBOOK W11).

The Fam-A views (W10) are not available yet on this feature set, so this run is
the documented **no-strong-view** variant: the quota and EM machinery is fully
active, the view-consistency terms are not.  ``lambda_u`` is therefore inert and
the ablation table says so instead of pretending otherwise.

Ablations: no EM (cell prior from the labelled rates), no prior-aware quota
(global confidence threshold = B7's rule), and a single round.

Writes ``results/w11_draws.csv`` and ``results/w11_past.csv`` (the paired table
``decide.py`` reads for gate G6).
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
from src.baselines import b7_naive_pseudo_label  # noqa: E402
from src.data import data_root  # noqa: E402
from src.draws import draw_labels, draw_seed, strict_eval_mask  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.mpp import MPP  # noqa: E402
from src.past import PAST  # noqa: E402

N_DRAWS = 50
K_PER_CELL = 10
VARIANTS = ("C1", "B7", "PAST", "PAST-noEM", "PAST-noQuota", "PAST-1round")

__all__ = ["run_one", "paired_table", "main"]


def run_one(payload) -> List[Dict[str, Any]]:
    (d, Z, y, plat, lang, cell, speaker, k_per_cell) = payload
    lab_pos, _ = draw_labels(cell, k_per_cell, draw_seed(d), cap_half=True)
    lab = np.zeros(len(y), bool)
    lab[lab_pos] = True
    ev = np.setdiff1d(np.arange(len(y)), lab_pos)
    if lab.sum() < 50 or ev.size < 100:
        return []
    strict = ev[strict_eval_mask(speaker, lab_pos, ev)]
    Zl, yl, pl, ll, cl = Z[lab], y[lab], plat[lab], lang[lab], cell[lab]
    Zu, pu, lu, cu = Z[~lab], plat[~lab], lang[~lab], cell[~lab]

    scores: Dict[str, np.ndarray] = {}
    scores["C1"] = MPP(sigma_p=0.5, sigma_gamma=0.5, lam=1.0).fit(
        Zl, yl, pl, ll, cl).predict_logit(Z, plat, lang, cell)
    s7, _ = b7_naive_pseudo_label(Zl, yl, Zu, tau=0.9, rounds=3)
    sc = np.zeros(len(y)); sc[~lab] = s7
    scores["B7"] = sc
    for name, kw in (
        ("PAST", {"rounds": 3}),
        ("PAST-noEM", {"rounds": 3, "use_em": False}),
        ("PAST-noQuota", {"rounds": 3, "use_quota": False}),
        ("PAST-1round", {"rounds": 1}),
    ):
        model = PAST(**kw).fit(Zl, yl, pl, ll, cl, Zu, pu, lu, cu)
        scores[name] = model.predict_logit(Z, plat, lang, cell)

    rows: List[Dict[str, Any]] = []
    for mask, idx in (("clips", ev), ("strict", strict)):
        if idx.size < 50:
            continue
        for name, s in scores.items():
            rows.append(summarize(y[idx], s[idx], plat[idx], extra={
                "protocol": "dev_v2", "region": "dev", "backbone": w6.BACKBONE, "spec": w6.SPEC,
                "method": name, "k": int(k_per_cell * 29), "draw": int(d), "mask": mask,
                "n_labeled": int(lab.sum()), "n_eval": int(idx.size), "views": "none",
            }))
    return rows


def _worker(payload):
    try:
        return payload[0], run_one(payload)
    except Exception as exc:  # pragma: no cover
        return payload[0], [{"error": f"{type(exc).__name__}: {exc}"}]


def paired_table(draws: pd.DataFrame, variants: Sequence[str] = VARIANTS) -> pd.DataFrame:
    main = draws[draws["mask"] == "clips"]
    rows = []
    for other in variants:
        if other == "PAST":
            continue
        for metric in ("pooled_eer", "within_macro_eer"):
            wa = main[main["method"] == "PAST"].set_index("draw")[metric]
            wb = main[main["method"] == other].set_index("draw")[metric]
            common = wa.index.intersection(wb.index)
            if len(common) < 3:
                continue
            diff = (wa.loc[common].astype(float) - wb.loc[common].astype(float)).dropna().values
            if diff.size < 3:
                continue
            rows.append({"method": "PAST", "other": other, "metric": metric,
                         "mean": float(diff.mean()), "lo": float(np.percentile(diff, 2.5)),
                         "hi": float(np.percentile(diff, 97.5)), "n": int(diff.size),
                         "backbone": w6.BACKBONE, "spec": w6.SPEC})
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W11 PAST")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--k-per-cell", type=int, default=K_PER_CELL)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    sub, F = w6.load_dev(root, data_root(args.data))
    Z = w6.preprocess(F)
    y = sub["y"].to_numpy()
    plat = sub["platform"].to_numpy()
    lang = sub["language"].to_numpy()
    cell = sub["cell"].to_numpy()
    speaker = sub["speaker"].to_numpy()
    print(f"dev clips={len(sub)} draws={args.draws} variants={VARIANTS} (views: none)")

    payloads = [(d, Z, y, plat, lang, cell, speaker, args.k_per_cell) for d in range(args.draws)]
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
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
                if d % 10 == 0:
                    print(f"  draw {d} done ({len(rows)} rows)", flush=True)
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
    df.to_csv(root / "results" / "w11_draws.csv", index=False)
    paired_tbl = paired_table(df)
    paired_tbl.to_csv(root / "results" / "w11_past.csv", index=False)
    summary = (df[df["mask"] == "clips"].groupby("method")
               [["pooled_eer", "within_macro_eer", "worst_platform_eer", "auc"]].mean().round(3))
    print(summary.to_string())
    print()
    print(paired_tbl.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""W14 - B9 label propagation and GraphSSL (TASKBOOK W14).

B9 is the plain transductive propagation; GraphSSL is ``C1 + eta * Y_prop`` with
``eta`` chosen inside ``L`` on the shared folds.  Both are transductive and the
rows say so, because the report has to.

Rows join the ``w6_draws_*`` family so S-F can pair them with C1.
"""

from __future__ import annotations

import argparse
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
from src.baselines import b9_label_propagation  # noqa: E402
from src.data import data_root  # noqa: E402
from src.draws import draw_labels, draw_seed, strict_eval_mask  # noqa: E402
from src.graph_ssl import GraphSSL  # noqa: E402
from src.metrics3 import summarize  # noqa: E402

N_DRAWS = 50
K_PER_CELL = 10

__all__ = ["run_one", "main"]


def run_one(payload) -> List[Dict[str, Any]]:
    (d, Z, y, plat, lang, cell, speaker, k_per_cell) = payload
    lab_pos, _ = draw_labels(cell, k_per_cell, draw_seed(d), cap_half=True)
    lab = np.zeros(len(y), bool)
    lab[lab_pos] = True
    ev = np.setdiff1d(np.arange(len(y)), lab_pos)
    if lab.sum() < 5 or len(np.unique(y[lab])) < 2 or ev.size < 100:
        return []
    s9, i9 = b9_label_propagation(Z, y[lab], lab, k=15, alpha=0.9)
    g = GraphSSL().fit(Z, y, plat, lang, cell, lab, select=True)
    scores = {"B9": s9, "GraphSSL": g.score()}
    rows = []
    for mask, idx in (("clips", ev), ("strict", ev[strict_eval_mask(speaker, lab_pos, ev)])):
        if idx.size < 50:
            continue
        for name, s in scores.items():
            rows.append(summarize(y[idx], s[idx], plat[idx], extra={
                "protocol": "dev_v2", "region": "dev", "backbone": w6.BACKBONE, "spec": w6.SPEC,
                "method": name, "k": int(k_per_cell * 29), "draw": int(d), "mask": mask,
                "n_labeled": int(lab.sum()), "n_eval": int(idx.size),
                "transductive": True, "eta": getattr(g, "eta", None),
            }))
    return rows


def _worker(payload):
    try:
        return payload[0], run_one(payload)
    except Exception as exc:  # pragma: no cover
        return payload[0], [{"error": f"{type(exc).__name__}: {exc}"}]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W14 GraphSSL")
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
    print(f"dev clips={len(sub)} draws={args.draws}")

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
    out = root / "results" / f"w6_draws_{w6.BACKBONE}_{w6.SPEC}_graphssl.csv"
    df.to_csv(out, index=False)
    summary = (df[df["mask"] == "clips"].groupby("method")
               [["pooled_eer", "within_macro_eer", "auc"]].mean().round(3))
    print(summary.to_string())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

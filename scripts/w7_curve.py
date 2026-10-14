#!/usr/bin/env python3
"""W7 - the label-budget curve (TASKBOOK W7), the data behind S-K.

Per-cell quotas m in {1, 3, 10, 20, 40} (k ~ 30 / 100 / 290 / 600 / 1200), the
same dev region, the same probe family and the same 50 draws per point, so the
curve isolates the budget.  A cell that cannot supply ``floor(n_c/2) >= 4``
clips is dropped and the actual labelled count is recorded.

Writes ``results/w7_draws.csv`` and ``results/w7_curve.csv``.
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
from src.data import data_root  # noqa: E402
from src.draws import BUDGET_M  # noqa: E402

N_DRAWS = 50
PER_CELL_M = (1, 3, 10, 20, 40)
METHODS = ("B1", "B2", "B3", "B11", "C1", "PAST")

__all__ = ["main"]


def _worker(payload):
    try:
        return payload[0], payload[1], w6.run_one_draw(payload)
    except Exception as exc:  # pragma: no cover
        return payload[0], payload[1], [{"error": f"{type(exc).__name__}: {exc}"}]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W7 budget curve")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--m", default=",".join(str(m) for m in PER_CELL_M))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    ms = tuple(int(x) for x in args.m.split(",") if x.strip())
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    sub, F = w6.load_dev(root, data_root(args.data))
    Z = w6.preprocess(F)
    y = sub["y"].to_numpy()
    plat = sub["platform"].to_numpy()
    lang = sub["language"].to_numpy()
    cell = sub["cell"].to_numpy()
    speaker = sub["speaker"].to_numpy()
    zp = data_root(args.data) / "last_scores_p0cal_raw_s0.npy"
    zero = np.load(zp)[sub["clip_id"].to_numpy()] if zp.exists() else np.zeros(len(sub))
    print(f"dev clips={len(sub)} per-cell m={ms} methods={methods}")

    payloads = [(d, Z, y, plat, lang, cell, speaker, zero, methods, m)
                for m in ms for d in range(args.draws)]
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    if args.jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        with ctx.Pool(args.jobs) as pool:
            for d, m, out in pool.imap_unordered(_worker, payloads, chunksize=1):
                for r in out:
                    if "error" in r:
                        errors.append(f"m={m} draw={d}: {r['error']}")
                    else:
                        rows.append(r)
    else:
        for p in payloads:
            d, m, out = _worker(p)
            for r in out:
                if "error" in r:
                    errors.append(f"m={m} draw={d}: {r['error']}")
                else:
                    rows.append(r)
    if errors:
        print("errors:", errors[:5], "...", len(errors))
    df = pd.DataFrame(rows)
    (root / "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "results" / "w7_draws.csv", index=False)
    curve = (df[df["mask"] == "clips"].groupby(["method", "k"])
             [["pooled_eer", "within_macro_eer", "worst_platform_eer", "auc", "n_labeled"]]
             .agg(["mean", "std"]))
    curve.columns = [f"{a}_{b}" for a, b in curve.columns]
    curve = curve.reset_index()
    curve.to_csv(root / "results" / "w7_curve.csv", index=False)
    piv = curve.pivot(index="k", columns="method", values="pooled_eer_mean").round(2)
    print(piv.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

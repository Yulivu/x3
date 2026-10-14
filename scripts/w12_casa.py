#!/usr/bin/env python3
"""W12 - CASA against the six comparison strategies (TASKBOOK W12).

Every strategy gets exactly the same budget (k = 100 and 290), the same C1 probe,
the same per-cell ceiling and the same in-cell dedup, so the only thing that
differs is *which* clips get labelled.  Evaluation is always the unlabelled rest
of the dev region, and the verdict comes from the paired per-draw difference
(gate G7: CASA beats uniform by >= 1.0 pooled at k=100 with the interval clear of 0).

Writes ``results/w12_draws.csv`` (per-draw rows) and ``results/w12_casa.csv``
(the paired table ``decide.py`` reads).
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
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import w6_baselines as w6  # noqa: E402
from src.alloc import CASA, STRATEGIES  # noqa: E402
from src.data import data_root  # noqa: E402
from src.draws import draw_seed, strict_eval_mask  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.mpp import MPP  # noqa: E402

N_DRAWS = 30
KS = (100, 290)

__all__ = ["run_cell", "main"]


def run_cell(payload) -> List[Dict[str, Any]]:
    (strategy, k, d, Z, y, plat, lang, cell, speaker) = payload
    sel = CASA(k=k, strategy=strategy, seed=draw_seed(d)).run(Z, y, plat, lang, cell)["selected"]
    lab = np.zeros(len(y), bool)
    lab[sel] = True
    ev = np.setdiff1d(np.arange(len(y)), sel)
    if ev.size < 100:
        return []
    model = MPP(sigma_p=0.5, sigma_gamma=0.5, lam=1.0).fit(
        Z[lab], y[lab], plat[lab], lang[lab], cell[lab]
    )
    sc = model.predict_logit(Z, plat, lang, cell)
    rows = []
    for mask, idx in (("clips", ev), ("strict", ev[strict_eval_mask(speaker, sel, ev)])):
        if idx.size < 50:
            continue
        rows.append(summarize(y[idx], sc[idx], plat[idx], extra={
            "protocol": "dev_v2", "region": "dev", "backbone": w6.BACKBONE, "spec": w6.SPEC,
            "method": strategy, "k": int(k), "draw": int(d), "mask": mask,
            "n_labeled": int(lab.sum()), "n_eval": int(idx.size),
        }))
    return rows


def _worker(payload):
    try:
        return payload[2], payload[0], payload[1], run_cell(payload)
    except Exception as exc:  # pragma: no cover
        return payload[2], payload[0], payload[1], [{"error": f"{type(exc).__name__}: {exc}"}]


def paired_table(draws: pd.DataFrame, base_methods: Sequence[str] = STRATEGIES) -> pd.DataFrame:
    main = draws[draws["mask"] == "clips"]
    rows = []
    for k in sorted(main["k"].unique()):
        sub_k = main[main["k"] == k]
        for other in base_methods:
            if other == "CASA":
                continue
            for metric in ("pooled_eer", "within_macro_eer"):
                wa = sub_k[sub_k["method"] == "CASA"].set_index("draw")[metric]
                wb = sub_k[sub_k["method"] == other].set_index("draw")[metric]
                common = wa.index.intersection(wb.index)
                if len(common) < 3:
                    continue
                diff = (wa.loc[common].astype(float) - wb.loc[common].astype(float)).dropna().values
                if diff.size < 3:
                    continue
                rows.append({
                    "method": "CASA", "other": other, "metric": metric, "k": int(k),
                    "mean": float(diff.mean()), "lo": float(np.percentile(diff, 2.5)),
                    "hi": float(np.percentile(diff, 97.5)), "n": int(diff.size),
                    "backbone": w6.BACKBONE, "spec": w6.SPEC,
                })
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W12 CASA")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--ks", default=",".join(str(k) for k in KS))
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--backbone", default=w6.BACKBONE,
                        help="default is the frozen Line A block; anything else reads "
                             "$DATA/x3/feats/<backbone>/")
    parser.add_argument("--spec", default=w6.SPEC)
    parser.add_argument("--split", default="speaker_split_v2.json",
                        help="frozen split defining the dev region")
    parser.add_argument("--suffix", default="",
                        help="suffix for the output files (never overwrite)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())

    sub, F = w6.load_spec(root, data_root(args.data), args.backbone, args.spec,
                          split_name=args.split)
    # the per-draw rows record the backbone from w6's module globals, so they have to
    # follow the CLI (this is exactly what w6's own main does)
    w6.BACKBONE, w6.SPEC = args.backbone, args.spec
    Z = w6.preprocess(F)
    y = sub["y"].to_numpy()
    plat = sub["platform"].to_numpy()
    lang = sub["language"].to_numpy()
    cell = sub["cell"].to_numpy()
    speaker = sub["speaker"].to_numpy()
    print(f"dev clips={len(sub)} strategies={strategies} ks={ks}")

    payloads = [(s, k, d, Z, y, plat, lang, cell, speaker)
                for s in strategies for k in ks for d in range(args.draws)]
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    if args.jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        with ctx.Pool(args.jobs) as pool:
            for d, s, k, out in pool.imap_unordered(_worker, payloads, chunksize=1):
                for r in out:
                    if "error" in r:
                        errors.append(f"{s} k={k} draw={d}: {r['error']}")
                    else:
                        rows.append(r)
    else:
        for p in payloads:
            d, s, k, out = _worker(p)
            for r in out:
                if "error" in r:
                    errors.append(f"{s} k={k} draw={d}: {r['error']}")
                else:
                    rows.append(r)
    if errors:
        print("errors:", errors[:5], "...", len(errors))
    df = pd.DataFrame(rows)
    (root / "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "results" / f"w12_draws{args.suffix}.csv", index=False)
    paired_tbl = paired_table(df)
    paired_tbl.to_csv(root / "results" / f"w12_casa{args.suffix}.csv", index=False)
    summary = (df[df["mask"] == "clips"].groupby(["method", "k"])
               [["pooled_eer", "within_macro_eer", "auc"]].mean().round(3))
    print(summary.to_string())
    print()
    print(paired_tbl.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

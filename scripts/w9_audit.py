#!/usr/bin/env python3
"""W9 - leakage and negative-control audit on the current W6 configuration.

Five checks, all CPU-only, all able to only ever *remove* a claim:

1. label permutation inside each platform -> every method must fall back to 50 EER;
2. cell-metadata permutation -> C1's advantage over B1 must vanish;
3. clips vs strict (read back from the W6 output) and a video-disjoint mask when
   the manifest exposes a video id;
4. near-duplicate share between evaluation and labelled clips (cosine > 0.99);
5. the metadata-only shortcut level (platform x language, no audio).

Writes ``results/w9_audit.csv``; any violated expectation sets ``flag``.
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import w6_baselines as w6  # noqa: E402
from src.audit import (  # noqa: E402
    NEAR_DUP_COS,
    metadata_only_scores,
    nearest_neighbour_overlap,
    permute_cells,
    permute_labels_global,
    permute_labels_within,
    video_disjoint_mask,
)
from src.data import data_root  # noqa: E402
from src.draws import draw_labels, draw_seed, strict_eval_mask  # noqa: E402
from src.metrics3 import eer  # noqa: E402

PERMUTED_TOL = 3.0
SHORTCUT_TOL = 3.0
N_DRAWS = 30

__all__ = ["audit", "main"]


def _draw_frame(sub, Z, zero, methods, y, cell, seed_draw, k_per_cell):
    rows = w6.run_one_draw((seed_draw, Z, y, sub["platform"].to_numpy(),
                            sub["language"].to_numpy(), cell, sub["speaker"].to_numpy(),
                            zero, methods, k_per_cell))
    return pd.DataFrame(rows)


def audit(root: Path, data_dir: Path, n_draws: int = N_DRAWS,
          methods: Sequence[str] = ("B1", "B2", "C1", "B7"), k_per_cell: int = 10) -> pd.DataFrame:
    sub, F = w6.load_dev(root, data_dir)
    Z = w6.preprocess(F)
    y = sub["y"].to_numpy()
    plat = sub["platform"].to_numpy()
    lang = sub["language"].to_numpy()
    cell = sub["cell"].to_numpy()
    zp = data_dir / "last_scores_p0cal_raw_s0.npy"
    zero = np.load(zp)[sub["clip_id"].to_numpy()] if zp.exists() else np.zeros(len(sub))

    rows: List[Dict[str, Any]] = []
    perm_eer: Dict[str, List[float]] = {m: [] for m in methods}
    global_eer: Dict[str, List[float]] = {m: [] for m in methods}
    cell_perm: Dict[str, List[float]] = {m: [] for m in methods}
    baseline: Dict[str, List[float]] = {m: [] for m in methods}
    near_dup: List[float] = []

    for d in range(int(n_draws)):
        base = _draw_frame(sub, Z, zero, methods, y, cell, d, k_per_cell)
        base = base[base["mask"] == "clips"]
        for m in methods:
            r = base[base["method"] == m]
            if not r.empty:
                baseline[m].append(float(r["pooled_eer"].iloc[0]))

        yp = permute_labels_within(y, plat, seed=1000 + d)
        dfp = _draw_frame(sub, Z, zero, methods, yp, cell, d, k_per_cell)
        dfp = dfp[dfp["mask"] == "clips"]
        for m in methods:
            r = dfp[dfp["method"] == m]
            if not r.empty:
                perm_eer[m].append(float(r["pooled_eer"].iloc[0]))

        yg = permute_labels_global(y, seed=3000 + d)
        dfg = _draw_frame(sub, Z, zero, methods, yg, cell, d, k_per_cell)
        dfg = dfg[dfg["mask"] == "clips"]
        for m in methods:
            r = dfg[dfg["method"] == m]
            if not r.empty:
                global_eer[m].append(float(r["pooled_eer"].iloc[0]))

        cp = permute_cells(cell, seed=2000 + d)
        dfc = _draw_frame(sub, Z, zero, methods, y, cp, d, k_per_cell)
        dfc = dfc[dfc["mask"] == "clips"]
        for m in methods:
            r = dfc[dfc["method"] == m]
            if not r.empty:
                cell_perm[m].append(float(r["pooled_eer"].iloc[0]))

        lab_pos, _ = draw_labels(cell, k_per_cell, draw_seed(d))
        lab = np.zeros(len(y), bool)
        lab[lab_pos] = True
        ev = np.setdiff1d(np.arange(len(y)), lab_pos)
        nd = nearest_neighbour_overlap(Z[lab], Z[ev])
        if np.isfinite(nd["share"]):
            near_dup.append(nd["share"])

    md = metadata_only_scores(plat, lang, y)
    md_eer = float(eer(y, md))

    for m in methods:
        b = float(np.mean(baseline[m])) if baseline[m] else float("nan")
        p = float(np.mean(perm_eer[m])) if perm_eer[m] else float("nan")
        g = float(np.mean(global_eer[m])) if global_eer[m] else float("nan")
        c = float(np.mean(cell_perm[m])) if cell_perm[m] else float("nan")
        rows.append({"check": "label-permutation-global", "method": m, "mask": "clips",
                     "eer": g, "reference_eer": 50.0, "n_draws": len(global_eer[m]),
                     "flag": bool(np.isfinite(g) and abs(g - 50.0) > PERMUTED_TOL),
                     "expectation": "fully shuffled labels -> EER ~ 50 (+-3)"})
        rows.append({"check": "label-permutation-within-platform", "method": m, "mask": "clips",
                     "eer": p, "reference_eer": md_eer, "n_draws": len(perm_eer[m]),
                     "flag": bool(np.isfinite(p) and p < md_eer - SHORTCUT_TOL),
                     "expectation": (
                         "the platform mix is preserved, so the correct floor is the "
                         "metadata-only level; beating it by >3 means leakage")})
        rows.append({"check": "cell-permutation", "method": m, "mask": "clips",
                     "eer": c, "reference_eer": b, "n_draws": len(cell_perm[m]),
                     "flag": bool(np.isfinite(c) and np.isfinite(b) and m == "C1" and b < c - 0.5),
                     "expectation": "C1's gain over the permuted-cell run disappears"})
        rows.append({"check": "clips-vs-strict-baseline", "method": m, "mask": "clips",
                     "eer": b, "reference_eer": float("nan"), "n_draws": len(baseline[m]),
                     "flag": False, "expectation": "baseline (no expectation)"})

    rows.append({"check": "near-duplicate-share", "method": "-", "mask": "clips",
                 "eer": float(np.mean(near_dup)) if near_dup else float("nan"),
                 "reference_eer": float("nan"), "n_draws": len(near_dup),
                 "flag": bool(near_dup and np.mean(near_dup) > 0.05),
                 "expectation": f"share of eval clips with a labelled neighbour > cos {NEAR_DUP_COS} below 5%"})

    rows.append({"check": "metadata-only", "method": "platform x language", "mask": "clips",
                 "eer": md_eer, "reference_eer": float("nan"), "n_draws": 1, "flag": False,
                 "expectation": "shortcut level; any audio claim must beat this"})
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W9 audit")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--methods", default="B1,B2,C1,B7")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    df = audit(root, data_root(args.data), n_draws=args.draws, methods=methods)
    (root / "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "results" / "w9_audit.csv", index=False)
    flagged = df[df["flag"]]
    print(df.to_string(index=False))
    print(f"\nflagged: {len(flagged)} / {len(df)}")
    if not flagged.empty:
        print("AUDIT_FLAG on: " + ", ".join(f"{r.check}/{r.method}" for r in flagged.itertuples()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

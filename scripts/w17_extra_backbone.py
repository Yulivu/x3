#!/usr/bin/env python3
"""W17 (extra backbones) - the paired second-backbone comparison.

W6 runs the same baseline suite per ``(backbone, spec)`` on the same frozen region
and the same draw seeds, so two runs can be paired *draw by draw*.  That is the
evidence S-F' and G11 need ("every claim on two backbones"): it is not enough for a
second backbone to reproduce the ranking on its own; the paired difference has to be
reported with its interval as well.

Only the sampling-level (a) interval is available here, because W6 stores per-draw
summaries rather than per-clip scores; that is recorded in the output.

Writes ``results/w17_extra_backbone.csv`` (means) and
``results/w17_extra_backbone_paired.csv`` (paired differences).
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

METRICS = ("pooled_eer", "within_macro_eer", "worst_platform_eer", "auc")

__all__ = ["pair_runs", "means_table", "bootstrap_mean_interval", "main"]


def bootstrap_mean_interval(d: np.ndarray, B: int = 2000, seed: int = 20261015
                            ) -> Tuple[float, float]:
    """The (a) interval: bootstrap of the *mean* difference over the draws."""
    d = np.asarray(d, float)
    if d.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    means = d[rng.integers(0, d.size, size=(int(B), d.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pair_runs(a: pd.DataFrame, b: pd.DataFrame, backbone_a: str, backbone_b: str,
              metrics: Sequence[str] = METRICS) -> pd.DataFrame:
    """Draw-paired ``backbone_a - backbone_b`` per (method, mask, metric)."""
    key = ["method", "mask", "draw"]
    rows: List[Dict[str, Any]] = []
    for metric in metrics:
        if metric not in a.columns or metric not in b.columns:
            continue
        left = a[key + [metric]].rename(columns={metric: "va"})
        right = b[key + [metric]].rename(columns={metric: "vb"})
        both = left.merge(right, on=key, how="inner")
        for (method, mask), part in both.groupby(["method", "mask"]):
            d = (part["va"] - part["vb"]).to_numpy(float)
            d = d[np.isfinite(d)]
            if d.size < 5:
                continue
            lo, hi = bootstrap_mean_interval(d)
            old_lo, old_hi = (float(np.percentile(d, 2.5)),
                              float(np.percentile(d, 97.5)))
            rows.append({"metric": metric, "method": method, "mask": mask,
                         "a": backbone_a, "b": backbone_b, "n_draws": int(d.size),
                         "mean_a": float(part["va"].mean()),
                         "mean_b": float(part["vb"].mean()),
                         "delta": float(d.mean()), "a_lo": lo, "a_hi": hi,
                         "old_lo": old_lo, "old_hi": old_hi,
                         "a_contains_zero": bool(lo <= 0.0 <= hi),
                         "b_interval_available": False,
                         "note": "W6 stores per-draw summaries, so the speaker-"
                                 "clustered (b) interval needs per-clip scores"})
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["metric", "method", "mask", "a", "b", "n_draws",
                                    "mean_a", "mean_b", "delta", "a_lo", "a_hi",
                                    "old_lo", "old_hi", "a_contains_zero",
                                    "b_interval_available", "note"])
    return out


def means_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (df[df["metric"] == "pooled_eer"]
            .pivot_table(index="method", columns="mask", values=["mean_a", "mean_b"])
            .round(3).reset_index())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W17 extra-backbone comparison")
    parser.add_argument("--root", default=".")
    parser.add_argument("--a-file", default="w6_draws_mms-1b_S3_v3_mms1b.csv")
    parser.add_argument("--b-file", default="w6_draws_xls-r-2b_S3_v3_xlsr2b.csv")
    parser.add_argument("--name-a", default="mms-1b|S3")
    parser.add_argument("--name-b", default="xls-r-2b|S3")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    pa = root / "results" / args.a_file
    pb = root / "results" / args.b_file
    if not pa.exists() or not pb.exists():
        missing = [str(p) for p in (pa, pb) if not p.exists()]
        print("missing: " + ", ".join(missing), file=sys.stderr)
        return 1
    a = pd.read_csv(pa)
    b = pd.read_csv(pb)
    paired = pair_runs(a, b, args.name_a, args.name_b)
    paired.to_csv(root / "results" / "w17_extra_backbone_paired.csv", index=False)
    means = means_table(paired)
    means.to_csv(root / "results" / "w17_extra_backbone.csv", index=False)
    print(means.to_string(index=False))
    print()
    sel = paired[paired["metric"] == "pooled_eer"]
    print(sel[["method", "mask", "mean_a", "mean_b", "delta", "a_lo", "a_hi",
               "a_contains_zero"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

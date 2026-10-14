#!/usr/bin/env python3
"""The v2 -> v3 comparison U1' has to report (NEW_TASKS 02).

Two things are compared, and they answer different questions:

* **the same draws, two interval rules** - ``results/u1_speaker_split.csv`` holds the
  v2 per-draw pooled EERs, so both the v1 rule (2.5/97.5 quantiles of the per-draw
  differences) and the new (a) bootstrap-of-the-mean can be computed on *identical*
  data.  The speaker-clustered (b) interval needs per-clip scores, which v2 never
  stored, so it is reported as "not computable" rather than guessed.
* **the same protocol, two regions** - v2 dev (68 speakers) vs v3 dev (122), for
  every method and mask.

Writes ``results/u1p_v2v3.csv`` (means) and ``results/u1p_v2_rule_compare.csv``.
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

from src.paired import two_intervals  # noqa: E402

PAIRS = (("B9", "B1"), ("C1", "B1"), ("B3", "B1"), ("B7", "B1"))
MASKS = ("seen", "unseen", "strict_draw")

__all__ = ["rule_compare", "region_compare", "main"]


def rule_compare(v2: pd.DataFrame, pairs: Sequence = PAIRS,
                 masks: Sequence[str] = MASKS) -> pd.DataFrame:
    """Old rule vs (a) on the v2 per-draw table (the (b) interval needs raw scores)."""
    rows: List[Dict[str, Any]] = []
    for mask in masks:
        for a, b in pairs:
            sa = v2[(v2["method"] == a) & (v2["mask"] == mask)].set_index("draw")["pooled_eer"]
            sb = v2[(v2["method"] == b) & (v2["mask"] == mask)].set_index("draw")["pooled_eer"]
            common = sa.index.intersection(sb.index)
            if len(common) < 5:
                continue
            d = (sa.loc[common].astype(float) - sb.loc[common].astype(float)).values
            old_lo, old_hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            rng = np.random.default_rng(20261015)
            means = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
            a_lo, a_hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
            rows.append({
                "region": "v2", "mask": mask, "method_a": a, "method_b": b,
                "n_draws": len(common), "delta": float(d.mean()),
                "old_lo": old_lo, "old_hi": old_hi,
                "old_contains_zero": bool(old_lo <= 0.0 <= old_hi),
                "a_lo": a_lo, "a_hi": a_hi,
                "a_contains_zero": bool(a_lo <= 0.0 <= a_hi),
                # the v1 rule said "no effect" whenever (a) is clearly non-zero
                "rules_disagree": bool((old_lo <= 0.0 <= old_hi) and not (a_lo <= 0.0 <= a_hi)),
                "b_lo": float("nan"), "b_hi": float("nan"),
                "b_not_computable": "v2 never stored per-clip scores",
            })
    return pd.DataFrame(rows)


def region_compare(v2: pd.DataFrame, v3: pd.DataFrame,
                   masks: Sequence[str] = MASKS) -> pd.DataFrame:
    """Method x mask: v2 dev (68 speakers) vs v3 dev (122 speakers)."""
    masks = list(masks)
    v2 = v2[v2["mask"].isin(masks)]
    v3 = v3[v3["mask"].isin(masks)]
    a = v2.groupby(["method", "mask"])["pooled_eer"].mean()
    b = v3[v3["k"] == 290].groupby(["method", "mask"])["pooled_eer"].mean()
    rows = []
    for (method, mask) in sorted(set(a.index) | set(b.index)):
        m2 = a.get((method, mask), np.nan)
        m3 = b.get((method, mask), np.nan)
        rows.append({"method": method, "mask": mask, "v2_pooled": float(m2),
                     "v3_pooled": float(m3), "delta_v3_minus_v2": float(m3 - m2)})
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 U1' v2 vs v3")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    v2 = pd.read_csv(root / "results" / "u1_speaker_split.csv")
    v3 = pd.read_csv(root / "results" / "u1p_speaker_split_v3.csv")
    rc = rule_compare(v2)
    rc.to_csv(root / "results" / "u1p_v2_rule_compare.csv", index=False)
    rcmp = region_compare(v2, v3)
    rcmp.to_csv(root / "results" / "u1p_v2v3.csv", index=False)
    print("v2 -> v3, pooled EER (k=290)\n" + rcmp.round(3).to_string(index=False))
    print("\nsame v2 draws, old rule vs (a)\n" + rc.round(3).to_string(index=False))
    n = int(rc["rules_disagree"].sum()) if not rc.empty else 0
    print(f"\nrule disagreements (old says 0 is inside, (a) says it is not): {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

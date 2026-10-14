"""Pins for the paired second-backbone comparison."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w17_extra_backbone as w17  # noqa: E402


def _run(offset=0.0, n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n):
        for mask in ("clips", "strict"):
            for method in ("B1", "C1"):
                rows.append({"method": method, "mask": mask, "draw": d,
                             "pooled_eer": 10.0 + offset + rng.normal(0, 0.5),
                             "within_macro_eer": 11.0 + offset,
                             "worst_platform_eer": 20.0 + offset, "auc": 0.9})
    return pd.DataFrame(rows)


def test_pairing_is_draw_by_draw_and_reports_the_sampling_interval():
    a = _run(offset=-1.0, seed=1)
    b = _run(offset=0.0, seed=2)
    paired = w17.pair_runs(a, b, "A", "B")
    row = paired[(paired["metric"] == "pooled_eer") & (paired["method"] == "B1")
                 & (paired["mask"] == "clips")].iloc[0]
    assert row["n_draws"] == 12
    assert abs(row["delta"] - (-1.0)) < 0.4               # a - b, draw-paired
    assert row["a_hi"] < 0.0                              # a is consistently better
    assert not bool(row["b_interval_available"])
    assert "per-clip scores" in row["note"]


def test_pairing_only_uses_draws_present_in_both_runs():
    a = _run(seed=3)
    b = _run(seed=4)
    b = b[b["draw"] < 10]                       # the second run stops two draws early
    paired = w17.pair_runs(a, b, "A", "B")
    row = paired[(paired["method"] == "B1") & (paired["mask"] == "clips")
                 & (paired["metric"] == "pooled_eer")].iloc[0]
    assert row["n_draws"] <= 10


def test_a_single_draw_is_not_reported_as_an_interval():
    a = _run(n=1, seed=5)
    b = _run(n=1, seed=6)
    paired = w17.pair_runs(a, b, "A", "B")
    assert paired.empty                                    # < 5 paired draws


def test_overlapping_runs_give_an_interval_that_contains_zero():
    a = _run(offset=0.0, seed=7)
    b = _run(offset=0.0, seed=8)
    paired = w17.pair_runs(a, b, "A", "B")
    row = paired[paired["metric"] == "pooled_eer"].iloc[0]
    assert row["a_contains_zero"] is True or row["a_contains_zero"] == True  # noqa: E712


def test_means_table_pivots_the_two_backbones():
    a = _run(offset=-1.0, seed=9)
    b = _run(offset=0.0, seed=10)
    paired = w17.pair_runs(a, b, "A", "B")
    means = w17.means_table(paired)
    cols = set(map(str, means.columns))
    assert any("mean_a" in c for c in cols) and any("mean_b" in c for c in cols)
    assert set(means["method"]) == {"B1", "C1"}

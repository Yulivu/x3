"""Pins for the v2 -> v3 comparison (means and the old-vs-(a) rule change)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import u1p_compare_v2v3 as cmp  # noqa: E402


def _v2_table(seed=0):
    """100 draws where B9 is consistently a little better than B1, and B3 worse."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(100):
        common = rng.normal(0.0, 0.8)
        for mask in ("seen", "unseen", "strict_draw"):
            b1 = 10.0 + common + rng.normal(0, 0.3)
            for method, delta in (("B1", 0.0), ("B9", -1.2), ("B3", 2.0),
                                  ("B7", -0.5), ("C1", 0.4)):
                rows.append({"method": method, "mask": mask, "draw": d,
                             "pooled_eer": b1 + delta + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


def test_rule_compare_shows_where_the_old_rule_hides_a_real_effect():
    v2 = _v2_table()
    rc = cmp.rule_compare(v2, pairs=[("B9", "B1")], masks=["seen"])
    row = rc.iloc[0]
    # the per-draw spread is much wider than the spread of the mean
    assert row["old_hi"] - row["old_lo"] > row["a_hi"] - row["a_lo"]
    assert not bool(row["a_contains_zero"])         # a consistent -1.2 point gain
    assert row["delta"] < -1.0
    assert row["b_not_computable"]
    assert bool(row["rules_disagree"]) == bool(row["old_contains_zero"])


def test_rule_compare_reports_a_zero_effect_as_zero_on_both_rules():
    v2 = _v2_table()
    rc = cmp.rule_compare(v2, pairs=[("B1", "B1")], masks=["seen"])
    row = rc.iloc[0]
    assert abs(row["delta"]) < 1e-9
    assert bool(row["old_contains_zero"]) and bool(row["a_contains_zero"])
    assert bool(row["rules_disagree"]) is False


def test_region_compare_aligns_methods_and_masks():
    v2 = _v2_table()
    v3 = v2.copy()
    v3["pooled_eer"] = v3["pooled_eer"] + 0.7
    v3["k"] = 290
    out = cmp.region_compare(v2, v3, masks=["seen", "unseen"])
    assert set(out["mask"]) == {"seen", "unseen"}
    assert set(out["method"]) == {"B1", "B9", "B3", "B7", "C1"}
    assert np.allclose(out["delta_v3_minus_v2"].dropna(), 0.7, atol=0.05)


def test_region_compare_skips_a_method_that_only_exists_in_one_region():
    v2 = _v2_table()
    v3 = v2.copy()
    v3["k"] = 290
    v3 = v3[v3["method"] != "C1"]
    out = cmp.region_compare(v2, v3, masks=["seen"])
    row = out[out["method"] == "C1"].iloc[0]
    assert np.isnan(row["v3_pooled"])

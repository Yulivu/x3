"""Pins for U5: the corruption operator and the degradation summary."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import u5_noise as u5  # noqa: E402


def test_corrupt_flips_exactly_the_requested_fraction_of_labelled_rows():
    y = np.tile([0, 1], 50)
    lab = np.zeros(len(y), bool)
    lab[:40] = True
    for rate in (0.05, 0.1, 0.2):
        got = u5.corrupt(y, lab, rate, seed=0)
        changed = int((got != y).sum())
        assert changed == int(round(rate * lab.sum())), (rate, changed)


def test_corrupt_never_touches_unlabelled_rows():
    y = np.tile([0, 1], 50)
    lab = np.zeros(len(y), bool)
    lab[:20] = True
    got = u5.corrupt(y, lab, 0.5, seed=3)
    assert np.array_equal(got[~lab], y[~lab])


def test_corrupt_is_seeded_and_zero_rate_is_the_identity():
    y = np.tile([0, 1], 50)
    lab = np.zeros(len(y), bool)
    lab[:30] = True
    assert np.array_equal(u5.corrupt(y, lab, 0.0), y)
    assert np.array_equal(u5.corrupt(y, lab, 0.2, seed=5),
                          u5.corrupt(y, lab, 0.2, seed=5))
    assert not np.array_equal(u5.corrupt(y, lab, 0.2, seed=5),
                              u5.corrupt(y, lab, 0.2, seed=6))


def test_summary_reports_degradation_against_the_clean_row():
    rows = []
    for method, base in (("B1", 10.0), ("C1", 9.0), ("M-ALP", 8.0)):
        for k in (100, 290):
            for rate, delta in ((0.0, 0.0), (0.05, 1.0), (0.1, 3.0), (0.2, 8.0)):
                rows.append({"method": method, "k": k, "noise_rate": rate,
                             "pooled_eer": base + delta + (1.0 if k == 290 else 0.0),
                             "draw": 0})
    summ = u5.summary(pd.DataFrame(rows))
    got = summ[(summ["method"] == "C1") & (summ["k"] == 100) & (summ["noise_rate"] == 0.1)]
    assert float(got.iloc[0]["degradation"]) == 3.0
    clean = summ[(summ["method"] == "C1") & (summ["k"] == 100) & (summ["noise_rate"] == 0)]
    assert float(clean.iloc[0]["degradation"]) == 0.0


def test_summary_handles_a_run_without_a_clean_row():
    df = pd.DataFrame([{"method": "B1", "k": 100, "noise_rate": 0.05,
                        "pooled_eer": 12.0, "draw": 0},
                       {"method": "B1", "k": 100, "noise_rate": 0.2,
                        "pooled_eer": 15.0, "draw": 0}])
    out = u5.summary(df)
    assert set(out["noise_rate"]) == {0.05, 0.2}
    assert float(out["degradation"].min()) == 0.0

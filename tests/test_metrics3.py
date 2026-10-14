"""Pins for src/metrics3.py against the three properties required by TASKBOOK hard rule 3."""

import numpy as np
import pytest

from src.metrics3 import (
    MIN_PER_CLASS,
    auc,
    eer,
    eer_legacy,
    eer_with_threshold,
    hter,
    platform_eers,
    summarize,
)

RNG = np.random.default_rng(20261015)


def _two_class(n=400, sep=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    s = rng.normal(size=n) + sep * y
    return y, s


def test_matches_legacy_on_random_data():
    worst = 0.0
    for seed in range(50):
        y, s = _two_class(n=200, sep=0.5, seed=seed)
        worst = max(worst, abs(eer(y, s) - eer_legacy(y, s)))
    assert worst < 0.1, f"max |eer - eer_legacy| = {worst}"


def test_constant_scores_are_fifty():
    y = np.array([0, 1] * 50)
    assert eer(y, np.full(y.size, 3.14)) == pytest.approx(50.0)
    assert eer(y, np.zeros(y.size)) == pytest.approx(50.0)
    assert eer_legacy(y, np.full(y.size, 3.14)) == pytest.approx(50.0)


def test_negated_scores_complement():
    for seed in range(20):
        y, s = _two_class(n=300, sep=0.8, seed=seed)
        assert eer(y, -s) == pytest.approx(100.0 - eer(y, s), abs=1e-9)


def test_perfect_and_anti_perfect():
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([RNG.normal(-3, 0.5, 100), RNG.normal(3, 0.5, 100)])
    assert eer(y, s) < 1e-9
    assert eer(y, -s) > 100 - 1e-9
    assert auc(y, s) > 0.999


def test_ties_are_stable():
    y = np.array([0, 1] * 100)
    s = np.tile(np.array([0.0, 1.0]), 100)  # every score appears many times
    assert eer(y, s) == eer(y, s.copy())
    assert 0.0 <= eer(y, s) <= 100.0


def test_threshold_convention_and_hter():
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([np.full(100, -1.0), np.full(100, 1.0)])
    e, thr = eer_with_threshold(y, s)
    assert e == pytest.approx(0.0)
    assert hter(y, s, thr) == pytest.approx(0.0)
    # deliberately bad threshold -> 50 % HTER
    assert hter(y, s, 5.0) == pytest.approx(50.0)


def test_single_class_is_neutral():
    y = np.ones(50, dtype=int)
    s = RNG.normal(size=50)
    assert eer(y, s) == pytest.approx(50.0)
    assert np.isnan(auc(y, s))
    assert np.isnan(hter(y, s, 0.0))


def test_platform_rules_and_macro():
    n_ok = MIN_PER_CLASS + 10
    y = np.concatenate([np.array([0] * n_ok + [1] * n_ok), np.array([0, 1] * 3)])
    plat = ["T"] * (2 * n_ok) + ["tiny"] * 6
    s = np.concatenate(
        [
            np.concatenate([RNG.normal(-2, 1, n_ok), RNG.normal(2, 1, n_ok)]),
            RNG.normal(size=6),
        ]
    )
    per = platform_eers(y, s, plat)
    assert set(per) == {"T"}  # "tiny" has too few clips per class
    row = summarize(y, s, plat, threshold=0.0)
    assert row["worst_platform"] == "T"
    assert row["within_macro_eer"] == pytest.approx(per["T"])
    assert row["eer_T"] == pytest.approx(per["T"])
    assert "eer_tiny" not in row
    assert row["n_eval"] == len(y)


def test_summarize_without_valid_platform_is_nan():
    y = np.array([0, 1, 0, 1])
    s = np.array([-1.0, 1.0, -0.5, 0.5])
    row = summarize(y, s, ["a"] * 4)
    assert np.isnan(row["within_macro_eer"])
    assert np.isnan(row["worst_platform_eer"])
    assert row["worst_platform"] == ""
    assert np.isnan(row["hter_pooled"])  # no threshold supplied

"""Pins for U10: Platt calibration, the calibrated average and the variant list."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import u10_fusion as u10  # noqa: E402


def test_platt_recovers_a_monotone_calibration():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    s = 2.0 * y + rng.normal(0, 1.0, size=400)
    a, b = u10.platt(s, y)
    assert a > 0.0                                  # higher score still means fake
    p = 1.0 / (1.0 + np.exp(-(a * s + b)))
    assert abs(float(np.mean(p[y == 1])) - 1.0) < 0.35
    assert float(np.mean(p[y == 0])) < float(np.mean(p[y == 1]))


def test_platt_is_identity_when_it_cannot_be_fitted():
    assert u10.platt([1.0, 2.0, 3.0], [1, 1, 1]) == (1.0, 0.0)


def test_fuse_members_averages_the_calibrated_logits():
    rng = np.random.default_rng(1)
    y = np.tile([0, 1], 100)
    s1 = 1.0 * y + rng.normal(0, 1, 200)
    s2 = 1.0 * y + rng.normal(0, 1, 200)
    fused = u10.fuse_members([(s1, s1, y), (s2, s2, y)])
    assert fused.shape == (200,)
    # averaging two noisy copies must beat either alone
    from src.metrics3 import eer

    assert eer(y, fused) < 0.5 * (eer(y, s1) + eer(y, s2))


def test_fuse_members_rejects_an_empty_member_list():
    try:
        u10.fuse_members([])
    except ValueError as exc:
        assert "no members" in str(exc)
    else:                                            # pragma: no cover
        raise AssertionError("an empty ensemble must raise")


def test_variant_defs_cover_singles_and_both_ensembles():
    got = u10.variant_defs(("a", "b"), ("S3", "S5"))
    names = [n for n, _ in got]
    assert names[:4] == ["a|S3", "a|S5", "b|S3", "b|S5"]
    assert "ENS2-S3" in names and "ENS2-S5" in names and "ENS2-mix" in names
    members = dict(got)
    assert members["ENS2-S3"] == ["a|S3", "b|S3"]
    assert members["ENS2-mix"] == ["a|S3", "a|S5", "b|S3", "b|S5"]

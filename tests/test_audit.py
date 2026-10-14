"""Pins for the W9 audit primitives."""

import numpy as np
import pytest

from src.audit import (
    audit_row,
    metadata_only_scores,
    nearest_neighbour_overlap,
    permute_cells,
    permute_labels_within,
    video_disjoint_mask,
)
from src.metrics3 import eer


def test_permute_labels_keeps_per_platform_counts():
    rng = np.random.default_rng(0)
    plat = np.array(["a"] * 100 + ["b"] * 60)
    y = (rng.random(160) < 0.7).astype(int)
    p = permute_labels_within(y, plat, seed=1)
    for name in ("a", "b"):
        m = plat == name
        assert int(p[m].sum()) == int(y[m].sum())
    assert not np.array_equal(p, y)
    assert set(np.unique(p)) <= {0, 1}


def test_permuted_labels_drive_eer_to_fifty():
    rng = np.random.default_rng(2)
    n = 2000
    y = (rng.random(n) < 0.5).astype(int)
    plat = np.array(["a" if i % 2 else "b" for i in range(n)])
    # a genuinely informative score
    s = y * 2.0 + rng.normal(0, 1.0, n)
    assert eer(y, s) < 30
    for seed in range(5):
        yp = permute_labels_within(y, plat, seed=seed)
        assert abs(eer(yp, s) - 50.0) < 3.0, f"seed {seed}: EER {eer(yp, s):.2f}"


def test_permute_cells_is_a_derangement_of_the_metadata():
    cell = np.array(["a"] * 50 + ["b"] * 50)
    p = permute_cells(cell, seed=3)
    assert set(p) == set(cell)
    assert not np.array_equal(p, cell)


def test_nearest_neighbour_overlap_detects_copies():
    Z = np.random.default_rng(0).normal(size=(50, 8))
    near = nearest_neighbour_overlap(Z, Z.copy())
    assert near["share"] == pytest.approx(1.0)
    far = nearest_neighbour_overlap(Z, np.random.default_rng(1).normal(size=(50, 8)))
    assert far["share"] < 0.1
    assert far["n_eval"] == 50


def test_video_disjoint_mask_excludes_labelled_videos():
    video = np.array(["v1", "v1", "v2", "v3", "v3"])
    labeled = np.array([True, False, False, True, False])
    ev = np.array([False, True, True, False, True])
    mask = video_disjoint_mask(video, labeled, ev)
    # eval clips are v1, v2, v3 -> v1 and v3 carry a label, only v2 survives
    assert mask.tolist() == [False, True, False]


def test_metadata_only_is_fifty_on_random_labels_and_strong_on_platform_labels():
    rng = np.random.default_rng(5)
    n = 1200
    plat = np.repeat(["a", "b", "c"], n // 3)
    lang = np.tile(["l0", "l1"], n // 2)
    y_random = (rng.random(n) < 0.5).astype(int)
    s = metadata_only_scores(plat, lang, y_random)
    assert abs(eer(y_random, s) - 50.0) < 8.0
    y_plat = np.isin(plat, ["c"]).astype(int)
    s2 = metadata_only_scores(plat, lang, y_plat)
    assert eer(y_plat, s2) < 20.0, "platform metadata should predict a platform-driven label"


def test_audit_row_flags_a_broken_permutation():
    y = np.array([0, 1] * 100)
    good = audit_row("label-permutation", y, np.zeros(200), expect_fifty=True)
    assert good["flag"] is False and good["eer"] == pytest.approx(50.0)
    # a score that still predicts the label after permutation is a leak: flag it
    bad = audit_row("label-permutation", y, y.astype(float) * 2 - 1, expect_fifty=True)
    assert bad["flag"] is True
    assert bad["eer"] == pytest.approx(0.0)

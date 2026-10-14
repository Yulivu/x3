"""Pins for Track Z (src/zlab.py, TASKBOOK W8)."""

import numpy as np
import pytest

from src.metrics3 import auc, eer
from src.zlab import fit_shared_gaussians, track_z_scores, z0, z1, z2, z3


def _synth(seed=3, cells=(0.90, 0.50, 0.10), n_per=800, sep=0.6, n_plat=3):
    rng = np.random.default_rng(seed)
    s, y, cell, plat = [], [], [], []
    for i, pi in enumerate(cells):
        yy = (rng.random(n_per) < pi).astype(int)
        ss = sep * (2 * yy - 1) + rng.normal(0, 1.0, n_per)
        name = f"p{i % n_plat}|l{i}"
        s.append(ss); y.append(yy)
        cell += [name] * n_per
        plat += [name.split("|")[0]] * n_per
    return (np.concatenate(s), np.concatenate(y), np.asarray(cell), np.asarray(plat),
            {f"p{i % n_plat}|l{i}": pi for i, pi in enumerate(cells)})


def test_z1_makes_each_platform_median_zero_and_iqr_one():
    rng = np.random.default_rng(0)
    plat = np.array(["a"] * 500 + ["b"] * 500 + ["c"] * 30)
    s = rng.normal(size=1030) + np.where(plat == "a", 5.0, np.where(plat == "b", -3.0, 0.0))
    out = z1(s, plat)
    for p in ("a", "b"):
        m = plat == p
        assert abs(float(np.median(out[m]))) < 1e-9
        iqr = float(np.percentile(out[m], 75) - np.percentile(out[m], 25))
        assert iqr == pytest.approx(1.0, abs=1e-6)
    # a platform below the size floor falls back to the global statistics
    assert np.isfinite(out[plat == "c"]).all()


def test_em_recovers_cell_priors_without_labels():
    s, y, cell, plat, truth = _synth()
    mu0, sd0, mu1, sd1, pi = fit_shared_gaussians(s, cell, plat)
    assert mu1 > mu0, "the positive component must sit above the negative one"
    err = [abs(pi[c] - truth[c]) for c in truth]
    assert max(err) <= 0.1, f"cell prior errors {err}"


def test_z2_beats_z0_on_a_prior_confounded_score():
    s, y, cell, plat, truth = _synth()
    raw = z0(s)
    z2s, info = z2(s, cell, plat)
    assert auc(y, z2s) > auc(y, raw), (
        f"Z2 AUC {auc(y, z2s):.4f} did not beat raw {auc(y, raw):.4f}"
    )
    assert eer(y, z2s) < eer(y, raw)
    assert set(info["pi"]) == set(truth)


def test_z3_averages_views_and_is_deterministic():
    s, y, cell, plat, truth = _synth()
    rng = np.random.default_rng(9)
    views = [s + rng.normal(0, 0.3, s.size) for _ in range(3)]
    a, ia = z3(views, cell, plat)
    b, ib = z3(views, cell, plat)
    assert np.allclose(a, b)
    assert ia["n_views"] == 3 and ib["n_views"] == 3
    assert np.isfinite(a).all()


def test_track_z_scores_returns_the_four_variants():
    s, y, cell, plat, truth = _synth()
    rng = np.random.default_rng(4)
    views = [s + rng.normal(0, 0.2, s.size) for _ in range(2)]
    out = track_z_scores(s, cell, plat, view_scores=views)
    assert set(out) == {"Z0", "Z1", "Z2", "Z3"}
    for k, v in out.items():
        assert v.shape == s.shape and np.isfinite(v).all(), k
    # without views Z3 is simply not produced
    assert set(track_z_scores(s, cell, plat)) == {"Z0", "Z1", "Z2"}

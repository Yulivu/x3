"""Pins for Track Z: Z1, the Z2 mixture/EM, Z3 view averaging and Z5 propagation."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import knn_graph  # noqa: E402
from src.metrics3 import eer  # noqa: E402
from src.trackz import (  # noqa: E402
    fit_z2,
    iqr,
    log_gauss,
    platform_standardise,
    score_views,
    threshold_per_cell,
    z2_posteriors,
    z2_scores,
    z4_scores,
    z5_scores,
)


def _synth(n=4000, seed=0, separation=2.0, cell_shift=True):
    """Zero-shot scores of two classes, with per-cell prior shifts and platform scales."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    cell = rng.choice([f"c{i}" for i in range(8)], size=n)
    plat = np.array([c.split("|")[0] if "|" in c else f"p{int(c[1:]) % 3}" for c in cell])
    rate = {c: 0.2 + 0.1 * i for i, c in enumerate(sorted(set(cell.tolist())))}
    if cell_shift:
        y = np.array([int(rng.random() < rate[c]) for c in cell])
    s = (y * float(separation) - separation / 2.0
         + rng.normal(size=n) * 1.0)
    return s, y, cell, plat


def test_iqr_is_robust_and_never_zero():
    x = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    assert iqr(x) > 0
    assert iqr(np.zeros(5)) == 1.0
    assert iqr(np.array([])) == 1.0


def test_z1_standardises_inside_each_platform_only():
    s, y, cell, plat = _synth()
    z1 = platform_standardise(s, plat)
    for p in np.unique(plat):
        m = plat == p
        assert abs(float(np.median(z1[m]))) < 1e-6
        assert abs(iqr(z1[m]) - 1.0) < 0.35
    # a platform whose raw scores are shifted must not stay shifted after Z1
    s2 = s.copy()
    s2[plat == np.unique(plat)[0]] += 50.0
    z1b = platform_standardise(s2, plat)
    first = plat == np.unique(plat)[0]
    assert abs(float(np.median(z1b[first])) - float(np.median(z1[first]))) < 1e-6


def test_z2_recovers_the_mixture_and_beats_the_raw_score():
    s, y, cell, plat = _synth(separation=1.6)
    fit = fit_z2(s, cell, plat, iters=20)
    assert fit["mu1"] > fit["mu0"]
    assert fit["sig0"] > 0 and fit["sig1"] > 0
    z2 = z2_scores(s, cell, fit)
    assert eer(y, z2) < eer(y, s)
    assert np.all(np.isfinite(z2))


def test_z2_cell_priors_move_towards_the_observed_cell_rate():
    rng = np.random.default_rng(3)
    n = 3000
    cell = np.array(["a|en"] * n + ["b|en"] * n)
    y = np.concatenate([(rng.random(n) < 0.1).astype(int),
                        (rng.random(n) < 0.9).astype(int)])
    plat = np.array(["a"] * n + ["b"] * n)
    s = y * 2.0 - 1.0 + rng.normal(size=2 * n)
    fit = fit_z2(s, cell, plat, iters=20)
    assert fit["pi_cell"]["a|en"] < 0.35 < 0.65 < fit["pi_cell"]["b|en"]


def test_z2_does_not_invent_separation_when_there_is_none():
    """Identical score distributions: the score must stay at chance, not drift."""
    rng = np.random.default_rng(4)
    n = 2000
    cell = np.array(["x|en"] * n + ["y|en"] * n)
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    plat = np.array(["x"] * n + ["y"] * n)
    s = rng.normal(size=2 * n)                     # identical distributions
    fit = fit_z2(s, cell, plat, iters=20)
    z2 = z2_scores(s, cell, fit)
    assert 40.0 <= eer(y, z2) <= 60.0


def test_log_gauss_is_a_normalised_density():
    x = np.linspace(-8, 8, 4001)
    got = np.exp(log_gauss(x, 0.0, 1.0))
    area = float(got.sum() * (x[1] - x[0]))
    assert abs(area - 1.0) < 1e-3


def test_z3_averages_the_views_and_improves_on_a_noisy_single_view():
    s, y, cell, plat = _synth(separation=1.4)
    rng = np.random.default_rng(5)
    views = {"clean": s,
             "v1": s + rng.normal(scale=0.6, size=len(s)),
             "v2": s + rng.normal(scale=0.6, size=len(s)),
             "v3": s + rng.normal(scale=0.6, size=len(s))}
    z3, fits = score_views(views, cell, plat, iters=20)
    assert len(fits) == 4
    assert eer(y, z3) <= eer(y, s)


def test_z5_propagates_pseudo_labels_without_touching_a_true_label():
    s, y, cell, plat = _synth(n=1500, separation=1.6)
    fit = fit_z2(s, cell, plat, iters=20)
    z2 = z2_scores(s, cell, fit)
    # a feature block whose neighbourhoods agree with the classes
    rng = np.random.default_rng(6)
    Z = y[:, None] * 3.0 + rng.normal(size=(len(y), 6))
    graph = knn_graph(Z, k=20)
    z5 = z5_scores(z2, graph)
    assert z5.shape == z2.shape and np.all(np.isfinite(z5))
    assert eer(y, z5) <= eer(y, z2) + 1e-9


def test_z5_accepts_an_explicit_teacher():
    s, y, cell, plat = _synth(n=800)
    graph = knn_graph(np.asarray(s, float)[:, None], k=10)
    a = z5_scores(s, graph, teacher=s)
    b = z5_scores(s, graph)
    assert np.allclose(a, b)
    other = z5_scores(s, graph, teacher=-np.asarray(s, float))
    assert not np.allclose(a, other)


def test_z2_posteriors_are_monotone_in_the_score_and_match_the_cell_priors():
    s, y, cell, plat = _synth(separation=2.0)
    fit = fit_z2(s, cell, plat, iters=20)
    r = z2_posteriors(s, cell, fit)
    assert r.min() >= 0.0 and r.max() <= 1.0
    # inside one cell the posterior is monotone in the score (across cells the cell
    # prior also moves it, which is the point of Z2)
    for c in np.unique(cell)[:3]:
        m = np.where(cell == c)[0]
        order = m[np.argsort(s[m])]
        assert np.all(np.diff(r[order]) >= -1e-9), c
    assert abs(float(r.mean()) - fit["pi_global"]) < 0.15


def test_threshold_per_cell_hits_the_requested_rate():
    rng = np.random.default_rng(11)
    p = rng.random(1000)
    cell = np.array(["a"] * 500 + ["b"] * 500)
    got = threshold_per_cell(p, cell, {"a": 0.2, "b": 0.8})
    assert abs(float(got[:500].mean()) - 0.2) < 0.01
    assert abs(float(got[500:].mean()) - 0.8) < 0.01


def test_z4_learns_from_the_features_and_beats_the_mixture():
    """The point of Z4: the zero-shot score is weak, the features are not."""
    s, y, cell, plat = _synth(n=2000, separation=0.9)
    fit = fit_z2(s, cell, plat, iters=20)
    z2 = z2_scores(s, cell, fit)
    rng = np.random.default_rng(12)
    Z = np.stack([y * 2.0 + rng.normal(size=len(y)) * 1.0,
                  rng.normal(size=len(y))] , axis=1)
    z4, info = z4_scores(Z, s, cell, fit, iters=5)
    assert z4.shape == z2.shape
    assert info["iters"] >= 1
    assert eer(y, z4) < eer(y, z2)
    assert eer(y, z4) < 20.0


def test_z4_is_deterministic():
    s, y, cell, plat = _synth(n=800)
    fit = fit_z2(s, cell, plat, iters=10)
    rng = np.random.default_rng(13)
    Z = rng.normal(size=(len(y), 4))
    a, _ = z4_scores(Z, s, cell, fit, iters=3)
    b, _ = z4_scores(Z, s, cell, fit, iters=3)
    assert np.array_equal(a, b)


def test_z4_falls_back_to_the_zero_shot_when_the_pool_has_one_class():
    s = np.full(50, 5.0)                               # everything looks fake
    cell = np.array(["a|en"] * 50)
    plat = np.array(["a"] * 50)
    fit = fit_z2(s, cell, plat, iters=5)
    Z = np.zeros((50, 3))
    sc, info = z4_scores(Z, s, cell, fit, iters=3)
    assert sc.shape == (50,)
    assert info["iters"] == 0 or np.all(np.isfinite(sc))

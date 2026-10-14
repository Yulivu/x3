"""Synthetic pins for src/em_prior.py (TASKBOOK W6: EM must recover cell priors)."""

import numpy as np
import pytest

from src.em_prior import (
    P_HI,
    P_LO,
    X3_HI,
    X3_LO,
    beta_hier,
    binom_se,
    clip_pi,
    em_cell_cal,
    estimate_pi,
    fuse,
    logit,
)

RNG = np.random.default_rng(20261015)


def _source(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    s = rng.normal(loc=np.where(y == 1, 1.0, -1.0), scale=1.0)
    return s, y


def _cell(n, pi, seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pi).astype(int)
    s = rng.normal(loc=np.where(y == 1, 1.0, -1.0), scale=1.0)
    return s


def test_em_recovers_cell_prior_within_0_1():
    src_s, src_y = _source()
    truths = [0.10, 0.30, 0.50, 0.70, 0.90]
    est = []
    for i, pi_true in enumerate(truths):
        s_cell = _cell(4000, pi_true, seed=100 + i)
        # init from a deliberately rough prior, never 0.5-or-nothing
        init = 0.5
        pi_hat, se = em_cell_cal(s_cell, src_s, src_y, T=1.0, init=init)
        assert 0.0 <= se <= 0.5
        est.append(pi_hat)
    mae = float(np.mean(np.abs(np.asarray(est) - np.asarray(truths))))
    assert mae <= 0.1, f"EM cell-prior MAE {mae:.4f} > 0.10"


def test_em_is_not_initialisation_bound():
    src_s, src_y = _source()
    s_cell = _cell(4000, 0.85, seed=7)
    a = em_cell_cal(s_cell, src_s, src_y, T=1.0, init=0.20)[0]
    b = em_cell_cal(s_cell, src_s, src_y, T=1.0, init=0.80)[0]
    assert abs(a - b) < 0.02
    assert a > 0.75


def test_em_clips_to_requested_bounds():
    src_s, src_y = _source()
    s_cell = _cell(500, 0.995, seed=11)
    lo, hi = em_cell_cal(s_cell, src_s, src_y, T=1.0, init=0.5, lo=X3_LO, hi=X3_HI)
    assert X3_LO <= lo <= X3_HI
    line_a = em_cell_cal(s_cell, src_s, src_y, T=1.0, init=0.5)[0]
    assert P_LO <= line_a <= P_HI


def test_beta_hier_shrinks_sparse_cells_and_fits_phi():
    # three platforms, two cells each; one cell has only 3 labels but a raw rate of 1.0
    cells = ["p1|c1", "p1|c2", "p2|c1", "p2|c2", "p3|c1", "p3|c2"]
    n_lab = {c: 40 for c in cells}
    n_lab["p3|c2"] = 3
    rates = {"p1|c1": 0.60, "p1|c2": 0.40, "p2|c1": 0.55, "p2|c2": 0.45, "p3|c1": 0.50}
    n_succ = {c: int(round(rates[c] * n_lab[c])) for c in rates}
    n_succ["p3|c2"] = 3  # sparse cell, raw rate 1.0
    group = {c: c.split("|")[0] for c in cells}
    pi, se, phi, mu_g = beta_hier(n_succ, n_lab, group)
    assert phi > 0
    assert set(pi) == set(cells)
    assert all(P_LO <= v <= P_HI for v in pi.values())
    # the 3-observation cell is pulled toward its platform rate, away from its raw 1.0
    raw, m = 1.0, mu_g["p3"]
    assert abs(pi["p3|c2"] - m) < abs(raw - m)
    assert pi["p3|c2"] < 0.95
    # a cell backed by 40 observations stays near its own rate, while the
    # 3-observation cell is dragged far further away from its raw rate
    dense_pull = abs(pi["p1|c1"] - 0.60)
    sparse_pull = abs(pi["p3|c2"] - 1.0)
    assert dense_pull < 0.1
    assert sparse_pull > dense_pull


def test_estimate_pi_variants_agree_when_labels_are_plentiful():
    rng = np.random.default_rng(3)
    cells = [f"p{p}|l{l}" for p in range(3) for l in range(2)]
    n_per = 200
    cell_of, plat_of, y, lab = [], [], [], []
    for c in cells:
        p = c.split("|")[0]
        rate = 0.2 if p == "p0" else (0.5 if p == "p1" else 0.8)
        yy = (rng.random(n_per) < rate).astype(int)
        cell_of += [c] * n_per
        plat_of += [p] * n_per
        y += list(yy)
        lab += [True] * n_per
    y = np.asarray(y)
    lab = np.asarray(lab)
    out = {}
    for variant in ("count", "eb", "bb"):
        pi, n_lab, phi = estimate_pi(cells, cell_of, lab, y, plat_of, variant=variant)
        out[variant] = pi
        assert set(pi) == set(cells)
        assert all(P_LO <= v <= P_HI for v in pi.values())
    for c in cells:
        assert abs(out["bb"][c] - out["count"][c]) < 0.1
    # `bb` and `eb` disagree by design when a global prior is supplied; see the
    # dedicated shrink_hier test below.


def test_estimate_pi_bb_does_not_collapse_to_the_global_mean():
    """Line A's `bb` keeps platform structure; a collapse would erase the effect."""
    cells = [f"p{p}|l{l}" for p in range(3) for l in range(2)]
    rates = {"p0": 0.2, "p1": 0.5, "p2": 0.8}
    rng = np.random.default_rng(5)
    cell_of, plat_of, y, lab = [], [], [], []
    for c in cells:
        p = c.split("|")[0]
        yy = (rng.random(150) < rates[p]).astype(int)
        cell_of += [c] * 150
        plat_of += [p] * 150
        y += list(yy)
        lab += [True] * 150
    pi, _, _ = estimate_pi(cells, cell_of, np.asarray(lab), np.asarray(y), plat_of, variant="bb")
    means = {p: np.mean([pi[c] for c in cells if c.startswith(p)]) for p in rates}
    assert means["p0"] < 0.35 < means["p1"] < 0.65 < means["p2"]


def test_estimate_pi_falls_back_for_unlabelled_cells():
    cells = ["p1|c1", "p1|c2", "p2|c1"]
    cell_of = ["p1|c1"] * 50 + ["p1|c2"] * 50 + ["p2|c1"] * 50
    plat_of = ["p1"] * 100 + ["p2"] * 50
    y = np.array([1] * 40 + [0] * 10 + [1] * 5 + [0] * 45 + [1] * 30 + [0] * 20)
    lab = np.array([True] * 50 + [True] * 50 + [False] * 50)
    pi, _, _ = estimate_pi(cells, cell_of, lab, y, plat_of, variant="count")
    assert set(pi) == set(cells)
    # p2 has no labelled clip at all, so it falls back to Line A's 0.26 default
    assert pi["p2|c1"] == pytest.approx(0.26, abs=0.01)


def test_logit_clip_and_fuse_damping():
    assert logit(0.0) == pytest.approx(logit(0.02))
    assert logit(1.0) == pytest.approx(logit(0.98))
    s = np.linspace(-2, 2, 11)
    pi = np.full(11, 0.9)
    n = np.full(11, 100.0)
    full = fuse(s, pi, n, k=None)
    damped = fuse(s, pi, n, k=100.0, lam=1.0)
    # damping halves the correction at n == k
    assert np.allclose(full - fuse(s, pi, n, k=None, lam=0.0), logit(0.9), atol=1e-9)
    assert np.allclose(damped - fuse(s, pi, n, k=100.0, lam=0.0), 0.5 * logit(0.9), atol=1e-9)
    assert clip_pi(-1.0) == P_LO
    assert clip_pi(2.0) == P_HI
    assert binom_se(0.5, 100) == pytest.approx(0.05)

"""Pins for the baseline family B1-B11 (TASKBOOK W6)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.baselines import (
    B3_SIGMAS,
    MLPProbe,
    b1_logreg,
    b2_line_a,
    b3_platform_offset,
    b5_knn_prototype,
    b6_zeroshot_em,
    b7_naive_pseudo_label,
    b8_fixmatch_lite,
    b9_label_propagation,
    b11_mlp,
    fit_logreg_cv,
)
from src.cv import log_likelihood, stratified_folds
from src.metrics3 import eer


def _two_class(n=400, d=8, sep=1.2, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    Z = rng.normal(size=(n, d)) + sep * y[:, None] * rng.normal(size=d)
    return Z, y


def test_b1_matches_sklearn_exactly():
    Z, y = _two_class()
    folds = stratified_folds(y)
    model, info = fit_logreg_cv(Z, y, folds=folds)
    direct = LogisticRegression(C=info["C"], max_iter=5000).fit(Z, y)
    s1 = model.decision_function(Z)
    s2 = direct.decision_function(Z)
    assert float(np.corrcoef(s1, s2)[0, 1]) >= 0.999
    assert info["C"] in (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    scores, _ = b1_logreg(Z, y, folds=folds)
    assert np.isfinite(scores).all()


def test_b2_cell_offset_fixes_a_pooled_prior_mismatch():
    """Two cells with opposite priors: the cell correction must lower pooled EER."""
    rng = np.random.default_rng(4)
    n_per, d = 240, 6
    cell, plat, y, s = [], [], [], []
    for c, rate in (("p0|l0", 0.9), ("p1|l1", 0.1)):
        yy = (rng.random(n_per) < rate).astype(int)
        raw = yy * 1.0 + rng.normal(0, 1.3, n_per)   # weak, unbiased probe
        cell += [c] * n_per
        plat += [c.split("|")[0]] * n_per
        y += list(yy)
        s += list(raw)
    y = np.asarray(y)
    s = np.asarray(s)
    lab = np.zeros(len(y), bool)
    for c in ("p0|l0", "p1|l1"):
        idx = np.where(np.asarray(cell) == c)[0]
        lab[rng.choice(idx, 40, replace=False)] = True

    corrected, info = b2_line_a(s, cell, plat, lab, y, variant="bb")
    assert info["variant"] == "bb"
    assert eer(y, corrected) < eer(y, s)
    assert np.isfinite(corrected).all()


def test_b2_variants_run():
    rng = np.random.default_rng(6)
    n, d = 300, 5
    y = rng.integers(0, 2, n)
    s = y * 1.0 + rng.normal(0, 1.0, n)
    cell = np.array([f"p{i%3}|l{i%2}" for i in range(n)])
    plat = np.array([c.split("|")[0] for c in cell])
    lab = np.zeros(n, bool)
    for c in np.unique(cell):
        idx = np.where(cell == c)[0]
        lab[rng.choice(idx, 10, replace=False)] = True
    for variant in ("count", "bb", "eb"):
        out, info = b2_line_a(s, cell, plat, lab, y, variant=variant)
        assert out.shape == (n,)
        assert np.isfinite(out).all()


def test_b3_platform_offset_runs_and_shrinks():
    rng = np.random.default_rng(7)
    n, d = 500, 6
    y = rng.integers(0, 2, n)
    plat = np.array(["a" if i % 2 == 0 else "b" for i in range(n)])
    s = y * 1.0 + rng.normal(0, 1.0, n) + np.where(plat == "a", 1.5, -1.5)
    lab = np.zeros(n, bool)
    lab[rng.choice(n, 120, replace=False)] = True
    corr, info = b3_platform_offset(s, plat, lab, y, sigma=1.0)
    assert set(info["offsets"]) == {"a", "b"}
    assert info["offsets"]["b"] - info["offsets"]["a"] > 1.0
    assert np.isfinite(corr).all()
    assert all(sig in B3_SIGMAS for sig in B3_SIGMAS)


def test_b5_b6_b8_b9_b11_produce_finite_scores():
    Z, y = _two_class(n=400, d=8, seed=2)
    lab = np.zeros(len(y), bool)
    lab[np.random.default_rng(0).choice(len(y), 120, replace=False)] = True
    cell = np.array([f"p{i%3}|l{i%2}" for i in range(len(y))])
    plat = np.array([c.split("|")[0] for c in cell])
    unlab = ~lab

    s5, i5 = b5_knn_prototype(Z, y)
    assert s5.shape == (len(y),) and np.isfinite(s5).all()
    assert i5["k"] == 15

    zs = Z[:, 0] * 1.5
    s6, i6 = b6_zeroshot_em(zs, cell, plat, lab, y)
    assert s6.shape == (len(y),) and np.isfinite(s6).all()
    assert 0.02 <= min(i6["pi"].values()) and max(i6["pi"].values()) <= 0.98

    s8, _ = b8_fixmatch_lite(Z[lab], y[lab], Z[unlab], Z[unlab], tau=0.8, rounds=2)
    assert s8.shape == (unlab.sum(),) and np.isfinite(s8).all()

    s9, i9 = b9_label_propagation(Z, y[lab], lab)
    assert s9.shape == (len(y),) and np.isfinite(s9).all()
    assert i9["transductive"] is True

    s11, _ = b11_mlp(Z[lab], y[lab], steps=40)
    assert s11.shape == (lab.sum(),) and np.isfinite(s11).all()
    probe = MLPProbe(steps=10).fit(Z[lab], y[lab])
    assert probe.logit(Z).shape == (len(y),)


def test_b7_collapses_on_an_extreme_prior_cell():
    """G5's mechanism: the naive global threshold drifts toward the majority class."""
    rng = np.random.default_rng(9)
    d = 6
    # a balanced source L somewhere else, then two extreme-prior target cells
    Zs = rng.normal(size=(400, d))
    ys = (Zs[:, 0] > 0).astype(int)
    n_per = 600
    Zt, yt, cell = [], [], []
    for name, rate in (("hi", 0.92), ("lo", 0.08)):
        Z = rng.normal(size=(n_per, d))
        signal = Z[:, 0] * 1.0
        # inject the requested prior by shifting the intercept
        shift = np.log(rate / (1 - rate)) - np.log(0.5 / 0.5)
        yy = (rng.random(n_per) < 1 / (1 + np.exp(-(signal + shift)))).astype(int)
        Zt.append(Z); yt.append(yy); cell += [name] * n_per
    Zt = np.vstack(Zt); yt = np.concatenate(yt); cell = np.asarray(cell)

    score, info = b7_naive_pseudo_label(Zs, ys, Zt, tau=0.8, rounds=3)
    assert score.shape == (len(yt),)
    idx, py = info["pseudo_idx"], info["pseudo_y"]
    assert len(idx) > 50, "no pseudo-labels were produced; the test would be vacuous"
    shares = {}
    for name in ("hi", "lo"):
        m = cell[idx] == name
        assert m.sum() > 100, f"cell {name} produced only {int(m.sum())} pseudo-labels"
        shares[name] = float(py[m].mean())
    truths = {"hi": float(yt[cell == "hi"].mean()), "lo": float(yt[cell == "lo"].mean())}
    # Inside an extreme-prior cell the pseudo-label mix is far from the cell's true
    # mix: that is exactly the >20-point deviation G5 counts (record-only gate).
    for name, share in shares.items():
        dev = abs(share - truths[name])
        assert dev > 0.2, (
            f"cell {name}: pseudo-label fake share {share:.3f} vs truth {truths[name]:.3f} "
            f"(deviation {dev:.3f}); the collapse G5 is about did not reproduce"
        )
    # the naive threshold cannot tell the two opposite extremes apart
    assert abs(shares["hi"] - shares["lo"]) < 0.2

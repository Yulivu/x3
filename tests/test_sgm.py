import numpy as np

from src.metrics3 import eer
from src.sgm import fit_1d, fit_sgm_lr


def _synthetic(seed=7):
    rng = np.random.default_rng(seed)
    n_spk, per, d = 24, 80, 12
    sp = np.repeat([f"s{i}" for i in range(n_spk)], per)
    plat = np.repeat([f"p{i%4}" for i in range(n_spk)], per)
    offset = np.repeat(rng.normal(0, 1.2, n_spk), per)
    prior = np.repeat(rng.uniform(.15, .85, n_spk), per)
    y = (rng.random(n_spk * per) < prior).astype(int)
    score = 2.0 * y + offset + rng.normal(0, .9, len(y))
    X = rng.normal(size=(len(y), d)); X[:, 0] += 1.8 * y + offset
    return X, score, sp, plat, y


def test_sgm_reduces_speaker_shift_and_keeps_polarity():
    X, score, sp, plat, y = _synthetic()
    fit = fit_1d(score, sp, plat)
    assert not fit.polarity_flip
    assert eer(y, fit.score) <= eer(y, score) - 2.0
    semi = np.full(len(y), -1); semi[::17] = y[::17]
    lr = fit_sgm_lr(X, score, sp, plat, semi, rounds=2)
    assert not lr.polarity_flip
    assert np.isfinite(lr.score).all()

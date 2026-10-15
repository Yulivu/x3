"""Speaker-conditional Gaussian-mixture adaptation (NEW_TASKS 05, §2.1).

The module deliberately separates the one-dimensional generative adaptation from
the experiment drivers.  It can therefore be tested without ML-ITW and used in
the Z (no labels), E0 (known-real enrolment), F (few labelled clips), and U
(labelled-source/unlabelled-target) tracks without changing its assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 0.02, 0.98)
    return np.log(p / (1.0 - p))


def _expit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def standardise(x: Sequence[float]) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / max(float(x.std()), EPS)


@dataclass
class SGMFit:
    posterior: np.ndarray
    score: np.ndarray
    offset: Dict[str, float]
    prior: Dict[str, float]
    mu0: float
    mu1: float
    sigma: float
    polarity_flip: bool


def fit_1d(
    score: Sequence[float], speaker: Sequence[object], platform: Sequence[object],
    labels: Optional[Sequence[int]] = None, *, kappa_offset: float = 20.0,
    kappa_prior: float = 20.0, iters: int = 30,
) -> SGMFit:
    """Fit the shared two-normal speaker-conditional mixture.

    ``labels`` uses ``-1`` for unknown, ``0`` for bona fide, and ``1`` for
    spoof.  Known labels are clamped in every E step.  This is the sole route by
    which an F/U driver may inject labels; Z passes all ``-1`` and E0 clamps only
    its enrolment clips to zero.
    """
    u = standardise(score)
    sp = np.asarray(speaker).astype(str)
    pl = np.asarray(platform).astype(str)
    n = len(u)
    if len(sp) != n or len(pl) != n:
        raise ValueError("score, speaker, and platform must have equal length")
    y = np.full(n, -1, int) if labels is None else np.asarray(labels, int).copy()
    if y.shape != (n,) or np.any((y < -1) | (y > 1)):
        raise ValueError("labels must be -1, 0, or 1")
    # The detector orientation is fixed before EM.  A later reversal is recorded
    # and callers can discard that draw rather than silently relabeling classes.
    pos, neg = u >= 0.0, u < 0.0
    mu1 = float(u[pos].mean()) if pos.any() else 0.5
    mu0 = float(u[neg].mean()) if neg.any() else -0.5
    sigma = max(float(u.std()), 0.25)
    speakers = np.unique(sp)
    platforms = np.unique(pl)
    offset = {s: 0.0 for s in speakers}
    prior = {s: 0.5 for s in speakers}
    r = np.where(y >= 0, y, (u >= 0).astype(float)).astype(float)

    for _ in range(int(iters)):
        o = np.array([offset[s] for s in sp])
        pi = np.array([prior[s] for s in sp])
        llr = _logit(pi) + ((u - o - mu0) ** 2 - (u - o - mu1) ** 2) / (2 * sigma**2)
        r = _expit(llr)
        r[y == 0] = 0.0
        r[y == 1] = 1.0
        w1, w0 = r, 1.0 - r
        n1, n0 = max(float(w1.sum()), EPS), max(float(w0.sum()), EPS)
        centered = u - o
        mu1 = float(np.dot(w1, centered) / n1)
        mu0 = float(np.dot(w0, centered) / n0)
        var = (np.dot(w1, (centered - mu1) ** 2) + np.dot(w0, (centered - mu0) ** 2)) / (n1 + n0)
        sigma = max(float(np.sqrt(var)), 0.05)
        for s in speakers:
            m = sp == s
            residual = u[m] - (r[m] * mu1 + (1.0 - r[m]) * mu0)
            offset[s] = float(residual.sum() / (m.sum() + float(kappa_offset)))
        global_pi = float(np.clip(r.mean(), 0.02, 0.98))
        pi_platform = {}
        for p in platforms:
            m = pl == p
            pi_platform[p] = float(np.clip((r[m].sum() + kappa_prior * global_pi) /
                                            (m.sum() + kappa_prior), 0.02, 0.98))
        for s in speakers:
            m = sp == s
            parent = pi_platform[pl[m][0]]
            prior[s] = float(np.clip((r[m].sum() + kappa_prior * parent) /
                                      (m.sum() + kappa_prior), 0.02, 0.98))
    o = np.array([offset[s] for s in sp])
    pi = np.array([prior[s] for s in sp])
    score_out = _logit(pi) + ((u - o - mu0) ** 2 - (u - o - mu1) ** 2) / (2 * sigma**2)
    r = _expit(score_out)
    r[y == 0], r[y == 1] = 0.0, 1.0
    return SGMFit(r, score_out, offset, prior, mu0, mu1, sigma, bool(mu1 < mu0))


def fit_sgm_lr(
    X: np.ndarray, zero_score: Sequence[float], speaker: Sequence[object], platform: Sequence[object],
    labels: Optional[Sequence[int]] = None, *, alpha: float = 0.5, rounds: int = 4,
    pca_dim: int = 64, kappa_offset: float = 20.0, kappa_prior: float = 20.0,
    prepared: Optional[np.ndarray] = None,
) -> SGMFit:
    """SGM-LR: alternate the 1-D mixture and a weighted ridge detector.

    Ridge is fitted to the continuous responsibilities, which is the specified
    soft-label objective.  Its output is only a logit-like ranking term; the
    mixture remains responsible for speaker offsets and priors.
    """
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[0] != len(zero_score):
        raise ValueError("X must be (n,d) and align with zero_score")
    # ``prepared`` lets a driver fit the unsupervised scaler+PCA once and reuse it
    # across draws.  Re-fitting it inside every draw is identical numerically (the
    # features do not change) but costs minutes per driver.
    Z = prepare_features(X, pca_dim) if prepared is None else np.asarray(prepared, float)
    labels_a = np.full(len(Z), -1, int) if labels is None else np.asarray(labels, int)
    base = fit_1d(zero_score, speaker, platform, labels_a, kappa_offset=kappa_offset,
                  kappa_prior=kappa_prior)
    fit = base
    for _ in range(int(rounds)):
        target = fit.posterior.copy()
        target[labels_a == 0], target[labels_a == 1] = 0.0, 1.0
        weight = np.abs(2.0 * target - 1.0)
        weight[labels_a >= 0] = 1.0
        reg = Ridge(alpha=1.0).fit(Z, target, sample_weight=np.maximum(weight, 0.02))
        detector = standardise(reg.predict(Z))
        mixed = float(alpha) * detector + (1.0 - float(alpha)) * standardise(zero_score)
        fit = fit_1d(mixed, speaker, platform, labels_a, kappa_offset=kappa_offset,
                     kappa_prior=kappa_prior)
    return fit


def prepare_features(X: np.ndarray, pca_dim: int = 64) -> np.ndarray:
    """Unsupervised standardise + PCA, the feature space every SGM-LR round uses."""
    X = np.asarray(X, float)
    Z = StandardScaler().fit_transform(X)
    dim = max(1, min(int(pca_dim), Z.shape[0] - 1, Z.shape[1]))
    return PCA(n_components=dim, random_state=20261015).fit_transform(Z)

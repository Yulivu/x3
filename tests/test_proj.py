"""Pins for the nuisance-subspace projection (U9 / N6)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.proj import (  # noqa: E402
    lda_directions,
    nuisance_basis,
    orthogonalise_against,
    orthonormalise,
    residualise,
)


def _speaker_platform_data(n_speakers=12, per_speaker=15, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    speaker, platform = [], []
    Z = []
    for s in range(n_speakers):
        centre = rng.normal(size=dim) * 3.0
        plat = "p0" if s % 2 == 0 else "p1"
        for _ in range(per_speaker):
            Z.append(centre + rng.normal(size=dim) * 0.3)
            speaker.append(f"s{s:02d}")
            platform.append(plat)
    return np.asarray(Z), np.asarray(speaker), np.asarray(platform)


def test_orthonormalise_returns_an_orthonormal_basis():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(4, 10))
    Q = orthonormalise(W)
    assert Q.shape[0] == 4
    assert np.allclose(Q @ Q.T, np.eye(4), atol=1e-10)
    # the row space is preserved
    assert np.allclose(np.linalg.matrix_rank(np.vstack([W, Q])), 4)


def test_orthonormalise_drops_a_dependent_row():
    W = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    assert orthonormalise(W).shape == (2, 2)


def test_lda_directions_separate_the_classes_they_were_fitted_on():
    Z, speaker, _ = _speaker_platform_data()
    W = lda_directions(Z, speaker, 6)
    assert W.shape == (6, Z.shape[1])
    assert np.allclose(W @ W.T, np.eye(6), atol=1e-8)
    proj = Z @ W.T
    # between-speaker spread survives in the projection, within-speaker spread stays small
    between = np.array([proj[speaker == s].mean(0) for s in np.unique(speaker)]).var(0).sum()
    within = np.mean([proj[speaker == s].var(0).sum() for s in np.unique(speaker)])
    assert between > 20 * within


def test_residualise_removes_the_subspace():
    Z, speaker, _ = _speaker_platform_data()
    W = lda_directions(Z, speaker, 4)
    R = residualise(Z, W)
    assert np.allclose(R @ W.T, 0.0, atol=1e-3)
    # a vector outside the subspace is only shortened, never re-created
    assert R.shape == Z.shape
    assert np.all(np.linalg.norm(R, axis=1) <= np.linalg.norm(Z, axis=1) + 1e-6)


def test_residualise_with_an_empty_basis_is_the_identity():
    Z, _, _ = _speaker_platform_data()
    empty = np.zeros((0, Z.shape[1]))
    assert np.array_equal(residualise(Z, empty), Z)


def test_orthogonalise_against_makes_the_platform_basis_disjoint():
    Z, speaker, platform = _speaker_platform_data()
    W_s = lda_directions(Z, speaker, 4)
    W_p = orthogonalise_against(lda_directions(Z, platform, 1), W_s)
    assert np.allclose(W_p @ W_s.T, 0.0, atol=1e-8)
    assert np.allclose(W_p @ W_p.T, np.eye(W_p.shape[0]), atol=1e-8)


def test_nuisance_basis_reports_what_it_actually_removed():
    Z, speaker, platform = _speaker_platform_data()
    W, info = nuisance_basis(Z, speaker, platform, m_s=4, m_p=2)
    # two platforms can only ever give one discriminant direction
    assert info["m_s_available"] == 4 and info["m_p_available"] == 1
    assert info["dim_removed"] == 5
    assert W.shape == (5, Z.shape[1])
    def ratio(X):
        W2 = lda_directions(X, speaker, 4)
        proj = X @ W2.T
        between = np.array([proj[speaker == s].mean(0)
                            for s in np.unique(speaker)]).var(0).sum()
        within = np.mean([proj[speaker == s].var(0).sum() for s in np.unique(speaker)])
        return between / max(within, 1e-12)

    R = residualise(Z, W)
    # removing rank-5 of 8 shrinks how much speaker identity a fresh LDA can find
    assert ratio(R) < ratio(Z) / 3.0


def test_m_zero_removes_nothing():
    Z, speaker, platform = _speaker_platform_data()
    W, info = nuisance_basis(Z, speaker, platform, m_s=0, m_p=0)
    assert W.shape == (0, Z.shape[1])
    assert info["dim_removed"] == 0
    assert np.array_equal(residualise(Z, W), Z)

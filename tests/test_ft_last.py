"""Pins for W13 (B10), reduced variant (no encoder needed)."""

import numpy as np
import pytest

from src.ft_last import FT_LRS, LastBlockHead, fit_b10, train_head
from src.metrics3 import auc


def _synth(seed=3, d=12, n=400, sep=1.1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, d)) + (2 * y[:, None] - 1) * sep
    return X, y


def test_train_head_learns_and_is_deterministic():
    X, y = _synth()
    a = train_head(X, y, lr=3e-5, epochs=5, seed=1)
    b = train_head(X, y, lr=3e-5, epochs=5, seed=1)
    assert np.allclose(a.logit(X), b.logit(X))
    c = train_head(X, y, lr=3e-5, epochs=5, seed=2)
    assert not np.allclose(a.logit(X), c.logit(X))


def test_reduced_b10_picks_a_declared_lr_and_produces_finite_scores():
    """The declared grid (5 epochs at <=3e-5) is deliberately conservative.

    B10 exists to be a *fair* head-only baseline, not a strong one, so this test
    pins the contract (grid member, variant label, finite scores) rather than an
    accuracy threshold.
    """
    X, y = _synth(seed=8, sep=1.3)
    head, info = fit_b10(X, y)
    assert info["lr"] in tuple(float(v) for v in FT_LRS)
    assert info["variant"] == "B10_REDUCED"
    assert set(info["cv"]) == {str(float(v)) for v in FT_LRS}
    s = head.logit(X)
    assert s.shape == (len(y),) and np.isfinite(s).all()
    assert auc(y, s) >= 0.5


def test_the_trainer_itself_can_fit_when_you_leave_the_declared_grid():
    """Sanity: a failure to learn above comes from the budget, not from the code."""
    X, y = _synth(seed=8, sep=1.3)
    head = train_head(X, y, lr=1e-3, epochs=60)
    assert auc(y, head.logit(X)) > 0.9


def test_last_block_head_has_a_layernorm_and_a_linear_tail():
    head = LastBlockHead(7)
    names = [type(m).__name__ for m in head.net]
    assert names[0] == "LayerNorm" and names[-1] == "Linear"
    assert head.logit(np.zeros((3, 7))).shape == (3,)

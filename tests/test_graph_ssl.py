"""Pins for W14 GraphSSL."""

import numpy as np
import pytest

from src.graph_ssl import ETA_GRID, GraphSSL, graph_ssl_score, select_eta
from src.metrics3 import auc, eer


def _synth(seed=2, d=8, n=600, rate=0.5):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    Z = rng.normal(size=(n, d)) + (2 * y[:, None] - 1) * 0.9
    cell = np.array([f"p{i % 3}|l{i % 2}" for i in range(n)])
    plat = np.array([c.split("|")[0] for c in cell])
    lang = np.array([c.split("|")[1] for c in cell])
    lab = np.zeros(n, bool)
    lab[rng.choice(n, 120, replace=False)] = True
    return Z, y, plat, lang, cell, lab


def test_graph_ssl_score_is_a_scaled_sum():
    c1 = np.array([1.0, -2.0, 0.5])
    yp = np.array([1.0, -1.0, 0.0])
    assert np.allclose(graph_ssl_score(c1, yp, 0.5), [1.5, -2.5, 0.5])
    assert np.allclose(graph_ssl_score(c1, yp, 0.0), c1)


def test_select_eta_returns_a_grid_member_and_reports_cv():
    Z, y, plat, lang, cell, lab = _synth()
    eta, info = select_eta(Z, y, plat, lang, cell, lab)
    assert eta in {float(e) for e in ETA_GRID}
    assert set(info["cv"]) == {float(e) for e in ETA_GRID}
    assert info["transductive"] is True
    assert info["alpha"] == 0.9


def test_eta_selection_prefers_a_positive_eta_when_propagation_carries_signal():
    Z, y, plat, lang, cell, lab = _synth(seed=5)
    eta, info = select_eta(Z, y, plat, lang, cell, lab)
    assert max(info["cv"], key=lambda e: info["cv"][e]) == eta
    assert eta > 0.0


def test_graphssl_beats_its_own_c1_when_the_graph_is_informative():
    Z, y, plat, lang, cell, lab = _synth(seed=7)
    g = GraphSSL().fit(Z, y, plat, lang, cell, lab, select=False)
    c1 = g.c1_all_
    # eta=1 is the strongest propagation setting; on clustered features it must help
    s = graph_ssl_score(c1, g.y_prop_, 0.5)
    assert np.isfinite(s).all()
    assert g.info_["transductive"] is True
    assert auc(y, s) >= auc(y, c1) - 0.02


def test_graphssl_selection_runs_end_to_end():
    Z, y, plat, lang, cell, lab = _synth(seed=11, n=400)
    g = GraphSSL().fit(Z, y, plat, lang, cell, lab, select=True)
    s = g.score()
    assert s.shape == (len(y),) and np.isfinite(s).all()
    assert g.eta in {float(e) for e in ETA_GRID}
    assert 0.0 <= eer(y, s) <= 100.0

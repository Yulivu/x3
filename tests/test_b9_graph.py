"""``knn_graph`` must reproduce B9 exactly (it only skips the neighbour search)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import b9_label_propagation, knn_graph  # noqa: E402


def _toy(n=120, d=8, k=6, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    y = (Z[:, 0] > 0).astype(int)
    lab = np.zeros(n, bool)
    lab[rng.choice(n, size=40, replace=False)] = True
    return Z, y, lab


def test_graph_reuse_matches_the_direct_implementation():
    Z, y, lab = _toy()
    direct, info_d = b9_label_propagation(Z, y[lab], lab, k=6, alpha=0.9)
    idx, vals = knn_graph(Z, k=6)
    reused, info_r = b9_label_propagation(Z, y[lab], lab, k=6, alpha=0.9,
                                          graph=(idx, vals))
    assert np.allclose(direct, reused, atol=1e-8)
    assert info_d == info_r


def test_graph_reuse_matches_for_a_smaller_k_than_the_cache():
    Z, y, lab = _toy()
    direct, _ = b9_label_propagation(Z, y[lab], lab, k=3, alpha=0.7)
    idx, vals = knn_graph(Z, k=6)          # cache with more neighbours than needed
    reused, info = b9_label_propagation(Z, y[lab], lab, k=3, alpha=0.7,
                                        graph=(idx, vals))
    assert np.allclose(direct, reused, atol=1e-8)
    assert info["k"] == 3


def test_knn_graph_excludes_self_and_is_sorted():
    Z, _, _ = _toy(n=40, k=5)
    idx, vals = knn_graph(Z, k=5)
    assert idx.shape == (40, 5) and vals.shape == (40, 5)
    for i in range(40):
        assert i not in set(idx[i].tolist())
        assert np.all(np.diff(vals[i]) <= 1e-12)       # descending


def test_knn_graph_handles_tiny_inputs():
    idx, vals = knn_graph(np.zeros((1, 3)), k=5)
    assert idx.shape == (1, 0) and vals.shape == (1, 0)

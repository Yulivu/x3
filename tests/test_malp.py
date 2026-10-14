"""Pins for the M-ALP learner and selector (U4)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import knn_graph  # noqa: E402
from src.malp import (  # noqa: E402
    DEDUP_COS,
    fuse,
    heat_start,
    hyperparam_grid,
    local_density,
    malp_scores,
    mass_normalise,
    propagate,
    select_batch,
    sparse_affinity,
)


def _toy(n=120, dim=8, k=8, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, dim))
    y = (Z[:, 0] > 0).astype(int)
    speaker = np.array([f"s{i // 6:02d}" for i in range(n)])
    idx, sim = knn_graph(Z, k=k)
    return Z, y, speaker, (idx, sim)


def test_affinity_is_symmetric_and_spectrally_bounded():
    Z, y, speaker, graph = _toy(n=80, k=6)
    S = sparse_affinity(graph[0], graph[1], 6)
    assert np.allclose((S - S.T).toarray(), 0.0, atol=1e-12)
    # D^-1/2 W D^-1/2 has spectral radius <= 1 by construction
    eig = np.linalg.eigvalsh(S.toarray())
    assert float(np.max(np.abs(eig))) <= 1.0 + 1e-9


def test_same_speaker_edges_can_be_upweighted():
    Z, y, speaker, graph = _toy(n=80, k=8)
    base = sparse_affinity(graph[0], graph[1], 8, speaker, w_s=1.0)
    up = sparse_affinity(graph[0], graph[1], 8, speaker, w_s=2.0)
    idx = graph[0][:, :8]
    same = np.zeros((len(Z), len(Z)), bool)
    same[np.arange(len(Z))[:, None], idx] = speaker[idx] == speaker[:, None]
    # same-speaker entries grow relative to different-speaker ones
    ratio_base = base.toarray()[same].sum() / max(base.toarray()[~same].sum(), 1e-9)
    ratio_up = up.toarray()[same].sum() / max(up.toarray()[~same].sum(), 1e-9)
    assert ratio_up > ratio_base


def test_propagation_matches_the_closed_form_solution():
    Z, y, speaker, graph = _toy(n=60, k=6)
    S = sparse_affinity(graph[0], graph[1], 6)
    lab = np.zeros(len(y), bool)
    lab[::4] = True
    got = propagate(S, y[lab], lab, alpha=0.9, iters=500)
    Y0 = np.zeros(len(y))
    Y0[lab] = 2.0 * y[lab] - 1.0
    want = np.linalg.solve(np.eye(len(y)) - 0.9 * S.toarray(), Y0)
    assert np.allclose(got, want, atol=1e-6)


def test_mass_normalisation_hits_the_requested_fake_rate():
    rng = np.random.default_rng(1)
    f = rng.normal(size=500)
    for pi in (0.1, 0.3, 0.5):
        got = mass_normalise(f, pi)
        assert abs(float((got > 0).mean()) - pi) < 0.01


def test_fuse_is_standardised_and_eta_scales_the_propagation_term():
    rng = np.random.default_rng(2)
    lr = rng.normal(size=300)
    f = rng.normal(size=300) * 5.0
    z = fuse(lr, f, 0.0)
    assert abs(float(z.mean())) < 1e-9 and abs(float(z.std()) - 1.0) < 1e-6
    both = fuse(lr, f, 1.0)
    # eta=0 only carries the LR term; raising eta hands weight to the propagation
    corr0 = float(np.corrcoef(z, f)[0, 1])
    corr1 = float(np.corrcoef(both, f)[0, 1])
    assert abs(corr0) < 0.05 and corr1 > 0.6
    assert float(np.corrcoef(both, lr)[0, 1]) < 0.8


def test_eta_zero_returns_b1_exactly():
    Z, y, speaker, graph = _toy()
    lab = np.zeros(len(y), bool)
    lab[::5] = True
    lr = np.asarray(Z @ np.ones(Z.shape[1]) * 0.1 + 0.3)
    got = malp_scores(Z, y, lab, graph, (10, 0.8, 1.0, 0.0), lr, speaker)
    assert np.array_equal(got, lr)


def test_hyperparameter_grid_respects_the_sixteen_config_budget():
    grid = hyperparam_grid()
    assert 0 < len(grid) <= 16
    assert len(set(grid)) == len(grid)
    assert any(c[3] == 0.0 for c in grid) and any(c[3] == 1.0 for c in grid)


def test_heat_start_is_deterministic_and_spreads_out():
    from src.malp import coverage

    # 20 well-separated blobs: a good warm start must touch every one of them
    rng = np.random.default_rng(3)
    centres = rng.normal(size=(20, 8)) * 8.0
    Z = np.repeat(centres, 10, axis=0) + rng.normal(size=(200, 8)) * 0.1
    a = heat_start(Z, n0=20, seed=7)
    b = heat_start(Z, n0=20, seed=7)
    assert np.array_equal(a, b)
    assert len(a) <= 20
    assert len(set(a.tolist())) == len(a)
    assert len(set((a // 10).tolist())) == 20        # one clip per blob
    rand = np.sort(np.random.default_rng(0).choice(200, 20, replace=False))
    pool = np.arange(len(Z))
    # ``coverage`` is the acquisition weight (1 = far from any label), so a better
    # warm start leaves a LOWER mean: fewer clips sit far from a labelled neighbour
    assert coverage(Z, a, pool).mean() < coverage(Z, rand, pool).mean()


def test_select_batch_prefers_uncertainty_and_dedups():
    Z, y, speaker, graph = _toy(n=100, k=8)
    scores = np.full(len(Z), 10.0)        # confident everywhere ...
    scores[0] = 0.0                       # ... except clip 0 (q = 0.5: most uncertain)
    scores[1] = 0.0
    Z = Z.copy()
    Z[1] = Z[0]                           # a near-duplicate of the uncertain clip
    idx, sim = knn_graph(Z, k=8)
    picks = select_batch(Z, scores, idx, sim, np.array([50]), np.arange(len(Z)),
                         batch=2, variant="U")
    assert picks[0] == 0
    assert 1 not in picks.tolist()        # the duplicate is filtered


def test_coverage_grows_as_clips_get_labelled():
    Z, y, speaker, _ = _toy(n=100, k=8)
    from src.malp import coverage

    far = coverage(Z, np.array([0]), np.arange(len(Z)))
    assert far.max() > 0.5
    many = coverage(Z, np.arange(50), np.arange(len(Z)))
    assert many.mean() <= far.mean()      # more labels -> better coverage everywhere

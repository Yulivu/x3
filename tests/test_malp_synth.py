"""The U4 synthetic check: on two classes whose clips carry speaker sub-clusters,
the M-ALP learner+selector (LP + density-weighted uncertainty) must beat uniform
label selection by >= 2 pooled EER points at a small budget.

What the synthetic does and does not show
-----------------------------------------
The gain is real and large when the evaluation clips come from speakers that ``L``
also covers (the ``clips`` mask, which 追加 02 makes one of the two required U4
masks): propagation walks the speaker sub-cluster and labels the siblings, so the
same 100 labels buy far more.  The first half of this file pins that claim across
several seeds.

The second half pins the *limit* of the claim: with the evaluation speakers held
out entirely (``strict_draw``), the same selector has no reliable advantage,
because the graph can only connect the new speakers to ``L`` through class
proximity, not through identity.  That is a fact about the mechanism, so it is
asserted instead of being hidden - the real decision has to come from the dev
numbers, not from this file.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import fit_logreg_cv, knn_graph  # noqa: E402
from src.cv import stratified_folds  # noqa: E402
from src.malp import (  # noqa: E402
    heat_start,
    hyperparam_grid,
    malp_scores,
    select_batch,
    select_hyperparams,
    sparse_affinity,
)
from src.metrics3 import eer  # noqa: E402


def synth(n_speakers=30, per_speaker=40, dim=12, class_gap=3.0, speaker_sd=1.5,
          noise=0.7, seed=0):
    """Two classes, each made of speaker sub-clusters: one clip never leaves its speaker."""
    rng = np.random.default_rng(seed)
    direction = np.zeros(dim)
    direction[0] = 1.0
    Z, y, speaker = [], [], []
    for s in range(n_speakers):
        label = s % 2
        offset = rng.normal(size=dim) * speaker_sd
        centre = class_gap * label * direction
        for _ in range(per_speaker):
            Z.append(centre + offset + rng.normal(size=dim) * noise)
            y.append(label)
            speaker.append(f"s{s:02d}")
    return (np.asarray(Z, np.float32), np.asarray(y, int), np.asarray(speaker))


def clip_split(y, n, seed=0, frac=0.25):
    """Held-out *clips* (the ``clips`` mask): every speaker stays in the pool."""
    rng = np.random.default_rng(1000 + seed)
    ev = rng.choice(n, size=int(n * frac), replace=False)
    return np.setdiff1d(np.arange(n), ev), ev


def speaker_split(speaker, n_eval=10):
    """Held-out *speakers* (the ``strict_draw`` mask)."""
    uniq = np.unique(speaker)
    ev_spk = set(uniq[-n_eval:].tolist())
    ev = np.array([s in ev_spk for s in speaker])
    return np.where(~ev)[0], np.where(ev)[0]


def uniform_pick(pool, k, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=min(k, len(pool)), replace=False))


def malp_pick(Z, y, pool, k, graph, speaker, n0=60, batches=4, variant="full", seed=0):
    """Heat start, then ``u * rho * c`` rounds, refitting the fused model each round."""
    n0 = int(min(n0, k))
    picked = [int(pool[i]) for i in heat_start(Z[pool], n0=n0, seed=seed)]
    batch = max(1, (int(k) - len(picked)) // int(batches))
    for _ in range(int(batches)):
        if len(picked) >= k:
            break
        mask = np.zeros(len(y), bool)
        mask[np.asarray(picked, int)] = True
        cfg, _ = select_hyperparams(Z, y, mask, graph, stratified_folds(y[picked]),
                                    speaker=speaker, grid=hyperparam_grid(8))
        lr, _ = fit_logreg_cv(Z[mask], y[mask])
        sc = malp_scores(Z, y, mask, graph, cfg, lr.decision_function(Z), speaker)
        new = select_batch(Z, sc, graph[0], graph[1], np.asarray(picked, int),
                           pool[~np.isin(pool, picked)], batch, variant=variant,
                           speaker=speaker)
        if new.size == 0:
            break
        picked.extend(new.tolist())
    return np.asarray(sorted(set(picked))[:k], int)


def score_on(Z, y, speaker, graph, labelled, ev, cfg=None):
    lr, _ = fit_logreg_cv(Z[labelled], y[labelled])
    logit = lr.decision_function(Z)
    if cfg is None:
        return eer(y[ev], logit[ev])
    mask = np.zeros(len(y), bool)
    mask[labelled] = True
    return eer(y[ev], malp_scores(Z, y, mask, graph, cfg, logit, speaker)[ev])


def _gains(seeds, eval_kind, variant="full", k=100):
    out = []
    for seed in seeds:
        Z, y, speaker = synth(seed=seed)
        graph = knn_graph(Z, k=30)
        pool, ev = (clip_split(y, len(y), seed=seed) if eval_kind == "clips"
                    else speaker_split(speaker))
        uniform = uniform_pick(pool, k, seed=seed)
        picked = malp_pick(Z, y, pool, k, graph, speaker, variant=variant, seed=seed)
        mask = np.zeros(len(y), bool)
        mask[picked] = True
        cfg, _ = select_hyperparams(Z, y, mask, graph, stratified_folds(y[picked]),
                                    speaker=speaker)
        out.append(score_on(Z, y, speaker, graph, uniform, ev)
                   - score_on(Z, y, speaker, graph, picked, ev, cfg=cfg))
    return np.asarray(out)


def test_malp_beats_uniform_by_two_points_on_the_clips_mask():
    gains = _gains(seeds=(0, 1, 2), eval_kind="clips")
    assert gains.min() >= 2.0, gains
    assert gains.mean() >= 5.0, gains


def test_the_propagation_term_is_what_carries_the_gain_on_the_clips_mask():
    """Same selection, learner forced to pure LR (eta = 0): the gain disappears."""
    Z, y, speaker = synth(seed=1)
    graph = knn_graph(Z, k=30)
    pool, ev = clip_split(y, len(y), seed=1)
    picked = malp_pick(Z, y, pool, 100, graph, speaker)
    fused = score_on(Z, y, speaker, graph, picked, ev, cfg=(10, 0.99, 2.0, 1.0))
    lr_only = score_on(Z, y, speaker, graph, picked, ev)
    assert lr_only > fused + 5.0, (lr_only, fused)


def test_the_gain_does_not_survive_a_full_speaker_holdout():
    """The honest limit: an unseen evaluation speaker cannot be filled in by the graph."""
    gains = _gains(seeds=(0, 1, 2), eval_kind="speakers")
    assert gains.min() < 2.0, gains


def test_the_selector_alone_is_not_worse_than_uniform_with_one_learner():
    Z, y, speaker = synth(seed=3)
    graph = knn_graph(Z, k=30)
    pool, ev = clip_split(y, len(y), seed=3)
    cfg = (10, 0.9, 1.0, 0.5)
    uniform = uniform_pick(pool, 100, seed=3)
    picked = malp_pick(Z, y, pool, 100, graph, speaker, variant="full", seed=3)
    assert (score_on(Z, y, speaker, graph, picked, ev, cfg=cfg)
            <= score_on(Z, y, speaker, graph, uniform, ev, cfg=cfg) + 1.0)


def test_the_graph_used_by_the_synthetic_check_is_symmetric_with_unit_radius():
    Z, y, speaker = synth(n_speakers=8, per_speaker=10)
    graph = knn_graph(Z, k=10)
    S = sparse_affinity(graph[0], graph[1], 10, speaker, w_s=2.0)
    eig = np.linalg.eigvalsh(S.toarray())
    assert float(np.max(np.abs(eig))) <= 1.0 + 1e-9
    assert np.allclose((S - S.T).toarray(), 0.0, atol=1e-12)

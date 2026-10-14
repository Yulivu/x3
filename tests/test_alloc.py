"""Pins for C3 (CASA) and the comparison strategies (TASKBOOK W12)."""

import numpy as np
import pytest

from src.alloc import CASA, STRATEGIES, allocate, bernoulli_entropy


def _pool(seed=5, d=10, sizes=(900, 700, 500, 80, 60), rates=(0.9, 0.6, 0.5, 0.2, 0.1)):
    rng = np.random.default_rng(seed)
    Z, y, cell, plat, lang = [], [], [], [], []
    for i, (n, r) in enumerate(zip(sizes, rates)):
        yy = (rng.random(n) < r).astype(int)
        Zc = rng.normal(size=(n, d)) + (2 * yy[:, None] - 1) * 0.8
        name = f"p{i}|l{i}"
        Z.append(Zc); y.append(yy)
        cell += [name] * n
        plat += [name.split("|")[0]] * n
        lang += [name.split("|")[1]] * n
    return (np.vstack(Z), np.concatenate(y), np.asarray(cell), np.asarray(plat), np.asarray(lang))


def test_entropy_shape():
    q = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    h = bernoulli_entropy(q)
    assert np.argmax(h) == 2
    assert h[2] == pytest.approx(np.log(2), abs=1e-6)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_returns_a_unique_batch_under_the_cap(strategy):
    Z, y, cell, plat, lang = _pool()
    rng = np.random.default_rng(0)
    unlab = np.arange(len(cell))
    q = np.full(len(cell), 0.5)
    pos = {c: 0 for c in np.unique(cell)}
    neg = {c: 0 for c in np.unique(cell)}
    picks = allocate(
        strategy, Z_pool=Z, cell_pool=cell, unlab_idx=unlab, q_pool=q,
        pos=pos, neg=neg, batch=100, rng=rng,
    )
    assert picks.size == 100, f"{strategy} returned {picks.size} clips"
    assert len(set(picks.tolist())) == 100
    counts = {c: int((cell[picks] == c).sum()) for c in np.unique(cell)}
    cap = max(1, int(np.ceil(2.0 * 100 / len(np.unique(cell)))))
    assert all(v <= cap for v in counts.values()), f"{strategy} broke the per-cell cap: {counts}"


def test_casa_favours_the_cell_we_are_least_certain_about():
    """V_c shrinks as a cell accumulates labels, so CASA should over-sample the
    equal-sized cell that already has far fewer labels, relative to proportional."""
    rng = np.random.default_rng(1)
    n_per, d = 400, 6
    Zc = rng.normal(size=(2 * n_per, d))
    cell = np.array(["rich"] * n_per + ["poor"] * n_per)
    q = np.full(2 * n_per, 0.5)
    # "rich" already carries 60 labels, "poor" carries none
    pos = {"rich": 30, "poor": 0}
    neg = {"rich": 30, "poor": 0}
    unlab = np.concatenate([np.arange(60, n_per), np.arange(n_per, 2 * n_per)])
    args = dict(Z_pool=Zc, cell_pool=cell, unlab_idx=unlab, q_pool=q, pos=pos, neg=neg, batch=100)
    p_pick = allocate("P", rng=np.random.default_rng(2), **args)
    casa_pick = allocate("CASA", rng=np.random.default_rng(2), rho=1.0, **args)
    assert int((cell[casa_pick] == "poor").sum()) > int((cell[p_pick] == "poor").sum())


def test_dedup_drops_near_duplicates():
    rng = np.random.default_rng(3)
    # 40 copies of the same direction plus 60 spread out; entropy is tied so the
    # tie-break order decides, and the filter must stop the copies filling the batch
    base = np.zeros((100, 4))
    base[:40] = 1.0
    base[40:] = rng.normal(size=(60, 4))
    Z = base
    cell = np.array(["c"] * 100)
    q = np.full(100, 0.5)
    picks = allocate(
        "E", Z_pool=Z, cell_pool=cell, unlab_idx=np.arange(100), q_pool=q,
        pos={"c": 0}, neg={"c": 0}, batch=30, rng=rng, cap=30,
    )
    from_dupes = int((picks < 40).sum())
    assert from_dupes <= 3, f"{from_dupes} near-duplicate clips survived the filter"


def test_casa_run_is_deterministic_and_fills_the_budget():
    Z, y, cell, plat, lang = _pool()
    for strategy in ("CASA", "U"):
        a = CASA(k=100, strategy=strategy, seed=7).run(Z, y, plat, lang, cell)
        b = CASA(k=100, strategy=strategy, seed=7).run(Z, y, plat, lang, cell)
        assert a["selected"].tolist() == b["selected"].tolist()
        assert a["selected"].size == 100
        assert a["strategy"] == strategy
        assert len(a["history"]) >= 2
        assert a["history"][-1]["n_selected"] >= a["history"][0]["n_selected"]


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        CASA(strategy="nope")
    with pytest.raises(ValueError):
        allocate(
            "nope", Z_pool=np.zeros((4, 2)), cell_pool=np.array(["a"] * 4),
            unlab_idx=np.arange(4), q_pool=np.zeros(4), pos={}, neg={}, batch=1,
            rng=np.random.default_rng(0),
        )

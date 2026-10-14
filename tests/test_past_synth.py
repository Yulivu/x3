"""Synthetic pin for C2 (PAST), TASKBOOK W11.

Generator: class-conditional (``y ~ Bern(pi_c)``, ``x | y ~ N(+-m, I)``) so the
detector LLR is well specified and the EM's target is unambiguous.  Cells carry
extreme priors (0.90 ... 0.06) and 10 labels each.

Two numbers must hold on this data: naive thresholded pseudo-labelling (B7) is
more than 20 points off on the cell prior, and PAST's EM prior is within 10
points of the truth.
"""

import numpy as np
import pytest

from src.baselines import b7_naive_pseudo_label
from src.past import PAST, estimate_cell_prior_em


def _synth(seed=13, d=8, cells=(0.90, 0.75, 0.50, 0.30, 0.12, 0.06),
           n_pool=600, n_lab=10, n_plat=3, sep=1.0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    m = sep * u
    Z, y, cell, plat, lang = [], [], [], [], []
    for i, pi in enumerate(cells):
        yy = (rng.random(n_pool) < pi).astype(int)
        Zc = rng.normal(size=(n_pool, d)) + (2 * yy[:, None] - 1) * m
        name = f"p{i % n_plat}|l{i}"
        Z.append(Zc)
        y.append(yy)
        cell += [name] * n_pool
        plat += [name.split("|")[0]] * n_pool
        lang += [name.split("|")[1]] * n_pool
    Z = np.vstack(Z)
    y = np.concatenate(y)
    cell = np.asarray(cell)
    plat = np.asarray(plat)
    lang = np.asarray(lang)

    lab = np.zeros(len(y), bool)
    rng2 = np.random.default_rng(seed + 1)
    for name in np.unique(cell):
        idx = np.where(cell == name)[0]
        lab[rng2.choice(idx, n_lab, replace=False)] = True
    truth = {name: float(y[cell == name].mean()) for name in np.unique(cell)}
    llr = 2.0 * (Z @ m)  # the exact LLR of this generator
    return dict(Z=Z, y=y, cell=cell, plat=plat, lang=lang, lab=lab, truth=truth, llr=llr)


def test_em_prior_recovers_extreme_cells():
    """With the exact LLR the EM must land within 10 points on every cell."""
    data = _synth()
    lab = data["lab"]
    pi, _, _ = estimate_cell_prior_em(
        data["llr"][~lab], data["cell"][~lab], data["plat"][~lab],
        data["y"][lab], data["cell"][lab], data["plat"][lab], kappa=10.0,
    )
    worst = max(abs(pi[c] - data["truth"][c]) for c in pi)
    mae = float(np.mean([abs(pi[c] - data["truth"][c]) for c in pi]))
    assert worst <= 0.1, f"worst cell-prior error {worst:.3f}; estimates {pi}"
    assert mae <= 0.1


def test_past_beats_naive_pseudo_label_on_cell_prior_error():
    data = _synth()
    lab = data["lab"]
    unlab = ~lab

    # B7's implied cell prior = the fake share of its pseudo-labels
    _, info = b7_naive_pseudo_label(
        data["Z"][lab], data["y"][lab], data["Z"][unlab], tau=0.8, rounds=3
    )
    idx, py = info["pseudo_idx"], info["pseudo_y"]
    cells_un = data["cell"][unlab]
    b7_err = [
        abs(float(py[cells_un[idx] == c].mean()) - data["truth"][c])
        for c in np.unique(cells_un)
        if (cells_un[idx] == c).sum() >= 20
    ]
    assert b7_err and max(b7_err) > 0.2, f"B7 did not collapse enough to be a contrast: {b7_err}"

    past = PAST(rounds=3, kappa=10.0, sigma_p=0.5, sigma_gamma=0.5, lam=1.0).fit(
        data["Z"][lab], data["y"][lab], data["plat"][lab], data["lang"][lab], data["cell"][lab],
        data["Z"][unlab], data["plat"][unlab], data["lang"][unlab], data["cell"][unlab],
    )
    past_err = [abs(past.pi_[c] - data["truth"][c]) for c in past.pi_]
    assert float(np.mean(past_err)) <= 0.1, (
        f"PAST cell-prior MAE {np.mean(past_err):.3f} > 0.10; per cell {past_err}"
    )
    assert max(past_err) <= 0.2, f"PAST worst cell-prior error {max(past_err):.3f}"


def test_past_quota_ramp_and_views():
    data = _synth(seed=17, cells=(0.85, 0.15), n_pool=300, n_lab=10)
    lab = data["lab"]
    unlab = ~lab
    rng = np.random.default_rng(0)
    views = [data["Z"][unlab] + rng.normal(0, 0.05, (int(unlab.sum()), data["Z"].shape[1]))]
    past = PAST(rounds=2, lambda_u=0.1).fit(
        data["Z"][lab], data["y"][lab], data["plat"][lab], data["lang"][lab], data["cell"][lab],
        data["Z"][unlab], data["plat"][unlab], data["lang"][unlab], data["cell"][unlab],
        views_unlab=views,
    )
    assert len(past.history_) == 2
    assert past.history_[1]["n_selected"] >= past.history_[0]["n_selected"]
    s = past.predict_logit(data["Z"], data["plat"], data["lang"], data["cell"])
    assert s.shape == (len(data["y"]),) and np.isfinite(s).all()

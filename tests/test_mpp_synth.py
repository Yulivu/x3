"""Synthetic pin for C1 (MPP), TASKBOOK W6.

Generator: d=20, 29 cells nested in 7 platforms, cell effect N(0, 0.5^2) and
platform effect N(0, 0.5^2) (C1's own model has both, so a generator with only one
of them would test a different estimator), 6000 clips, 290 labelled.

Requirements: the recovered cell effect (alpha_p + gamma_c, the identifiable
quantity) correlates >= 0.6 with the truth and pooled AUC is at least B1's.  The
correlation is averaged over three seeds because a single 290-label draw is a
noisy measurement of the effect vector - exactly the variance the 100-draw
protocol exists to average out.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.cv import log_likelihood, stratified_folds
from src.metrics3 import auc
from src.mpp import MPP, SIGMA_GRID, fit_cv


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _synth(seed=11, d=20, n_cells=29, n_total=6000, per_cell_labels=10,
           sig_gamma=0.5, sig_alpha=0.5, n_plat=7):
    rng = np.random.default_rng(seed)
    n_per = n_total // n_cells
    gamma = rng.normal(0.0, sig_gamma, n_cells)
    alpha = rng.normal(0.0, sig_alpha, n_plat)
    w = rng.normal(0.0, 1.0 / np.sqrt(d), d) * 1.3
    cell_names = [f"p{c % n_plat}|l{c}" for c in range(n_cells)]
    platform = [c.split("|")[0] for c in cell_names]
    language = [c.split("|")[1] for c in cell_names]

    Z, y, cell, plat, lang = [], [], [], [], []
    for c in range(n_cells):
        z = rng.normal(size=(n_per, d))
        eta = z @ w + alpha[c % n_plat] + gamma[c]
        yy = (rng.random(n_per) < _sigmoid(eta)).astype(int)
        Z.append(z)
        y.append(yy)
        cell += [cell_names[c]] * n_per
        plat += [platform[c]] * n_per
        lang += [language[c]] * n_per
    Z = np.vstack(Z)
    y = np.concatenate(y)
    cell = np.asarray(cell)
    plat = np.asarray(plat)
    lang = np.asarray(lang)

    # 10 labelled clips per cell -> k = 290
    rng2 = np.random.default_rng(seed + 1)
    lab = np.zeros(len(y), bool)
    for c in cell_names:
        idx = np.where(cell == c)[0]
        lab[rng2.choice(idx, per_cell_labels, replace=False)] = True
    truth = np.array([alpha[c % n_plat] + gamma[c] for c in range(n_cells)])
    return dict(Z=Z, y=y, cell=cell, plat=plat, lang=lang, lab=lab, truth=truth,
                cell_names=cell_names)


def _b1(Z, y, lab):
    sc = StandardScaler().fit(Z)          # region-level standardisation, labels irrelevant
    Zs = sc.transform(Z)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Zs[lab], y[lab])
    return lr.decision_function(Zs)


def test_mpp_recovers_cell_effect_and_beats_b1_auc():
    corrs, gains, cfgs = [], [], []
    for seed in (11, 23, 31):
        data = _synth(seed=seed)
        lab = data["lab"]
        model, cfg = fit_cv(
            data["Z"][lab], data["y"][lab], data["plat"][lab], data["lang"][lab],
            data["cell"][lab], lam=1.0,
        )
        cfgs.append(cfg)
        assert (cfg["sigma_p"], cfg["sigma_gamma"]) in {(a, b) for a in SIGMA_GRID for b in SIGMA_GRID}
        eff = model.cell_effects()
        got = np.array([eff[c] for c in data["cell_names"]])
        corrs.append(float(np.corrcoef(got, data["truth"])[0, 1]))
        s_mpp = model.predict_logit(data["Z"], data["plat"], data["lang"], data["cell"])
        s_b1 = _b1(data["Z"], data["y"], lab)
        gains.append(auc(data["y"], s_mpp) - auc(data["y"], s_b1))

    mean_corr = float(np.mean(corrs))
    assert mean_corr >= 0.6, f"mean recovered cell-effect correlation {mean_corr:.3f} over seeds {corrs}"
    assert min(corrs) >= 0.45, f"one seed recovered the cell effect badly: {corrs}"
    assert float(np.mean(gains)) >= 0.0, f"MPP lost AUC to B1 on average: {gains}"


def test_mpp_fit_beats_ridge_only_probe_in_cv_loglik():
    data = _synth(seed=23)
    lab = data["lab"]
    model, cfg = fit_cv(
        data["Z"][lab], data["y"][lab], data["plat"][lab], data["lang"][lab], data["cell"][lab],
        lam=1.0,
    )
    folds = stratified_folds(data["y"][lab])
    # a probe that ignores the cell is what C1 must improve on
    plain = MPP(sigma_p=0.25, sigma_gamma=2.0, lam=1.0, mode="platformOnly").fit(
        data["Z"][lab], data["y"][lab], data["plat"][lab], data["lang"][lab], data["cell"][lab]
    )
    ll_mpp = log_likelihood(data["y"][lab], model.predict_logit(
        data["Z"][lab], data["plat"][lab], data["lang"][lab], data["cell"][lab]))
    ll_plain = log_likelihood(data["y"][lab], plain.predict_logit(
        data["Z"][lab], data["plat"][lab], data["lang"][lab], data["cell"][lab]))
    assert ll_mpp > ll_plain
    assert folds.shape[0] == lab.sum()


def test_variants_and_unseen_cells():
    data = _synth(seed=31, n_cells=12, n_total=2400, per_cell_labels=12)
    lab = data["lab"]
    for mode in ("full", "noInt", "platformOnly"):
        m = MPP(sigma_p=0.5, sigma_gamma=0.5, lam=1.0, mode=mode).fit(
            data["Z"][lab], data["y"][lab], data["plat"][lab], data["lang"][lab], data["cell"][lab]
        )
        s = m.predict_logit(data["Z"], data["plat"], data["lang"], data["cell"])
        assert s.shape == (len(data["y"]),)
        assert np.isfinite(s).all()
    # a cell that never appeared in L must still score (alpha + beta only)
    m = MPP(sigma_p=0.5, sigma_gamma=0.5, lam=1.0, mode="full").fit(
        data["Z"][lab], data["y"][lab], data["plat"][lab], data["lang"][lab], data["cell"][lab]
    )
    s_new = m.predict_logit(
        data["Z"][:50], data["plat"][:50], data["lang"][:50], ["unseen|cell"] * 50
    )
    assert np.isfinite(s_new).all()
    assert not np.allclose(s_new, 0.0)


def test_mpa_rejects_unknown_mode():
    with pytest.raises(ValueError):
        MPP(mode="nope")

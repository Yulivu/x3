"""C1 - the multi-level probe (MPP), TASKBOOK section 7.2.

    eta_i = w'z_i + mu + alpha_{p(i)} + beta_{l(i)} + gamma_{c(i)}

with ``alpha_p, beta_l ~ N(0, sigma_p^2)``, ``gamma_c ~ N(0, sigma_gamma^2)`` and
``w ~ N(0, 1/lambda)``; ``mu`` is unpenalised.  The penalised negative
log-likelihood is minimised with L-BFGS-B using the analytic gradient.

The point of the joint fit is that ``w`` only has to explain the variation that
is left after the cell-level prior is accounted for, so the probe stops using
the cell prior twice.  That is the hypothesis G4 and G10 test - it is not
assumed here.

``(sigma_p, sigma_gamma)`` ranges over the 4x4 grid from the TASKBOOK (16
configurations, the hard-rule-6 maximum) and is chosen by the shared stratified
folds inside ``L``; ``lambda`` is supplied by the caller (B1's CV choice).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .cv import N_SPLITS, log_likelihood, stratified_folds

SIGMA_GRID = (0.25, 0.5, 1.0, 2.0)

__all__ = ["MPP", "SIGMA_GRID", "MPP_MODES", "fit_cv"]

MPP_MODES = ("full", "noInt", "platformOnly", "mlp")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _codes(values: Sequence[Any], mapping: Optional[Dict[Any, int]] = None):
    if mapping is None:
        mapping = {v: i for i, v in enumerate(sorted(set(values), key=str))}
    codes = np.array([mapping.get(v, -1) for v in values], dtype=np.int64)
    return codes, mapping


class MPP:
    """Multi-level probe with platform / language / cell random effects."""

    def __init__(
        self,
        sigma_p: float = 1.0,
        sigma_gamma: float = 1.0,
        lam: float = 1.0,
        mode: str = "full",
        max_iter: int = 500,
    ) -> None:
        if mode not in MPP_MODES:
            raise ValueError(f"unknown MPP mode {mode!r}")
        if mode == "platformOnly":
            sigma_gamma = float("inf")   # gamma effectively unpenalised -> drop it
        self.sigma_p = float(sigma_p)
        self.sigma_gamma = float(sigma_gamma)
        self.lam = float(lam)
        self.mode = mode
        self.max_iter = int(max_iter)
        self.fitted_ = False

    # ------------------------------------------------------------------ fit
    def _layout(self, d: int) -> Tuple[int, int, int]:
        if self.mode == "platformOnly":
            return 0, 0, 0
        if self.mode == "noInt":
            return len(self.platform_index), len(self.language_index), 0
        return len(self.platform_index), len(self.language_index), len(self.cell_index)

    def fit(self, Z, y, platform, language, cell, sample_weight=None) -> "MPP":
        """``y`` may be soft in [0, 1] (PAST feeds posterior targets) and
        ``sample_weight`` scales each row's likelihood term."""
        Z = np.asarray(Z, float)
        y = np.asarray(y, float)
        if Z.ndim != 2:
            raise ValueError("Z must be 2-D")
        if sample_weight is None:
            w_row = np.ones(Z.shape[0], float)
        else:
            w_row = np.asarray(sample_weight, float).reshape(-1)
            if w_row.shape[0] != Z.shape[0]:
                raise ValueError("sample_weight must have one entry per row")
        self.d_ = Z.shape[1]
        pc, self.platform_index = _codes(platform)
        lc, self.language_index = _codes(language)
        cc, self.cell_index = _codes(cell)
        if (np.array([pc.min() if pc.size else -1, lc.min() if lc.size else -1,
                      cc.min() if cc.size else -1]) < 0).any():
            raise ValueError("fit() received a category it was not given")
        P, L, C = self._layout(self.d_)
        n = Z.shape[0]
        size = self.d_ + 1 + P + L + C

        zs_p2 = 1.0 / max(self.sigma_gamma, 1e-6) ** 2
        zs_l2 = 1.0 / max(self.sigma_p, 1e-6) ** 2
        inv_p2 = 1.0 / max(self.sigma_p, 1e-6) ** 2

        def unpack(theta):
            i = 0
            w = theta[i:i + self.d_]; i += self.d_
            mu = theta[i]; i += 1
            alp = theta[i:i + P]; i += P
            bet = theta[i:i + L]; i += L
            gam = theta[i:i + C]
            return w, mu, alp, bet, gam

        def obj(theta):
            w, mu, alp, bet, gam = unpack(theta)
            eta = Z @ w + mu
            if P:
                eta = eta + alp[pc]
            if L:
                eta = eta + bet[lc]
            if C:
                eta = eta + gam[cc]
            r = w_row * (_sigmoid(eta) - y)             # dNLL/deta
            nll = float(np.sum(w_row * (np.logaddexp(0.0, eta) - y * eta)))
            nll += 0.5 * self.lam * float(w @ w)
            if P:
                nll += 0.5 * inv_p2 * float(alp @ alp)
            if L:
                nll += 0.5 * inv_p2 * float(bet @ bet)
            if C:
                nll += 0.5 * zs_p2 * float(gam @ gam)
            grad = np.empty_like(theta)
            i = 0
            grad[i:i + self.d_] = Z.T @ r + self.lam * w; i += self.d_
            grad[i] = float(r.sum()); i += 1
            if P:
                grad[i:i + P] = np.bincount(pc, weights=r, minlength=P) + inv_p2 * alp; i += P
            if L:
                grad[i:i + L] = np.bincount(lc, weights=r, minlength=L) + inv_p2 * bet; i += L
            if C:
                grad[i:i + C] = np.bincount(cc, weights=r, minlength=C) + zs_p2 * gam
            return nll, grad

        from scipy.optimize import minimize

        theta0 = np.zeros(size)
        p0 = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
        theta0[self.d_] = float(np.log(p0 / (1 - p0)))
        res = minimize(obj, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.theta_ = res.x
        self.fitted_ = True
        self.nll_ = float(res.fun)
        return self

    # -------------------------------------------------------------- predict
    def _terms(self):
        return self._layout(self.d_)

    def predict_logit(self, Z, platform, language, cell) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("MPP.predict_logit called before fit()")
        Z = np.asarray(Z, float)
        P, L, C = self._terms()
        pc, _ = _codes(platform, self.platform_index)
        lc, _ = _codes(language, self.language_index)
        cc, _ = _codes(cell, self.cell_index)
        w, mu, alp, bet, gam = self._unpack()
        eta = Z @ w + mu
        if P:
            m = pc >= 0
            eta[m] = eta[m] + alp[pc[m]]
        if L:
            m = lc >= 0
            eta[m] = eta[m] + bet[lc[m]]
        if C:
            m = cc >= 0
            eta[m] = eta[m] + gam[cc[m]]
        return eta

    def _unpack(self):
        P, L, C = self._terms()
        theta = self.theta_
        i = 0
        w = theta[i:i + self.d_]; i += self.d_
        mu = theta[i]; i += 1
        alp = theta[i:i + P]; i += P
        bet = theta[i:i + L]; i += L
        gam = theta[i:i + C]
        return w, mu, alp, bet, gam

    # -------------------------------------------------------------- effects
    def cell_effects(self) -> Dict[Any, float]:
        """``alpha_p + gamma_c`` per cell - the quantity that is identifiable."""
        if not self.fitted_:
            raise RuntimeError("cell_effects called before fit()")
        P, L, C = self._terms()
        _w, _mu, alp, _bet, gam = self._unpack()
        out = {}
        for c, j in self.cell_index.items():
            plat = str(c).split("|")[0]
            a = alp[self.platform_index[plat]] if (P and plat in self.platform_index) else 0.0
            g = gam[j] if C else 0.0
            out[c] = float(a + g)
        return out

    def detector_logit(self, Z) -> np.ndarray:
        """``w'z + mu`` - the detector part ONLY.

        PAST needs this: ``gamma_c`` already absorbs the cell's prior-driven
        intercept, so feeding ``predict_logit`` into the EM would count the cell
        prior twice and bias every estimate toward the middle.  TASKBOOK 7.3
        specifies ``d_i = w'z_i + mu - logit(pi_bar_L)`` for exactly this reason.
        """
        if not self.fitted_:
            raise RuntimeError("detector_logit called before fit()")
        w, mu, _alp, _bet, _gam = self._unpack()
        return np.asarray(Z, float) @ w + mu

    def platform_effects(self) -> Dict[Any, float]:
        P, _L, _C = self._terms()
        _w, _mu, alp, _bet, _gam = self._unpack()
        return {p: float(alp[i]) for p, i in self.platform_index.items()} if P else {}


def fit_cv(
    Z,
    y,
    platform,
    language,
    cell,
    lam: float = 1.0,
    mode: str = "full",
    folds: Optional[Sequence[int]] = None,
    n_splits: int = N_SPLITS,
) -> Tuple[MPP, Dict[str, Any]]:
    """Choose ``(sigma_p, sigma_gamma)`` by the shared folds, then refit on all of ``L``."""
    y = np.asarray(y, float)
    folds = np.asarray(folds) if folds is not None else stratified_folds(y, n_splits=n_splits)
    grid = [(sp, sg) for sp in SIGMA_GRID for sg in SIGMA_GRID]
    if mode == "platformOnly":
        grid = [(sp, sg) for (sp, sg) in grid if sg == SIGMA_GRID[0]]
    best = None
    for sp, sg in grid:
        scores = []
        for f in np.unique(folds):
            tr = folds != f
            te = ~tr
            if len(np.unique(y[tr])) < 2 or te.sum() == 0:
                continue
            model = MPP(sigma_p=sp, sigma_gamma=sg, lam=lam, mode=mode).fit(
                Z[tr], y[tr], np.asarray(platform)[tr], np.asarray(language)[tr], np.asarray(cell)[tr]
            )
            scores.append(
                log_likelihood(
                    y[te], model.predict_logit(Z[te], np.asarray(platform)[te],
                                               np.asarray(language)[te], np.asarray(cell)[te])
                )
            )
        if not scores:
            continue
        mean_ll = float(np.mean(scores))
        if best is None or mean_ll > best[0]:
            best = (mean_ll, sp, sg)
    if best is None:
        best = (float("nan"), SIGMA_GRID[2], SIGMA_GRID[2])
    _ll, sp, sg = best
    model = MPP(sigma_p=sp, sigma_gamma=sg, lam=lam, mode=mode).fit(Z, y, platform, language, cell)
    return model, {"sigma_p": sp, "sigma_gamma": sg, "lam": lam, "mode": mode, "cv_loglik": _ll}

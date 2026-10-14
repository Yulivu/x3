"""C2 - PAST, prior-aware self-training (TASKBOOK section 7.3).

The whole point is the difference from B7.  B7 thresholds a global confidence and
therefore reproduces whatever class mix the detector happens to be confident
about; in a cell whose true fake rate is far from the detector's operating point
that mix is wrong by tens of points (see ``tests/test_baselines.py``).

PAST instead estimates the cell prior with EM on the detector LLR, and takes a
QUOTA of pseudo-labels per cell that matches that prior, anchored to ``L`` and
shrunk toward the platform rate.  Everything else (quota ramp, confidence floor,
strong-view consistency) is bookkeeping around that one idea.

Steps per round ``r = 1..R``:

1. EM cell prior: ``q_i = sigma(d_i + logit(pi_c))``,
   ``pi_c <- (sum_{U_c} q_i + sum_{L_c} y_i + kappa*pi_p) / (|U_c| + |L_c| + kappa)``
2. quota pseudo-labels: ``rho_r = 0.1 r`` of each cell's unlabelled clips, split
   between the classes in proportion to ``(pi_c, 1 - pi_c)``, with confidence
   floor ``tau_r = 0.75 + 0.05 r``; cells with fewer than 10 unlabelled clips are skipped
3. strong-view consistency: the Fam-A view of every chosen clip joins training
   (weight 0.25) and every unlabelled clip contributes its view with the current
   posterior as a SOFT target (weight ``lambda_u``) - the KL term, written as
   extra Bernoulli terms
4. refit C1 on ``L`` (weight 1) + quota pseudo-labels (weight 0.5) + the view rows

``pi`` is clipped to ``[0.02, 0.98]`` (hard rule 11.10).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cv import stratified_folds
from .em_prior import X3_HI, X3_LO, logit
from .mpp import MPP, fit_cv

PAST_ROUNDS = 4
PAST_KAPPAS = (5, 10, 20)
PAST_WEIGHT_PSEUDO = 0.5
PAST_WEIGHT_VIEW = 0.25
PAST_LAMBDA_U = 0.1
PAST_MIN_UNLAB = 10

__all__ = ["PAST", "PAST_ROUNDS", "PAST_KAPPAS", "estimate_cell_prior_em"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def estimate_cell_prior_em(
    d_u: Sequence[float],
    cell_u: Sequence[Any],
    platform_u: Sequence[Any],
    y_lab: Sequence[float],
    cell_lab: Sequence[Any],
    platform_lab: Sequence[Any],
    kappa: float = 10.0,
    iters: int = 20,
    init: Optional[Dict[Any, float]] = None,
    clip: Tuple[float, float] = (X3_LO, X3_HI),
) -> Tuple[Dict[Any, float], Dict[Any, float], Dict[Any, float]]:
    """EM cell prior with a platform level and a global level.

    Returns ``(pi_cell, pi_platform, pi_global)``.  ``pi_global`` is the labelled
    prior (an anchor, never estimated from the unlabelled scores alone).
    """
    d_u = np.asarray(d_u, float)
    cell_u = np.asarray(cell_u)
    platform_u = np.asarray(platform_u)
    y_lab = np.asarray(y_lab, float)
    cell_lab = np.asarray(cell_lab)
    platform_lab = np.asarray(platform_lab)

    cells = list(dict.fromkeys(list(cell_u)))
    pi_global = float(np.clip(y_lab.mean(), clip[0], clip[1])) if y_lab.size else 0.5
    pi = {}
    for c in cells:
        if init and c in init:
            pi[c] = float(np.clip(init[c], clip[0], clip[1]))
            continue
        m = cell_lab == c
        pi[c] = float(np.clip(y_lab[m].mean(), clip[0], clip[1])) if m.sum() >= 3 else pi_global

    plat_u = {}
    for c in cells:
        plat_u.setdefault(str(c).split("|")[0], []).append(c)

    for _ in range(iters):
        # platform prior = weighted mean of its cells plus its own labelled rate
        pi_plat = {}
        for p, kids in plat_u.items():
            w = np.array([max(1, int((cell_u == c).sum())) for c in kids], float)
            est = float((w * np.array([pi[c] for c in kids])).sum() / w.sum())
            m = platform_lab == p
            if m.sum() >= 5:
                est = float((est * w.sum() + float(y_lab[m].mean()) * m.sum()) / (w.sum() + m.sum()))
            pi_plat[p] = float(np.clip(est, clip[0], clip[1]))

        new = {}
        for c in cells:
            m = cell_u == c
            if m.sum() == 0:
                new[c] = pi[c]
                continue
            p = str(c).split("|")[0]
            q = _sigmoid(d_u[m] + logit(pi[c], eps=1e-3))
            ml = cell_lab == c
            num = float(q.sum()) + float(y_lab[ml].sum()) + kappa * pi_plat.get(p, pi_global)
            den = float(m.sum()) + float(ml.sum()) + kappa
            new[c] = float(np.clip(num / max(den, 1e-9), clip[0], clip[1]))
        delta = max(abs(new[c] - pi[c]) for c in cells) if cells else 0.0
        pi = new
        if delta < 1e-6:
            break
    return pi, {c: pi_plat.get(str(c).split("|")[0], pi_global) for c in cells}, {c: pi_global for c in cells}


class PAST:
    """Prior-aware self-training on top of C1."""

    def __init__(
        self,
        rounds: int = PAST_ROUNDS,
        kappa: float = 10.0,
        lambda_u: float = PAST_LAMBDA_U,
        weight_pseudo: float = PAST_WEIGHT_PSEUDO,
        weight_view: float = PAST_WEIGHT_VIEW,
        quota_r0: float = 0.1,
        quota_step: float = 0.1,
        tau0: float = 0.75,
        tau_step: float = 0.05,
        min_unlab: int = PAST_MIN_UNLAB,
        sigma_p: float = 0.5,
        sigma_gamma: float = 0.5,
        lam: float = 1.0,
        cv_select: bool = False,
        seed: int = 20261015,
    ) -> None:
        self.__dict__.update(
            rounds=rounds, kappa=kappa, lambda_u=lambda_u, weight_pseudo=weight_pseudo,
            weight_view=weight_view, quota_r0=quota_r0, quota_step=quota_step,
            tau0=tau0, tau_step=tau_step, min_unlab=min_unlab, sigma_p=sigma_p,
            sigma_gamma=sigma_gamma, lam=lam, cv_select=cv_select, seed=seed,
        )
        self.history_: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        Z_lab, y_lab, plat_lab, lang_lab, cell_lab,
        Z_unlab, plat_unlab, lang_unlab, cell_unlab,
        views_unlab: Optional[Sequence[np.ndarray]] = None,
    ) -> "PAST":
        Z_lab = np.asarray(Z_lab, float)
        y_lab = np.asarray(y_lab, float)
        Z_unlab = np.asarray(Z_unlab, float)
        views = [np.asarray(v, float) for v in (views_unlab or [])]

        base = self._fit_base(Z_lab, y_lab, plat_lab, lang_lab, cell_lab)
        model = base
        pi_global = float(np.clip(y_lab.mean(), X3_LO, X3_HI)) if y_lab.size else 0.5
        pi: Dict[Any, float] = {}

        for r in range(1, self.rounds + 1):
            # detector part only (w'z + mu): the cell effect already carries the
            # prior-driven intercept, so including it here would double count.
            d_u = model.detector_logit(Z_unlab)
            d_l = model.detector_logit(Z_lab)
            T = self._fit_temperature(d_l, y_lab)
            d_u_cal = d_u / T - logit(pi_global, eps=1e-3)

            pi, pi_plat, _ = estimate_cell_prior_em(
                d_u_cal, cell_unlab, plat_unlab, y_lab, cell_lab, plat_lab, kappa=self.kappa
            )
            sel, hard = self._quota_select(d_u_cal, cell_unlab, pi, r)
            if len(sel) == 0 and not views:
                break

            X = [Z_lab, Z_unlab[sel]]
            t = [y_lab, hard.astype(float)]
            w = [np.ones(len(y_lab)), np.full(len(sel), self.weight_pseudo)]
            pct = [np.asarray(plat_lab), np.asarray(plat_unlab)[sel]]
            lct = [np.asarray(lang_lab), np.asarray(lang_unlab)[sel]]
            cct = [np.asarray(cell_lab), np.asarray(cell_unlab)[sel]]

            for V in views:
                X.append(V[sel])
                t.append(hard.astype(float))
                w.append(np.full(len(sel), self.weight_view))
                pct.append(np.asarray(plat_unlab)[sel])
                lct.append(np.asarray(lang_unlab)[sel])
                cct.append(np.asarray(cell_unlab)[sel])
            if views:
                q_u = _sigmoid(d_u_cal + np.array([logit(pi[c], eps=1e-3) for c in cell_unlab]))
                for V in views:
                    X.append(V)
                    t.append(q_u)
                    w.append(np.full(len(q_u), self.lambda_u))
                    pct.append(np.asarray(plat_unlab))
                    lct.append(np.asarray(lang_unlab))
                    cct.append(np.asarray(cell_unlab))

            Xa = np.vstack(X)
            ta = np.concatenate(t)
            wa = np.concatenate(w)
            models = self._fit_augmented(Xa, ta, wa, np.concatenate(pct), np.concatenate(lct), np.concatenate(cct))
            model = models
            self.history_.append({
                "round": r,
                "T": T,
                "n_selected": int(len(sel)),
                "pi_range": (float(min(pi.values())), float(max(pi.values()))) if pi else None,
            })
        self.model_ = model
        self.pi_ = pi
        self.pi_global_ = pi_global
        return self

    # -------------------------------------------------------------- plumbing
    def _fit_base(self, Z, y, plat, lang, cell) -> MPP:
        if self.cv_select:
            model, cfg = fit_cv(Z, y, plat, lang, cell, lam=self.lam)
            self.cv_cfg_ = cfg
            return model
        return MPP(sigma_p=self.sigma_p, sigma_gamma=self.sigma_gamma, lam=self.lam).fit(
            Z, y, plat, lang, cell
        )

    def _fit_augmented(self, X, t, w, plat, lang, cell) -> MPP:
        # hyper-parameters are re-selected inside L only (here: the base choice),
        # never on the augmented rows
        return MPP(sigma_p=self.sigma_p, sigma_gamma=self.sigma_gamma, lam=self.lam).fit(
            X, t, plat, lang, cell, sample_weight=w
        )

    @staticmethod
    def _fit_temperature(d_l: Sequence[float], y: Sequence[float]) -> float:
        from scipy.optimize import minimize_scalar

        d = np.asarray(d_l, float)
        y = np.asarray(y, float)
        if len(np.unique(y)) < 2:
            return 1.0

        def nll(logT):
            tt = np.exp(logT)
            p = np.clip(_sigmoid(d / tt), 1e-9, 1 - 1e-9)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        return float(np.exp(minimize_scalar(nll, bounds=(-3, 6), method="bounded").x))

    def _quota_select(self, d_u, cell_u, pi, r) -> Tuple[np.ndarray, np.ndarray]:
        """Quota pseudo-labels per cell, split by ``(pi_c, 1 - pi_c)``."""
        cell_u = np.asarray(cell_u)
        d_u = np.asarray(d_u, float)
        rho = min(1.0, self.quota_r0 * r)
        tau = self.tau0 + self.tau_step * r
        keep: List[int] = []
        hard: List[int] = []
        for c in np.unique(cell_u):
            idx = np.where(cell_u == c)[0]
            if idx.size < self.min_unlab:
                continue
            p = float(np.clip(pi.get(c, 0.5), X3_LO, X3_HI))
            n_fake = int(round(rho * idx.size * p))
            n_real = int(round(rho * idx.size * (1 - p)))
            order = idx[np.argsort(-d_u[idx])]
            fake_pick = order[:n_fake]
            real_pick = order[-n_real:] if n_real > 0 else np.array([], dtype=int)
            for j in fake_pick:
                if float(_sigmoid(d_u[j] + logit(p, eps=1e-3))) >= tau:
                    keep.append(int(j)); hard.append(1)
            for j in real_pick:
                if float(_sigmoid(d_u[j] + logit(p, eps=1e-3))) <= 1 - tau:
                    keep.append(int(j)); hard.append(0)
        return np.asarray(keep, int), np.asarray(hard, int)

    # -------------------------------------------------------------- predict
    def predict_logit(self, Z, plat, lang, cell) -> np.ndarray:
        d = self.model_.predict_logit(Z, plat, lang, cell)
        if not self.pi_:
            return d
        off = np.array([
            logit(self.pi_[c], eps=1e-3) if c in self.pi_ else logit(self.pi_global_, eps=1e-3)
            for c in cell
        ])
        return d + off

"""Cell-prior estimation and fusion.

This is a faithful port of Line A's prior machinery
(``wx2/scripts/pi_est2.py`` and ``wx2/scripts/fsc2.py::estimate``): the
Beta-Binomial empirical-Bayes hierarchy (``beta_hier``), the DerSimonian-Laird
empirical-Bayes shrinkage (``shrink_hier``), the Saerens EM on calibrated logits
(``em_cell_cal``) and the damped Bayes fusion (``fuse``).

It exists so that X3's B2 reproduces the frozen ML-ITW tables bit-for-bit, and so
that X3's own Track Z / PAST estimators reuse the same primitives instead of a
re-implementation that drifts.

Line A clips ``pi`` to ``[0.05, 0.95]`` and applies the correction with weight
``lambda = 1``.  X3's Track Z / PAST use the wider clip from TASKBOOK rule 11.10;
both are parameters here, never constants hidden in the code.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

P_LO, P_HI = 0.05, 0.95          # Line A's clip
X3_LO, X3_HI = 0.02, 0.98        # TASKBOOK rule 11.10
RIDGE = 1e-2
MIN_SE = 1e-3
PHI_GRID = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 80.0, 120.0)

__all__ = [
    "P_LO", "P_HI", "X3_LO", "X3_HI", "binom_se", "clip_pi", "logit",
    "beta_hier", "shrink_hier", "em_cell_cal", "fuse", "estimate_pi",
]


def clip_pi(p: float, lo: float = P_LO, hi: float = P_HI) -> float:
    return float(np.clip(p, lo, hi))


def logit(p, eps: float = 0.02):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def binom_se(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 1e-9) / max(n, 1)))


def _dl_tau2(theta: np.ndarray, se: np.ndarray) -> float:
    """DerSimonian-Laird moment estimate of the between-group variance."""
    theta, se = np.asarray(theta, float), np.asarray(se, float)
    if len(theta) < 2:
        return 0.0
    w = 1.0 / se ** 2
    fixed = float((w * theta).sum() / w.sum())
    q = float((w * (theta - fixed) ** 2).sum())
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    return max(0.0, (q - (len(theta) - 1)) / c) if c > 1e-12 else 0.0


def beta_hier(
    n_succ: Mapping[Any, float],
    n_lab: Mapping[Any, float],
    group: Mapping[Any, Any],
    global_p: Optional[float] = None,
    phi: Optional[float] = None,
    phi_grid: Sequence[float] = PHI_GRID,
    lo: float = P_LO,
    hi: float = P_HI,
) -> Tuple[Dict[Any, float], Dict[Any, float], float, Dict[Any, float]]:
    """Beta-Binomial empirical Bayes, three levels: cell -> group -> global.

    ``n_succ[c] / n_lab[c]`` are the successes / observations behind ``pi_hat_c``
    (literal counts for labelled data, ``n_c * pi_hat_c`` and ``n_c`` for a
    label-free estimate).  The shared prior strength ``phi`` is fitted by maximum
    marginal likelihood on ``phi_grid`` unless given.

    Returns ``(pi_hat, se, phi, group_means)``.  Verbatim port of Line A.
    """
    from scipy.optimize import minimize_scalar  # noqa: F401  (kept for API parity)
    from scipy.special import gammaln

    cells = list(n_succ)
    groups = sorted({group[c] for c in cells})
    if global_p is None:
        global_p = float(sum(n_succ[c] for c in cells) / max(sum(n_lab[c] for c in cells), 1))
    g_succ = {g: float(sum(n_succ[c] for c in cells if group[c] == g)) for g in groups}
    g_n = {g: float(max(sum(n_lab[c] for c in cells if group[c] == g), 1.0)) for g in groups}

    def marg_ll(f: float) -> float:
        ll = 0.0
        for g in groups:
            mu = (g_succ[g] + f * global_p) / (g_n[g] + f)
            a, b = f * mu, f * (1 - mu)
            for c in cells:
                if group[c] != g:
                    continue
                k, n = float(n_succ[c]), float(n_lab[c])
                ll += (
                    gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
                    + gammaln(k + a) + gammaln(n - k + b) - gammaln(n + a + b)
                    + gammaln(a + b) - gammaln(a) - gammaln(b)
                )
        return ll

    if phi is None:
        phi = float(max((marg_ll(f), f) for f in phi_grid)[1])
    mu_g = {
        g: float(np.clip((g_succ[g] + phi * global_p) / (g_n[g] + phi), lo, hi)) for g in groups
    }
    pi, se = {}, {}
    for c in cells:
        g = group[c]
        m = mu_g[g]
        p = (float(n_succ[c]) + phi * m) / (float(n_lab[c]) + phi)
        pi[c] = clip_pi(p, lo, hi)
        se[c] = min(
            float(np.sqrt(max(pi[c] * (1 - pi[c]), 1e-9) / (float(n_lab[c]) + phi + 1.0))), 0.5
        )
    return pi, se, phi, mu_g


def shrink_hier(
    pi: Mapping[Any, float],
    se: Mapping[Any, float],
    group: Mapping[Any, Any],
    parent_pi: Optional[Mapping[Any, float]] = None,
    parent_se: Optional[Mapping[Any, float]] = None,
    global_pi: Optional[float] = None,
    global_se: Optional[float] = None,
    to_global: bool = True,
    lo: float = P_LO,
    hi: float = P_HI,
) -> Tuple[Dict[Any, float], Dict[str, Any]]:
    """Empirical-Bayes shrinkage cell -> group -> global (DerSimonian-Laird).

    Shrink BEFORE fusing: the prior variance is the between-group variance, never
    the sampling variance of the global mean (that bug cost Line A 6 EER points).
    """
    cells = list(pi)
    groups = sorted({group[c] for c in cells})
    se_c = {c: max(float(se[c]), MIN_SE) for c in cells}
    g_est, g_se, tau_c = {}, {}, {}
    for g in groups:
        cs = [c for c in cells if group[c] == g]
        if parent_pi is not None and g in parent_pi:
            w_dir = 1.0 / max(float(parent_se[g]), MIN_SE) ** 2
            if len(cs) >= 2:
                w_i = np.array([1.0 / se_c[c] ** 2 for c in cs])
                th = np.array([pi[c] for c in cs])
                pool = float((w_i * th).sum() / w_i.sum())
                tau_c[g] = _dl_tau2(th, np.array([se_c[c] for c in cs]))
                w_pool = float(w_i.sum())
            else:
                pool, w_pool, tau_c[g] = (
                    (pi[cs[0]], 1.0 / se_c[cs[0]] ** 2, 0.0) if cs else (float(parent_pi[g]), 0.0, 0.0)
                )
            g_est[g] = float((w_dir * float(parent_pi[g]) + w_pool * pool) / (w_dir + w_pool))
            g_se[g] = float(1.0 / np.sqrt(w_dir + w_pool))
        elif len(cs) >= 2:
            th = np.array([pi[c] for c in cs])
            sd = np.array([se_c[c] for c in cs])
            tau_c[g] = _dl_tau2(th, sd)
            w_i = 1.0 / (sd ** 2 + tau_c[g])
            g_est[g] = float((w_i * th).sum() / w_i.sum())
            g_se[g] = float(1.0 / np.sqrt(w_i.sum()))
        else:
            tau_c[g] = 0.0
            g_est[g] = clip_pi(pi[cs[0]], lo, hi)
            g_se[g] = min(se_c[cs[0]], 0.5)

    if global_pi is not None:
        glob, glob_se, tau_g = float(global_pi), max(float(global_se), MIN_SE), 0.0
    else:
        th = np.array([g_est[g] for g in groups])
        sd = np.array([g_se[g] for g in groups])
        tau_g = _dl_tau2(th, sd)
        w_i = 1.0 / (sd ** 2 + tau_g)
        glob = float((w_i * th).sum() / w_i.sum())
        glob_se = float(1.0 / np.sqrt(w_i.sum()))

    g_shrunk = {}
    for g in groups:
        if to_global and len(groups) > 1:
            w = tau_g / (tau_g + g_se[g] ** 2) if tau_g > 0 else 0.0
            g_shrunk[g] = clip_pi(w * g_est[g] + (1 - w) * glob, lo, hi)
        else:
            g_shrunk[g] = clip_pi(g_est[g], lo, hi)

    out = {}
    for c in cells:
        g = group[c]
        between = tau_c.get(g, 0.0)
        w = between / (between + se_c[c] ** 2)
        out[c] = clip_pi(w * pi[c] + (1 - w) * g_shrunk[g], lo, hi)
    diag = dict(
        tau_cell=tau_c,
        tau_group=tau_g,
        group_est=g_est,
        group_shrunk=g_shrunk,
        group_se=g_se,
        global_est=glob,
        global_se=glob_se,
    )
    return out, diag


def _gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd + 1e-12)


def em_cell_cal(
    s_cell: Sequence[float],
    src_s: Sequence[float],
    src_y: Sequence[float],
    T: float,
    init: float,
    iters: int = 300,
    alpha: float = 0.0,
    lo: float = P_LO,
    hi: float = P_HI,
) -> Tuple[float, float]:
    """Saerens EM on CALIBRATED logits, initialised from a prior estimate.

    Never initialise at 0.5 (Line A: the EM collapse was a calibration bug).
    ``alpha > 0`` adds a symmetric Dirichlet(alpha) prior on pi.
    """
    ls, lc = np.asarray(src_s, float) / T, np.asarray(s_cell, float) / T
    src_y = np.asarray(src_y)
    mu, sd = {}, {}
    for y in (0, 1):
        v = ls[src_y == y]
        mu[y], sd[y] = float(v.mean()), float(v.std() + 1e-6)
    pi = clip_pi(init, lo, hi)
    l1, l0 = _gauss(lc, mu[1], sd[1]), _gauss(lc, mu[0], sd[0])
    w_last = None
    for _ in range(iters):
        w = pi * l1 / (pi * l1 + (1 - pi) * l0 + 1e-300)
        if alpha > 0:
            new = (w.sum() + alpha - 1) / (len(w) + 2 * (alpha - 1))
        else:
            new = w.mean()
        new = clip_pi(new, lo, hi)
        if abs(new - pi) < 1e-7:
            pi = new
            break
        pi, w_last = new, w
    if w_last is None:
        w_last = pi * l1 / (pi * l1 + (1 - pi) * l0 + 1e-300)
    info = float(np.sum((np.log(l1 + 1e-300) - np.log(l0 + 1e-300)) ** 2 * w_last * (1 - w_last)))
    se = float(np.sqrt(1.0 / info)) if info > 1e-9 else 0.5
    return pi, min(se, 0.5)


def fuse(
    s: Sequence[float],
    pi_per_clip: Sequence[float],
    n_per_clip: Sequence[float],
    k: float = 20.0,
    lam: float = 1.0,
    eps: float = 0.02,
) -> np.ndarray:
    """Damped Bayes correction: ``z + lam * (n/(n+k)) * logit(pi)``.

    ``z`` is the standardised score.  ``k = None`` gives the undamped correction.
    """
    s = np.asarray(s, float)
    z = (s - s.mean()) / (s.std() + 1e-9)
    w = (
        np.ones_like(z)
        if k is None
        else (np.asarray(n_per_clip, float) / (np.asarray(n_per_clip, float) + k))
    )
    return z + lam * w * logit(np.asarray(pi_per_clip, float), eps=eps)


def estimate_pi(
    all_cells: Sequence[Any],
    cell_of_clip: Sequence[Any],
    labeled: Sequence[bool],
    y: Sequence[float],
    platform_of_clip: Sequence[Any],
    variant: str = "bb",
    global_default: float = 0.26,
) -> Tuple[Dict[Any, float], Dict[Any, Optional[int]], Optional[float]]:
    """Line A's ``fsc2.estimate`` (variants ``count`` / ``eb`` / ``bb``).

    Returns ``(pi_per_cell, n_labelled_per_cell, phi)``; ``phi`` is only set for
    the ``bb`` variant, matching Line A's diagnostic output.
    """
    ck = np.asarray(cell_of_clip)
    plat = np.asarray(platform_of_clip)
    y = np.asarray(y, float)
    lab = np.asarray(labeled, bool)
    all_cells = list(all_cells)

    cell_pi, cell_se, cell_n = {}, {}, {}
    for c in np.unique(ck):
        m = (ck == c) & lab
        n = int(m.sum())
        if n < 3:
            continue
        p = float(y[m].mean())
        cell_pi[c], cell_se[c], cell_n[c] = p, binom_se(p, n), n
    plat_pi, plat_se = {}, {}
    for p_ in np.unique(plat):
        m = (plat == p_) & lab
        rate = float(y[m].mean()) if m.sum() >= 5 else global_default
        plat_pi[p_] = rate
        plat_se[p_] = binom_se(rate, int(max(m.sum(), 1)))
    g_pi = float(y[lab].mean()) if lab.sum() >= 5 else global_default
    g_se = binom_se(g_pi, int(max(lab.sum(), 1)))

    phi = None
    if variant.startswith("count"):
        pi = {c: cell_pi.get(c, plat_pi.get(str(c).split("|")[0], g_pi)) for c in all_cells}
        n_lab = {c: cell_n.get(c, 3) for c in all_cells}
    elif variant.startswith("bb"):
        n_s = {c: cell_pi[c] * cell_n[c] for c in cell_pi}
        bb, _se, phi, mu_g = beta_hier(
            n_s,
            {c: cell_n[c] for c in cell_pi},
            {c: str(c).split("|")[0] for c in cell_pi},
            global_p=g_pi,
        )
        pi = {c: bb.get(c, mu_g.get(str(c).split("|")[0], g_pi)) for c in all_cells}
        n_lab = {c: cell_n.get(c, 3) for c in all_cells}
    else:
        if not cell_pi:
            pi = {c: g_pi for c in all_cells}
        else:
            pi, _diag = shrink_hier(
                cell_pi,
                cell_se,
                {c: str(c).split("|")[0] for c in cell_pi},
                parent_pi=plat_pi,
                parent_se=plat_se,
                global_pi=g_pi,
                global_se=g_se,
            )
            pi = {c: pi.get(c, plat_pi.get(str(c).split("|")[0], g_pi)) for c in all_cells}
        n_lab = {c: cell_n.get(c, 3) for c in all_cells}
    return pi, n_lab, phi

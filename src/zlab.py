"""Track Z - zero-label adaptation (TASKBOOK W8, section 7.5).

``k = 0``: no labels at all, only the zero-shot detector scores, the cell
metadata and the unlabelled audio.  The evaluation region is the whole dev
region.

* ``Z0`` - the raw AntiDeepfake zero-shot logit difference.
* ``Z1`` - per-platform standardisation against the platform median and IQR.
* ``Z2`` - two-component Gaussian mixture with shared class conditionals and a
  free per-cell mixing weight.  The cell prior is estimated by EM (initialised
  from the sign of the zero-shot score, never from 0.5) and shrunk toward the
  platform and the global level with ``kappa``; the score is
  ``log N(s|mu1,sd1) - log N(s|mu0,sd0) + logit(pi_c)``.
* ``Z3`` - ``Z2`` applied to the average of the Fam-A view scores (W10).

Nothing here reads a label outside ``L`` (there is no ``L``), so Track Z can
never leak; the only choice is which zero-shot backbone the caller hands in.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .em_prior import X3_HI, X3_LO, logit

Z2_ITERS = 20
Z2_KAPPA = 10.0
Z_CLIP = (X3_LO, X3_HI)

__all__ = ["z0", "z1", "z2", "z3", "fit_shared_gaussians", "track_z_scores",
           "Z2_ITERS", "Z2_KAPPA"]


def _log_gauss(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    sd = max(float(sd), 1e-6)
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd)


def z0(scores: Sequence[float]) -> np.ndarray:
    """The raw zero-shot score, standardised over the whole region."""
    s = np.asarray(scores, float)
    return (s - s.mean()) / (s.std() + 1e-9)


def z1(
    scores: Sequence[float],
    platform_of: Sequence[Any],
    min_per_platform: int = 20,
) -> np.ndarray:
    """Per-platform robust standardisation, ``(s - median_p) / IQR_p``."""
    s = np.asarray(scores, float)
    plat = np.asarray(platform_of)
    out = np.zeros_like(s)
    global_med = float(np.median(s))
    global_iqr = float(np.percentile(s, 75) - np.percentile(s, 25)) or 1.0
    for p in np.unique(plat):
        m = plat == p
        if m.sum() < min_per_platform:
            out[m] = (s[m] - global_med) / (global_iqr + 1e-9)
            continue
        med = float(np.median(s[m]))
        iqr = float(np.percentile(s[m], 75) - np.percentile(s[m], 25))
        out[m] = (s[m] - med) / (iqr + 1e-9)
    return out


def fit_shared_gaussians(
    s: Sequence[float],
    cell_of: Sequence[Any],
    platform_of: Sequence[Any],
    iters: int = Z2_ITERS,
    kappa: float = Z2_KAPPA,
    clip: Tuple[float, float] = Z_CLIP,
) -> Tuple[float, float, float, float, Dict[Any, float]]:
    """EM for shared class conditionals + free per-cell mixing weight.

    Returns ``(mu0, sd0, mu1, sd1, pi_cell)``.  Initialisation uses the sign of
    the zero-shot score (a soft split at the median), not 0.5.
    """
    s = np.asarray(s, float)
    cell = np.asarray(cell_of)
    plat = np.asarray(platform_of)
    med = float(np.median(s))
    scale = float(np.std(s)) or 1.0
    w = 1.0 / (1.0 + np.exp(-2.0 * (s - med) / scale))     # soft sign split

    def wstats(weights: np.ndarray) -> Tuple[float, float]:
        tot = float(weights.sum())
        if tot < 1e-6:
            return med, scale
        mu = float((weights * s).sum() / tot)
        var = float((weights * (s - mu) ** 2).sum() / tot)
        return mu, max(np.sqrt(var), 1e-3)

    mu1, sd1 = wstats(w)
    mu0, sd0 = wstats(1.0 - w)
    pi = {}
    for c in np.unique(cell):
        m = cell == c
        pi[c] = float(np.clip(w[m].mean(), clip[0], clip[1]))
    pi_global = float(np.clip(pi and np.mean(list(pi.values())) or 0.5, clip[0], clip[1]))

    for _ in range(int(iters)):
        l1 = np.exp(_log_gauss(s, mu1, sd1))
        l0 = np.exp(_log_gauss(s, mu0, sd0))
        p = np.array([pi[c] for c in cell], float)
        w = p * l1 / (p * l1 + (1 - p) * l0 + 1e-300)
        mu1, sd1 = wstats(w)
        mu0, sd0 = wstats(1.0 - w)
        new = {}
        for c in np.unique(cell):
            m = cell == c
            p_plat = float(np.mean([pi[k] for k in np.unique(cell[plat == str(c).split("|")[0]])])) \
                if np.any(plat == str(c).split("|")[0]) else pi_global
            num = float(w[m].sum()) + kappa * p_plat
            den = float(m.sum()) + kappa
            new[c] = float(np.clip(num / max(den, 1e-9), clip[0], clip[1]))
        delta = max(abs(new[c] - pi[c]) for c in pi) if pi else 0.0
        pi = new
        if delta < 1e-7:
            break
    return mu0, sd0, mu1, sd1, pi


def z2(
    scores: Sequence[float],
    cell_of: Sequence[Any],
    platform_of: Sequence[Any],
    iters: int = Z2_ITERS,
    kappa: float = Z2_KAPPA,
    clip: Tuple[float, float] = Z_CLIP,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Z2 score: shared class conditionals + cell prior, fitted by EM."""
    s = np.asarray(scores, float)
    z = (s - s.mean()) / (s.std() + 1e-9)
    mu0, sd0, mu1, sd1, pi = fit_shared_gaussians(z, cell_of, platform_of, iters, kappa, clip)
    lr = _log_gauss(z, mu1, sd1) - _log_gauss(z, mu0, sd0)
    pi_clip = np.array([pi[c] for c in cell_of], float)
    score = lr + logit(pi_clip, eps=1e-3)
    return score, {
        "mu0": mu0, "sd0": sd0, "mu1": mu1, "sd1": sd1,
        "pi": {str(k): v for k, v in pi.items()},
    }


def z3(
    scores_per_view: Sequence[Sequence[float]],
    cell_of: Sequence[Any],
    platform_of: Sequence[Any],
    iters: int = Z2_ITERS,
    kappa: float = Z2_KAPPA,
    clip: Tuple[float, float] = Z_CLIP,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Z3: average the zero-shot scores over the Fam-A views, then run Z2."""
    if not scores_per_view:
        raise ValueError("z3 needs at least one view")
    stacked = np.vstack([z0(v) for v in scores_per_view])
    mean_score = stacked.mean(axis=0)
    score, info = z2(mean_score, cell_of, platform_of, iters=iters, kappa=kappa, clip=clip)
    info["n_views"] = int(len(scores_per_view))
    return score, info


def track_z_scores(
    zeroshot: Sequence[float],
    cell_of: Sequence[Any],
    platform_of: Sequence[Any],
    view_scores: Optional[Sequence[Sequence[float]]] = None,
    iters: int = Z2_ITERS,
    kappa: float = Z2_KAPPA,
) -> Dict[str, np.ndarray]:
    """All four Track Z scores for one backbone, in one call."""
    out = {
        "Z0": z0(zeroshot),
        "Z1": z1(zeroshot, platform_of),
    }
    out["Z2"], _ = z2(zeroshot, cell_of, platform_of, iters=iters, kappa=kappa)
    if view_scores:
        out["Z3"], _ = z3(view_scores, cell_of, platform_of, iters=iters, kappa=kappa)
    return out

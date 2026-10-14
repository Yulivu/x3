"""Track Z - the zero-label track (TASKBOOK W8, plus Z5 from NEW_TASKS 01 U6).

``k = 0``: no labels at all, only metadata and unlabelled audio.

* **Z0** - the backbone's zero-shot score itself.
* **Z1** - per-platform standardisation ``(s - med_p) / IQR_p``, both statistics
  taken from that platform's unlabelled clips, so nothing is learned from labels.
* **Z2** - a two-component Gaussian mixture on the zero-shot score.  The class
  parameters ``(mu0, sig0, mu1, sig1)`` are shared globally; the *cell* priors are
  free and shrunk towards the platform and then the global prior with strength
  ``kappa``.  Initialised by the sign of the zero-shot score, 20 EM steps, and the
  score is ``log N(s|mu1,sig1) - log N(s|mu0,sig0) + logit(pi_c)``.
* **Z5** (U6) - initialise pseudo-labels from the Z2 score and propagate them on the
  cosine kNN graph.  Still label-free: the pseudo-labels come from the detector.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "platform_standardise", "iqr", "fit_z2", "z2_scores", "log_gauss",
    "z5_scores", "score_views", "z2_posteriors", "threshold_per_cell", "z4_scores",
]


def iqr(x: Sequence[float]) -> float:
    x = np.asarray(x, float)
    if x.size == 0:
        return 1.0
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25) if (q75 - q25) > 1e-9 else 1.0


def platform_standardise(s: Sequence[float], platform: Sequence[object]
                         ) -> np.ndarray:
    """**Z1**: ``(s - median_p) / IQR_p`` computed inside each platform."""
    s = np.asarray(s, float)
    plat = np.asarray(platform)
    out = np.empty_like(s)
    for p in np.unique(plat):
        m = plat == p
        out[m] = (s[m] - float(np.median(s[m]))) / iqr(s[m])
    return out


def log_gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * ((np.asarray(x, float) - mu)
                                                          / sigma) ** 2


def fit_z2(s: Sequence[float], cell: Sequence[object], platform: Sequence[object],
           iters: int = 20, kappa: float = 10.0) -> Dict[str, Any]:
    """EM for the shared-normal / free-cell-prior mixture of W8's Z2."""
    s = np.asarray(s, float)
    cell = np.asarray(cell)
    plat = np.asarray(platform)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    sd = float(s.std()) + 1e-6

    # initialisation from the sign of the zero-shot score (never from labels)
    pos, neg = s > 0, s <= 0
    mu1 = float(s[pos].mean()) if pos.any() else sd
    mu0 = float(s[neg].mean()) if neg.any() else -sd
    sig1 = float(s[pos].std()) + 1e-6 if pos.any() else sd
    sig0 = float(s[neg].std()) + 1e-6 if neg.any() else sd
    pi_cell = {c: float(np.clip((s[cell == c] > 0).mean(), 0.02, 0.98))
               for c in np.unique(cell)}
    loglik = float("nan")
    for _ in range(int(iters)):
        pi_clip = np.array([pi_cell[c] for c in cell])
        l1 = log_gauss(s, mu1, sig1) + np.log(np.clip(pi_clip, 1e-6, 1))
        l0 = log_gauss(s, mu0, sig0) + np.log(np.clip(1 - pi_clip, 1e-6, 1))
        m = np.maximum(l1, l0)
        r = np.exp(l1 - m) / (np.exp(l1 - m) + np.exp(l0 - m))
        loglik = float(np.mean(m + np.log(np.exp(l1 - m) + np.exp(l0 - m))))
        w1, w0 = r, 1.0 - r
        n1, n0 = float(w1.sum()), float(w0.sum())
        if n1 > 1 and n0 > 1:
            mu1 = float((w1 * s).sum() / n1)
            mu0 = float((w0 * s).sum() / n0)
            sig1 = float(np.sqrt((w1 * (s - mu1) ** 2).sum() / n1)) + 1e-6
            sig0 = float(np.sqrt((w0 * (s - mu0) ** 2).sum() / n0)) + 1e-6
        pi_global = float(r.mean())
        pi_plat = {}
        for p in np.unique(plat):
            m_p = plat == p
            new = (float(r[m_p].sum()) + kappa * pi_global) / (float(m_p.sum()) + kappa)
            pi_plat[p] = float(np.clip(new, 0.01, 0.99))
        for c in np.unique(cell):
            m_c = cell == c
            parent = pi_plat[plat[m_c][0]]
            new = (float(r[m_c].sum()) + kappa * parent) / (float(m_c.sum()) + kappa)
            pi_cell[c] = float(np.clip(new, 0.01, 0.99))
    return {"mu0": mu0, "mu1": mu1, "sig0": sig0, "sig1": sig1,
            "pi_cell": pi_cell, "pi_platform": pi_plat, "pi_global": pi_global,
            "loglik": loglik, "iters": int(iters), "kappa": float(kappa)}


def z2_scores(s: Sequence[float], cell: Sequence[object], fit: Dict[str, Any]
              ) -> np.ndarray:
    """``log N(s|mu1,sig1) - log N(s|mu0,sig0) + logit(pi_c)``."""
    s = np.asarray(s, float)
    cell = np.asarray(cell)
    pi = np.array([fit["pi_cell"][c] for c in cell], float)
    return (log_gauss(s, fit["mu1"], fit["sig1"])
            - log_gauss(s, fit["mu0"], fit["sig0"])
            + np.log(np.clip(pi, 1e-6, 1 - 1e-6) / np.clip(1 - pi, 1e-6, 1)))


def z2_posteriors(s: Sequence[float], cell: Sequence[object], fit: Dict[str, Any]
                  ) -> np.ndarray:
    """The responsibility ``P(class 1 | s, cell)`` of the Z2 mixture."""
    s = np.asarray(s, float)
    cell = np.asarray(cell)
    pi = np.array([fit["pi_cell"][c] for c in cell], float)
    l1 = log_gauss(s, fit["mu1"], fit["sig1"]) + np.log(np.clip(pi, 1e-12, 1))
    l0 = log_gauss(s, fit["mu0"], fit["sig0"]) + np.log(np.clip(1 - pi, 1e-12, 1))
    m = np.maximum(l1, l0)
    return np.exp(l1 - m) / (np.exp(l1 - m) + np.exp(l0 - m))


def threshold_per_cell(p: Sequence[float], cell: Sequence[object],
                       target_rate: Dict[Any, float]) -> np.ndarray:
    """Label-free class-mass normalisation: pick a per-cell threshold that makes the
    predicted fake rate equal the prior ``target_rate[cell]``."""
    p = np.asarray(p, float)
    cell = np.asarray(cell)
    out = np.zeros(len(p), bool)
    for c in np.unique(cell):
        m = cell == c
        rate = float(target_rate.get(c, 0.5))
        if m.sum() == 0:
            continue
        cut = float(np.quantile(p[m], 1.0 - rate))
        out[m] = p[m] >= cut
    if out.all() or not out.any():                    # degenerate: fall back to the median
        out = p >= float(np.median(p))
    return out


def z4_scores(Z: np.ndarray, s: Sequence[float], cell: Sequence[object],
              fit: Dict[str, Any], iters: int = 5, C: float = 1.0,
              seed: int = 20261015) -> Tuple[np.ndarray, Dict[str, Any]]:
    """**Z4**: label-free PAST - self-train an LR on the Z2 posterior, with the cell
    priors held fixed, and iterate.  No label is ever read.

    Each round: fit ``LR(Z -> pseudo)``, then re-threshold its probability per cell so
    the predicted fake rate matches the mixture's own prior.  The returned score is
    the logit of the final LR.
    """
    from sklearn.linear_model import LogisticRegression

    Z = np.asarray(Z, float)
    r = z2_posteriors(s, cell, fit)
    target = dict(fit["pi_cell"])
    pseudo = r > 0.5
    history: List[Dict[str, Any]] = []
    model = None
    for t in range(int(iters)):
        if len(np.unique(pseudo)) < 2:                # nothing to learn from
            break
        model = LogisticRegression(C=float(C), max_iter=5000,
                                   random_state=int(seed)).fit(Z, pseudo.astype(int))
        p = model.predict_proba(Z)[:, 1]
        new = threshold_per_cell(p, cell, target)
        history.append({"iter": t, "n_pos": int(new.sum())})
        if np.array_equal(new, pseudo):
            pseudo = new
            break
        pseudo = new
    if model is None:                                 # pragma: no cover
        return np.asarray(s, float), {"iters": 0, "history": history}
    return (model.decision_function(Z),
            {"iters": len(history), "history": history,
             "n_pseudo_pos": int(pseudo.sum())})


def z5_scores(z2: Sequence[float], graph, alpha: float = 0.9, iters: int = 50,
              teacher: Optional[Sequence[float]] = None) -> np.ndarray:
    """**Z5**: pseudo-label the pool from Z2, propagate on the graph, keep the field.

    ``teacher`` (default: the Z2 score) decides the pseudo-label sign; no true label
    is read anywhere.
    """
    from .malp import mass_normalise, propagate, sparse_affinity

    t = np.asarray(z2 if teacher is None else teacher, float)
    pseudo = (t > 0).astype(float)
    mask = np.ones(len(t), bool)
    S = sparse_affinity(graph[0], graph[1], graph[0].shape[1])
    f = propagate(S, pseudo, mask, alpha=alpha, iters=iters)
    return mass_normalise(f, float(pseudo.mean()))


def score_views(logits: Dict[str, np.ndarray], cell: Sequence[object],
                platform: Sequence[object], iters: int = 20, kappa: float = 10.0
                ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """**Z3**: fit Z2 per view and average the resulting scores.

    Each view supplies its own zero-shot logit (W10 stores one per view), so the
    mixture is refitted per view and the average is taken in score space.
    """
    parts = []
    fits: Dict[str, Any] = {}
    for name, s in logits.items():
        fit = fit_z2(s, cell, platform, iters=iters, kappa=kappa)
        fits[name] = {k: v for k, v in fit.items() if k != "pi_cell"}
        parts.append(z2_scores(s, cell, fit))
    if not parts:                                       # pragma: no cover
        raise ValueError("no view scores were given")
    return np.mean(np.vstack(parts), axis=0), fits

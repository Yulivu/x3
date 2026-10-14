"""M-ALP - active label propagation (NEW_TASKS 01 U4, revised by 追加 02).

Two pieces, both pre-registered in ``claims_v2.yaml``:

**Learner L-GRAPH.** A cosine kNN graph in the PCA-256 space with edge weight
``exp(sim / T)`` (``T`` = median edge similarity), symmetrised and normalised to
``S = D^-1/2 W D^-1/2``; same-speaker edges can be up-weighted by ``w_s``.  Labels
propagate with ``F <- alpha S F + (1 - alpha) Y0`` (the fixed point of the resolvent
form ``F = (I - alpha S)^-1 Y0``, which is what a 15k-node graph can actually
solve), then the score is class-mass normalised (Zhu 2003) and fused with the LR
logit: ``score = z(logit_LR) + eta * z(F_mass)``.  ``eta = 0`` is B1.

**Selector.** ``kmeans`` heat start (n0 = 60), then four batches with
``score_i = u_i * rho_i * c_i`` - uncertainty ``u = 1 - |2q - 1|`` from the current
fused model, local density ``rho`` (mean similarity to the clip's neighbours) and a
coverage term ``c = 1 - exp(-d / tau)`` with ``tau`` the median distance to the
nearest labelled clip - then greedy with a cosine-dedup pass.  ``SC`` enforces
speaker coverage first and ``CC`` enforces cell coverage.

Hyper-parameters are chosen inside ``L`` with the shared folds.  U4's original text
asked for leave-one-labelled-node-out; 追加 02 V-3 replaced that for anything
reported on ``unseen`` / ``strict_draw`` with leave-one-speaker-out group folds, and
this module follows V-3 (the newer rule).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .baselines import knn_graph
from .cv import log_likelihood, select_config

KG_GRID = (10, 15, 30)
ALPHA_GRID = (0.8, 0.9, 0.99)
WS_GRID = (1.0, 2.0)
ETA_GRID = (0.0, 0.25, 0.5, 1.0)
DEDUP_COS = 0.1
N0 = 60
BATCHES = 4

__all__ = [
    "KG_GRID", "ALPHA_GRID", "WS_GRID", "ETA_GRID", "DEDUP_COS", "N0", "BATCHES",
    "sparse_affinity", "propagate", "mass_normalise", "fuse", "malp_scores",
    "hyperparam_grid", "local_density", "coverage", "select_batch", "heat_start",
]


def sparse_affinity(idx: np.ndarray, sim: np.ndarray, k_g: int,
                    speaker: Optional[Sequence[object]] = None,
                    w_s: float = 1.0):
    """Symmetric, degree-normalised affinity ``S`` from a cached kNN table."""
    from scipy.sparse import csr_matrix

    n = idx.shape[0]
    k = int(min(k_g, idx.shape[1]))
    if k <= 0:
        return csr_matrix((n, n))
    nbr = np.asarray(idx)[:, :k]
    val = np.asarray(sim, float)[:, :k]
    T = float(np.median(val)) if val.size else 1.0
    w = np.exp(val / max(T, 1e-6))
    if speaker is not None and float(w_s) != 1.0:
        speaker = np.asarray(speaker)
        same = speaker[nbr] == speaker[:, None]
        w = np.where(same, w * float(w_s), w)
    rows = np.repeat(np.arange(n), k)
    W = csr_matrix((w.ravel(), (rows, nbr.ravel())), shape=(n, n))
    W = W.maximum(W.T).tocsr()
    d = np.asarray(W.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    S = csr_matrix((W.data * dinv[W.indices] * np.repeat(dinv, np.diff(W.indptr)),
                    W.indices, W.indptr), shape=W.shape)
    return S.tocsr()


def propagate(S, y_lab: Sequence[float], lab_mask: Sequence[bool], alpha: float = 0.9,
              iters: int = 50) -> np.ndarray:
    """The fixed point of ``F = (I - alpha S)^-1 Y0`` (iterated; Y0 = +-1 on L).

    The iteration is ``F <- alpha S F + Y0``.  (The ``(1 - alpha) Y`` variant seen in
    some papers converges to the same vector times ``(1 - alpha)``; the score is
    standardised before fusion, so the two agree, but this form matches the task
    text's ``F = (I - alpha S)^-1 Y`` exactly.)
    """
    lab_mask = np.asarray(lab_mask, bool)
    y = np.asarray(y_lab, float)
    n = S.shape[0]
    Y0 = np.zeros(n)
    Y0[lab_mask] = 2.0 * y - 1.0
    F = Y0.copy()
    for _ in range(int(iters)):
        F = alpha * (S @ F) + Y0
    return F


def mass_normalise(f: Sequence[float], pi: float) -> np.ndarray:
    """Shift ``f`` so that the predicted fake fraction equals ``pi`` (Zhu 2003)."""
    f = np.asarray(f, float)
    pi = float(min(max(pi, 1e-6), 1 - 1e-6))
    t = float(np.quantile(f, 1.0 - pi))
    return f - t


def fuse(lr_logit: Sequence[float], f_mass: Sequence[float], eta: float) -> np.ndarray:
    """``z(logit_LR) + eta * z(F)``; both terms are standardised over the region."""
    a = np.asarray(lr_logit, float)
    b = np.asarray(f_mass, float)
    za = (a - a.mean()) / (a.std() + 1e-9)
    zb = (b - b.mean()) / (b.std() + 1e-9)
    return za + float(eta) * zb


def hyperparam_grid(max_configs: int = 16) -> List[Tuple[int, float, float, float]]:
    """``(k_g, alpha, w_s, eta)``, at most 16 entries (hard rule 6)."""
    out: List[Tuple[int, float, float, float]] = []
    for k_g in KG_GRID:
        for alpha in ALPHA_GRID:
            for w_s in WS_GRID:
                for eta in ETA_GRID:
                    out.append((int(k_g), float(alpha), float(w_s), float(eta)))
    # keep the grid's corners: eta = 0 (pure LR) and eta = 1 (pure LP) for every
    # (k_g, alpha, w_s), because those are the two ends the fusion has to beat
    keep = [c for c in out if c[3] in (0.0, 1.0)]
    rest = [c for c in out if c[3] not in (0.0, 1.0)]
    ordered = keep + rest
    return ordered[:int(max_configs)]


def malp_scores(Z: np.ndarray, y: np.ndarray, lab_mask: np.ndarray, graph,
                cfg: Tuple[int, float, float, float], lr_logit: np.ndarray,
                speaker: Optional[Sequence[object]] = None) -> np.ndarray:
    """Fused M-ALP score for one configuration (``eta=0`` returns the LR logit)."""
    k_g, alpha, w_s, eta = cfg
    if float(eta) == 0.0:
        return np.asarray(lr_logit, float)
    S = sparse_affinity(graph[0], graph[1], k_g, speaker, w_s)
    f = propagate(S, y[lab_mask], lab_mask, alpha=alpha)
    pi = float(np.mean(y[lab_mask])) if lab_mask.any() else 0.5
    return fuse(lr_logit, mass_normalise(f, pi), eta)


def local_density(idx: np.ndarray, sim: np.ndarray, k: int = 15) -> np.ndarray:
    """Mean similarity to a clip's own neighbours (a label-free density proxy)."""
    k = int(min(k, np.asarray(idx).shape[1]))
    if k <= 0:
        return np.ones(np.asarray(idx).shape[0])
    return np.asarray(sim, float)[:, :k].mean(axis=1)


def coverage(Z: np.ndarray, labelled: np.ndarray, pool: np.ndarray,
             tau: Optional[float] = None) -> np.ndarray:
    """``1 - exp(-d / tau)`` where ``d`` is the distance to the nearest labelled clip."""
    Zu = np.asarray(Z, float)
    Zu = Zu / (np.linalg.norm(Zu, axis=1, keepdims=True) + 1e-9)
    if labelled.size == 0:
        return np.ones(len(Zu))
    Zl = Zu[np.asarray(labelled, int)]
    d = 1.0 - np.max(Zu @ Zl.T, axis=1)
    tau = float(np.median(d)) if tau is None else float(tau)
    return 1.0 - np.exp(-d / max(tau, 1e-6))


def heat_start(Z: np.ndarray, n0: int = N0, seed: int = 20261015) -> np.ndarray:
    """One clip per k-means cluster: the clip closest to the cluster centre."""
    from .alloc import _kmeans_pp

    Z = np.asarray(Z, float)
    n0 = int(min(n0, len(Z)))
    if n0 <= 0:
        return np.array([], int)
    rng = np.random.default_rng(int(seed))
    centres = _kmeans_pp(Z, np.arange(len(Z)), n0, rng)
    Zu = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    picks = []
    for c in centres:
        d = 1.0 - Zu @ Zu[int(c)]
        picks.append(int(np.argmin(d)))
    return np.unique(np.asarray(picks, int))


def select_batch(Z: np.ndarray, scores: np.ndarray, idx: np.ndarray, sim: np.ndarray,
                 labelled: np.ndarray, pool: np.ndarray, batch: int,
                 variant: str = "full", speaker: Optional[Sequence[object]] = None,
                 cell: Optional[Sequence[object]] = None,
                 min_per_group: int = 30) -> np.ndarray:
    """One acquisition round: ``u * rho * c`` with coverage constraints and dedup."""
    pool = np.asarray(pool, int)
    if pool.size == 0 or batch <= 0:
        return np.array([], int)
    batch = int(min(batch, pool.size))
    q = 1.0 / (1.0 + np.exp(-np.asarray(scores, float)))
    u = 1.0 - np.abs(2.0 * q - 1.0)
    rho = local_density(idx, sim, k=15)
    c = coverage(Z, labelled, pool)
    base = np.zeros(len(Z))
    base[pool] = u[pool] * rho[pool] * c[pool]
    if variant == "U":
        base[pool] = u[pool]
    elif variant == "D":
        base[pool] = u[pool] * rho[pool]
    picks: List[int] = []

    def _take(candidates: np.ndarray, want: int) -> None:
        order = candidates[np.argsort(-base[candidates])]
        for i in order:
            if want <= 0:
                break
            if i in picks:
                continue
            if picks:
                zu = np.asarray(Z, float)[i]
                zu = zu / (np.linalg.norm(zu) + 1e-9)
                zs = np.asarray(Z, float)[picks]
                zs = zs / (np.linalg.norm(zs, axis=1, keepdims=True) + 1e-9)
                if np.max(zs @ zu) > 1.0 - DEDUP_COS:
                    continue
            picks.append(int(i))
            want -= 1

    if variant in ("SC", "CC") and len(picks) < batch:
        group = speaker if variant == "SC" else cell
        if group is not None:
            group = np.asarray(group)
            big = [g for g in np.unique(group[pool])
                   if int((group == g).sum()) >= int(min_per_group)]
            for g in big:                       # one clip per group before the score
                if len(picks) >= batch:
                    break
                cand = pool[(group[pool] == g) & ~np.isin(pool, picks)]
                if cand.size == 0:
                    continue
                _take(cand, 1)
    _take(pool[~np.isin(pool, picks)], batch - len(picks))
    return np.asarray(picks[:batch], int)


def select_hyperparams(Z: np.ndarray, y: np.ndarray, lab_mask: np.ndarray, graph,
                       folds: np.ndarray, speaker: Optional[Sequence[object]] = None,
                       grid: Optional[Sequence] = None, max_configs: int = 16,
                       C: float = 1.0
                       ) -> Tuple[Tuple[int, float, float, float], Dict[str, Any]]:
    """V-3 selection of ``(k_g, alpha, w_s, eta)`` inside ``L``.

    ``eta = 0`` is scored once (it is B1) and every other configuration reuses the
    same fold models, so the 16-point budget buys 16 real evaluations.
    """
    from sklearn.linear_model import LogisticRegression

    grid = list(grid or hyperparam_grid(max_configs))
    lab_idx = np.where(lab_mask)[0]
    ylab = np.asarray(y, float)[lab_idx]
    folds = np.asarray(folds)
    if folds.shape[0] != lab_idx.shape[0]:
        raise ValueError(f"folds must have one entry per labelled clip "
                         f"({lab_idx.shape[0]}), got {folds.shape[0]}")
    best = None
    history: List[Dict[str, Any]] = []
    fold_models = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or len(np.unique(ylab[tr])) < 2:
            continue
        m = LogisticRegression(C=C, max_iter=5000).fit(Z[lab_idx[tr]], ylab[tr])
        fold_models.append((tr, te, m))
    if not fold_models:                                 # pragma: no cover
        return grid[0], {"cv_loglik": float("nan"), "grid": history}
    for cfg in grid:
        k_g, alpha, w_s, eta = cfg
        S = None if float(eta) == 0.0 else sparse_affinity(graph[0], graph[1], k_g,
                                                          speaker, w_s)
        lls = []
        for tr, te, model in fold_models:
            lr_all = model.decision_function(Z)
            if float(eta) == 0.0:
                sc = lr_all[lab_idx[te]]
            else:
                mask = np.zeros(len(y), bool)
                mask[lab_idx[tr]] = True
                f = propagate(S, ylab[tr], mask, alpha=alpha)
                pi = float(np.mean(ylab[tr]))
                sc = fuse(lr_all, mass_normalise(f, pi), eta)[lab_idx[te]]
            lls.append(log_likelihood(ylab[te], sc))
        mean_ll = float(np.mean(lls)) if lls else float("-inf")
        history.append({"config": list(cfg), "cv_loglik": mean_ll})
        if best is None or mean_ll > best[0]:
            best = (mean_ll, cfg)
    return best[1], {"cv_loglik": best[0], "grid": history}

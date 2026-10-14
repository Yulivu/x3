"""The baseline family B1-B11 from TASKBOOK section 7.1.

Every method here consumes features that were already preprocessed ONCE for the
region (standardisation + PCA fitted on the unlabelled region features, shared by
all methods), and selects its hyper-parameters with the shared stratified folds
inside ``L`` (``src.cv``).  Nothing in this module may look at test-region rows.

Scores are always logits / logit differences: larger means "more spoof".
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .cv import log_likelihood, stratified_folds
from .em_prior import (
    X3_HI,
    X3_LO,
    binom_se,
    clip_pi,
    em_cell_cal,
    estimate_pi,
    fuse,
    logit,
)

LOGREG_CS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
B3_SIGMAS = (0.25, 0.5, 1.0, 2.0)
B7_TAUS = (0.8, 0.9, 0.95)
B6_TAUS = (0.8, 0.9, 0.95)
KNN_K = 15

__all__ = [
    "LOGREG_CS", "B3_SIGMAS", "B7_TAUS", "B6_TAUS", "KNN_K",
    "MLPProbe", "fit_logreg_cv", "b1_logreg", "b2_line_a", "b3_platform_offset",
    "b4_gp", "b5_knn_prototype", "b6_zeroshot_em", "b7_naive_pseudo_label",
    "b8_fixmatch_lite", "b9_label_propagation", "b11_mlp",
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _zscore(s: Sequence[float]) -> np.ndarray:
    s = np.asarray(s, float)
    return (s - s.mean()) / (s.std() + 1e-9)


# ------------------------------------------------------------------ B1 / B11
def fit_logreg_cv(
    Z: np.ndarray,
    y: np.ndarray,
    folds: Optional[Sequence[int]] = None,
    grid: Sequence[float] = LOGREG_CS,
):
    """B1: L2 logistic regression, ``C`` chosen by the shared folds (hard rule 6)."""
    from sklearn.linear_model import LogisticRegression

    Z = np.asarray(Z, float)
    y = np.asarray(y)
    folds = np.asarray(folds) if folds is not None else stratified_folds(y)
    best = None
    for C in grid:
        scores = []
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            if len(np.unique(y[tr])) < 2 or te.sum() == 0:
                continue
            m = LogisticRegression(C=C, max_iter=5000).fit(Z[tr], y[tr])
            scores.append(log_likelihood(y[te], m.decision_function(Z[te])))
        if scores:
            ll = float(np.mean(scores))
            if best is None or ll > best[0]:
                best = (ll, float(C))
    C = best[1] if best else 1.0
    model = LogisticRegression(C=C, max_iter=5000).fit(Z, y)
    return model, {"C": C, "cv_loglik": best[0] if best else float("nan")}


def b1_logreg(Z, y, folds=None) -> Tuple[np.ndarray, Dict[str, Any]]:
    model, info = fit_logreg_cv(Z, y, folds=folds)
    return model.decision_function(np.asarray(Z, float)), info


class MLPProbe:
    """B11: one hidden layer (GELU), dropout, weight decay, full-batch Adam."""

    def __init__(self, hidden: int = 256, dropout: float = 0.3, weight_decay: float = 1e-2,
                 lr: float = 1e-3, steps: int = 100, seed: int = 20261015) -> None:
        self.hidden, self.dropout = int(hidden), float(dropout)
        self.weight_decay, self.lr, self.steps = float(weight_decay), float(lr), int(steps)
        self.seed = int(seed)
        self.net_ = None

    def fit(self, Z, y) -> "MLPProbe":
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        Z = np.asarray(Z, np.float32)
        y = np.asarray(y, np.float32)
        self.mu_, self.sd_ = Z.mean(0), Z.std(0) + 1e-6
        X = torch.from_numpy((Z - self.mu_) / self.sd_)
        t = torch.from_numpy(y)
        net = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        )
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lossf = nn.BCEWithLogitsLoss()
        net.train()
        for _ in range(self.steps):
            opt.zero_grad()
            loss = lossf(net(X).squeeze(-1), t)
            loss.backward()
            opt.step()
        self.net_ = net
        return self

    def logit(self, Z) -> np.ndarray:
        import torch

        if self.net_ is None:
            raise RuntimeError("MLPProbe.logit called before fit()")
        X = torch.from_numpy(((np.asarray(Z, np.float32) - self.mu_) / self.sd_))
        self.net_.eval()
        with torch.no_grad():
            return self.net_(X).squeeze(-1).numpy().astype(float)


def b11_mlp(Z, y, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    probe = MLPProbe(**kwargs).fit(Z, y)
    return probe.logit(Z), {"hidden": probe.hidden, "dropout": probe.dropout}


# ------------------------------------------------------------------- B2 / B3
def b2_line_a(
    scores_all: Sequence[float],
    cell_of: Sequence[Any],
    platform_of: Sequence[Any],
    labeled_mask: Sequence[bool],
    y_all: Sequence[float],
    variant: str = "bb",
    k: Optional[float] = None,
    lam: float = 1.0,
    lo: float = X3_LO,
    hi: float = X3_HI,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """B2 = Line A's M2: standardised probe score + ``logit(pi_cell)``.

    ``k=None`` is the undamped Bayes correction Line A uses; ``k`` finite gives
    the uncertainty-damped variant.  Mirrors ``fsc2.estimate`` + ``pi_est2.fuse``.
    """
    scores_all = np.asarray(scores_all, float)
    y_all = np.asarray(y_all, float)
    labeled_mask = np.asarray(labeled_mask, bool)
    cells = list(dict.fromkeys(list(cell_of)))
    pi, n_lab, phi = estimate_pi(
        cells, cell_of, labeled_mask, y_all, platform_of, variant=variant
    )
    pi_clip = np.array([pi[c] for c in cell_of], float)
    n_clip = np.array([n_lab[c] if n_lab[c] is not None else 3 for c in cell_of], float)
    corrected = fuse(scores_all, pi_clip, n_clip, k=k, lam=lam)
    return corrected, {"variant": variant, "phi": phi, "k": k, "lam": lam, "lo": lo, "hi": hi}


def b3_platform_offset(
    scores_all: Sequence[float],
    platform_of: Sequence[Any],
    labeled_mask: Sequence[bool],
    y_all: Sequence[float],
    sigma: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """B3: freeze the probe, fit ONE scalar per platform with an ``N(0, sigma^2)`` prior."""
    from scipy.optimize import minimize_scalar

    z = _zscore(scores_all)
    y = np.asarray(y_all, float)
    plat = np.asarray(platform_of)
    lab = np.asarray(labeled_mask, bool)
    offsets: Dict[Any, float] = {}
    for p in np.unique(plat):
        m = (plat == p) & lab
        if m.sum() == 0:
            offsets[p] = 0.0
            continue
        zz, yy = z[m], y[m]

        def obj(a):
            return float(np.sum(np.logaddexp(0.0, zz + a) - yy * (zz + a)) + a * a / (2 * sigma ** 2))

        res = minimize_scalar(obj, bounds=(-8.0, 8.0), method="bounded")
        offsets[p] = float(res.x)
    corrected = z + np.array([offsets[p] for p in plat], float)
    return corrected, {"sigma": sigma, "offsets": offsets}


# -------------------------------------------------------------------- B4/B5/B6
def b4_gp(Z, y, n_pca: int = 64, restarts: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """B4: sklearn RBF Gaussian process on PCA-64 (a stand-in for ADD-GP, not ADD-GP)."""
    from sklearn.decomposition import PCA
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    Z = np.asarray(Z, float)
    n_comp = int(min(n_pca, Z.shape[1], max(1, Z.shape[0] - 1)))
    pca = PCA(n_components=n_comp, random_state=20261015).fit(Z)
    P = pca.transform(Z)
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gp = GaussianProcessClassifier(
        kernel=kernel, optimizer=None if restarts == 0 else "fmin_l_bfgs_b",
        random_state=20261015, n_restarts_optimizer=restarts,
    ).fit(P, y)
    return gp.predict_proba(P)[:, 1], P, {"n_pca": n_comp, "note": "sklearn GPC, not ADD-GP"}


def b5_knn_prototype(
    Z: np.ndarray, y: np.ndarray, k: int = KNN_K, calibrate: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """B5: cosine k-NN distance-weighted vote, Platt-calibrated inside ``L``."""
    Z = np.asarray(Z, float)
    y = np.asarray(y)
    unit = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    kk = int(min(k, max(1, len(Z) - 1)))
    idx = np.argsort(-sim, axis=1)[:, :kk]
    w = np.take_along_axis(sim, idx, axis=1)
    lab = y[idx]
    vote = (w * (2 * lab - 1)).sum(1) / (np.abs(w).sum(1) + 1e-9)
    if calibrate:
        a, b = _platt(vote, y)
        score = a * vote + b
    else:
        score = vote
        a = b = None
    return score, {"k": kk, "calibrate": calibrate, "platt": (a, b)}


def _platt(score: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """1-D logistic calibration (Platt) by Newton/IRLS on the log-odds."""
    from sklearn.linear_model import LogisticRegression

    s = np.asarray(score, float).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(s, np.asarray(y))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def b6_zeroshot_em(
    zs_all: Sequence[float],
    cell_of: Sequence[Any],
    platform_of: Sequence[Any],
    labeled_mask: Sequence[bool],
    y_all: Sequence[float],
    unlabeled_mask: Optional[Sequence[bool]] = None,
    iters: int = 20,
    shrink_kappa: float = 10.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """B6: Platt-calibrated zero-shot logit + per-cell Saerens EM prior, shrunk cell->platform->global."""
    zs = np.asarray(zs_all, float)
    y = np.asarray(y_all, float)
    lab = np.asarray(labeled_mask, bool)
    cells = np.asarray(cell_of)
    plat = np.asarray(platform_of)
    unlab = ~lab if unlabeled_mask is None else np.asarray(unlabeled_mask, bool)

    a, b = _platt(zs[lab], y[lab]) if lab.sum() > 5 else (1.0, 0.0)
    cal = a * zs + b
    T = max(1e-3, _temperature(cal[lab], y[lab])) if lab.sum() > 5 else 1.0

    pi: Dict[Any, float] = {}
    for c in np.unique(cells):
        m = (cells == c) & unlab
        n_lab_c = int(((cells == c) & lab).sum())
        if m.sum() < 5:
            pi[c] = float(np.clip(y[lab].mean(), X3_LO, X3_HI)) if lab.sum() else 0.5
            continue
        init = (float(y[(cells == c) & lab].mean()) if n_lab_c >= 3
                else float(np.clip(y[lab].mean(), X3_LO, X3_HI)) if lab.sum() else 0.5)
        p, _ = em_cell_cal(cal[m], cal[lab], y[lab], T=T, init=init, iters=iters * 10, lo=X3_LO, hi=X3_HI)
        pi[c] = p
    # shrink cell -> platform -> global
    glob = float(np.clip((y[lab].mean() if lab.sum() else 0.5), X3_LO, X3_HI))
    plat_pi = {}
    for p in np.unique(plat):
        kids = [c for c in pi if str(c).split("|")[0] == p]
        if kids:
            wsum = sum(1.0 for _ in kids)
            plat_pi[p] = float((sum(pi[c] for c in kids) + shrink_kappa * glob) / (wsum + shrink_kappa))
        else:
            plat_pi[p] = glob
    out_pi = {}
    for c in np.unique(cells):
        p = str(c).split("|")[0]
        out_pi[c] = float(np.clip((pi[c] + shrink_kappa * plat_pi.get(p, glob)) / (1.0 + shrink_kappa),
                                  X3_LO, X3_HI))
    score = cal + np.array([logit(out_pi[c], eps=1e-3) - logit(glob, eps=1e-3) for c in cells])
    return score, {"platt": (a, b), "T": T, "pi": out_pi}


def _temperature(logit_s: Sequence[float], y: Sequence[float]) -> float:
    from scipy.optimize import minimize_scalar

    s = np.asarray(logit_s, float)
    y = np.asarray(y, float)

    def nll(logT):
        t = np.exp(logT)
        p = np.clip(_sigmoid(s / t), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 6), method="bounded").x))


# --------------------------------------------------------------------- B7/B8
def b7_naive_pseudo_label(
    Z_lab, y_lab, Z_unlab, tau: float = 0.9, rounds: int = 3, weight: float = 0.5
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """B7: global-confidence-threshold pseudo-labelling, NO class-ratio control.

    This is the baseline PAST has to beat: on a cell whose true fake rate is far
    from 0.5 it drifts toward the majority class.
    """
    from sklearn.linear_model import LogisticRegression

    Z_lab = np.asarray(Z_lab, float)
    y_lab = np.asarray(y_lab)
    Z_unlab = np.asarray(Z_unlab, float)
    model = LogisticRegression(C=1.0, max_iter=5000).fit(Z_lab, y_lab)
    X, t = Z_lab, y_lab
    added = 0
    pseudo_idx = np.array([], dtype=int)
    pseudo_y = np.array([], dtype=int)
    for _ in range(rounds):
        p = _sigmoid(model.decision_function(Z_unlab))
        conf = np.maximum(p, 1 - p)
        keep = conf >= tau
        if keep.sum() == 0:
            break
        pseudo_idx = np.where(keep)[0]
        pseudo_y = (p[keep] >= 0.5).astype(int)
        X = np.vstack([Z_lab, Z_unlab[keep]])
        t = np.concatenate([y_lab, pseudo_y])
        w = np.concatenate([np.ones(len(y_lab)), np.full(keep.sum(), weight)])
        model = LogisticRegression(C=1.0, max_iter=5000).fit(X, t, sample_weight=w)
        added = int(keep.sum())
    score = model.decision_function(Z_unlab)
    return score, {
        "tau": tau,
        "rounds": rounds,
        "added_last": added,
        "pseudo_idx": pseudo_idx,
        "pseudo_y": pseudo_y,
    }


def b8_fixmatch_lite(
    Z_lab, y_lab, Z_unlab_weak, Z_unlab_strong, tau: float = 0.9, rounds: int = 3,
    weight: float = 0.5, noise: float = 0.0, feature_dropout: float = 0.2,
    seed: int = 20261015,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """B8: weak-view hard label for the strong view; Gaussian noise + feature dropout before W10."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    Z_lab = np.asarray(Z_lab, float)
    y_lab = np.asarray(y_lab)
    Zw = np.asarray(Z_unlab_weak, float)
    Zs = np.asarray(Z_unlab_strong, float)
    model = LogisticRegression(C=1.0, max_iter=5000).fit(Z_lab, y_lab)
    for _ in range(rounds):
        p = _sigmoid(model.decision_function(Zw))
        conf = np.maximum(p, 1 - p)
        keep = conf >= tau
        if keep.sum() == 0:
            break
        hard = (p[keep] >= 0.5).astype(int)
        strong = Zs[keep].copy()
        if noise > 0:
            strong = strong + rng.normal(0, noise, strong.shape)
        if feature_dropout > 0:
            mask = rng.random(strong.shape) >= feature_dropout
            strong = strong * mask
        X = np.vstack([Z_lab, strong])
        t = np.concatenate([y_lab, hard])
        w = np.concatenate([np.ones(len(y_lab)), np.full(strong.shape[0], weight)])
        model = LogisticRegression(C=1.0, max_iter=5000).fit(X, t, sample_weight=w)
    return model.decision_function(Zw), {"tau": tau, "rounds": rounds}


# --------------------------------------------------------------------- B9
def b9_label_propagation(
    Z: np.ndarray, y_lab: np.ndarray, lab_mask: Sequence[bool], k: int = KNN_K,
    alpha: float = 0.9, iters: int = 20,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """B9: symmetric kNN affinity ``exp(cos)`` graph, label propagation from ``L``."""
    Z = np.asarray(Z, float)
    lab_mask = np.asarray(lab_mask, bool)
    y_lab = np.asarray(y_lab, float)
    unit = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)
    kk = int(min(k, max(1, len(Z) - 1)))
    idx = np.argsort(-sim, axis=1)[:, :kk]
    W = np.zeros_like(sim)
    rows = np.repeat(np.arange(len(Z)), kk)
    W[rows, idx.ravel()] = np.exp(np.take_along_axis(sim, idx, axis=1).ravel())
    W = np.maximum(W, W.T)
    d = W.sum(1) + 1e-12
    S = W / d[:, None]
    Y0 = np.zeros(len(Z))
    Y0[lab_mask] = 2 * y_lab - 1
    Y = Y0.copy()
    for _ in range(iters):
        Y = alpha * (S @ Y) + (1 - alpha) * Y0
    return Y, {"k": kk, "alpha": alpha, "iters": iters, "transductive": True}

"""W14 - label propagation (B9) and GraphSSL.

B9 is the plain transductive propagation already implemented in
``src.baselines.b9_label_propagation``.  GraphSSL is the C1 logit plus a scaled
propagated signal::

    score = C1_logit + eta * Y_prop

``eta`` is a design decision chosen inside ``L`` with the shared folds, by
refitting C1 on the training part of each fold and scoring the held-out part.
The graph contains the evaluation clips (this is a transductive method) and the
report must say so.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .baselines import KNN_K, b9_label_propagation
from .cv import log_likelihood, stratified_folds
from .metrics3 import eer
from .mpp import MPP

ETA_GRID = (0.25, 0.5, 1.0)

__all__ = ["ETA_GRID", "graph_ssl_score", "select_eta", "GraphSSL"]


def graph_ssl_score(c1_logit: Sequence[float], y_prop: Sequence[float], eta: float) -> np.ndarray:
    """``C1 logit + eta * Y_prop`` - the GraphSSL score."""
    return np.asarray(c1_logit, float) + float(eta) * np.asarray(y_prop, float)


def select_eta(
    Z: np.ndarray,
    y: np.ndarray,
    platform: Sequence[Any],
    language: Sequence[Any],
    cell: Sequence[Any],
    labeled_mask: Sequence[bool],
    eta_grid: Sequence[float] = ETA_GRID,
    folds: Optional[Sequence[int]] = None,
    refit: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    select_by: str = "log_likelihood",
    alpha: float = 0.9,
    knn: int = KNN_K,
) -> Tuple[float, Dict[str, Any]]:
    """Choose ``eta`` inside ``L`` with the shared folds.

    ``refit(tr_idx)`` must return a C1 logit for **every** row after training on
    the rows in ``tr_idx``; the default refits an MPP on the training part.  The
    graph is rebuilt on the whole pool for each fold, so the propagation term is
    transductive by construction and never sees a label outside ``tr_idx``.
    """
    Z = np.asarray(Z, float)
    y = np.asarray(y)
    labeled_mask = np.asarray(labeled_mask, bool)
    folds_all = (
        np.asarray(folds) if folds is not None
        else stratified_folds(y[labeled_mask])
    )
    lab_idx = np.where(labeled_mask)[0]

    scores: Dict[float, list] = {float(e): [] for e in eta_grid}
    for f in np.unique(folds_all):
        hold = lab_idx[folds_all == f]
        tr = np.ones(len(y), bool)
        tr[hold] = False
        c1 = refit(tr) if refit is not None else _refit_mpp(
            Z, y, platform, language, cell, tr, labeled_mask
        )
        y_prop, _ = b9_label_propagation(Z, y[tr & labeled_mask], tr & labeled_mask,
                                         k=knn, alpha=alpha)
        for eta in eta_grid:
            s = graph_ssl_score(c1[hold], y_prop[hold], eta)
            if select_by == "log_likelihood":
                scores[float(eta)].append(log_likelihood(y[hold], s))
            else:
                scores[float(eta)].append(-eer(y[hold], s))
    means = {e: float(np.mean(v)) if v else float("nan") for e, v in scores.items()}
    best = max(means, key=lambda e: means[e]) if means else float(ETA_GRID[0])
    return best, {"eta": best, "cv": means, "select_by": select_by,
                  "transductive": True, "alpha": alpha, "k": knn}


def _refit_mpp(Z, y, platform, language, cell, tr_mask, labeled_mask) -> np.ndarray:
    train = tr_mask & labeled_mask
    model = MPP(sigma_p=0.5, sigma_gamma=0.5, lam=1.0).fit(
        Z[train], y[train], np.asarray(platform)[train], np.asarray(language)[train],
        np.asarray(cell)[train],
    )
    return model.predict_logit(Z, platform, language, cell)


class GraphSSL:
    """C1 + ``eta * Y_prop`` with ``eta`` chosen on ``L``."""

    def __init__(self, eta: float = 0.5, alpha: float = 0.9, knn: int = KNN_K,
                 sigma_p: float = 0.5, sigma_gamma: float = 0.5, lam: float = 1.0) -> None:
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.knn = int(knn)
        self.sigma_p, self.sigma_gamma, self.lam = sigma_p, sigma_gamma, lam
        self.info_: Dict[str, Any] = {}

    def fit(self, Z, y, platform, language, cell, labeled_mask, select: bool = True) -> "GraphSSL":
        Z = np.asarray(Z, float)
        labeled_mask = np.asarray(labeled_mask, bool)
        self.model_ = MPP(sigma_p=self.sigma_p, sigma_gamma=self.sigma_gamma, lam=self.lam).fit(
            Z[labeled_mask], np.asarray(y)[labeled_mask],
            np.asarray(platform)[labeled_mask], np.asarray(language)[labeled_mask],
            np.asarray(cell)[labeled_mask],
        )
        self.c1_all_ = self.model_.predict_logit(Z, platform, language, cell)
        if select:
            self.eta, self.info_ = select_eta(
                Z, y, platform, language, cell, labeled_mask,
                alpha=self.alpha, knn=self.knn,
            )
            self.model_ = MPP(sigma_p=self.sigma_p, sigma_gamma=self.sigma_gamma, lam=self.lam).fit(
                Z[labeled_mask], np.asarray(y)[labeled_mask],
                np.asarray(platform)[labeled_mask], np.asarray(language)[labeled_mask],
                np.asarray(cell)[labeled_mask],
            )
            self.c1_all_ = self.model_.predict_logit(Z, platform, language, cell)
        self.y_prop_, _ = b9_label_propagation(
            Z, np.asarray(y)[labeled_mask], labeled_mask, k=self.knn, alpha=self.alpha
        )
        self.info_.setdefault("eta", self.eta)
        self.info_["transductive"] = True
        return self

    def score(self) -> np.ndarray:
        return graph_ssl_score(self.c1_all_, self.y_prop_, self.eta)

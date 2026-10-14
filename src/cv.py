"""Shared cross-validation folds (TASKBOOK hard rule 6).

Every method - baselines included - must select its hyper-parameters on the same
stratified folds inside ``L``.  Put the fold generator here so that requirement is
structural rather than a convention someone has to remember.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

CV_SEED = 20261015          # identical to the draw seed base, by design
N_SPLITS = 5
MAX_CONFIGS = 16            # hard rule 6: every method searches at most 16 configs

__all__ = ["CV_SEED", "N_SPLITS", "MAX_CONFIGS", "stratified_folds", "log_likelihood"]


def stratified_folds(
    y: Sequence[float], n_splits: int = N_SPLITS, seed: int = CV_SEED
) -> np.ndarray:
    """Fold assignment per labelled clip; deterministic for a given ``y`` and seed."""
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    folds = np.zeros(len(y), dtype=int)
    for i, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[te] = i
    return folds


def log_likelihood(y: Sequence[float], logit: Sequence[float]) -> float:
    """Mean Bernoulli log-likelihood of a logit prediction (the CV objective)."""
    y = np.asarray(y, float)
    z = np.asarray(logit, float)
    # numerically stable: -log(1+e^-z) - (1-y)*... rewritten as -softplus(-z) - (1-y)z
    ll = -np.logaddexp(0.0, -z) - (1.0 - y) * z
    return float(np.mean(ll))

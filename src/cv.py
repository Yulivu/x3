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

__all__ = [
    "CV_SEED", "N_SPLITS", "MAX_CONFIGS", "stratified_folds", "speaker_grouped_folds",
    "select_config", "log_likelihood",
]


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


def speaker_grouped_folds(
    speaker: Sequence[object], y: Sequence[float], n_splits: int = N_SPLITS,
    seed: int = CV_SEED,
) -> np.ndarray:
    """Group K-fold whose groups are SPEAKERS (NEW_TASKS 02, V-3).

    Every speaker lands in exactly one fold, so a held-out fold never shares a
    speaker with the folds it is scored against - that is what "simulate an unseen
    speaker while choosing the hyper-parameters" means.  Speakers are dealt
    round-robin inside each majority-label stratum so every fold still sees both
    classes whenever the stratum is large enough.
    """
    speaker = np.asarray(speaker)
    y = np.asarray(y, float)
    if speaker.shape != y.shape:
        raise ValueError("speaker and y must have the same shape")
    folds = np.zeros(len(speaker), dtype=int)
    uniq = np.unique(speaker)
    n_splits = int(max(1, min(int(n_splits), len(uniq))))
    rng = np.random.default_rng(int(seed))
    strata: dict = {}
    for s in uniq:
        m = speaker == s
        strata.setdefault(int(np.mean(y[m]) >= 0.5), []).append(s)
    for lab in sorted(strata):
        group = list(strata[lab])
        rng.shuffle(group)
        for i, s in enumerate(group):
            folds[speaker == s] = i % n_splits
    return folds


def select_config(
    configs: Sequence[object],
    evaluate,
    y: Sequence[float],
    folds: Sequence[int],
    default: int = 0,
):
    """Pick the config with the best mean held-out log-likelihood (hard rule 6).

    ``evaluate(cfg, train_mask, test_mask)`` must refit on the training folds and
    return scores for the held-out clips.  The history is returned so a report can
    show the whole grid instead of only the winner.
    """
    y = np.asarray(y, float)
    folds = np.asarray(folds)
    best = None
    history = []
    for cfg in configs:
        lls = []
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            lls.append(log_likelihood(y[te], evaluate(cfg, tr, te)))
        mean_ll = float(np.mean(lls)) if lls else float("-inf")
        history.append({"config": cfg, "cv_loglik": mean_ll})
        if best is None or mean_ll > best[0]:
            best = (mean_ll, cfg)
    if best is None:                                   # pragma: no cover
        return configs[int(default)], {"cv_loglik": float("nan"), "grid": history}
    return best[1], {"cv_loglik": best[0], "grid": history}

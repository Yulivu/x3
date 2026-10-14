"""W9 - leakage and negative-control audit (TASKBOOK W9).

Four checks that can only ever *remove* a claim:

1. label permutation inside each platform - every method must fall back to ~50 EER;
2. cell-metadata permutation - C1's gain over B1 must disappear;
3. the strict mask (drop evaluation clips whose speaker is labelled) vs the clips mask,
   plus a video-disjoint mask when the manifest exposes a video id;
4. near-duplicate detection - the share of evaluation clips whose nearest labelled
   neighbour has cosine > 0.99.

Plus the shortcut reference: how well platform x language alone predicts the label.
That number is the floor any "the model learned something about the audio" claim has
to clear.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .metrics3 import eer

NEAR_DUP_COS = 0.99
PERMUTED_EER_TOL = 3.0

__all__ = [
    "permute_labels_within",
    "permute_cells",
    "nearest_neighbour_overlap",
    "metadata_only_scores",
    "video_disjoint_mask",
    "audit_row",
    "NEAR_DUP_COS",
]


def permute_labels_within(y: Sequence[float], platform: Sequence[Any], seed: int) -> np.ndarray:
    """Shuffle labels inside each platform: the shortcut signal survives, the rest dies."""
    rng = np.random.default_rng(int(seed))
    y = np.asarray(y).copy()
    plat = np.asarray(platform)
    out = y.copy()
    for p in np.unique(plat):
        m = np.where(plat == p)[0]
        out[m] = rng.permutation(y[m])
    return out


def permute_cells(cell: Sequence[Any], seed: int) -> np.ndarray:
    """Shuffle the cell id across clips (the metadata becomes uninformative)."""
    rng = np.random.default_rng(int(seed))
    cell = np.asarray(cell)
    return cell[rng.permutation(len(cell))]


def nearest_neighbour_overlap(
    Z_lab: np.ndarray, Z_eval: np.ndarray, threshold: float = NEAR_DUP_COS
) -> Dict[str, float]:
    """Share of evaluation clips with a labelled neighbour above ``threshold`` cosine."""
    Z_lab = np.asarray(Z_lab, float)
    Z_eval = np.asarray(Z_eval, float)
    if Z_lab.size == 0 or Z_eval.size == 0:
        return {"share": float("nan"), "n_eval": 0}
    a = Z_lab / (np.linalg.norm(Z_lab, axis=1, keepdims=True) + 1e-9)
    b = Z_eval / (np.linalg.norm(Z_eval, axis=1, keepdims=True) + 1e-9)
    best = (b @ a.T).max(axis=1)
    return {"share": float((best > threshold).mean()), "n_eval": int(best.size)}


def video_disjoint_mask(
    video_of: Sequence[Any], labeled_mask: Sequence[bool], eval_mask: Sequence[bool]
) -> np.ndarray:
    """True for evaluation clips whose source video carries no label."""
    v = np.asarray(video_of)
    lab_videos = set(v[np.asarray(labeled_mask, bool)].tolist())
    ev = np.asarray(eval_mask, bool)
    return np.array([x not in lab_videos for x in v[ev]], dtype=bool)


def metadata_only_scores(
    platform: Sequence[Any],
    language: Sequence[Any],
    y: Sequence[float],
    folds: Optional[Sequence[int]] = None,
    seed: int = 20261015,
) -> np.ndarray:
    """Out-of-fold scores of a platform x language only classifier.

    TASKBOOK names LightGBM; scikit-learn's histogram gradient booster is the
    same model family and is already a dependency, so it is used here.  The
    classifier never sees the audio, so its EER is the shortcut level.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold

    plat = np.asarray(platform)
    lang = np.asarray(language)
    y = np.asarray(y)
    cats = sorted({f"{p}|{l}" for p, l in zip(plat, lang)})
    index = {c: i for i, c in enumerate(cats)}
    X = np.array([[index[f"{p}|{l}"]] for p, l in zip(plat, lang)], float)

    if folds is None:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = np.zeros(len(y), int)
        for i, (_, te) in enumerate(skf.split(X, y)):
            folds[te] = i
    folds = np.asarray(folds)

    scores = np.zeros(len(y), float)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            scores[te] = 0.0
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, categorical_features=[0], random_state=seed
        ).fit(X[tr], y[tr])
        scores[te] = clf.predict_proba(X[te])[:, 1]
    return scores


def audit_row(
    name: str,
    y: Sequence[float],
    scores: Sequence[float],
    *,
    expect_fifty: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """One audit line; ``flag`` is set when the expectation is violated."""
    e = float(eer(y, scores))
    flag = bool(expect_fifty and abs(e - 50.0) > PERMUTED_EER_TOL)
    row: Dict[str, Any] = {"check": name, "eer": e, "flag": flag}
    if extra:
        row.update(dict(extra))
    return row

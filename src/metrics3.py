"""X3 three-view metrics (TASKBOOK section 2, hard rules 3-4).

Conventions
-----------
* spoof is the positive class: ``y = 1`` means spoof.
* scores are logits (or logit differences): a larger score means "more spoof".
* EER is reported as a percentage; identical scores return ``50.0``.
* every number is always reported in three views at once:
  ``pooled`` (all evaluation clips), ``within_macro`` (per-platform EER averaged
  with equal weight over platforms that carry >= ``MIN_PER_CLASS`` clips of
  *both* classes) and ``worst`` (the largest such platform EER).

``eer_legacy`` is a verbatim copy of Line A's implementation
(``wx2/scripts/eval_mlitw.py::eer``) and is kept unchanged so X3 numbers can be
reconciled against the frozen Line A tables.  ``eer`` is the tie-stable
``sklearn.metrics.roc_curve`` version and is what the X3 pipeline uses; the unit
tests pin the two together on random data.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

__all__ = [
    "MIN_PER_CLASS",
    "eer",
    "eer_legacy",
    "eer_with_threshold",
    "hter",
    "auc",
    "platform_eers",
    "summarize",
]

# A platform needs at least this many clips of *each* class to be scored
# (Line A rule, kept unchanged).
MIN_PER_CLASS = 30


def _arrays(y: Sequence[float], s: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    yy = np.asarray(y).astype(np.int64).ravel()
    ss = np.asarray(s, dtype=float).ravel()
    if yy.shape != ss.shape:
        raise ValueError(f"y and s must have the same shape, got {yy.shape} and {ss.shape}")
    return yy, ss


def eer_legacy(y: Sequence[float], s: Sequence[float]) -> float:
    """EER in %; spoof = positive; larger score = more spoof.

    Verbatim copy of Line A (``wx2/scripts/eval_mlitw.py``).  Do not edit: it is
    the reconciliation reference.
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    if len(np.unique(s)) < 2:
        return 50.0
    order = np.argsort(s)
    y = y[order]
    ranks = np.arange(len(y), 0, -1)
    tp = np.cumsum(y[::-1])[::-1].astype(float)  # positives above threshold
    fp = ranks.astype(float) - tp
    P = max(1, int(y.sum()))
    Nn = max(1, len(y) - int(y.sum()))
    fnr = 1.0 - tp / P
    fpr = fp / Nn
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2 * 100)


def eer(y: Sequence[float], s: Sequence[float]) -> float:
    """Tie-stable EER in %, computed from ``sklearn.metrics.roc_curve``.

    * identical scores -> 50.0
    * a single-class ``y`` is undefined for EER -> 50.0
    * ``eer(y, -s) == 100 - eer(y, s)``
    """
    yy, ss = _arrays(y, s)
    if yy.size == 0:
        return float("nan")
    if np.unique(ss).size < 2 or np.unique(yy).size < 2:
        return 50.0
    fpr, tpr, _ = roc_curve(yy, ss, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2 * 100)


def eer_with_threshold(y: Sequence[float], s: Sequence[float]) -> Tuple[float, float]:
    """Return ``(eer_pct, threshold)`` where a clip is spoof iff ``score >= threshold``."""
    yy, ss = _arrays(y, s)
    if yy.size == 0:
        return float("nan"), float("nan")
    if np.unique(ss).size < 2 or np.unique(yy).size < 2:
        return 50.0, float(np.median(ss)) if ss.size else float("nan")
    fpr, tpr, thr = roc_curve(yy, ss, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2 * 100), float(thr[i])


def hter(y: Sequence[float], s: Sequence[float], threshold: float) -> float:
    """Half total error rate in % at a *fixed* threshold (from the labelled set)."""
    yy, ss = _arrays(y, s)
    if yy.size == 0:
        return float("nan")
    pos = yy == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    pred = ss >= float(threshold)
    fnr = float(np.mean(~pred[pos]))
    fpr = float(np.mean(pred[neg]))
    return (fnr + fpr) / 2 * 100


def auc(y: Sequence[float], s: Sequence[float]) -> float:
    """ROC AUC in [0, 1]; NaN when only one class is present."""
    yy, ss = _arrays(y, s)
    if yy.size == 0 or np.unique(yy).size < 2:
        return float("nan")
    return float(roc_auc_score(yy, ss))


def platform_eers(
    y: Sequence[float],
    s: Sequence[float],
    platform: Sequence[Any],
    min_per_class: int = MIN_PER_CLASS,
) -> Dict[Any, float]:
    """Per-platform EER for platforms that carry >= ``min_per_class`` of both classes."""
    yy, ss = _arrays(y, s)
    plat = np.asarray(platform).ravel()
    if plat.shape[0] != yy.shape[0]:
        raise ValueError("platform must have the same length as y")
    out: Dict[Any, float] = {}
    for p in np.unique(plat):
        m = plat == p
        n_pos = int((yy[m] == 1).sum())
        n_neg = int((yy[m] == 0).sum())
        if min(n_pos, n_neg) < min_per_class:
            continue
        out[p] = eer(yy[m], ss[m])
    return out


def summarize(
    y: Sequence[float],
    s: Sequence[float],
    platform: Sequence[Any],
    threshold: Optional[float] = None,
    min_per_class: int = MIN_PER_CLASS,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The full three-view row used by every ``results/*.csv`` file.

    Returns ``pooled_eer``, ``within_macro_eer``, ``worst_platform``,
    ``worst_platform_eer``, ``auc``, ``hter_pooled`` plus ``eer_<platform>`` for
    every scorable platform.  Missing/undefined numbers are ``nan`` so a reader
    can never mistake "not measured" for "perfect".
    """
    yy, ss = _arrays(y, s)
    plat = np.asarray(platform).ravel()
    if plat.shape[0] != yy.shape[0]:
        raise ValueError("platform must have the same length as y")

    per = platform_eers(yy, ss, plat, min_per_class=min_per_class)
    row: Dict[str, Any] = {}
    row["pooled_eer"] = eer(yy, ss)
    row["within_macro_eer"] = float(np.mean(list(per.values()))) if per else float("nan")
    if per:
        worst_key = max(per, key=lambda k: per[k])
        row["worst_platform"] = worst_key
        row["worst_platform_eer"] = float(per[worst_key])
    else:
        row["worst_platform"] = ""
        row["worst_platform_eer"] = float("nan")
    row["auc"] = auc(yy, ss)
    row["hter_pooled"] = hter(yy, ss, threshold) if threshold is not None else float("nan")
    row["n_eval"] = int(yy.size)
    row["n_pos"] = int((yy == 1).sum())
    for p in sorted(per, key=lambda k: str(k)):
        row[f"eer_{p}"] = float(per[p])
    if extra:
        for k, v in extra.items():
            row[k] = v
    return row

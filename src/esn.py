"""ESN - enrolled bona fide speaker normalisation (NEW_TASKS 02, U8 / N5).

The U1 gap (unseen speakers cost 1.7-2.2x) says speaker identity is the main
nuisance.  A deployed system that protects a public figure can get a handful of
recordings that are *known to be genuine* - an enrollment set ``R_s`` - without
any spoof label.  ESN uses exactly that:

* ``z~_i = [z_i, z_i - mu_s]`` where ``mu_s`` is the mean of ``R_s`` in the same
  (post-PCA) space, so the second block is the speaker-centred residual.  The gate
  version ``ESN-P`` uses the same-platform mean when ``R_s`` holds at least two
  clips from the clip's own platform;
* ``OC`` is the one-class control: the nearest-neighbour cosine *distance* to
  ``R_s`` with no training at all;
* ``OC+B1`` fuses the standardised OC distance with B1's standardised logit, and
  the weight is chosen inside ``L``;
* ``B1+enroll`` is the "just add the enrollment clips as extra genuine labels"
  control, which is the cheap alternative to a feature change.

Enrollment clips never enter ``L`` and never enter ``E``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "choose_enrollment", "speaker_means", "esn_features", "one_class_scores",
    "standardise", "fuse",
]


def choose_enrollment(speaker: Sequence[object], y: Sequence[float], m: int,
                      seed: int = 20261015) -> np.ndarray:
    """Pick ``min(m, floor(n_real / 2))`` bona fide clips per speaker.

    The ``floor(n_real/2)`` cap is the taskbook's rule: enrollment must not eat the
    speaker's whole genuine supply, which also has to serve as evidence in ``E``.
    """
    speaker = np.asarray(speaker)
    y = np.asarray(y)
    enroll = np.zeros(len(speaker), bool)
    if int(m) <= 0:
        return enroll
    rng = np.random.default_rng(int(seed))
    for s in np.unique(speaker):
        pos = np.where((speaker == s) & (y == 0))[0]
        take = int(min(int(m), pos.size // 2))
        if take <= 0:
            continue
        enroll[rng.choice(pos, size=take, replace=False)] = True
    return enroll


def speaker_means(Z: np.ndarray, speaker: Sequence[object], enroll: Sequence[bool],
                  platform: Optional[Sequence[object]] = None,
                  m: int = 0) -> Dict[str, Any]:
    """Per-speaker (and per speaker x platform) means of the enrollment features."""
    Z = np.asarray(Z, float)
    speaker = np.asarray(speaker)
    enroll = np.asarray(enroll, bool)
    by_speaker: Dict[Any, np.ndarray] = {}
    by_platform: Dict[Tuple[Any, Any], np.ndarray] = {}
    n_platform: Dict[Tuple[Any, Any], int] = {}
    plat = None if platform is None else np.asarray(platform)
    for s in np.unique(speaker):
        idx = np.where(enroll & (speaker == s))[0]
        if idx.size == 0:
            continue
        by_speaker[s] = Z[idx].mean(0)
        if plat is None:
            continue
        for p in np.unique(plat[idx]):
            j = idx[plat[idx] == p]
            by_platform[(s, p)] = Z[j].mean(0)
            n_platform[(s, p)] = int(j.size)
    global_mu = Z[enroll].mean(0) if enroll.any() else np.zeros(Z.shape[1])
    return {"speaker": by_speaker, "platform": by_platform,
            "platform_counts": n_platform, "global": global_mu,
            "n_enrolled_speakers": int(len(by_speaker)), "m": int(m)}


def esn_features(Z: np.ndarray, speaker: Sequence[object], means: Dict[str, Any],
                 platform: Optional[Sequence[object]] = None,
                 variant: str = "plain", min_platform_clips: int = 2) -> np.ndarray:
    """``[z, z - mu]`` for every clip; ``variant='platform'`` is ESN-P.

    A speaker with no enrollment at all falls back to the population mean of the
    enrollment features, which is recorded by the caller (never hidden).
    """
    Z = np.asarray(Z, float)
    speaker = np.asarray(speaker)
    if variant not in ("plain", "platform"):
        raise ValueError(f"unknown ESN variant {variant!r}")
    if int(means.get("m", 0)) <= 0:
        return Z                     # m=0: byte-identical to B1's design matrix
    global_mu = np.asarray(means["global"], float)
    mu = np.empty_like(Z)
    plat = None if platform is None else np.asarray(platform)
    for i, s in enumerate(speaker):
        row = means["speaker"].get(s)
        if row is None:
            row = global_mu
        if variant == "platform" and plat is not None:
            key = (s, plat[i])
            if means["platform_counts"].get(key, 0) >= int(min_platform_clips):
                row = means["platform"][key]
        mu[i] = row
    centred = Z - mu
    return np.ascontiguousarray(np.concatenate([Z, centred], axis=1), dtype=np.float32)


def _cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A @ B.T


def one_class_scores(Z: np.ndarray, speaker: Sequence[object],
                     enroll: Sequence[bool]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """``OC``: negative maximum cosine similarity to the speaker's enrollment set.

    Higher means "further from the known-genuine recordings", i.e. more likely
    spoof - the same polarity as every other score in X3.
    """
    Z = np.asarray(Z, float)
    speaker = np.asarray(speaker)
    enroll = np.asarray(enroll, bool)
    out = np.full(len(Z), np.nan)
    fallback = np.where(enroll)[0]
    n_fallback = 0
    for s in np.unique(speaker):
        idx = np.where(speaker == s)[0]
        ref = np.where(enroll & (speaker == s))[0]
        if ref.size == 0:
            ref = fallback
            n_fallback += int(idx.size)
        if ref.size == 0:                             # pragma: no cover
            continue
        out[idx] = -_cosine(Z[idx], Z[ref]).max(axis=1)
    return out, {"n_enrolled": int(enroll.sum()),
                 "n_clips_without_own_enrollment": int(n_fallback)}


def standardise(s: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-sd over the region (label-free, like the PCA step)."""
    s = np.asarray(s, float)
    finite = np.isfinite(s)
    if finite.sum() < 2:
        return s - np.nanmean(s)
    mu = float(np.mean(s[finite]))
    sd = float(np.std(s[finite])) + 1e-9
    return (s - mu) / sd


def fuse(s_a: np.ndarray, s_b: np.ndarray, w: float) -> np.ndarray:
    """``z(s_a) + w * z(s_b)``: the "logit fusion" of the task."""
    return standardise(s_a) + float(w) * standardise(s_b)

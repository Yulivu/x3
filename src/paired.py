"""The two intervals a difference needs (NEW_TASKS 02, V-1).

The v1 rule ("2.5/97.5 quantiles of the per-draw differences") describes
*single-draw fluctuation*, not the uncertainty of the mean difference, so it
declares a real but small effect "not significant".  Every comparison now carries
both:

* **(a) sampling level** - bootstrap the per-draw differences (B=2000) and take the
  2.5/97.5 quantiles of the resampled MEAN.  This is the "how much would another
  label draw move it" interval.  Diagnostic.
* **(b) population level, speaker-clustered** - inside each draw resample the
  evaluation SPEAKERS with replacement (B_s=200), giving ``delta_d^(b)``; average
  each replicate index over the draws (``mean_b delta_d^(b)``) and take the
  2.5/97.5 quantiles of those B_s means.  This is the "how much would a different
  set of speakers move it" interval, and a gate that says "the interval excludes 0"
  uses this one.

Replicate ``b`` uses the SAME resample recipe in every draw (the draw-pairing is
what the taskbook asks for: "for each b, average over the 100 draws"), which is
exact whenever the draws share their evaluation speakers - the ``strict_draw``
case.  A draw whose cluster count differs falls back to its own resample.

Both need the per-clip scores of every draw, so the drivers persist them with
``save_draw_scores`` and the intervals are recomputable without re-running a model.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics3 import eer

BOOT_DRAWS = 2000
SPEAKER_BOOTS = 200
SEED = 20261015

__all__ = [
    "BOOT_DRAWS", "SPEAKER_BOOTS", "pooled_eer", "two_intervals",
    "save_draw_scores", "load_draw_scores",
]


def pooled_eer(y: np.ndarray, s: np.ndarray, speaker: Optional[np.ndarray] = None) -> float:
    return float(eer(y, s))


def _speaker_indices(speaker: np.ndarray) -> List[np.ndarray]:
    out = []
    for sp in np.unique(speaker):
        out.append(np.where(speaker == sp)[0])
    return out


def two_intervals(
    per_draw: Sequence[Dict[str, np.ndarray]],
    metric_fn: Callable[..., float] = pooled_eer,
    B: int = BOOT_DRAWS,
    B_s: int = SPEAKER_BOOTS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Both intervals for ``metric(a) - metric(b)`` on draw-paired data.

    ``per_draw[i]`` must hold ``y``, ``sa``, ``sb`` and ``speaker`` for one draw,
    all over the SAME evaluation clips.
    """
    if not per_draw:
        return {"mean": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    deltas = np.array([
        metric_fn(d["y"], d["sa"], d["speaker"]) - metric_fn(d["y"], d["sb"], d["speaker"])
        for d in per_draw
    ], float)
    # (a) bootstrap the mean over draws
    means = deltas[rng.integers(0, len(deltas), size=(int(B), len(deltas)))].mean(axis=1)
    a_lo, a_hi = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    # the v1 rule, kept so every report can show old vs new side by side
    old_lo, old_hi = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))

    # (b) speaker-clustered: replicate b uses the same resample recipe in every draw
    groups = [_speaker_indices(d["speaker"]) for d in per_draw]
    k_max = max(len(g) for g in groups)
    recipe = rng.integers(0, k_max, size=(int(B_s), k_max))
    boot = np.empty(int(B_s), float)
    for b in range(int(B_s)):
        vals = np.empty(len(per_draw), float)
        for i, d in enumerate(per_draw):
            gs = groups[i]
            pick = (recipe[b] if len(gs) == k_max
                    else rng.integers(0, len(gs), size=len(gs)))
            idx = np.concatenate([gs[k] for k in pick])
            vals[i] = (metric_fn(d["y"][idx], d["sa"][idx], d["speaker"][idx])
                       - metric_fn(d["y"][idx], d["sb"][idx], d["speaker"][idx]))
        boot[b] = vals.mean()
    b_lo, b_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {
        "mean": float(deltas.mean()),
        "sd": float(deltas.std()),
        "a_lo": a_lo, "a_hi": a_hi,
        "old_lo": old_lo, "old_hi": old_hi,
        "old_contains_zero": bool(old_lo <= 0.0 <= old_hi),
        "b_lo": b_lo, "b_hi": b_hi,
        # the gate reads (b) only; False means "the interval is clear of 0"
        "b_contains_zero": bool(b_lo <= 0.0 <= b_hi),
        "b_excludes_zero": bool(not (b_lo <= 0.0 <= b_hi)),
        "n_draws": int(len(per_draw)),
        "n_a": int(B), "n_b": int(B_s),
    }


def save_draw_scores(path, draw: int, y: np.ndarray, scores: Dict[str, np.ndarray],
                     speaker: np.ndarray, extra: Optional[Dict[str, Any]] = None) -> None:
    """Persist one draw so the intervals can be recomputed without a model.

    ``speaker`` is coerced to a fixed-width unicode array: a pandas object column
    would be written as an object array, and ``np.load`` refuses those without
    ``allow_pickle`` (which no analysis script should need).
    """
    import json
    from pathlib import Path

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    sp = np.asarray(speaker)
    if sp.dtype.kind in "OU" or sp.dtype == object:
        sp = sp.astype(str)
    np.savez_compressed(path / f"draw{draw:04d}.npz", y=y, speaker=sp,
                        **{f"s_{k}": v for k, v in scores.items()})
    if extra is not None:
        (path / f"draw{draw:04d}.json").write_text(json.dumps(extra, default=str),
                                                   encoding="utf-8")


def load_draw_scores(path, method_a: str, method_b: str) -> List[Dict[str, np.ndarray]]:
    """Load every persisted draw as the ``per_draw`` list ``two_intervals`` wants."""
    from pathlib import Path

    out: List[Dict[str, np.ndarray]] = []
    for f in sorted(Path(path).glob("draw*.npz")):
        with np.load(f) as z:
            ka, kb = f"s_{method_a}", f"s_{method_b}"
            if ka not in z.files or kb not in z.files:
                continue
            out.append({"y": z["y"], "speaker": z["speaker"],
                        "sa": z[ka], "sb": z[kb]})
    return out

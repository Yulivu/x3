"""Label-draw protocol (TASKBOOK hard rule 5; W3/W5/W7).

Draw ``d`` always uses ``seed = 20261015 + d`` and its sampled clip ids are
written to ``draws/<protocol>_<d>.json`` so that every method inside one draw
shares exactly the same labelled set ``L`` and evaluation set ``E``.

Cell quotas follow W5: take at most ``per_cell`` clips per cell, but never more
than ``floor(n_c / 2)`` so that ``E`` keeps at least as many clips as ``L``.  A
cell whose ``floor(n_c / 2)`` is below ``MIN_CELL`` cannot supply a usable
label/eval pair at all and is dropped; the actual quota used is recorded for
every cell either way (so the small ``k`` budget points stay runnable).

The protocol is region-agnostic: pass the dev-region rows only, and the draw is a
dev-region draw.  Nothing here ever looks at a test-region row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

SEED_BASE = 20261015
MIN_CELL = 4
# W7 budget curve: per-cell quotas m_c for the five k values.
BUDGET_M = {30: 1, 100: 3, 290: 10, 600: 20, 1200: 40}

__all__ = [
    "SEED_BASE",
    "MIN_CELL",
    "BUDGET_M",
    "draw_seed",
    "draw_labels",
    "strict_eval_mask",
    "Dataset",
    "build_draw",
]


def draw_seed(d: int) -> int:
    return SEED_BASE + int(d)


def draw_labels(
    cell_of_clip: Sequence[Any],
    per_cell: Union[int, Mapping[Any, int]],
    seed: int,
    cap_half: bool = True,
    min_cell: int = MIN_CELL,
) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
    """Sample the labelled clips for one draw.

    Returns ``(labeled_positions, info)`` where ``labeled_positions`` indexes the
    array that was passed in and ``info`` records, per cell, the pool size, the
    quota actually used and whether the cell was dropped.
    """
    cell_of_clip = np.asarray(cell_of_clip)
    rng = np.random.default_rng(int(seed))
    chosen: List[np.ndarray] = []
    info: Dict[str, Dict[str, Any]] = {}
    for c in np.unique(cell_of_clip):
        idx = np.where(cell_of_clip == c)[0]
        n = int(idx.size)
        want = int(per_cell[c]) if isinstance(per_cell, Mapping) else int(per_cell)
        if n // 2 < min_cell:
            info[str(c)] = {"n": n, "m": 0, "requested": want, "skipped": True}
            continue
        m = min(want, n // 2) if cap_half else min(want, n)
        chosen.append(rng.choice(idx, size=m, replace=False))
        info[str(c)] = {"n": n, "m": m, "requested": want, "skipped": False}
    labeled = np.sort(np.concatenate(chosen)) if chosen else np.array([], dtype=int)
    return labeled, info


def strict_eval_mask(
    speaker_of_clip: Sequence[Any],
    labeled_positions: Sequence[int],
    eval_positions: Sequence[int],
) -> np.ndarray:
    """Clips-only mask vs the strict mask (drop evaluation clips whose speaker is labelled)."""
    sp = np.asarray(speaker_of_clip)
    labeled_speakers = set(np.asarray(sp)[np.asarray(labeled_positions, dtype=int)].tolist())
    ev = np.asarray(eval_positions, dtype=int)
    return np.array([s not in labeled_speakers for s in sp[ev]], dtype=bool)


@dataclass
class Dataset:
    """The rows one draw works on (already restricted to the dev region)."""

    clip_id: List[Any]
    cell: List[Any]
    platform: List[Any]
    speaker: List[Any]

    def __post_init__(self) -> None:
        n = len(self.clip_id)
        for name in ("cell", "platform", "speaker"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, expected {n}")

    def __len__(self) -> int:
        return len(self.clip_id)


@dataclass
class Draw:
    protocol: str
    d: int
    seed: int
    per_cell: Dict[str, Any] = field(default_factory=dict)
    labeled_ids: List[Any] = field(default_factory=list)
    info: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_pool: int = 0

    @property
    def n_labeled(self) -> int:
        return len(self.labeled_ids)

    def to_json(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "d": self.d,
            "seed": self.seed,
            "per_cell": self.per_cell,
            "labeled": self.labeled_ids,
            "info": self.info,
            "n_pool": self.n_pool,
            "n_labeled": self.n_labeled,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)


def build_draw(
    ds: Dataset,
    protocol: str,
    d: int,
    per_cell: Union[int, Mapping[Any, int]],
    cap_half: bool = True,
    min_cell: int = MIN_CELL,
) -> Draw:
    """One draw over ``ds``; the seed is derived from ``d`` (never passed in)."""
    seed = draw_seed(d)
    pos, info = draw_labels(ds.cell, per_cell, seed, cap_half=cap_half, min_cell=min_cell)
    quota = (
        {str(k): int(v) for k, v in per_cell.items()}
        if isinstance(per_cell, Mapping)
        else {"*": int(per_cell)}
    )
    return Draw(
        protocol=protocol,
        d=int(d),
        seed=seed,
        per_cell=quota,
        labeled_ids=[ds.clip_id[i] for i in pos.tolist()],
        info=info,
        n_pool=len(ds),
    )

"""Shared, label-discipline helpers for NEW_TASKS 05 drivers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def zero_scores(data_dir: Path, backbone: str, clip_id: np.ndarray) -> np.ndarray:
    d = pd.read_csv(Path(data_dir) / "x3" / "feats" / backbone / "scores.csv")
    key = dict(zip(d["clip_id"].astype(int), d["logit_fake_minus_real"].astype(float)))
    return np.asarray([key[int(i)] for i in clip_id], float)


def speaker_split(speaker: np.ndarray, draw: int, frac: float = .5) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic disjoint speaker masks; labels never participate."""
    sp = np.asarray(speaker).astype(str)
    uniq = np.array(sorted(set(sp)))
    rng = np.random.default_rng(20261015 + int(draw))
    rng.shuffle(uniq)
    left = set(uniq[: max(1, int(round(len(uniq) * frac)))])
    return np.isin(sp, list(left)), ~np.isin(sp, list(left))


def enrol_mask(y: np.ndarray, speaker: np.ndarray, m: int, draw: int) -> np.ndarray:
    """Simulated known-real enrolment.  Its true label is used only to construct R_s."""
    y, sp = np.asarray(y), np.asarray(speaker).astype(str)
    out = np.zeros(len(y), bool)
    rng = np.random.default_rng(20261015 + int(draw))
    for s in np.unique(sp):
        idx = np.where((sp == s) & (y == 0))[0]
        take = min(int(m), len(idx) // 2)
        if take:
            out[rng.choice(idx, take, replace=False)] = True
    return out


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

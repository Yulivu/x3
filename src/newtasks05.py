"""Shared, label-discipline helpers for NEW_TASKS 05 drivers."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - sklearn always ships it
    threadpool_limits = None


@contextlib.contextmanager
def one_thread_blas():
    """Keep one pool worker on one core; 16 vCPU cannot host 7 x N BLAS threads."""
    if threadpool_limits is None:  # pragma: no cover
        yield
        return
    with threadpool_limits(limits=1):
        yield


def worker_slots() -> int:
    """Workers a single driver may use, from the shared CPU budget.

    The AutoDL container is 16 vCPU / 80 GiB even though ``nproc`` reports the
    208 cores of the host, so every driver has to read the budget instead of
    trusting ``cpu_count``.  The budget is split between the two CPU lanes.
    """
    raw = os.environ.get("X3_CPU_BUDGET")
    if not raw:
        path = Path(__file__).resolve().parents[1] / "logs" / "x3_cpu_budget"
        raw = path.read_text(encoding="utf-8").strip() if path.exists() else "8"
    try:
        budget = int(str(raw).strip())
    except (TypeError, ValueError):
        budget = 8
    return max(1, budget // 2)


def zero_scores(data_dir: Path, backbone: str, clip_id: np.ndarray) -> np.ndarray:
    path = Path(data_dir) / "x3" / "feats" / backbone / "scores.csv"
    if not path.exists():
        # The frozen Line A block has no anti-deepfake head, so its "zero-shot"
        # column is an explicit constant rather than a missing file.
        return np.zeros(len(clip_id), float)
    d = pd.read_csv(path)
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

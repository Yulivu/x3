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


def oc_scores(Xn: np.ndarray, speaker: np.ndarray, enrol: np.ndarray) -> np.ndarray:
    """One-class score: ``+min`` cosine distance to the speaker's enrolment set.

    Sign convention (NEW_TASKS 06 section 0): every score is "larger = more
    spoof", so the distance is used with a POSITIVE sign.  ``Xn`` must already
    be L2-normalised, which makes ``1 - x.y`` the cosine distance.
    """
    Xn = np.asarray(Xn, float)
    sp = np.asarray(speaker).astype(str)
    en = np.asarray(enrol, bool)
    if Xn.shape[0] != sp.shape[0] or en.shape != sp.shape:
        raise ValueError("Xn, speaker and enrol must share the first dimension")
    out = np.full(sp.shape[0], np.nan)
    glob = np.where(en)[0]
    glob_mat = Xn[glob] if glob.size else None
    for s in np.unique(sp):
        idx = np.where(sp == s)[0]
        r = idx[en[idx]]
        ref = Xn[r] if r.size else glob_mat
        if ref is None or ref.size == 0:
            continue
        out[idx] = (1.0 - Xn[idx] @ np.asarray(ref, float).T).min(axis=1)
    if np.isnan(out).any():
        fill = float(np.nanmedian(out)) if np.isfinite(out).any() else 0.0
        out = np.where(np.isnan(out), fill, out)
    return out


def l2_normalise(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def enrol_norm(s: np.ndarray, speaker: np.ndarray, enrol: np.ndarray) -> np.ndarray:
    """E0-norm: z-score the zero-shot logit inside each speaker's enrolment set."""
    s = np.asarray(s, float)
    sp = np.asarray(speaker).astype(str)
    en = np.asarray(enrol, bool)
    out = np.full(s.shape[0], np.nan)
    glob_sd = float(np.std(s[en])) if en.any() else 0.0
    glob_sd = glob_sd if glob_sd > 1e-9 else 1.0
    for k in np.unique(sp):
        idx = np.where(sp == k)[0]
        r = idx[en[idx]]
        if r.size == 0:
            out[idx] = (s[idx] - float(np.mean(s[en]) if en.any() else np.mean(s))) / glob_sd
            continue
        sd = float(np.std(s[r]))
        out[idx] = (s[idx] - float(np.mean(s[r]))) / (sd if sd > 1e-9 else glob_sd)
    return out

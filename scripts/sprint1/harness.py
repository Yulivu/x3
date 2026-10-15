#!/usr/bin/env python3
"""S1-A: the device-agnostic numerics every later package reuses.

Section 1 wants the heavy work on the GPU - batched cosine distances, EER by
sorting, speaker-clustered bootstrap - and the CPU left to scheduling, IO and small
tables.  This module is that layer:

* it runs on CUDA when the box has it and on the CPU otherwise (same code path);
* its metrics are pinned to ``src.metrics3`` by ``harness_selftest.py``, so a number
  computed here and a number recomputed with numpy/sklearn agree;
* every library distance masks the query clip's own speaker (section 0.6 test b).

Nothing here reads data: the callers pass tensors/arrays, which is what keeps the
same functions usable by P1 (dev only) and by P3 after the freeze.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def pick_device(prefer: str = "auto") -> str:
    """``cuda`` when a GPU is usable, else ``cpu`` (import-safe without torch)."""
    if prefer not in ("auto", "cuda", "cpu"):
        raise ValueError(f"unknown device preference {prefer!r}")
    if prefer == "cpu":
        return "cpu"
    try:
        import torch
    except Exception:
        return "cpu"
    if prefer == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_threads(n: Optional[int] = None) -> int:
    """Cap torch's CPU threads to the lane budget; returns the value in force."""
    try:
        import torch
    except Exception:
        return 0
    if n is None:
        raw = os.environ.get("OMP_NUM_THREADS") or os.environ.get("X3_THREADS_A") or "5"
        n = max(1, int(raw))
    torch.set_num_threads(int(n))
    return int(n)


def as_tensor(x: Any, device: str = "cpu", dtype: str = "float64"):
    import torch

    t = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
    return t.to(device=device, dtype=getattr(torch, dtype))


def l2n(X: Any, device: str = "cpu"):
    import torch

    T = as_tensor(X, device, "float32")
    return T / T.norm(dim=1, keepdim=True).clamp_min(1e-12)


def eer_t(scores: Any, labels: Any) -> float:
    """EER in %, identical to ``src.metrics3.eer`` on tie-free scores.

    Recipe: walk the thresholds downwards (predict spoof iff ``score >= t``),
    keep one point per distinct score, and take the point where |FNR - FPR| is
    smallest - the same choice ``metrics3.eer`` makes on sklearn's ROC curve.
    """
    import torch

    s = as_tensor(scores, "cpu", "float64").ravel()
    y = as_tensor(labels, "cpu", "float64").ravel()
    if s.numel() == 0:
        return float("nan")
    if torch.unique(s).numel() < 2 or torch.unique(y).numel() < 2:
        return 50.0
    order = torch.argsort(s, descending=True, stable=True)
    ss, ys = s[order], y[order]
    tp = torch.cumsum(ys, 0)
    fp = torch.cumsum(1 - ys, 0)
    P = ys.sum().clamp_min(1.0)
    N = (ys.numel() - ys.sum()).clamp_min(1.0)
    fnr = 1.0 - tp / P
    fpr = fp / N
    # one point per distinct score: the last index of every tie run
    keep = torch.ones_like(ys, dtype=torch.bool)
    keep[:-1] = ss[:-1] != ss[1:]
    fpr, fnr = fpr[keep], fnr[keep]
    # the (0, 0) operating point sklearn also emits (threshold = +inf)
    fpr = torch.cat([torch.zeros(1, dtype=fpr.dtype), fpr])
    fnr = torch.cat([torch.ones(1, dtype=fnr.dtype), fnr])
    i = int(torch.argmin((fnr - fpr).abs()))
    return float((fpr[i] + fnr[i]) / 2 * 100)


def auc_t(scores: Any, labels: Any) -> float:
    """ROC AUC with average ranks (equals ``sklearn.roc_auc_score``)."""
    import torch

    s = as_tensor(scores, "cpu", "float64").ravel()
    y = as_tensor(labels, "cpu", "float64").ravel()
    pos = y == 1
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s, stable=True)
    ranks = torch.empty_like(s)
    sorted_s = s[order]
    # average ranks inside tie groups
    i = 0
    while i < sorted_s.numel():
        j = i
        while j + 1 < sorted_s.numel() and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def platform_eers_t(scores: Any, labels: Any, platform: Sequence[Any],
                    min_per_class: int = 30) -> Dict[str, float]:
    import torch

    s = as_tensor(scores, "cpu", "float64").ravel()
    y = as_tensor(labels, "cpu", "float64").ravel()
    out: Dict[str, float] = {}
    plat = np.asarray(platform).astype(str).ravel()
    for p in sorted(set(plat)):
        idx = np.where(plat == p)[0]
        yy = y[idx]
        if float(yy.sum()) < min_per_class or float((1 - yy).sum()) < min_per_class:
            continue
        out[str(p)] = eer_t(s[idx], yy)
    return out


def summarize_t(scores: Any, labels: Any, platform: Optional[Sequence[Any]] = None,
                min_per_class: int = 30, extra: Optional[Dict[str, Any]] = None
                ) -> Dict[str, Any]:
    """The three-view row of ``metrics3.summarize``, computed through torch."""
    import torch

    s = as_tensor(scores, "cpu", "float64").ravel()
    y = as_tensor(labels, "cpu", "float64").ravel()
    row: Dict[str, Any] = {"pooled_eer": eer_t(s, y), "auc": auc_t(s, y),
                           "n_eval": int(y.numel()), "n_pos": int(y.sum())}
    if platform is not None:
        per = platform_eers_t(s, y, platform, min_per_class)
        row["within_macro_eer"] = float(np.mean(list(per.values()))) if per else float("nan")
        if per:
            worst = max(per, key=lambda k: per[k])
            row["worst_platform"] = worst
            row["worst_platform_eer"] = float(per[worst])
        else:
            row["worst_platform"] = ""
            row["worst_platform_eer"] = float("nan")
        row["n_platforms_scored"] = len(per)
        for p, v in per.items():
            row[f"eer_{p}"] = float(v)
    if extra:
        row.update(extra)
    return row


def speaker_recipes(speaker: Sequence[Any], B: int = 200, seed: int = 20261015
                    ) -> Tuple[List[str], np.ndarray]:
    """Speaker-clustered resample recipes shared by every (b) interval."""
    sp = np.asarray(speaker).astype(str).ravel()
    names = sorted(set(sp.tolist()))
    rng = np.random.default_rng(int(seed))
    return names, rng.integers(0, len(names), size=(int(B), len(names)))


def library_distance(X: Any, query_rows: Sequence[int], lib_rows: Sequence[int],
                     query_speaker: Sequence[Any], lib_speaker: Sequence[Any],
                     device: str = "cpu", block: int = 8192,
                     mask_same_speaker: bool = True, exclude_self: bool = False):
    """``+min`` cosine distance from every query clip to the library.

    Section 0.6 test b: the distance matrix of a query clip never contains a library
    member of its own speaker, so the score cannot peek at its own identity.
    """
    import torch

    Zn = l2n(X, device)
    q_rows = np.asarray(query_rows, int)
    l_rows = np.asarray(lib_rows, int)
    q_sp = np.asarray(query_speaker).astype(str)[q_rows]
    l_sp = np.asarray(lib_speaker).astype(str)[l_rows]
    out = torch.full((q_rows.size,), float("nan"), dtype=torch.float32, device=device)
    if l_rows.size == 0:
        return out.cpu().numpy()
    L = Zn[torch.as_tensor(l_rows, device=device)]
    Lsp = l_sp
    for s in np.unique(q_sp):
        q_idx = np.where(q_sp == s)[0]
        keep = ~(Lsp == s) if mask_same_speaker else np.ones(Lsp.shape, bool)
        if not keep.any():
            continue
        Ls = L[torch.as_tensor(np.where(keep)[0], device=device)]
        rows = torch.as_tensor(q_rows[q_idx], device=device)
        best = torch.empty(q_idx.size, dtype=torch.float32, device=device)
        for start in range(0, q_idx.size, block):
            chunk = rows[start:start + block]
            d = 1.0 - Zn[chunk] @ Ls.T
            best[start:start + block] = d.min(dim=1).values
        out[torch.as_tensor(q_idx, device=device)] = best
    if exclude_self:
        # a query never counts itself as a library member even when the rows overlap
        for i, r in enumerate(q_rows):
            if r in set(int(v) for v in l_rows):
                pos = int(np.where(l_rows == r)[0][0])
                d = 1.0 - (Zn[torch.as_tensor(int(r), device=device)]
                           * L[pos]).sum()
                if float(d) <= float(out[i]) + 1e-12:
                    others = [k for k in range(l_rows.size) if k != pos]
                    if others:
                        idx = torch.as_tensor(others, device=device)
                        out[i] = (1.0 - Zn[torch.as_tensor(int(r), device=device)] @ L[idx].T).min()
    return out.cpu().numpy()


def logreg_lbfgs(X: Any, y: Any, l2: float = 1e-2, max_iter: int = 200,
                 device: str = "cpu", weights: Any = None):
    """Binary logistic regression by torch LBFGS; returns ``(logits, info)``."""
    import torch

    T = as_tensor(X, device, "float32")
    yy = as_tensor(y, device, "float32").ravel()
    w = None if weights is None else as_tensor(weights, device, "float32").ravel()
    W = torch.zeros(T.shape[1], 1, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=int(max_iter),
                            tolerance_grad=1e-7, tolerance_change=1e-9,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logit = (T @ W + b).ravel()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, yy, weight=w, reduction="mean")
        loss = loss + float(l2) * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        logits = (T @ W + b).ravel()
    info = {"l2": float(l2), "n": int(T.shape[0]), "dim": int(T.shape[1]),
            "w_norm": float(W.detach().norm()), "device": device}
    return logits.detach().cpu().numpy(), info

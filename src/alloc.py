"""C3 - CASA, cell-aware active label selection (TASKBOOK section 7.4), plus the
six comparison strategies it has to beat.

Warm start ``n0`` clips spread evenly over the cells; then ``batches`` rounds of
``B = (k - n0) / batches`` clips each, refitting the probe after every round.

    score_i = H(q_i) + rho * V_c * (N_c / N)

``H`` is the Bernoulli entropy (an uninformative clip is worth labelling) and
``V_c = a b / ((a+b)^2 (a+b+1))`` with ``a = 1 + pos_c``, ``b = 1 + neg_c`` is the
posterior variance of the cell's fake rate - a cell with almost no labels is
worth sampling even if its clips look easy.

Every strategy is subject to the same per-cell ceiling (``2x`` the average) and
the same in-cell redundancy filter (cosine distance < 0.1 in PCA space), so the
comparison in W12 is about the selection rule alone.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .mpp import MPP

CASA_N0 = 60
CASA_BATCHES = 4
CASA_RHOS = (0.5, 1.0, 2.0)
DEDUP_COS = 0.1
STRATEGIES = ("U", "P", "N", "E", "K", "BADGE", "CASA")

__all__ = ["CASA", "STRATEGIES", "CASA_N0", "CASA_RHOS", "allocate", "bernoulli_entropy"]


def bernoulli_entropy(q: Sequence[float]) -> np.ndarray:
    q = np.clip(np.asarray(q, float), 1e-9, 1 - 1e-9)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _per_cell_cap(n_cells: int, batch: int) -> int:
    """``2x`` the average per-cell allocation, but never so tight it blocks a batch."""
    avg = batch / max(n_cells, 1)
    return max(1, int(np.ceil(2.0 * avg)))


def _quota_from_weights(weights: Dict[Any, float], batch: int, cap: int) -> Dict[Any, int]:
    """Largest-remainder rounding of cell weights into integer quotas under ``cap``."""
    total = float(sum(weights.values()))
    if total <= 0:
        weights = {c: 1.0 for c in weights}
        total = float(len(weights))
    raw = {c: batch * w / total for c, w in weights.items()}
    quota = {c: min(int(np.floor(v)), cap) for c, v in raw.items()}
    left = batch - sum(quota.values())
    order = sorted(raw, key=lambda c: (-(raw[c] - np.floor(raw[c])), str(c)))
    i = 0
    while left > 0 and order:
        c = order[i % len(order)]
        if quota[c] < cap:
            quota[c] += 1
            left -= 1
        i += 1
        if i > 100 * len(order) + batch:
            break
    return quota


def _dedup(greedy: List[int], Z_unit: np.ndarray, cos: float) -> List[int]:
    out: List[int] = []
    for j in greedy:
        if all(float(Z_unit[j] @ Z_unit[k]) < 1.0 - cos for k in out):
            out.append(j)
    return out


def allocate(
    strategy: str,
    *,
    Z_pool: np.ndarray,
    cell_pool: Sequence[Any],
    unlab_idx: np.ndarray,
    q_pool: np.ndarray,
    pos: Dict[Any, int],
    neg: Dict[Any, int],
    batch: int,
    rng: np.random.Generator,
    rho: float = 1.0,
    Z_pca: Optional[np.ndarray] = None,
    grad_emb: Optional[np.ndarray] = None,
    cap: Optional[int] = None,
) -> np.ndarray:
    """Return ``batch`` pool indices chosen by ``strategy`` from ``unlab_idx``."""
    cell_pool = np.asarray(cell_pool)
    unlab_idx = np.asarray(unlab_idx, int)
    if unlab_idx.size == 0 or batch <= 0:
        return np.array([], int)
    batch = int(min(batch, unlab_idx.size))
    cells = list(dict.fromkeys(cell_pool[unlab_idx].tolist()))
    n_pool = len(cell_pool)
    cap = cap if cap is not None else _per_cell_cap(len(cells), batch)
    avail = {c: int((cell_pool[unlab_idx] == c).sum()) for c in cells}
    sizes = {c: int((cell_pool == c).sum()) for c in cells}
    Z_unit = Z_pool / (np.linalg.norm(Z_pool, axis=1, keepdims=True) + 1e-9)
    pca = Z_pca if Z_pca is not None else Z_pool
    Zp_unit = pca / (np.linalg.norm(pca, axis=1, keepdims=True) + 1e-9)

    if strategy == "U":
        weights = {c: 1.0 for c in cells}
        quota = _quota_from_weights(weights, batch, cap)
        picks: List[int] = []
        for c, m in quota.items():
            idx = unlab_idx[cell_pool[unlab_idx] == c]
            picks += list(rng.choice(idx, size=min(m, idx.size), replace=False))
        return np.asarray(picks, int)

    if strategy in ("P", "N"):
        if strategy == "P":
            weights = {c: float(avail[c]) for c in cells}
        else:
            weights = {}
            for c in cells:
                a, b = 1 + pos.get(c, 0), 1 + neg.get(c, 0)
                pi = a / (a + b)
                weights[c] = float(avail[c]) * float(np.sqrt(max(pi * (1 - pi), 1e-9)))
        quota = _quota_from_weights(weights, batch, cap)
        picks = []
        for c, m in quota.items():
            idx = unlab_idx[cell_pool[unlab_idx] == c]
            picks += list(rng.choice(idx, size=min(m, idx.size), replace=False))
        return np.asarray(picks, int)

    if strategy == "E":
        score = bernoulli_entropy(q_pool)[unlab_idx]
    elif strategy == "K":
        score = np.zeros(unlab_idx.size)
        for c in cells:
            m = cell_pool[unlab_idx] == c
            idx = unlab_idx[m]
            if idx.size == 0:
                continue
            start = idx[int(np.argmax(np.linalg.norm(Zp_unit[idx], axis=1)))]
            chosen = [start]
            while len(chosen) < min(cap, idx.size):
                d = 1.0 - Zp_unit[idx] @ Zp_unit[chosen].T
                chosen.append(int(idx[int(np.argmax(d.min(axis=1)))]))
            score[np.isin(unlab_idx, chosen)] = 1.0
        # keep the k-center set exactly
        keep = np.where(score > 0)[0]
        return unlab_idx[keep][:batch]
    elif strategy == "BADGE":
        emb = grad_emb if grad_emb is not None else Z_pool
        return _kmeans_pp(emb[unlab_idx], unlab_idx, batch, rng)
    elif strategy == "CASA":
        score = bernoulli_entropy(q_pool)[unlab_idx].copy()
        cell_un = cell_pool[unlab_idx]
        bonus = np.zeros(unlab_idx.size)
        for c in cells:
            m = cell_un == c
            a, b = 1 + pos.get(c, 0), 1 + neg.get(c, 0)
            v = (a * b) / ((a + b) ** 2 * (a + b + 1))
            bonus[m] = rho * v * (sizes.get(c, 0) / max(n_pool, 1))
        score = score + bonus
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    # Scan the whole ranking, enforcing the per-cell cap and the in-cell
    # redundancy filter as we go, so a cluster of near-duplicates cannot fill the
    # batch and the top-up comes from genuinely different clips.
    order = unlab_idx[np.argsort(-score, kind="stable")]
    taken: Dict[Any, int] = {}
    per_cell_sel: Dict[Any, List[int]] = {}
    picked: List[int] = []
    for j in order:
        if len(picked) >= batch:
            break
        c = cell_pool[j]
        if taken.get(c, 0) >= cap:
            continue
        if any(
            float(Zp_unit[j] @ Zp_unit[k]) >= 1.0 - DEDUP_COS for k in per_cell_sel.get(c, [])
        ):
            continue
        picked.append(int(j))
        taken[c] = taken.get(c, 0) + 1
        per_cell_sel.setdefault(c, []).append(int(j))
    return np.asarray(picked, int)


def _kmeans_pp(emb: np.ndarray, idx: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding (BADGE-lite) on the gradient embeddings."""
    n = emb.shape[0]
    if n <= k:
        return idx.copy()
    first = int(rng.integers(n))
    centers = [first]
    d2 = np.linalg.norm(emb - emb[first], axis=1) ** 2
    while len(centers) < k:
        p = d2 / (d2.sum() + 1e-12)
        nxt = int(rng.choice(n, p=p))
        if nxt in centers:
            nxt = int(np.argmax(d2))
        centers.append(nxt)
        d2 = np.minimum(d2, np.linalg.norm(emb - emb[nxt], axis=1) ** 2)
    return idx[np.asarray(centers, int)]


class CASA:
    """Warm start + ``batches`` rounds of active selection, refitting C1 each round."""

    def __init__(
        self,
        k: int = 100,
        n0: int = CASA_N0,
        batches: int = CASA_BATCHES,
        rho: float = 1.0,
        strategy: str = "CASA",
        sigma_p: float = 0.5,
        sigma_gamma: float = 0.5,
        lam: float = 1.0,
        seed: int = 20261015,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        self.k = int(k)
        self.n0 = int(min(n0, k))
        self.batches = int(batches)
        self.rho = float(rho)
        self.strategy = strategy
        self.sigma_p, self.sigma_gamma, self.lam = sigma_p, sigma_gamma, lam
        self.seed = int(seed)
        self.history_: List[Dict[str, Any]] = []

    def _model(self, Z, y, plat, lang, cell) -> MPP:
        return MPP(sigma_p=self.sigma_p, sigma_gamma=self.sigma_gamma, lam=self.lam).fit(
            Z, y, plat, lang, cell
        )

    def run(
        self,
        Z_pool, y_pool, plat_pool, lang_pool, cell_pool,
        Z_pca: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Select ``k`` clips.  ``y_pool`` is read ONLY for the chosen indices."""
        rng = np.random.default_rng(self.seed)
        Z_pool = np.asarray(Z_pool, float)
        cell_pool = np.asarray(cell_pool)
        y_pool = np.asarray(y_pool)
        n = len(cell_pool)
        cells = list(dict.fromkeys(cell_pool.tolist()))

        # warm start: n0 spread evenly over cells
        warm = _quota_from_weights({c: 1.0 for c in cells}, self.n0, max(1, self.n0))
        selected: List[int] = []
        for c, m in warm.items():
            idx = np.where(cell_pool == c)[0]
            selected += list(rng.choice(idx, size=min(m, idx.size), replace=False))
        selected = selected[: self.n0]

        per_batch = max(1, (self.k - len(selected)) // max(self.batches, 1))
        model = None
        for b in range(self.batches):
            lab_idx = np.asarray(sorted(selected), int)
            unlab_idx = np.setdiff1d(np.arange(n), lab_idx, assume_unique=False)
            if unlab_idx.size == 0:
                break
            model = self._model(
                Z_pool[lab_idx], y_pool[lab_idx], plat_pool[lab_idx], lang_pool[lab_idx], cell_pool[lab_idx]
            )
            q_pool = _sigmoid(model.predict_logit(Z_pool, plat_pool, lang_pool, cell_pool))
            pos = {c: int(((cell_pool[lab_idx] == c) & (y_pool[lab_idx] == 1)).sum()) for c in cells}
            neg = {c: int(((cell_pool[lab_idx] == c) & (y_pool[lab_idx] == 0)).sum()) for c in cells}
            grad = None
            if self.strategy == "BADGE":
                w = model._unpack()[0]
                grad = (q_pool - 0.5)[:, None] * (Z_pool * 0) + (q_pool - 0.5)[:, None] * Z_pool
                grad = np.hstack([grad, np.ones((n, 1)) * (q_pool - 0.5)[:, None]])
                del w
            take = max(1, min(per_batch, self.k - len(selected)))
            new_idx = allocate(
                self.strategy,
                Z_pool=Z_pool, cell_pool=cell_pool, unlab_idx=unlab_idx, q_pool=q_pool,
                pos=pos, neg=neg, batch=take, rng=rng, rho=self.rho,
                Z_pca=Z_pca, grad_emb=grad,
            )
            selected += list(new_idx)
            self.history_.append({
                "batch": b + 1,
                "n_selected": len(selected),
                "cells_touched": int(len(set(cell_pool[np.asarray(selected, int)].tolist()))),
            })
            if len(selected) >= self.k:
                break
        self.selected_ = np.asarray(sorted(selected[: self.k]), int)
        return {"selected": self.selected_, "history": self.history_, "strategy": self.strategy}

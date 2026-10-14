"""W13 - B10, fine-tuning the last block on cached features.

Full version: cache the ``L-1`` layer output for every dev clip as ragged fp16
(``[T_i, D]`` plus an offset table), then train the encoder's last block + final
LayerNorm + mean pooling + the model's own linear head with AdamW
(``lr in {1e-5, 3e-5}`` chosen on ``L`` inside CV, 5 epochs, batch 16, no class
weights).  That part needs fairseq and only runs on the box.

Reduced version (``B10_REDUCED``): the same trainer applied to a LayerNorm + linear
head on the pooled feature vector.  That is what ``tests/test_ft_last.py`` pins,
because it needs no encoder.

Both variants share the same optimiser, schedule, batch size and selection rule
so the comparison stays honest.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .cv import log_likelihood, stratified_folds

FT_LRS = (1e-5, 3e-5)
FT_EPOCHS = 5
FT_BATCH = 16
FT_WEIGHT_DECAY = 1e-4

__all__ = ["FT_LRS", "FT_EPOCHS", "FT_BATCH", "LastBlockHead", "train_head",
           "fit_b10", "B10_REDUCED_NOTE"]

B10_REDUCED_NOTE = (
    "B10_REDUCED: only the final LayerNorm + linear head were trained; the encoder "
    "block was not fine-tuned. Report this label whenever the full variant did not run."
)


def _torch():
    import torch

    return torch


class LastBlockHead:
    """LayerNorm -> linear, i.e. the trainable tail of the backbone."""

    def __init__(self, d: int, seed: int = 20261015) -> None:
        torch = _torch()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(int(d)), torch.nn.Linear(int(d), 1))
        self.d = int(d)

    def parameters(self):
        return self.net.parameters()

    def logit(self, X) -> np.ndarray:
        torch = _torch()
        self.net.eval()
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(X, np.float32))
            return self.net(t).squeeze(-1).numpy().astype(float)


def train_head(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 1e-5,
    epochs: int = FT_EPOCHS,
    batch: int = FT_BATCH,
    weight_decay: float = FT_WEIGHT_DECAY,
    seed: int = 20261015,
) -> LastBlockHead:
    """AdamW on the head, full-batch-ordered mini-batches, no class weights."""
    torch = _torch()
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32)
    head = LastBlockHead(X.shape[1], seed=seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    head.net.train()
    for _ in range(int(epochs)):
        order = rng.permutation(len(X))
        for start in range(0, len(order), int(batch)):
            idx = torch.from_numpy(order[start:start + int(batch)])
            opt.zero_grad()
            out = head.net(Xt[idx]).squeeze(-1)
            loss = lossf(out, yt[idx])
            loss.backward()
            opt.step()
    return head


def fit_b10(
    X: np.ndarray,
    y: np.ndarray,
    folds: Optional[Sequence[int]] = None,
    grid: Sequence[float] = FT_LRS,
    epochs: int = FT_EPOCHS,
    batch: int = FT_BATCH,
    seed: int = 20261015,
) -> Tuple[LastBlockHead, Dict[str, Any]]:
    """Choose ``lr`` on the shared folds, then refit on all of ``L``."""
    X = np.asarray(X, np.float32)
    y = np.asarray(y)
    folds = np.asarray(folds) if folds is not None else stratified_folds(y)
    best: Optional[Tuple[float, float]] = None
    cv: Dict[str, float] = {}
    for lr in grid:
        vals = []
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            if len(np.unique(y[tr])) < 2 or te.sum() == 0:
                continue
            head = train_head(X[tr], y[tr], lr=lr, epochs=epochs, batch=batch, seed=seed)
            vals.append(log_likelihood(y[te], head.logit(X[te])))
        if vals:
            ll = float(np.mean(vals))
            cv[str(lr)] = ll
            if best is None or ll > best[0]:
                best = (ll, float(lr))
    lr = best[1] if best else float(grid[0])
    head = train_head(X, y, lr=lr, epochs=epochs, batch=batch, seed=seed)
    return head, {"lr": lr, "epochs": epochs, "batch": batch, "cv": cv,
                  "variant": "B10_REDUCED" if X.ndim == 2 else "B10_FULL"}

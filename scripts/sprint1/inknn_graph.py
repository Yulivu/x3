#!/usr/bin/env python3
"""P-4: the platform-internal (in-stream) kNN graph M-F walks on.

Section 3 asks for a kNN graph inside every (platform) stream, built on the P-3
PCA projections so the graph never sees raw features and never crosses platforms.
The graph is cached because M-F and the backlog probes reuse it with different
seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sprint1 import config as cfg  # noqa: E402
from scripts.sprint1 import state as st  # noqa: E402


def knn(Z: np.ndarray, k: int, block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Cosine kNN (self excluded) via blocked matrix products."""
    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    n = Zn.shape[0]
    k = int(min(k, max(1, n - 1)))
    idx = np.empty((n, k), np.int32)
    sim = np.empty((n, k), np.float32)
    for start in range(0, n, block):
        stop = min(n, start + block)
        S = Zn[start:stop] @ Zn.T
        for r in range(stop - start):
            S[r, start + r] = -np.inf      # a clip is never its own neighbour
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(S, part, axis=1)
        order = np.argsort(-vals, axis=1)
        idx[start:stop] = np.take_along_axis(part, order, axis=1)
        sim[start:stop] = np.take_along_axis(vals, order, axis=1)
    return idx, sim


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X3 sprint 1 P-4 in-stream kNN graph")
    ap.add_argument("--out-dir", default="sprint1/graph")
    ap.add_argument("--pca", default="sprint1/pca")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else cfg.REPO_ROOT
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pca_dir = Path(args.pca)
    pca_dir = pca_dir if pca_dir.is_absolute() else root / pca_dir
    index = json.loads((pca_dir / "pca_index.json").read_text(encoding="utf-8"))

    graph_index: Dict[str, Any] = {"k": int(args.k), "built_from": str(pca_dir),
                                   "backbones": {}}
    for bb, per_target in index["backbones"].items():
        per_bb: Dict[str, Any] = {}
        for p, meta in per_target.items():
            npz = np.load(root / meta["path"])
            Z = npz["Z_target"]
            idx, sim = knn(Z, args.k, args.block)
            path = out_dir / f"graph_{bb}_{p}.npz"
            np.savez_compressed(path, rows=npz["target_rows"], idx=idx, sim=sim)
            per_bb[p] = {"path": str(path.relative_to(root)), "n": int(Z.shape[0]),
                         "k": int(idx.shape[1])}
            print(f"{bb} {p}: n={Z.shape[0]} k={idx.shape[1]}", flush=True)
        graph_index["backbones"][bb] = per_bb
    (out_dir / "graph_index.json").write_text(
        json.dumps({**graph_index, "created": st.now_iso()}, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "targets": list(graph_index["backbones"]
                                                              .get(cfg.BACKBONES[0], {}))},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

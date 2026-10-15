#!/usr/bin/env python3
"""P-3: per-fold PCA-256 and standardisation, fitted on the source platforms only.

Section 0.3 is the point of this package: PCA and standardisation are *inductive* -
fitted on the source side of each P1 fold and then applied to the target platform.
The same code path serves the leave-one-platform-out folds used to pick
hyperparameters, so a hyperparameter chosen on the source is the only kind the
target ever sees.

Outputs (per backbone and target platform):

* ``sprint1/pca/pca_<backbone>_<platform>.npz`` - standardised PCA projections of the
  target rows, of the full source, and of every leave-one-source-platform-out fold;
* ``sprint1/pca/pca_index.json`` - the index the graph and method packages read.
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


def standardise(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def fit_pca(Z_train: np.ndarray, dim: int, seed: int):
    from sklearn.decomposition import PCA

    dim = int(min(dim, min(Z_train.shape)))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=int(seed))
    pca.fit(Z_train)
    return pca


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X3 sprint 1 P-3 per-fold PCA")
    ap.add_argument("--out-dir", default="sprint1/pca")
    ap.add_argument("--tables", default="sprint1/tables")
    ap.add_argument("--backbones", default=",".join(cfg.BACKBONES))
    ap.add_argument("--spec", default="S3")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--data", default=None)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else cfg.REPO_ROOT
    data_root = cfg.data_root(args.data)
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = Path(args.tables)
    tables = tables if tables.is_absolute() else root / tables
    p1 = json.loads((tables / "p1_folds.json").read_text(encoding="utf-8"))
    backbones = [b.strip() for b in str(args.backbones).split(",") if b.strip()]

    from src.regions import load_spec_block

    index: Dict[str, Any] = {"spec": args.spec, "dim": int(args.dim),
                             "fitted_on": "source platforms only (inductive)",
                             "backbones": {}}
    for bb in backbones:
        per_bb: Dict[str, Any] = {}
        for p, block in p1["targets"].items():
            src_rows = np.asarray(block["source_rows"], int)
            tgt_rows = np.asarray(block["target_rows"], int)
            rows = np.unique(np.concatenate([src_rows, tgt_rows]))
            F = load_spec_block(data_root, bb, args.spec, rows)
            pos = {int(r): i for i, r in enumerate(rows)}
            F_src = F[[pos[int(r)] for r in src_rows]]
            F_tgt = F[[pos[int(r)] for r in tgt_rows]]
            mu, sd = standardise(F_src)
            pca = fit_pca((F_src - mu) / sd, args.dim, cfg.DRAW_BASE + abs(hash(p)) % 9973)
            payload: Dict[str, np.ndarray] = {
                "target_rows": tgt_rows,
                "source_rows": src_rows,
                "mu": mu, "sd": sd,
                "components": pca.components_.astype(np.float32),
                "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
                "Z_target": pca.transform((F_tgt - mu) / sd).astype(np.float32),
                "Z_source": pca.transform((F_src - mu) / sd).astype(np.float32),
            }
            folds = p1["lopo_folds"].get(p, {})
            for q, fold_rows in folds.items():
                fold_rows = np.asarray(fold_rows, int)
                if fold_rows.size == 0:
                    continue
                fold_pos = [pos[int(r)] for r in fold_rows]
                F_fold = F[fold_pos]
                mu_f, sd_f = standardise(F_fold)
                pca_f = fit_pca((F_fold - mu_f) / sd_f, args.dim,
                                cfg.DRAW_BASE + abs(hash((p, q))) % 9973)
                payload[f"fold_{q}_rows"] = fold_rows
                payload[f"fold_{q}_mu"] = mu_f
                payload[f"fold_{q}_sd"] = sd_f
                payload[f"fold_{q}_Z_target"] = pca_f.transform(
                    (F_tgt - mu_f) / sd_f).astype(np.float32)
                payload[f"fold_{q}_Z_source"] = pca_f.transform(
                    (F_fold - mu_f) / sd_f).astype(np.float32)
            path = out_dir / f"pca_{bb}_{p}.npz"
            np.savez_compressed(path, **payload)
            per_bb[p] = {"path": str(path.relative_to(root)),
                         "n_target": int(len(tgt_rows)), "n_source": int(len(src_rows)),
                         "n_folds": int(len(folds)),
                         "files": sorted(payload)}
            print(f"{bb} {p}: target {len(tgt_rows)} source {len(src_rows)} "
                  f"folds {len(folds)}", flush=True)
        index["backbones"][bb] = per_bb
    (out_dir / "pca_index.json").write_text(
        json.dumps({**index, "created": st.now_iso()}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "backbones": backbones,
                      "targets": list(p1["targets"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

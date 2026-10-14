#!/usr/bin/env python3
"""W5 - spec search on a new backbone, ranked against the original features (G3).

Every candidate (backbone, spec) is scored with the SAME dev draws, so the
comparison is draw-paired rather than two independent means.  Candidates:

* ``xlsr300m-linea/S1-lineA`` - the frozen Line A feature block, the incumbent
* ``<backbone>/S1`` - the Line A layer fractions applied to the new backbone
* ``S2`` - last hidden state, mean + std
* ``S3`` - layer ``floor(0.75 L)``, mean + std
* ``S4`` - layers ``floor(0.5 L)``, ``floor(0.75 L)``, ``L``, means concatenated
* ``S5`` - S3 plus the zero-shot logit

For each candidate and draw we fit an L2 logistic probe (``C`` chosen on the
shared folds inside ``L``) and then apply the Line A cell-prior correction, i.e.
the M0 / M1 / M2 triple.  G3 switches backbone only when the paired within-macro
gain is at least 0.5 points with the interval clear of zero.

Writes ``results/backbone_rank.csv``, ``results/w5_draws.csv`` and, with
``--select``, ``chosen_backbone.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.baselines import fit_logreg_cv  # noqa: E402
from src.data import data_root, load_mlitw  # noqa: E402
from src.draws import draw_labels, draw_seed, strict_eval_mask  # noqa: E402
from src.em_prior import estimate_pi, fuse, logit  # noqa: E402
from src.metrics3 import summarize  # noqa: E402

N_DRAWS = 30
K_PER_CELL = 10
PCA_DIM = 256
LINEA_FRACTIONS = (0.32, 0.48, 0.64, 0.72, 0.80)   # Line A's [8,12,16,18,20] / 25

__all__ = ["specs_for", "load_candidates", "run_candidate", "rank", "main"]


def specs_for(mean: np.ndarray, std: np.ndarray, zero: np.ndarray,
              n_layers: int) -> Dict[str, np.ndarray]:
    """The five specs of TASKBOOK W5, built from a ``mean (N, L+1, D)`` block."""
    hs = mean.shape[1] - 1                     # hidden state 0 is the projection
    L = int(n_layers)
    frac = [max(1, min(hs, int(round(f * hs)))) for f in LINEA_FRACTIONS]
    s1 = np.concatenate([mean[:, i, :] for i in frac], axis=1)
    last = hs
    q3 = max(1, min(hs, int(round(0.75 * L))))
    q2 = max(1, min(hs, int(round(0.5 * L))))
    sidx = min(std.shape[1] - 1, max(0, int(round(0.75 * L)) - 1))
    return {
        "S1": s1,
        "S2": np.concatenate([mean[:, last, :], std[:, -1, :]], axis=1),
        "S3": np.concatenate([mean[:, q3, :], std[:, sidx, :]], axis=1),
        "S4": np.concatenate([mean[:, q2, :], mean[:, q3, :], mean[:, last, :]], axis=1),
        "S5": np.concatenate([mean[:, q3, :], std[:, sidx, :], zero[:, None]], axis=1),
    }


def _pca(Z: np.ndarray, dim: int = PCA_DIM, seed: int = 20261015) -> np.ndarray:
    """Standardise + PCA, fitted on the dev region's UNLABELLED features only."""
    from sklearn.decomposition import PCA

    mu, sd = Z.mean(0, keepdims=True), Z.std(0, keepdims=True) + 1e-6
    Zs = (Z - mu) / sd
    d = int(min(dim, Zs.shape[1], max(1, Zs.shape[0] - 1)))
    return PCA(n_components=d, svd_solver="randomized", random_state=seed).fit_transform(Zs).astype(np.float32)


def _dev_rows(root: Path, meta: pd.DataFrame) -> np.ndarray:
    split = json.loads((root / "splits" / "speaker_split_v2.json").read_text(encoding="utf-8"))
    return meta["speaker"].isin(set(split["dev_speakers"])).to_numpy()


def load_candidates(root: Path, data_dir: Path, backbone: str) -> Dict[str, Dict[str, Any]]:
    """Build every candidate on the SAME dev rows, in the same order."""
    meta = load_mlitw(data_dir)
    keep = _dev_rows(root, meta)
    sub = meta.loc[keep].reset_index(drop=True)
    out: Dict[str, Dict[str, Any]] = {}
    old = data_dir / "mlitw_feats.npy"
    if old.exists():
        layers = [8, 12, 16, 18, 20]
        F = np.asarray(np.load(old, mmap_mode="r")[keep][:, layers, :], dtype=np.float32)
        out["xlsr300m-linea/S1-lineA"] = {
            "backbone": "xlsr300m-linea", "spec": "S1-lineA",
            "Z": _pca(F.reshape(len(sub), -1)),
            "zero": np.zeros(len(sub), np.float32),
        }
    fdir = data_dir / "x3" / "feats" / backbone
    if (fdir / "mean.npy").exists():
        mean = np.asarray(np.load(fdir / "mean.npy", mmap_mode="r")[keep], np.float32)
        std = np.asarray(np.load(fdir / "std.npy", mmap_mode="r")[keep], np.float32)
        info = json.loads((fdir / "layers.json").read_text(encoding="utf-8"))
        zero = pd.read_csv(fdir / "scores.csv")["logit_fake_minus_real"].to_numpy()[keep]
        for name, F in specs_for(mean, std, zero, info["n_layers"]).items():
            out[f"{backbone}/{name}"] = {
                "backbone": backbone, "spec": name, "Z": _pca(F),
                "zero": zero.astype(np.float32),
            }
    return out


def run_candidate(payload) -> List[Dict[str, Any]]:
    (key, Z, zero, y, plat, cell, speaker, d) = payload
    lab_pos, _ = draw_labels(cell, K_PER_CELL, draw_seed(d), cap_half=True)
    lab = np.zeros(len(y), bool)
    lab[lab_pos] = True
    ev = np.setdiff1d(np.arange(len(y)), lab_pos)
    if lab.sum() < 50 or ev.size < 100:
        return []
    model, _ = fit_logreg_cv(Z[lab], y[lab])
    raw = model.decision_function(Z)
    z_all = (raw - raw.mean()) / (raw.std() + 1e-9)
    pi, _n_lab, _phi = estimate_pi(list(dict.fromkeys(cell.tolist())), cell, lab, y, plat, variant="bb")
    pi_clip = np.array([pi[c] for c in cell], float)
    m1 = np.array([logit(pi[c], eps=1e-3) for c in cell], float)
    m2 = fuse(z_all, pi_clip, np.ones(len(y)), k=None, lam=1.0)
    backbone, spec = key.split("/", 1)
    rows = []
    for mask, idx in (("clips", ev), ("strict", ev[strict_eval_mask(speaker, lab_pos, ev)])):
        if idx.size < 50:
            continue
        for method, sc in (("M0", z_all), ("M1", m1), ("M2", m2)):
            rows.append(summarize(y[idx], sc[idx], plat[idx], extra={
                "protocol": "dev_v2", "region": "dev", "backbone": backbone, "spec": spec,
                "method": method, "k": int(K_PER_CELL * 29), "draw": int(d), "mask": mask,
                "n_labeled": int(lab.sum()), "n_eval": int(idx.size),
            }))
    return rows


def _worker(payload):
    try:
        return payload[0], payload[-1], run_candidate(payload)
    except Exception as exc:  # pragma: no cover
        return payload[0], payload[-1], [{"error": f"{type(exc).__name__}: {exc}"}]


def rank(draws: pd.DataFrame, incumbent: str = "xlsr300m-linea/S1-lineA") -> pd.DataFrame:
    """Long table: one row per (candidate, metric), with the paired diff vs the incumbent."""
    main = draws[(draws["mask"] == "clips") & (draws["method"] == "M2")]
    inc = main[(main["backbone"] + "/" + main["spec"]) == incumbent]
    rows = []
    for (backbone, spec), grp in main.groupby(["backbone", "spec"]):
        key = f"{backbone}/{spec}"
        paired = {}
        if key != incumbent and not inc.empty:
            for metric in ("pooled_eer", "within_macro_eer", "worst_platform_eer"):
                a = grp.set_index("draw")[metric]
                b = inc.set_index("draw")[metric]
                common = a.index.intersection(b.index)
                if len(common) >= 3:
                    d = (a.loc[common].astype(float) - b.loc[common].astype(float)).dropna().values
                    if d.size >= 3:
                        paired[metric] = (float(d.mean()), float(np.percentile(d, 2.5)),
                                          float(np.percentile(d, 97.5)), int(d.size))
        for metric in ("pooled_eer", "within_macro_eer", "worst_platform_eer", "auc"):
            pm, plo, phi, pn = paired.get(metric, (np.nan, np.nan, np.nan, 0))
            rows.append({
                "backbone": backbone, "spec": spec, "metric": metric,
                "mean": float(grp[metric].mean()), "sd": float(grp[metric].std()),
                "paired_mean": pm, "paired_lo": plo, "paired_hi": phi, "paired_n": pn,
                "n_draws": int(grp["draw"].nunique()),
            })
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W5 spec search / backbone rank")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    data_dir = data_root(args.data)
    meta = load_mlitw(data_dir)
    keep = _dev_rows(root, meta)
    sub = meta.loc[keep].reset_index(drop=True)
    y = sub["y"].to_numpy()
    plat = sub["platform"].to_numpy()
    cell = sub["cell"].to_numpy()
    speaker = sub["speaker"].to_numpy()
    cands = load_candidates(root, data_dir, args.backbone)
    print(f"dev clips={len(sub)} candidates={list(cands)}")
    payloads = [(k, v["Z"], v["zero"], y, plat, cell, speaker, d)
                for k, v in cands.items() for d in range(args.draws)]
    rows, errors = [], []
    if args.jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        with ctx.Pool(args.jobs) as pool:
            for key, d, out in pool.imap_unordered(_worker, payloads, chunksize=1):
                for r in out:
                    if "error" in r:
                        errors.append(f"{key} draw={d}: {r['error']}")
                    else:
                        rows.append(r)
    else:
        for p in payloads:
            key, d, out = _worker(p)
            for r in out:
                if "error" in r:
                    errors.append(f"{key} draw={d}: {r['error']}")
                else:
                    rows.append(r)
    if errors:
        print("errors:", errors[:5], "...", len(errors))
    df = pd.DataFrame(rows)
    (root / "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "results" / "w5_draws.csv", index=False)
    table = rank(df)
    table.to_csv(root / "results" / "backbone_rank.csv", index=False)
    show = table[table["metric"] == "within_macro_eer"].sort_values("mean")
    print(show[["backbone", "spec", "mean", "sd", "paired_mean", "paired_lo", "paired_hi"]]
          .round(3).to_string(index=False))
    if args.select:
        from src.pipeline import gate_g3

        print("G3 ->", json.dumps(gate_g3(root, select=True), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

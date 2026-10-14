#!/usr/bin/env python3
"""W6 - baseline family plus C1 on the dev region, k=290.

Runs FIRST on Line A's existing XLS-R features (spec ``S1-lineA``) so the CPU lane
never waits for the GPU lane (TASKBOOK 0.4).  When G3 names a new (backbone, spec)
the same script is re-run with ``--spec``/``--features`` to produce the ``final``
rows; nothing else changes, which is the point of keeping it parameterised.

Provisional scope notes (recorded in the report, never hidden):

* ``B4`` (Gaussian process) is skipped in provisional mode - an exact GP on ~7 k
  clips is not tractable; it runs on the final spec with the PCA-64 cap.
* ``B8`` (FixMatch-lite) needs the Fam-A views from W10 and is skipped here.
* ``B10`` is W13.  ``B6`` uses Line A's probe score as its zero-shot detector.

Writes ``results/w6_draws_<backbone>_<spec>.csv``, ``w6_summary.csv``,
``w6_paired.csv`` and ``w6_collapse.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.baselines import (  # noqa: E402
    b1_logreg,
    b2_line_a,
    b3_platform_offset,
    b5_knn_prototype,
    b6_zeroshot_em,
    b7_naive_pseudo_label,
    b9_label_propagation,
    b11_mlp,
)
from src.cv import stratified_folds  # noqa: E402
from src.data import data_root, load_mlitw  # noqa: E402
from src.draws import draw_labels, draw_seed, strict_eval_mask  # noqa: E402
from src.em_prior import logit  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.mpp import MPP  # noqa: E402
from src.past import PAST  # noqa: E402

LINEA_LAYERS = (8, 12, 16, 18, 20)
SPEC = "S1-lineA"
BACKBONE = "xlsr300m-linea"
K_PER_CELL = 10
N_DRAWS = 100
PCA_DIM = 256
PROVISIONAL_METHODS = ("B1", "B2", "B3", "B5", "B6", "B7", "B9", "B11", "C1")

__all__ = ["load_dev", "preprocess", "run_one_draw", "summarize_draws", "main"]


# ------------------------------------------------------------------ loading
def load_dev(root: Path, data_dir: Path, layers: Sequence[int] = LINEA_LAYERS):
    """Dev-region rows only: the split is frozen and nothing here reads test."""
    split = json.loads((root / "splits" / "speaker_split_v2.json").read_text(encoding="utf-8"))
    dev_speakers = set(split["dev_speakers"])
    meta = load_mlitw(data_dir)
    keep = meta["speaker"].isin(dev_speakers).to_numpy()
    feats = np.load(data_dir / "mlitw_feats.npy", mmap_mode="r")[:, list(layers), :]
    sub = meta.loc[keep].reset_index(drop=True)
    # each selected layer is ALREADY mean-pooled over time by extract_ssl.py;
    # flattening the layers is what Line A's probe consumed (5 x 1024 = 5120 dims)
    F = np.asarray(feats[keep], dtype=np.float32).reshape(len(sub), -1)
    return sub, F


def load_spec(root: Path, data_dir: Path, backbone: str, spec: str):
    """Dev rows plus the feature block for ``(backbone, spec)``.

    ``backbone == BACKBONE_LINEA`` reads the frozen Line A block; anything else
    reads ``$DATA/x3/feats/<backbone>/`` and rebuilds the W5 spec menu.
    """
    if backbone == BACKBONE:
        return load_dev(root, data_dir)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import w5_backbones as w5

    split = json.loads((root / "splits" / "speaker_split_v2.json").read_text(encoding="utf-8"))
    meta = load_mlitw(data_dir)
    keep = meta["speaker"].isin(set(split["dev_speakers"])).to_numpy()
    sub = meta.loc[keep].reset_index(drop=True)
    fdir = data_dir / "x3" / "feats" / backbone
    mean = np.asarray(np.load(fdir / "mean.npy", mmap_mode="r")[keep], np.float32)
    std = np.asarray(np.load(fdir / "std.npy", mmap_mode="r")[keep], np.float32)
    info = json.loads((fdir / "layers.json").read_text(encoding="utf-8"))
    zero = pd.read_csv(fdir / "scores.csv")["logit_fake_minus_real"].to_numpy()[keep]
    specs = w5.specs_for(mean, std, zero, info["n_layers"])
    if spec not in specs:
        raise KeyError(f"spec {spec!r} not in {sorted(specs)}")
    return sub, specs[spec]


def preprocess(F: np.ndarray, dim: int = PCA_DIM, seed: int = 20261015) -> np.ndarray:
    """Standardise then PCA, fitted on the UNLABELLED region features only."""
    from sklearn.decomposition import PCA

    mu, sd = F.mean(0, keepdims=True), F.std(0, keepdims=True) + 1e-6
    Z = (F - mu) / sd
    dim = int(min(dim, Z.shape[1], max(1, Z.shape[0] - 1)))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=seed).fit(Z)
    return pca.transform(Z).astype(np.float32)


# ------------------------------------------------------------------- a draw
def run_one_draw(args) -> List[Dict[str, Any]]:
    """One draw: label per cell, fit every method, score both masks."""
    (d, Z, y, plat, lang, cell, speaker, zero, methods, k_per_cell) = args
    lab_pos, info = draw_labels(cell, k_per_cell, draw_seed(d), cap_half=True)
    lab = np.zeros(len(y), bool)
    lab[lab_pos] = True
    ev_clips = np.setdiff1d(np.arange(len(y)), lab_pos)
    ev_strict = ev_clips[strict_eval_mask(speaker, lab_pos, ev_clips)]
    # a low floor only: the budget curve deliberately goes down to m=1 (k~30),
    # and refusing to score it would silently drop the smallest curve point
    if lab.sum() < 5 or len(np.unique(y[lab])) < 2 or ev_clips.size < 100:
        return []
    folds = stratified_folds(y[lab])
    scores: Dict[str, np.ndarray] = {}
    infos: Dict[str, Dict[str, Any]] = {}
    Zl, yl = Z[lab], y[lab]
    pl, ll, cl = plat[lab], lang[lab], cell[lab]
    Zu = Z[~lab]

    if "B1" in methods:
        from src.baselines import fit_logreg_cv

        model, i = fit_logreg_cv(Zl, yl, folds=folds)
        scores["B1"] = model.decision_function(Z)   # every clip, not just L
        infos["B1"] = i
    if "B11" in methods:
        from src.baselines import MLPProbe

        probe = MLPProbe(steps=100).fit(Zl, yl)
        scores["B11"] = probe.logit(Z)
        infos["B11"] = {"hidden": probe.hidden}
    if "C1" in methods:
        m = MPP(sigma_p=0.5, sigma_gamma=0.5, lam=1.0).fit(Zl, yl, pl, ll, cl)
        scores["C1"] = m.predict_logit(Z, plat, lang, cell); infos["C1"] = {"sigma": 0.5}
    if "B2" in methods or "B3" in methods:
        base = scores.get("B1")
        if base is not None:
            if "B2" in methods:
                sc, i = b2_line_a(base, cell, plat, lab, y, variant="bb")
                scores["B2"] = sc; infos["B2"] = i
            if "B3" in methods:
                sc, i = b3_platform_offset(base, plat, lab, y, sigma=1.0)
                scores["B3"] = sc; infos["B3"] = i
    if "B5" in methods:
        s, i = b5_knn_prototype(Z, y, k=15, labeled_mask=lab)
        scores["B5"] = s; infos["B5"] = i
    if "B6" in methods:
        s, i = b6_zeroshot_em(zero, cell, plat, lab, y)
        scores["B6"] = s; infos["B6"] = i
    if "B7" in methods:
        s, i = b7_naive_pseudo_label(Zl, yl, Zu, tau=0.9, rounds=3)
        scores["B7"] = np.zeros(len(y)); scores["B7"][~lab] = s
        infos["B7"] = {"pseudo_idx": i["pseudo_idx"], "pseudo_y": i["pseudo_y"]}
    if "B9" in methods:
        s, i = b9_label_propagation(Z, yl, lab, k=15, alpha=0.9)
        scores["B9"] = s; infos["B9"] = i
    if "PAST" in methods:
        from src.past import PAST as _PAST

        model = _PAST(rounds=3).fit(Zl, yl, pl, ll, cl, Zu, plat[~lab], lang[~lab], cell[~lab])
        scores["PAST"] = model.predict_logit(Z, plat, lang, cell)
        infos["PAST"] = {"rounds": 3, "views": "none"}

    rows: List[Dict[str, Any]] = []
    for mask, ev in (("clips", ev_clips), ("strict", ev_strict)):
        if ev.size < 50:
            continue
        for name, sc in scores.items():
            if name in ("B6", "B9"):
                pass  # transductive: the score IS defined on the whole pool
            rows.append(summarize(y[ev], sc[ev], plat[ev], extra={
                "protocol": "dev_v2", "region": "dev", "backbone": BACKBONE, "spec": SPEC,
                "method": name, "k": int(k_per_cell * 29), "draw": d, "mask": mask,
                "n_labeled": int(lab.sum()), "n_eval": int(ev.size),
                "n_cells_labelled": int(sum(1 for v in info.values() if not v["skipped"])),
            }))
    # G5's raw material: B7's pseudo-label mix per cell vs the cell's true rate
    if "B7" in scores:
        idx, py = infos["B7"]["pseudo_idx"], infos["B7"]["pseudo_y"]
        cu = cell[~lab]
        for c in np.unique(cu):
            m = cu[idx] == c
            if m.sum() >= 20:
                rows.append({
                    "protocol": "dev_v2", "region": "dev", "backbone": BACKBONE, "spec": SPEC,
                    "method": "B7", "k": int(k_per_cell * 29), "draw": d, "mask": "collapse",
                    "cell": str(c), "true_rate": float(y[cell == c].mean()),
                    "pl_rate": float(py[m].mean()), "n_pseudo": int(m.sum()),
                })
    return rows


def _worker(payload):
    d = payload[0]
    try:
        return d, run_one_draw(payload)
    except Exception as exc:  # pragma: no cover - reported, never fatal
        return d, [{"error": f"{type(exc).__name__}: {exc}", "draw": d}]


def summarize_draws(rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main = rows[rows["mask"].isin(["clips", "strict"])].copy()
    # LONG format on purpose: decide.py's gates read (method, metric, mean), so a
    # wide table silently makes every gate report NOT_RUN.
    metrics = ["pooled_eer", "within_macro_eer", "worst_platform_eer", "auc"]
    agg = main.groupby(["backbone", "spec", "method", "mask"])[metrics].agg(["mean", "std"])
    long = agg.stack(level=0, future_stack=True).reset_index()
    long = long.rename(columns={"level_4": "metric"})
    long = long[["backbone", "spec", "method", "mask", "metric", "mean", "std"]]

    def paired(a: str, b: str, metric: str) -> Optional[Dict[str, Any]]:
        wa = main[main["method"] == a].set_index("draw")[metric]
        wb = main[main["method"] == b].set_index("draw")[metric]
        common = wa.index.intersection(wb.index)
        if len(common) < 3:
            return None
        d = (wa.loc[common].astype(float) - wb.loc[common].astype(float)).dropna().values
        if d.size < 3:
            return None
        return {"method": a, "other": b, "metric": metric, "mean": float(d.mean()),
                "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
                "n": int(d.size)}

    rows_paired = []
    bb = str(main["backbone"].iloc[0]) if "backbone" in main.columns and len(main) else BACKBONE
    sp = str(main["spec"].iloc[0]) if "spec" in main.columns and len(main) else SPEC
    if "C1" in set(main["method"]):
        for other in ("B1", "B2", "B3", "B11"):
            if other not in set(main["method"]):
                continue
            for metric in ("pooled_eer", "within_macro_eer", "worst_platform_eer"):
                p = paired("C1", other, metric)
                if p:
                    p.update({"backbone": bb, "spec": sp})
                    rows_paired.append(p)
    collapse = rows[rows["mask"] == "collapse"].copy()
    return long, pd.DataFrame(rows_paired), collapse


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W6 baselines")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--k-per-cell", type=int, default=K_PER_CELL)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--methods", default=",".join(PROVISIONAL_METHODS))
    parser.add_argument("--pca", type=int, default=PCA_DIM)
    parser.add_argument("--backbone", default=BACKBONE,
                        help=f"{BACKBONE} = the frozen Line A block; anything else reads "
                             "$DATA/x3/feats/<backbone>/")
    parser.add_argument("--spec", default=SPEC)
    parser.add_argument("--suffix", default="", help="suffix for the draws filename")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    data_dir = data_root(args.data)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    sub, F = load_spec(root, data_dir, args.backbone, args.spec)
    global BACKBONE, SPEC
    backbone, spec = args.backbone, args.spec
    Z = preprocess(F, dim=args.pca)
    y = sub["y"].to_numpy()
    plat = sub["platform"].to_numpy()
    lang = sub["language"].to_numpy()
    cell = sub["cell"].to_numpy()
    speaker = sub["speaker"].to_numpy()
    zp = data_dir / "last_scores_p0cal_raw_s0.npy"
    zero = np.load(zp)[sub["clip_id"].to_numpy()] if zp.exists() else np.zeros(len(sub))
    print(f"dev clips={len(sub)} classes={np.bincount(y)} pca={Z.shape[1]} methods={methods} "
          f"backbone={backbone} spec={spec}")

    payloads = [(d, Z, y, plat, lang, cell, speaker, zero, methods, args.k_per_cell)
                for d in range(args.draws)]
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    if args.jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        with ctx.Pool(args.jobs) as pool:
            for d, out in pool.imap_unordered(_worker, payloads, chunksize=1):
                for r in out:
                    if "error" in r:
                        errors.append(f"draw {d}: {r['error']}")
                    else:
                        rows.append(r)
                if d % 10 == 0:
                    print(f"  draw {d} done ({len(rows)} rows)", flush=True)
    else:
        for p in payloads:
            d, out = _worker(p)
            for r in out:
                if "error" in r:
                    errors.append(f"draw {d}: {r['error']}")
                else:
                    rows.append(r)
    if errors:
        print("errors:", errors[:5], "...", len(errors))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["backbone"] = backbone
        df["spec"] = spec
    (root / "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "results" / f"w6_draws_{backbone}_{spec}{args.suffix}.csv", index=False)
    summary, paired_tbl, collapse = summarize_draws(df)
    summary.to_csv(root / "results" / "w6_summary.csv", index=False)
    if not paired_tbl.empty:
        paired_tbl.to_csv(root / "results" / "w6_paired.csv", index=False)
    if not collapse.empty:
        collapse.to_csv(root / "results" / "w6_collapse.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

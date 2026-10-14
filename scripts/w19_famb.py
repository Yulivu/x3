#!/usr/bin/env python3
"""W19 - robustness on an UNSEEN transcoding chain (TASKBOOK W19).

Training always happens on the clean features; the question is what happens to the
score when the same clips arrive through ``famB1`` (Opus 24k -> MP3 64k, optional
8 kHz band limit), a chain that never appears in training.  The metric is
``dEER = EER(famB1) - EER(clean)`` on the SAME evaluation clips, per method.

``PAST+TTA`` averages the clean score with the two Fam-A view scores (test-time
augmentation) - that is the only method here that is allowed to look at a view at
inference time, and the table says so.

Writes ``results/w19_famb.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import w6_baselines as w6  # noqa: E402
from src.baselines import b2_line_a, fit_logreg_cv  # noqa: E402
from src.data import data_root  # noqa: E402
from src.draws import draw_labels, draw_seed, strict_eval_mask  # noqa: E402
from src.metrics3 import eer, summarize  # noqa: E402
from src.mpp import MPP  # noqa: E402
from src.past import PAST  # noqa: E402

N_DRAWS = 50
K_PER_CELL = 10
METHODS = ("B1", "B2", "C1", "PAST", "PAST+TTA")

__all__ = ["load_tiles", "run_one", "main"]


def load_tiles(data_dir: Path, backbone: str, views: Sequence[str],
               transform, n_rows: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for view in views:
        p = data_dir / "x3" / "views" / view / backbone / "spec_feats.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        tile = np.load(p)
        if tile.shape[0] != n_rows:
            raise ValueError(f"{p} has {tile.shape[0]} rows, expected {n_rows}")
        out[view] = transform(np.asarray(tile, np.float32))
    return out


def _evaluate(y, score_clean, score_shift, plat, ev, name, draw) -> List[Dict[str, Any]]:
    rows = []
    for tag, sc in (("clean", score_clean), ("famb", score_shift)):
        r = summarize(y[ev], sc[ev], plat[ev], extra={
            "protocol": "dev_v2", "region": "dev", "method": name, "draw": int(draw),
            "mask": tag, "n_eval": int(ev.size),
        })
        rows.append(r)
    return rows


def run_one(payload) -> List[Dict[str, Any]]:
    (d, Z, tiles, y, plat, lang, cell, speaker, k_per_cell) = payload
    lab_pos, _ = draw_labels(cell, k_per_cell, draw_seed(d), cap_half=True)
    lab = np.zeros(len(y), bool)
    lab[lab_pos] = True
    ev = np.setdiff1d(np.arange(len(y)), lab_pos)
    if lab.sum() < 50 or ev.size < 100:
        return []
    Zl, yl, pl, ll, cl = Z[lab], y[lab], plat[lab], lang[lab], cell[lab]
    rows: List[Dict[str, Any]] = []

    lr, _ = fit_logreg_cv(Zl, yl)
    clean_b1 = lr.decision_function(Z)
    famb_b1 = lr.decision_function(tiles["famB1"])
    rows += _evaluate(y, clean_b1, famb_b1, plat, ev, "B1", d)
    sc, _ = b2_line_a(clean_b1, cell, plat, lab, y, variant="bb")
    rows += _evaluate(y, sc, famb_b1, plat, ev, "B2", d)   # prior fitted on clean only

    c1 = MPP(sigma_p=0.5, sigma_gamma=0.5, lam=1.0).fit(Zl, yl, pl, ll, cl)
    clean_c1 = c1.predict_logit(Z, plat, lang, cell)
    famb_c1 = c1.predict_logit(tiles["famB1"], plat, lang, cell)
    rows += _evaluate(y, clean_c1, famb_c1, plat, ev, "C1", d)

    past = PAST(rounds=3).fit(Zl, yl, pl, ll, cl, Z[~lab], plat[~lab], lang[~lab], cell[~lab])
    clean_past = past.predict_logit(Z, plat, lang, cell)
    famb_past = past.predict_logit(tiles["famB1"], plat, lang, cell)
    rows += _evaluate(y, clean_past, famb_past, plat, ev, "PAST", d)

    # test-time augmentation: average the clean score with both Fam-A view scores
    view_a1 = past.predict_logit(tiles["famA1"], plat, lang, cell)
    view_a2 = past.predict_logit(tiles["famA2"], plat, lang, cell)
    # symmetric TTA: the clean arm averages the clean channel with the two training
    # views, the shifted arm swaps the clean channel for the unseen chain
    tta_clean = (clean_past + view_a1 + view_a2) / 3.0
    tta_shift = (famb_past + view_a1 + view_a2) / 3.0
    rows += _evaluate(y, tta_clean, tta_shift, plat, ev, "PAST+TTA", d)
    return rows


def _worker(payload):
    try:
        return payload[0], run_one(payload)
    except Exception as exc:  # pragma: no cover
        return payload[0], [{"error": f"{type(exc).__name__}: {exc}"}]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W19 Fam-B robustness")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--views", default="famA1,famA2,famB1")
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    data_dir = data_root(args.data)
    sub, F = w6.load_spec(root, data_dir, args.backbone, args.spec)
    transform, Z = w6.fit_preprocess(F)
    tiles = load_tiles(data_dir, args.backbone,
                       [v.strip() for v in args.views.split(",") if v.strip()],
                       transform, len(sub))
    print(f"dev clips={len(sub)} tiles={list(tiles)}")
    y = sub["y"].to_numpy()
    plat = sub["platform"].to_numpy()
    lang = sub["language"].to_numpy()
    cell = sub["cell"].to_numpy()
    speaker = sub["speaker"].to_numpy()
    payloads = [(d, Z, tiles, y, plat, lang, cell, speaker, K_PER_CELL)
                for d in range(args.draws)]
    rows, errors = [], []
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
    (root / "results").mkdir(parents=True, exist_ok=True)
    piv = df.groupby(["method", "mask"])["pooled_eer"].mean().unstack()
    piv["dEER"] = piv["famb"] - piv["clean"]
    piv = piv.reset_index()
    piv["backbone"] = args.backbone
    piv["spec"] = args.spec
    piv.to_csv(root / "results" / "w19_famb.csv", index=False)
    df.to_csv(root / "results" / "w19_draws.csv", index=False)
    print(piv.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

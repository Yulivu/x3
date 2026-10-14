#!/usr/bin/env python3
"""W20 - sensitivity of the conclusions to the pipeline's free choices (TASKBOOK W20).

One knob at a time, everything else at its default, 30 draws per setting, four
methods (B1, B2, C1, PAST):

* ``pca``           - 64 / 128 / 256 / 512 components
* ``cell``          - platform only / language only / platform x language (default)
* ``label_noise``   - 0 % / 5 % / 10 % flipped labels inside ``L``
* ``label_draw``    - uniform per cell (default) / globally random / all labels
                      drawn from 20 speakers

One-at-a-time rather than a full cross product: with 4 methods and 30 draws the
cross product would be ~5x the compute for a question the marginals already answer,
and the report says so instead of pretending the knobs are independent.

Writes ``results/w20_sensitivity.csv``.
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
from src.metrics3 import summarize  # noqa: E402
from src.mpp import MPP  # noqa: E402
from src.past import PAST  # noqa: E402

N_DRAWS = 30
K_PER_CELL = 10
METHODS = ("B1", "B2", "C1", "PAST")
PCA_GRID = (64, 128, 256, 512)
CELL_VARIANTS = ("platform", "language", "platform_x_language")
NOISE_GRID = (0.0, 0.05, 0.10)
DRAW_VARIANTS = ("per_cell", "global", "speaker_clustered")

__all__ = ["select_labels", "run_config", "main"]


def select_labels(kind: str, cell, speaker, k_per_cell: int, seed: int) -> np.ndarray:
    """Return the labelled positions for one draw under one selection rule."""
    n = len(cell)
    rng = np.random.default_rng(seed)
    if kind == "per_cell":
        pos, _ = draw_labels(cell, k_per_cell, seed, cap_half=True)
        return pos
    if kind == "global":
        k = int(k_per_cell * len(np.unique(cell)))
        return np.sort(rng.choice(n, size=min(k, n), replace=False))
    if kind == "speaker_clustered":
        speakers = np.unique(speaker)
        chosen = rng.choice(speakers, size=min(20, len(speakers)), replace=False)
        pool = np.where(np.isin(speaker, chosen))[0]
        k = int(k_per_cell * len(np.unique(cell)))
        return np.sort(rng.choice(pool, size=min(k, pool.size), replace=False))
    raise ValueError(f"unknown label-draw rule {kind!r}")


def run_config(payload) -> List[Dict[str, Any]]:
    (knob, value, d, Z, y, plat, lang, cell_default, speaker) = payload
    cell = cell_default
    y_used = y
    if knob == "cell":
        # only the PRIOR GROUPING changes: the label budget stays 290 spread over
        # the platform x language cells, otherwise the knob would also change k
        cell = (plat if value == "platform" else lang if value == "language" else cell_default)
    draw_kind = value if knob == "label_draw" else "per_cell"
    lab_pos = select_labels(draw_kind, cell_default, speaker, K_PER_CELL, draw_seed(d))
    lab = np.zeros(len(y), bool)
    lab[lab_pos] = True
    ev = np.setdiff1d(np.arange(len(y)), lab_pos)
    if lab.sum() < 50 or ev.size < 100:
        return []
    if knob == "label_noise" and float(value) > 0:
        rng = np.random.default_rng(999 + d)
        flip = rng.random(int(lab.sum())) < float(value)
        y_used = y.copy()
        idx = np.where(lab)[0]
        y_used[idx[flip]] = 1 - y_used[idx[flip]]
    Zl, yl, pl, ll, cl = Z[lab], y_used[lab], plat[lab], lang[lab], cell[lab]
    Zu = Z[~lab]
    scores: Dict[str, np.ndarray] = {}
    lr, _ = fit_logreg_cv(Zl, yl)
    base = lr.decision_function(Z)
    scores["B1"] = base
    b2, _ = b2_line_a(base, cell, plat, lab, y_used, variant="bb")
    scores["B2"] = b2
    c1 = MPP(sigma_p=0.5, sigma_gamma=0.5, lam=1.0).fit(Zl, yl, pl, ll, cl)
    scores["C1"] = c1.predict_logit(Z, plat, lang, cell)
    past = PAST(rounds=3).fit(Zl, yl, pl, ll, cl, Zu, plat[~lab], lang[~lab], cell[~lab])
    scores["PAST"] = past.predict_logit(Z, plat, lang, cell)
    rows = []
    for name, sc in scores.items():
        rows.append(summarize(y[ev], sc[ev], plat[ev], extra={
            "protocol": "dev_v2", "region": "dev", "knob": knob, "value": str(value),
            "method": name, "draw": int(d), "mask": "clips", "n_labeled": int(lab.sum()),
            "n_eval": int(ev.size),
        }))
    return rows


def _worker(payload):
    try:
        return payload[2], payload[0], payload[1], run_config(payload)
    except Exception as exc:  # pragma: no cover
        return payload[2], payload[0], payload[1], [{"error": f"{type(exc).__name__}: {exc}"}]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W20 sensitivity")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    data_dir = data_root(args.data)
    sub, F = w6.load_spec(root, data_dir, args.backbone, args.spec)
    y = sub["y"].to_numpy()
    plat = sub["platform"].to_numpy()
    lang = sub["language"].to_numpy()
    cell_default = sub["cell"].to_numpy()
    speaker = sub["speaker"].to_numpy()

    payloads = []
    for value in PCA_GRID:
        Z = w6.preprocess(F, dim=int(value))
        payloads += [("pca", f"{value}", d, Z, y, plat, lang, cell_default, speaker)
                     for d in range(args.draws)]
    Z256 = w6.preprocess(F)
    for value in CELL_VARIANTS:
        payloads += [("cell", value, d, Z256, y, plat, lang, cell_default, speaker)
                     for d in range(args.draws)]
    for value in NOISE_GRID:
        payloads += [("label_noise", f"{value}", d, Z256, y, plat, lang, cell_default, speaker)
                     for d in range(args.draws)]
    for value in DRAW_VARIANTS:
        payloads += [("label_draw", value, d, Z256, y, plat, lang, cell_default, speaker)
                     for d in range(args.draws)]
    print(f"dev clips={len(sub)} configs={len({(p[0], p[1]) for p in payloads})} "
          f"tasks={len(payloads)}")
    rows, errors = [], []
    if args.jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        with ctx.Pool(args.jobs) as pool:
            for d, knob, value, out in pool.imap_unordered(_worker, payloads, chunksize=1):
                for r in out:
                    if "error" in r:
                        errors.append(f"{knob}={value} draw={d}: {r['error']}")
                    else:
                        rows.append(r)
    else:
        for p in payloads:
            d, knob, value, out = _worker(p)
            for r in out:
                if "error" in r:
                    errors.append(f"{knob}={value} draw={d}: {r['error']}")
                else:
                    rows.append(r)
    if errors:
        print("errors:", errors[:5], "...", len(errors))
    df = pd.DataFrame(rows)
    (root / "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "results" / "w20_draws.csv", index=False)
    summary = (df.groupby(["knob", "value", "method"])
               [["pooled_eer", "within_macro_eer", "worst_platform_eer", "auc"]]
               .agg(["mean", "std"]))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary["backbone"] = args.backbone
    summary["spec"] = args.spec
    summary.to_csv(root / "results" / "w20_sensitivity.csv", index=False)
    piv = summary.pivot_table(index=["knob", "value"], columns="method",
                              values="within_macro_eer_mean")
    print(piv.round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

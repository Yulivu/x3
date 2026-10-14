#!/usr/bin/env python3
"""W18 - C4, ensembles and test-time augmentation (TASKBOOK W18).

* ``E2`` - the Fam-A TTA arm: average the clean-channel logits with both Fam-A
  view logits, for C1 and for PAST.  This is the variant that only needs one
  backbone, so it is what runs now.
* ``E1`` (top-2 backbone average) and ``E3`` (E1+E2) need a second backbone and
  are reported as ``NOT_RUN(needs_2nd_backbone)`` rather than quietly dropped.

Every arm is compared against its own no-TTA counterpart on the same draws, so
the table answers "does TTA help?" rather than "is TTA good?".

Writes ``results/w18_c4.csv`` and ``results/w18_views.csv``.
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
from src.data import data_root  # noqa: E402
from src.draws import draw_labels, draw_seed, strict_eval_mask  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.mpp import MPP  # noqa: E402
from src.past import PAST  # noqa: E402

N_DRAWS = 50
K_PER_CELL = 10
ARMS = ("C1", "C1+TTA", "PAST", "PAST+TTA")

__all__ = ["run_one", "paired", "main"]


def run_one(payload) -> List[Dict[str, Any]]:
    (d, Z, tiles, y, plat, lang, cell, speaker, k_per_cell) = payload
    lab_pos, _ = draw_labels(cell, k_per_cell, draw_seed(d), cap_half=True)
    lab = np.zeros(len(y), bool)
    lab[lab_pos] = True
    ev = np.setdiff1d(np.arange(len(y)), lab_pos)
    if lab.sum() < 50 or ev.size < 100:
        return []
    Zl, yl, pl, ll, cl = Z[lab], y[lab], plat[lab], lang[lab], cell[lab]

    c1 = MPP(sigma_p=0.5, sigma_gamma=0.5, lam=1.0).fit(Zl, yl, pl, ll, cl)
    clean_c1 = c1.predict_logit(Z, plat, lang, cell)
    tta_c1 = (clean_c1 + c1.predict_logit(tiles["famA1"], plat, lang, cell)
              + c1.predict_logit(tiles["famA2"], plat, lang, cell)) / 3.0
    past = PAST(rounds=3).fit(Zl, yl, pl, ll, cl, Z[~lab], plat[~lab], lang[~lab], cell[~lab])
    clean_past = past.predict_logit(Z, plat, lang, cell)
    tta_past = (clean_past + past.predict_logit(tiles["famA1"], plat, lang, cell)
                + past.predict_logit(tiles["famA2"], plat, lang, cell)) / 3.0

    rows: List[Dict[str, Any]] = []
    for name, sc in (("C1", clean_c1), ("C1+TTA", tta_c1),
                     ("PAST", clean_past), ("PAST+TTA", tta_past)):
        for mask, idx in (("clips", ev), ("strict", ev[strict_eval_mask(speaker, lab_pos, ev)])):
            if idx.size < 50:
                continue
            rows.append(summarize(y[idx], sc[idx], plat[idx], extra={
                "protocol": "dev_v2", "region": "dev", "method": name, "draw": int(d),
                "mask": mask, "n_labeled": int(lab.sum()), "n_eval": int(idx.size),
                "arm": "E2" if name.endswith("+TTA") else "none",
            }))
    return rows


def _worker(payload):
    try:
        return payload[0], run_one(payload)
    except Exception as exc:  # pragma: no cover
        return payload[0], [{"error": f"{type(exc).__name__}: {exc}"}]


def paired(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df["mask"] == "clips"]
    rows = []
    for tta, base in (("C1+TTA", "C1"), ("PAST+TTA", "PAST")):
        for metric in ("pooled_eer", "within_macro_eer"):
            a = main[main["method"] == tta].set_index("draw")[metric]
            b = main[main["method"] == base].set_index("draw")[metric]
            common = a.index.intersection(b.index)
            if len(common) < 3:
                continue
            diff = (a.loc[common].astype(float) - b.loc[common].astype(float)).dropna().values
            if diff.size < 3:
                continue
            rows.append({"method": tta, "other": base, "metric": metric, "arm": "E2",
                         "mean": float(diff.mean()), "lo": float(np.percentile(diff, 2.5)),
                         "hi": float(np.percentile(diff, 97.5)), "n": int(diff.size)})
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W18 C4 ensemble / TTA")
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
    transform, Z = w6.fit_preprocess(F)
    tiles = {}
    for view in ("famA1", "famA2"):
        p = data_dir / "x3" / "views" / view / args.backbone / "spec_feats.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        tiles[view] = transform(np.asarray(np.load(p), np.float32))
    if len({t.shape for t in tiles.values()}) != 1 or next(iter(tiles.values())).shape[0] != len(sub):
        raise ValueError("view tiles are not aligned with the dev rows")
    print(f"dev clips={len(sub)} views={list(tiles)} arms={ARMS}")
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
    df.to_csv(root / "results" / "w18_views.csv", index=False)
    summary = (df[df["mask"] == "clips"].groupby("method")
               [["pooled_eer", "within_macro_eer", "worst_platform_eer", "auc"]].mean().round(3))
    print(summary.to_string())
    tbl = paired(df)
    if not tbl.empty:
        tbl["backbone"] = args.backbone
        tbl["spec"] = args.spec
    # the arms that need a second backbone are stated, not dropped
    extra = pd.DataFrame([
        {"method": "E1", "other": "top-2 backbones", "metric": "pooled_eer", "arm": "E1",
         "mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0, "backbone": args.backbone,
         "spec": args.spec, "note": "NOT_RUN(needs_2nd_backbone)"},
        {"method": "E3", "other": "E1+E2", "metric": "pooled_eer", "arm": "E3",
         "mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0, "backbone": args.backbone,
         "spec": args.spec, "note": "NOT_RUN(needs_2nd_backbone)"},
    ])
    out = pd.concat([tbl, extra], ignore_index=True, sort=False) if not tbl.empty else extra
    out.to_csv(root / "results" / "w18_c4.csv", index=False)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

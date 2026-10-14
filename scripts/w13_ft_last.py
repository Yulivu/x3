#!/usr/bin/env python3
"""W13 - B10, fine-tuning the last block (REDUCED variant on pooled features).

The full variant needs the encoder's last block, i.e. the AntiDeepfake backbone and
its fairseq module, so it belongs to the GPU lane.  This script runs the documented
fallback: LayerNorm + linear head trained with the same optimiser, budget and CV
rule (``B10_REDUCED``).  Every row carries ``variant="B10_REDUCED"`` so the report
can never present it as the full fine-tune.

Rows are written into the ``w6_draws_*`` family with the identical draw protocol,
so S-F can pair B10 against C1 draw by draw.
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
from src.ft_last import B10_REDUCED_NOTE, fit_b10  # noqa: E402
from src.metrics3 import summarize  # noqa: E402

N_DRAWS = 50
K_PER_CELL = 10

__all__ = ["run_one", "main"]


def run_one(payload) -> List[Dict[str, Any]]:
    (d, Z, y, plat, lang, cell, speaker, k_per_cell) = payload
    lab_pos, _ = draw_labels(cell, k_per_cell, draw_seed(d), cap_half=True)
    lab = np.zeros(len(y), bool)
    lab[lab_pos] = True
    ev = np.setdiff1d(np.arange(len(y)), lab_pos)
    if lab.sum() < 5 or len(np.unique(y[lab])) < 2 or ev.size < 100:
        return []
    head, info = fit_b10(Z[lab], y[lab])
    sc = head.logit(Z)
    rows = []
    for mask, idx in (("clips", ev), ("strict", ev[strict_eval_mask(speaker, lab_pos, ev)])):
        if idx.size < 50:
            continue
        rows.append(summarize(y[idx], sc[idx], plat[idx], extra={
            "protocol": "dev_v2", "region": "dev", "backbone": w6.BACKBONE, "spec": w6.SPEC,
            "method": "B10", "k": int(k_per_cell * 29), "draw": int(d), "mask": mask,
            "n_labeled": int(lab.sum()), "n_eval": int(idx.size),
            "variant": info["variant"], "lr": info["lr"], "note": B10_REDUCED_NOTE,
        }))
    return rows


def _worker(payload):
    try:
        return payload[0], run_one(payload)
    except Exception as exc:  # pragma: no cover
        return payload[0], [{"error": f"{type(exc).__name__}: {exc}"}]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W13 B10 (reduced)")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--k-per-cell", type=int, default=K_PER_CELL)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    sub, F = w6.load_dev(root, data_root(args.data))
    Z = w6.preprocess(F)
    y = sub["y"].to_numpy()
    plat = sub["platform"].to_numpy()
    lang = sub["language"].to_numpy()
    cell = sub["cell"].to_numpy()
    speaker = sub["speaker"].to_numpy()
    print(f"dev clips={len(sub)} draws={args.draws} variant=B10_REDUCED")

    payloads = [(d, Z, y, plat, lang, cell, speaker, args.k_per_cell) for d in range(args.draws)]
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
    out = root / "results" / f"w6_draws_{w6.BACKBONE}_{w6.SPEC}_b10.csv"
    df.to_csv(out, index=False)
    summary = df[df["mask"] == "clips"][["pooled_eer", "within_macro_eer", "auc"]].mean().round(3)
    print(summary.to_string())
    print(f"wrote {out}  ({B10_REDUCED_NOTE})")
    auc_mean = float(df[df["mask"] == "clips"]["auc"].mean())
    if auc_mean < 0.5:
        # A head that scores BELOW chance is not a weak baseline, it is a broken
        # one: with a random init the declared budget (5 epochs at <=3e-5) barely
        # moves the weights, so the score is dominated by the first principal
        # direction. Record it rather than quietly reporting "we beat B10".
        note = {
            "item": "B10_REDUCED is not a usable baseline",
            "what": (
                f"B10_REDUCED scored mean AUC {auc_mean:.3f} (< 0.5) over {args.draws} draws: "
                "the head never trained. With a random initialisation the declared budget "
                "(5 epochs, lr <= 3e-5, AdamW) leaves the weights essentially at their random "
                "values, so the score is a random projection dominated by PC1 and comes out "
                "anti-correlated with the label. The full B10 (W13 on the GPU lane) is supposed "
                "to start from the backbone's own pretrained head, which this variant cannot do."
            ),
            "impact": (
                "Any 'we beat B10' statement made with these rows is vacuous. Either run the full "
                "B10 on the encoder, or report B10 as NOT_MEANINGFUL and keep it out of the S-F "
                "comparison."
            ),
            "source": "W13 on the box, 2026-10-14",
        }
        dev_path = root / "results" / "deviations.csv"
        existing = pd.read_csv(dev_path) if dev_path.exists() else pd.DataFrame()
        pd.concat([existing, pd.DataFrame([note])], ignore_index=True).to_csv(dev_path, index=False)
        print(f"DEVIATION recorded: B10_REDUCED mean AUC {auc_mean:.3f} < 0.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

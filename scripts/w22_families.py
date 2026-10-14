#!/usr/bin/env python3
"""W22 item 6 - the transcode-family robustness table.

The taskbook's backlog asks for **more transcode-chain families**.  ``src/views.py``
now holds four, all rendered on the frozen v3 dev region into suffixed view
directories (``views/<family>_v3/``) so the original v2 views stay untouched:

* ``famA1`` - the in-family reference (Opus branch with gain / pink-noise /
  downsampling jitter), i.e. the kind of degradation a probe is trained on;
* ``famB1`` - the taskbook's held-out chain (Opus 24 -> MP3 64, optional 8 kHz band);
* ``famB2`` - a *longer* chain at low bitrate (AAC 32 -> MP3 32 -> Opus 16, 6 kHz);
* ``famB3`` - a *narrow-band* chain (MP3 48 -> Opus 16, 4 kHz).

Same clips, same draws, same probe: the only thing that changes is the channel, so
the table reads directly as "what does this chain cost".

Writes ``results/w22_families.csv`` (per draw) and ``results/w22_families_summary.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if sys.path.append(str(_p)) is None and _p not in sys.path:
        sys.path.insert(0, str(_p))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import u1_speaker_split as u1  # noqa: E402
import w6_baselines as w6  # noqa: E402
import w22_backlog as w22  # noqa: E402
from src import regions as R  # noqa: E402
from src.baselines import fit_logreg_cv  # noqa: E402
from src.cv import speaker_grouped_folds, stratified_folds  # noqa: E402
from src.data import data_root  # noqa: E402
from src.draws import draw_labels, draw_seed  # noqa: E402
from src.metrics3 import summarize  # noqa: E402

FAMILIES = ("famA1", "famB1", "famB2", "famB3")
MASKS = ("clips", "strict_draw")
K_PER_CELL = 10
N_DRAWS = 50
DEFAULT_SUFFIX = "_v3"

_G: Dict[str, Any] = {}

__all__ = ["view_dir", "load_view_specs", "run_one", "summarise", "main"]


def view_dir(data_dir: Path, family: str, backbone: str, suffix: str = DEFAULT_SUFFIX
             ) -> Path:
    return Path(data_dir) / "x3" / "views" / f"{family}{suffix}" / backbone


def load_view_specs(data_dir: Path, backbone: str, families: Sequence[str] = FAMILIES,
                    suffix: str = DEFAULT_SUFFIX) -> Dict[str, np.ndarray]:
    """The S3 block of every family, plus the clean reference, row-aligned."""
    out: Dict[str, np.ndarray] = {}
    for family in families:
        path = view_dir(data_dir, family, backbone, suffix) / "spec_feats.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; render the family first with "
                f"scripts/w10_views.py --views {family} --split speaker_split_v3.json "
                f"--out-suffix {suffix}")
        out[family] = np.asarray(np.load(path, mmap_mode="r"), np.float32)
    return out


def run_one(payload) -> Tuple[int, str, List[Dict[str, Any]]]:
    d, mask = payload
    y, plat, cell, speaker = _G["y"], _G["plat"], _G["cell"], _G["speaker"]
    rows: List[Dict[str, Any]] = []
    pool, ev = _G["masks"][d][mask]
    if ev.size < 50:
        return d, mask, rows
    idx, _info = draw_labels(cell[pool], K_PER_CELL, draw_seed(d), cap_half=False)
    lab = pool[idx]
    if lab.size < 20 or len(np.unique(y[lab])) < 2:
        return d, mask, rows
    folds = (stratified_folds(y[lab]) if mask == "clips"
             else speaker_grouped_folds(speaker[lab], y[lab]))
    for arm, block in _G["blocks"].items():
        # the standardiser is fitted on the unlabelled region of the SAME family, so
        # no arm sees labels its peers do not
        model, info = fit_logreg_cv(block[lab], y[lab], folds=folds)
        scores = model.decision_function(block)
        rows.append(summarize(y[ev], scores[ev], plat[ev], extra={
            "protocol": "dev_v3_families", "region": "dev", "mask": mask,
            "method": f"B1-{arm}", "arm": arm, "k": int(lab.size), "draw": int(d),
            "backbone": _G["backbone"], "n_eval": int(ev.size),
            "n_labeled": int(lab.size),
            "cv_folds": "stratified" if mask == "clips" else "speaker_group",
            "C": info["C"]}))
    return d, mask, rows


def _worker(payload):
    try:
        return run_one(payload)
    except Exception as exc:                            # pragma: no cover
        return payload[0], payload[1], [{"error": f"{payload}: {exc}"}]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled EER per (arm, mask) plus the cost against the in-family reference."""
    if df.empty:
        return df
    piv = df.groupby(["arm", "mask"])["pooled_eer"].mean().unstack()
    out = piv.copy()
    for mask in piv.columns:
        if "famA1" in piv.index:
            out[f"cost_vs_famA1_{mask}"] = piv[mask] - piv.loc["famA1", mask]
        if "clean" in piv.index:
            out[f"cost_vs_clean_{mask}"] = piv[mask] - piv.loc["clean", mask]
    return out.round(3).reset_index()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W22 transcode families")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--split", default=R.SPLIT_V3)
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    data_dir = data_root(args.data)
    families = [f.strip() for f in str(args.families).split(",") if f.strip()]
    sub, F = R.load_region(root, data_dir, args.backbone, args.spec, args.split)
    transform, Z = w6.fit_preprocess(F)
    y = sub["y"].to_numpy()
    lang = sub["language"].to_numpy()
    cell = np.asarray([f"{p}|{l}" for p, l in zip(sub["platform"], lang)])
    speaker = sub["speaker"].to_numpy()
    blocks: Dict[str, np.ndarray] = {"clean": Z}
    for family, block in load_view_specs(data_dir, args.backbone, families,
                                         args.suffix).items():
        if block.shape[0] != len(sub):
            raise ValueError(f"{family}: {block.shape[0]} rows for {len(sub)} dev "
                             "clips; the views were rendered on a different region")
        blocks[family] = w6.fit_preprocess(block)[1]
    _G.update({"y": y, "plat": sub["platform"].to_numpy(), "cell": cell,
               "speaker": speaker, "backbone": args.backbone, "blocks": blocks,
               "masks": {d: w22.masks_for(d, Z, y, cell, speaker)
                         for d in range(int(args.draws))}})
    print(f"dev clips={len(sub)} arms={list(blocks)} draws={args.draws}", flush=True)
    payloads = [(d, mask) for mask in MASKS for d in range(int(args.draws))]
    rows = w22._pooled([], int(args.jobs), payloads)
    df = pd.DataFrame(rows)
    df.to_csv(root / "results" / "w22_families.csv", index=False)
    summary = summarise(df)
    summary.to_csv(root / "results" / "w22_families_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

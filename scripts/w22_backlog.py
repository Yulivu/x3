#!/usr/bin/env python3
"""W22 - the P3 backlog (TASKBOOK 6), run in the order the taskbook lists it.

The taskbook says this work package runs "until somebody stops it".  Three of its
six items are done elsewhere and are *recorded* here rather than duplicated:

* **Z4** (label-free PAST) lives in ``src/trackz.py`` and is reported as a row of
  ``results/w8_zlabel.csv``;
* **100 -> 200 draws** is a rerun of the headline config; this driver delegates it to
  ``w6_baselines.py --draws 200 --suffix _d200`` so no stored file is overwritten
  (``--heavy`` turns it on, because it is the expensive item);
* **more transcode-chain families** needs new augmented views, i.e. a GPU pass of
  W10, and is recorded with that reason.

Two items are computed here because nothing else produces them:

* **more k points** - the budget curve at k in {30, 60, 100, 200, 290, 600} on the
  frozen v3 region, for B1/B9/C1, on both allowed masks (50 draws each);
* **spec / layer sensitivity** - B1 and C1 on every W5 spec (S1..S5) with the same
  features and draws, so a spec choice can be read off directly.

Writes ``results/w22_kcurve.csv``, ``results/w22_spec_sensitivity.csv`` and
``results/w22_backlog.csv`` (one row per backlog item with its status).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import u1_speaker_split as u1  # noqa: E402
import u1p_speaker_split_v3 as u1p  # noqa: E402
import w6_baselines as w6  # noqa: E402
from src import regions as R  # noqa: E402
from src.baselines import b9_label_propagation, fit_logreg_cv, knn_graph  # noqa: E402
from src.cv import speaker_grouped_folds, stratified_folds  # noqa: E402
from src.data import data_root  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.mpp import MPP  # noqa: E402

K_PER_CELL = {30: 1, 60: 2, 100: 3, 200: 6, 290: 10, 600: 20}
METHODS = ("B1", "B9", "C1")
SPECS = ("S1", "S2", "S3", "S4", "S5")
MASKS = ("clips", "strict_draw")
STRICT_SEED = 777000
C1_FIXED = (0.5, 0.5, 1.0)
HEAVY_DRAWS = 200

_G: Dict[str, Any] = {}

__all__ = ["masks_for", "run_unit", "kcurve", "spec_sensitivity", "backlog_table",
           "delegate_heavy", "main"]


def masks_for(d: int, Z, y, cell, speaker) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """``(labelled, eval)`` for the two masks a selector/learner may be compared on."""
    n = len(y)
    rng = np.random.default_rng(1000 + d)
    ev = np.sort(rng.choice(n, size=int(0.25 * n), replace=False))
    pool = np.setdiff1d(np.arange(n), ev)
    s_l, s_e = u1.split_strict(speaker, seed=STRICT_SEED + d)
    return {"clips": (pool, ev), "strict_draw": (np.where(s_l)[0], np.where(s_e)[0])}


def run_unit(payload) -> Tuple[int, str, int, List[Dict[str, Any]]]:
    d, mask, k = payload
    Z, y, plat, cell, speaker = (_G["Z"], _G["y"], _G["plat"], _G["cell"],
                                 _G["speaker"])
    graph = _G["graph"]
    rows: List[Dict[str, Any]] = []
    pool, ev = _G["masks"][d][mask]
    if ev.size < 50 or pool.size < k:
        return d, mask, k, rows
    from src.draws import draw_labels, draw_seed

    m = K_PER_CELL.get(k, max(1, k // 29))
    idx, _info = draw_labels(cell[pool], m, draw_seed(d), cap_half=False)
    lab = pool[idx]
    if lab.size < 20 or len(np.unique(y[lab])) < 2:
        return d, mask, k, rows
    folds = (stratified_folds(y[lab]) if mask == "clips"
             else speaker_grouped_folds(speaker[lab], y[lab]))
    lr, info = fit_logreg_cv(Z[lab], y[lab], folds=folds)
    scores = {"B1": lr.decision_function(Z)}
    lab_mask = np.zeros(len(y), bool)
    lab_mask[lab] = True
    s9, _ = b9_label_propagation(Z, y[lab], lab_mask, k=15, alpha=0.9, graph=graph)
    scores["B9"] = s9
    c1 = MPP(sigma_p=C1_FIXED[0], sigma_gamma=C1_FIXED[1], lam=C1_FIXED[2])
    c1.fit(Z[lab], y[lab], plat[lab], _G["lang"][lab], cell[lab])
    scores["C1"] = c1.predict_logit(Z, plat, _G["lang"], cell)
    for name, sc in scores.items():
        rows.append(summarize(y[ev], sc[ev], plat[ev], extra={
            "protocol": "dev_v3_w22", "region": "dev", "mask": mask, "method": name,
            "k": int(lab.size), "k_nominal": int(k), "draw": int(d),
            "backbone": _G["backbone"], "spec": _G["spec"], "n_eval": int(ev.size),
            "n_labeled": int(lab.size), "C": info["C"] if name == "B1" else None,
            "cv_folds": "stratified" if mask == "clips" else "speaker_group"}))
    return d, mask, k, rows


def _worker(payload):
    try:
        return run_unit(payload)
    except Exception as exc:                            # pragma: no cover
        return payload[0], payload[1], payload[2], [{"error": f"{payload}: {exc}"}]


def _pooled(rows: List[Dict[str, Any]], jobs: int, payloads) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    errors: List[str] = []
    if jobs > 1:
        import multiprocessing as mp

        ctx = (mp.get_context("fork") if sys.platform != "win32"
               else mp.get_context("spawn"))
        with ctx.Pool(jobs) as pool:
            for i, (*_head, res) in enumerate(pool.imap_unordered(_worker, payloads,
                                                                  chunksize=1), 1):
                for r in res:
                    if "error" in r:
                        errors.append(str(r["error"]))
                    else:
                        out.append(r)
    else:
        for pl in payloads:
            *_head, res = _worker(pl)
            for r in res:
                if "error" in r:
                    errors.append(str(r["error"]))
                else:
                    out.append(r)
    if errors:
        print(f"  {len(errors)} errors; first: {errors[:2]}")
    return out


def kcurve(root: Path, data_dir: Path, backbone: str, spec: str, split: str,
           draws: int = 50, jobs: int = 8) -> pd.DataFrame:
    sub, F = R.load_region(root, data_dir, backbone, spec, split)
    transform, Z = w6.fit_preprocess(F)
    y = sub["y"].to_numpy()
    lang = sub["language"].to_numpy()
    cell = np.asarray([f"{p}|{l}" for p, l in zip(sub["platform"], lang)])
    _G.update({"Z": Z, "y": y, "plat": sub["platform"].to_numpy(), "lang": lang,
               "cell": cell, "speaker": sub["speaker"].to_numpy(),
               "backbone": backbone, "spec": spec,
               "graph": knn_graph(Z, k=15),
               "masks": {d: masks_for(d, Z, y, cell, sub["speaker"].to_numpy())
                         for d in range(int(draws))}})
    payloads = [(d, mask, k) for k in K_PER_CELL for mask in MASKS
                for d in range(int(draws))]
    rows = _pooled([], jobs, payloads)
    return pd.DataFrame(rows)


def spec_sensitivity(root: Path, data_dir: Path, backbone: str, split: str,
                     specs: Sequence[str] = SPECS, draws: int = 30, jobs: int = 8
                     ) -> pd.DataFrame:
    """B1 on every W5 spec, same draws, so the spec choice is directly readable."""
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        sub, F = R.load_region(root, data_dir, backbone, spec, split)
        transform, Z = w6.fit_preprocess(F)
        y = sub["y"].to_numpy()
        lang = sub["language"].to_numpy()
        cell = np.asarray([f"{p}|{l}" for p, l in zip(sub["platform"], lang)])
        speaker = sub["speaker"].to_numpy()
        plat = sub["platform"].to_numpy()
        _G.update({"Z": Z, "y": y, "plat": plat, "lang": lang, "cell": cell,
                   "speaker": speaker, "backbone": backbone, "spec": spec,
                   "masks": {d: masks_for(d, Z, y, cell, speaker)
                             for d in range(int(draws))}})
        payloads = [(d, mask, 290) for mask in MASKS for d in range(int(draws))]
        rows += _pooled([], jobs, payloads)
    return pd.DataFrame(rows)


def backlog_table(kcurve_df: pd.DataFrame, spec_df: pd.DataFrame,
                  root: Path) -> pd.DataFrame:
    """One row per taskbook backlog item, with its status and the evidence path."""
    rows: List[Dict[str, Any]] = [
        {"item": "1. 100 -> 200 draws (headline configs)",
         "status": "DELEGATED",
         "evidence": "w6_baselines.py --draws 200 --suffix _d200 writes "
                     "results/w6_draws_*_d200.csv and results/w6_summary_d200.csv",
         "note": "run with --heavy; no stored file is overwritten"},
        {"item": "2. more k points",
         "status": "DONE" if not kcurve_df.empty else "NOT_RUN",
         "evidence": "results/w22_kcurve.csv",
         "note": f"k in {sorted(K_PER_CELL)} on {len(MASKS)} masks"},
        {"item": "3. Z4 label-free PAST",
         "status": "DONE",
         "evidence": "results/w8_zlabel.csv (row Z4)",
         "note": "LR self-trained on the Z2 posterior, cell priors fixed"},
        {"item": "4. spec / layer sensitivity",
         "status": "DONE" if not spec_df.empty else "NOT_RUN",
         "evidence": "results/w22_spec_sensitivity.csv",
         "note": f"specs {list(SPECS)} x {len(MASKS)} masks at k=290"},
        {"item": "5. second backbone for the P1 experiments",
         "status": "PARTIAL",
         "evidence": "results/w6_summary_v3_mms1b.csv (all baselines) and "
                     "results/w17_extra_backbone_paired.csv",
         "note": "W11 accepts --backbone, so PAST can follow; CASA (W12) has no "
                 "backbone switch yet and its second-backbone run is still open"},
        {"item": "6. more transcode-chain families",
         "status": "NOT_RUN",
         "evidence": "-",
         "note": "needs new augmented views, i.e. a W10 GPU pass; W19 already covers "
                 "Fam-B1 (Opus24 + MP3 64 + 8 kHz)"},
    ]
    df = pd.DataFrame(rows)
    df["has_kcurve"] = not kcurve_df.empty
    df["has_spec"] = not spec_df.empty
    return df


def delegate_heavy(root: Path, backbone: str, spec: str, split: str,
                   draws: int = HEAVY_DRAWS, jobs: int = 8) -> Dict[str, Any]:
    """The 200-draw headline rerun, as a subprocess with its own suffix."""
    cmd = [sys.executable, str(root / "scripts" / "w6_baselines.py"),
           "--root", str(root), "--backbone", backbone, "--spec", spec,
           "--split", split, "--suffix", "_d200", "--draws", str(int(draws)),
           "--jobs", str(int(jobs))]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=6 * 3600)
    except Exception as exc:                            # pragma: no cover
        return {"ran": False, "error": f"{type(exc).__name__}: {exc}", "cmd": cmd}
    return {"ran": out.returncode == 0, "returncode": out.returncode, "cmd": cmd,
            "tail": (out.stdout or "")[-400:]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W22 backlog")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--split", default=R.SPLIT_V3)
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--spec-draws", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--heavy", action="store_true",
                        help="also run the 200-draw headline rerun")
    parser.add_argument("--kcurve-only", action="store_true")
    parser.add_argument("--spec-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    data_dir = data_root(args.data)
    kdf = spec_df = pd.DataFrame()
    if not args.spec_only:
        print("backlog item 2: more k points", flush=True)
        kdf = kcurve(root, data_dir, args.backbone, args.spec, args.split,
                     args.draws, args.jobs)
        kdf.to_csv(root / "results" / "w22_kcurve.csv", index=False)
        if not kdf.empty:
            print(kdf.pivot_table(index=["method", "k"], columns="mask",
                                  values="pooled_eer").round(3).to_string())
    if not args.kcurve_only:
        print("\nbacklog item 4: spec / layer sensitivity", flush=True)
        spec_df = spec_sensitivity(root, data_dir, args.backbone, args.split,
                                   SPECS, args.spec_draws, args.jobs)
        spec_df.to_csv(root / "results" / "w22_spec_sensitivity.csv", index=False)
        if not spec_df.empty:
            print(spec_df.pivot_table(index=["spec", "method"], columns="mask",
                                      values="pooled_eer").round(3).to_string())
    table = backlog_table(kdf, spec_df, root)
    if args.heavy:
        result = delegate_heavy(root, args.backbone, args.spec, args.split)
        table.loc[table["item"].str.startswith("1."), "status"] = (
            "DONE" if result.get("ran") else "FAILED")
        table.loc[table["item"].str.startswith("1."), "note"] = json.dumps(
            {k: v for k, v in result.items() if k != "cmd"})[:300]
    table.to_csv(root / "results" / "w22_backlog.csv", index=False)
    print("\n" + table[["item", "status", "note"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""A-OC variants and sub-metrics (NEW_TASKS 07 section 1).

Explains *why* the one-class arm is strong, on two settings:

* Track E0 (no labels at all, only enrolment) - the U12R setting
* Track U8 (the frozen U8 recipe: 290 labels plus enrolment, strict_draw)

Variants: ``o1`` 1-NN cosine distance, ``o3`` mean of the three smallest,
``oc`` distance to the enrolment centroid, ``om`` Mahalanobis distance to that
centroid with a Ledoit-Wolf pooled covariance (whitened once, then Euclidean).

Sub-metrics: language-match vs language-mismatch subsets, and the Line A
platform x language cell macro average.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from src import regions as R  # noqa: E402
from src.data import data_root  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.newtasks05 import (  # noqa: E402
    enrol_mask, l2_normalise, one_thread_blas, worker_slots, write_csv_atomic,
    zero_scores,
)

M_LIST = (3, 5, 10)
N_DRAWS = 50
MIN_CELL = 30
_G: Dict[str, Any] = {}


def compute_variants(Xn: np.ndarray, Xw: np.ndarray, sp: np.ndarray,
                     enrol: np.ndarray) -> Dict[str, np.ndarray]:
    """o1 / o3 / oc / om for every clip, one enrolment set."""
    n = Xn.shape[0]
    out = {k: np.full(n, np.nan) for k in ("o1", "o3", "oc", "om")}
    glob = np.where(enrol)[0]
    for s in np.unique(sp):
        idx = np.where(sp == s)[0]
        r = idx[enrol[idx]]
        if r.size == 0:
            r = glob
        if r.size == 0:
            continue
        d = 1.0 - Xn[idx] @ Xn[r].T                      # cosine distance, (n_s, m)
        out["o1"][idx] = d.min(axis=1)
        k = min(3, d.shape[1])
        out["o3"][idx] = np.sort(d, axis=1)[:, :k].mean(axis=1)
        c = Xn[r].mean(axis=0)
        c = c / max(float(np.linalg.norm(c)), 1e-12)
        out["oc"][idx] = 1.0 - Xn[idx] @ c
        cm = Xw[r].mean(axis=0)
        out["om"][idx] = np.linalg.norm(Xw[idx] - cm, axis=1)
    for k, v in out.items():
        if np.isnan(v).any():
            fill = float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0
            out[k] = np.where(np.isnan(v), fill, v)
    return out


def _cell_macro(y, s, cell, min_per_class: int = MIN_CELL) -> float:
    vals = []
    for c in np.unique(cell):
        m = cell == c
        if min(int((y[m] == 1).sum()), int((y[m] == 0).sum())) < min_per_class:
            continue
        vals.append(summarize(y[m], s[m], np.full(int(m.sum()), "c"))["pooled_eer"])
    return float(np.mean(vals)) if vals else float("nan")


def run_unit(payload):
    with one_thread_blas():
        return _run_unit(payload)


def _run_unit(payload):
    m, d = payload
    y, sp, pl = _G["y"], _G["sp"], _G["pl"]
    cell, lang = _G["cell"], _G["lang"]
    Xn, Xw, s0 = _G["Xn"], _G["Xw"], _G["zero"]
    en = enrol_mask(y, sp, int(m), int(d))
    ev = ~en
    sc = compute_variants(Xn, Xw, sp, en)
    sc["Z0"] = s0
    rows: List[Dict[str, Any]] = []

    # speaker's majority enrolment language, for the match / mismatch split
    maj = {}
    for k in np.unique(sp):
        r = np.where((sp == k) & en)[0]
        if r.size:
            vals, cnt = np.unique(lang[r], return_counts=True)
            maj[k] = vals[int(np.argmax(cnt))]
    match = np.array([maj.get(k, None) == lang[i] for i, k in enumerate(sp)], bool)

    for name, v in sc.items():
        rows.append(summarize(y[ev], v[ev], pl[ev], extra={
            "protocol": "a_oc_variants", "region": "dev", "setting": "E0",
            "method": name, "m": int(m), "draw": int(d), "mask": "E0_all",
            "n_labeled": 0, "n_eval": int(ev.sum()), "n_enroll": int(en.sum()),
            "cell_macro": _cell_macro(y[ev], v[ev], cell[ev])}))
        for tag, sel in (("lang_match", ev & match), ("lang_mismatch", ev & ~match)):
            if sel.sum() > 50 and len(np.unique(y[sel])) == 2:
                rows.append(summarize(y[sel], v[sel], pl[sel], extra={
                    "protocol": "a_oc_variants", "region": "dev", "setting": "E0",
                    "method": name, "m": int(m), "draw": int(d), "mask": tag,
                    "n_labeled": 0, "n_eval": int(sel.sum())}))
    return rows


def _stream(fn, payloads, jobs):
    import multiprocessing as mp

    if jobs > 1:
        ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        with ctx.Pool(int(jobs)) as pool:
            yield from pool.imap_unordered(fn, payloads, chunksize=1)
    else:
        for p in payloads:
            yield fn(p)


def main(argv=None):
    ap = argparse.ArgumentParser(description="X3 A-OC variants")
    ap.add_argument("--root", default=".")
    ap.add_argument("--data")
    ap.add_argument("--backbone", default="xls-r-2b")
    ap.add_argument("--spec", default="S3")
    ap.add_argument("--split", default=R.SPLIT_V3)
    ap.add_argument("--ms", default=",".join(map(str, M_LIST)))
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--jobs", type=int, default=worker_slots())
    ap.add_argument("--out", default="results/a_oc_variants.csv")
    a = ap.parse_args(argv)

    root = Path(a.root)
    data = data_root(a.data)
    sub, X = R.load_region(root, data, a.backbone, a.spec, a.split)
    Xn = l2_normalise(X)
    print("whitening for the Mahalanobis variant (one Ledoit-Wolf fit) ...", flush=True)
    res = X.copy()
    for k in np.unique(sub["speaker"].to_numpy()):
        idx = np.where(sub["speaker"].to_numpy() == k)[0]
        res[idx] -= res[idx].mean(axis=0)
    lw = LedoitWolf(assume_centered=True).fit(res)
    w, V = np.linalg.eigh(lw.covariance_)
    Xw = (res @ V) / np.sqrt(np.maximum(w, 1e-8))
    print(f"  whitened to {Xw.shape}", flush=True)

    _G.update({"y": sub["y"].to_numpy(int), "sp": sub["speaker"].to_numpy(),
               "pl": sub["platform"].to_numpy(), "cell": sub["cell"].to_numpy(),
               "lang": (sub["language"].to_numpy() if "language" in sub.columns
                        else np.array([""] * len(sub))),
               "zero": zero_scores(data, a.backbone, sub["clip_id"].to_numpy()),
               "Xn": Xn, "Xw": Xw})
    ms = [int(x) for x in str(a.ms).split(",") if x.strip()]
    payloads = [(m, d) for m in ms for d in range(int(a.draws))]
    rows: List[Dict[str, Any]] = []
    for i, out in enumerate(_stream(run_unit, payloads, a.jobs), 1):
        rows.extend(out)
        if i % 25 == 0:
            print(f"  {i}/{len(payloads)} units ({len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    write_csv_atomic(df, root / a.out)
    print(df[df["mask"] == "E0_all"]
          .groupby(["m", "method"])["pooled_eer"].mean().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

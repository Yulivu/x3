#!/usr/bin/env python3
"""W2-lite 2.1: pseudo-enrolment (NEW_TASKS 07 section 2.1).

No labels and no real enrolment: each speaker's ``m'`` most genuine-looking
clips under the base score become a *pseudo* enrolment set, and the one-class
score is computed against them exactly as U12R does with real enrolment.

Controls in the same run: randomly chosen pseudo sets, and a variant that first
drops the 20 % most spoof-looking clips of the pseudo set (Z0 filter) before
taking ``o3``.  The real-enrolment number is U12R's and is quoted in the report,
not recomputed here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from src import regions as R  # noqa: E402
from src.data import data_root  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.newtasks05 import (  # noqa: E402
    l2_normalise, one_thread_blas, worker_slots, write_csv_atomic, zero_scores,
)

M_LIST = (5, 10)
N_DRAWS = 20
_G: Dict[str, Any] = {}


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / max(float(x.std()), 1e-9)


def _oc_from(Xn: np.ndarray, sp: np.ndarray, ref: np.ndarray, mode: str) -> np.ndarray:
    n = Xn.shape[0]
    out = np.full(n, np.nan)
    for s in np.unique(sp):
        idx = np.where(sp == s)[0]
        r = idx[ref[idx]]              # global row ids of this speaker's pseudo set
        if r.size == 0:
            continue
        d = 1.0 - Xn[idx] @ Xn[r].T
        if mode == "o3z":
            out[idx] = np.sort(d, axis=1)[:, : min(3, d.shape[1])].mean(axis=1)
        else:
            out[idx] = d.min(axis=1)
    if np.isnan(out).any():
        fill = float(np.nanmedian(out)) if np.isfinite(out).any() else 0.0
        out = np.where(np.isnan(out), fill, out)
    return out


def run_unit(draw):
    with one_thread_blas():
        return _run_unit(draw)


def _run_unit(draw):
    y, sp, pl = _G["y"], _G["sp"], _G["pl"]
    lang = _G["lang"]
    Xn, s0 = _G["Xn"], _G["zero"]
    rng = np.random.default_rng(20261015 + int(draw))
    rows: List[Dict[str, Any]] = []
    for m in _G["ms"]:
        pseudo = np.zeros(len(y), bool)
        rnd = np.zeros(len(y), bool)
        for s in np.unique(sp):
            idx = np.where(sp == s)[0]
            k = min(int(m), max(1, idx.size // 2))
            order = np.argsort(s0[idx])
            pseudo[idx[order[:k]]] = True
            rnd[idx[rng.choice(idx.size, size=min(k, idx.size), replace=False)]] = True
        # Z0 filter: drop the 20 % most spoof-looking clips of the pseudo set
        pf = pseudo.copy()
        for s in np.unique(sp):
            own = np.where(pseudo & (sp == s))[0]
            if own.size >= 3:
                drop = own[np.argsort(s0[own])[-max(1, int(0.2 * own.size)):]]
                pf[drop] = False

        o_p = _oc_from(Xn, sp, pseudo, "o1")
        o_r = _oc_from(Xn, sp, rnd, "o1")
        o_z = _oc_from(Xn, sp, pf, "o3z")
        variants = {
            "Z0": s0,
            "pseudo-o1": _z(s0) + _z(o_p),
            "random-o1": _z(s0) + _z(o_r),
            "z0filter-o3": _z(s0) + _z(o_z),
        }
        maj = {}
        for s in np.unique(sp):
            own = np.where(pseudo & (sp == s))[0]
            if own.size:
                v, c = np.unique(lang[own], return_counts=True)
                maj[s] = v[int(np.argmax(c))]
        match = np.array([maj.get(s, None) == lang[i] for i, s in enumerate(sp)], bool)
        for name, v in variants.items():
            rows.append(summarize(y, v, pl, extra={
                "protocol": "w2lite_pseudo", "region": "dev", "setting": "TrackZ",
                "method": name, "m": int(m), "draw": int(draw), "mask": "all",
                "n_labeled": 0, "n_eval": int(len(y)),
                "n_pseudo": int(pseudo.sum()),
                "backbone": _G["backbone"], "spec": _G["spec"]}))
            for tag, sel in (("lang_match", match), ("lang_mismatch", ~match)):
                if sel.sum() > 50 and len(np.unique(y[sel])) == 2:
                    rows.append(summarize(y[sel], v[sel], pl[sel], extra={
                        "protocol": "w2lite_pseudo", "region": "dev", "setting": "TrackZ",
                        "method": name, "m": int(m), "draw": int(draw), "mask": tag,
                        "n_labeled": 0, "n_eval": int(sel.sum()),
                        "backbone": _G["backbone"], "spec": _G["spec"]}))
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
    ap = argparse.ArgumentParser(description="X3 W2-lite 2.1 pseudo-enrolment")
    ap.add_argument("--root", default=".")
    ap.add_argument("--data")
    ap.add_argument("--backbone", default="xls-r-2b")
    ap.add_argument("--spec", default="S3")
    ap.add_argument("--split", default=R.SPLIT_V3)
    ap.add_argument("--ms", default=",".join(map(str, M_LIST)))
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--jobs", type=int, default=worker_slots())
    ap.add_argument("--out", default="results/w2lite_pseudo.csv")
    a = ap.parse_args(argv)

    root = Path(a.root)
    data = data_root(a.data)
    sub, X = R.load_region(root, data, a.backbone, a.spec, a.split)
    _G.update({"y": sub["y"].to_numpy(int), "sp": sub["speaker"].to_numpy(),
               "pl": sub["platform"].to_numpy(),
               "lang": (sub["language"].to_numpy() if "language" in sub.columns
                        else np.array([""] * len(sub))),
               "zero": zero_scores(data, a.backbone, sub["clip_id"].to_numpy()),
               "Xn": l2_normalise(X), "ms": [int(x) for x in str(a.ms).split(",") if x],
               "backbone": a.backbone, "spec": a.spec})
    rows: List[Dict[str, Any]] = []
    for out in _stream(run_unit, list(range(int(a.draws))), a.jobs):
        rows.extend(out)
    df = pd.DataFrame(rows)
    write_csv_atomic(df, root / a.out)
    print(df[df["mask"] == "all"].groupby(["m", "method"])["pooled_eer"]
          .mean().round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

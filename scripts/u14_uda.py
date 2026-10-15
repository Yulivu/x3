#!/usr/bin/env python3
"""U14 Track-U: source-speaker labelled / target-speaker unlabelled adaptation.

Every arm is trained with the same label budget (all source labels) on the same
PCA-256 block, and only SGM-LR additionally models the speaker shift.  The
baselines are real implementations, not aliases:

* ``B1-full``  source LR (the supervised upper bound, trained on source speakers)
* ``CORAL``    whiten on the source, recolor with the target covariance, retrain
* ``SWS``      per-speaker standardisation, then the source LR
* ``EntMin``   source LR, then Tent-style bias/scale entropy minimisation on target
* ``PL``       source LR, then 3 rounds of global-threshold pseudo-labelling
* ``SGM-LR``   the speaker-conditional mixture of NEW_TASKS 05 section 2.1
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse
import contextlib
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from src import regions as R  # noqa: E402
from src.data import data_root  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.newtasks05 import speaker_split, worker_slots, write_csv_atomic, zero_scores  # noqa: E402
from src.sgm import fit_sgm_lr, prepare_features  # noqa: E402

C_GRID = (0.01, 0.1, 1.0, 10.0)
_G: Dict[str, Any] = {}

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None


@contextlib.contextmanager
def one_thread_blas():
    if threadpool_limits is None:  # pragma: no cover
        yield
        return
    with threadpool_limits(limits=1):
        yield


def _pick_C(Zs, ys, groups):
    """≤4 configs, grouped by speaker on the source domain only."""
    n_splits = min(3, len(np.unique(groups)))
    if n_splits < 2:
        return 1.0
    best, best_ll = 1.0, -np.inf
    for C in C_GRID:
        ll = 0.0
        for tr, te in GroupKFold(n_splits=n_splits).split(Zs, ys, groups):
            m = LogisticRegression(C=C, max_iter=3000).fit(Zs[tr], ys[tr])
            p = np.clip(m.predict_proba(Zs[te])[:, 1], 1e-9, 1 - 1e-9)
            ll += float(np.sum(ys[te] * np.log(p) + (1 - ys[te]) * np.log(1 - p)))
        if ll > best_ll:
            best, best_ll = C, ll
    return best


def _entropy_min(logits_t, steps=200, lr=0.05):
    """Tent-lite: only a scale and a bias move, fitted on the target logits."""
    a, b = 1.0, 0.0
    for _ in range(int(steps)):
        z = a * logits_t + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        # d/dz mean entropy
        g = (1.0 + np.log(np.clip(p, 1e-9, 1))) * (1 - p) \
            - np.log(np.clip(1 - p, 1e-9, 1)) * p
        ga = float(np.mean(z * g))
        gb = float(np.mean(g))
        a -= lr * ga
        b -= lr * gb
        a = float(np.clip(a, 0.1, 10.0))
    return a, b


def _speaker_standardise(Z, groups):
    out = np.empty_like(Z)
    for g in np.unique(groups):
        m = groups == g
        mu = Z[m].mean(0)
        sd = Z[m].std(0) + 1e-6
        out[m] = (Z[m] - mu) / sd
    return out


def run_unit(draw):
    with one_thread_blas():
        return _run_unit(draw)


def _run_unit(draw):
    y, sp, pl = _G["y"], _G["speaker"], _G["platform"]
    Z, s = _G["Z256"], _G["zero"]
    src, tgt = speaker_split(sp, draw, 0.6)
    Zs, Zt = Z[src], Z[tgt]
    ys, yt = y[src], y[tgt]
    rows: List[Dict[str, Any]] = []

    def add(method, score):
        rows.append(summarize(yt, np.asarray(score, float), pl[tgt], extra={
            "protocol": "u14_track_u_v3", "draw": int(draw), "track": "U",
            "method": method, "k": int(src.sum()), "n_labeled": int(src.sum()),
            "mask": "target_speakers", "backbone": _G["backbone"],
            "spec": _G["spec"], "n_eval": int(tgt.sum()),
        }))

    C = _pick_C(Zs, ys, sp[src])
    base = LogisticRegression(C=C, max_iter=3000).fit(Zs, ys)
    add("B1-full", base.decision_function(Zt))

    # CORAL: whiten on the source, recolor with the target covariance.
    def inv_sqrt(m):
        w, v = np.linalg.eigh(m)
        return v @ np.diag(1.0 / np.sqrt(np.clip(w, 1e-8, None))) @ v.T

    Cs = np.cov(Zs, rowvar=False) + 1e-3 * np.eye(Zs.shape[1])
    Ct = np.cov(Zt, rowvar=False) + 1e-3 * np.eye(Zt.shape[1])
    ms, mt = Zs.mean(0), Zt.mean(0)
    Wh = inv_sqrt(Cs)
    Zs_c = (Zs - ms) @ Wh
    Zt_c = (Zt - mt) @ Wh
    coral = LogisticRegression(C=C, max_iter=3000).fit(Zs_c, ys)
    add("CORAL", coral.decision_function(Zt_c))

    # SWS: per-speaker standardisation of both domains, then the source LR.
    Zs_s = _speaker_standardise(Zs, sp[src])
    Zt_s = _speaker_standardise(Zt, sp[tgt])
    sws = LogisticRegression(C=C, max_iter=3000).fit(Zs_s, ys)
    add("SWS", sws.decision_function(Zt_s))

    # EntMin: source LR, then entropy minimisation on the target.
    a, b = _entropy_min(base.decision_function(Zt))
    add("EntMin", a * base.decision_function(Zt) + b)

    # PL: naive global-threshold pseudo-labelling, 3 rounds.
    pl_model = base
    for _ in range(3):
        p = pl_model.predict_proba(Zt)[:, 1]
        conf = (p >= 0.9) | (p <= 0.1)
        if conf.sum() < 20:
            break
        Ztr = np.vstack([Zs, Zt[conf]])
        ytr = np.concatenate([ys, (p[conf] >= 0.5).astype(int)])
        w = np.concatenate([np.ones(len(ys)), np.full(int(conf.sum()), 0.5)])
        pl_model = LogisticRegression(C=C, max_iter=3000).fit(Ztr, ytr, sample_weight=w)
    add("PL", pl_model.decision_function(Zt))

    lab = np.full(len(y), -1, int)
    lab[src] = y[src]
    sgm = fit_sgm_lr(Z, s, sp, pl, lab, prepared=Z, pca_dim=Z.shape[1])
    add("SGM-LR", sgm.score[tgt])
    return rows


def _stream(fn, payloads, jobs):
    if jobs > 1:
        ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        with ctx.Pool(int(jobs)) as pool:
            yield from pool.imap_unordered(fn, payloads, chunksize=1)
    else:
        for p in payloads:
            yield fn(p)


def main(argv=None):
    p = argparse.ArgumentParser(description="X3 U14 Track U")
    p.add_argument("--root", default=".")
    p.add_argument("--data")
    p.add_argument("--backbone", default="xls-r-2b")
    p.add_argument("--spec", default="S3")
    p.add_argument("--split", default=R.SPLIT_V3)
    p.add_argument("--draws", type=int, default=100)
    p.add_argument("--jobs", type=int, default=worker_slots())
    p.add_argument("--out", default="results/u14_uda.csv")
    a = p.parse_args(argv)

    root = Path(a.root)
    data = data_root(a.data)
    sub, X = R.load_region(root, data, a.backbone, a.spec, a.split)
    _G.update({"y": sub["y"].to_numpy(int), "speaker": sub["speaker"].to_numpy(),
               "platform": sub["platform"].to_numpy(),
               "zero": zero_scores(data, a.backbone, sub["clip_id"].to_numpy()),
               "backbone": a.backbone, "spec": a.spec})
    _G["Z256"] = prepare_features(X, 256)
    print(f"U14 {a.backbone}|{a.spec}: clips={len(sub)} "
          f"PCA={_G['Z256'].shape[1]} draws={a.draws} jobs={a.jobs}", flush=True)

    rows: List[Dict[str, Any]] = []
    payloads = list(range(int(a.draws)))
    for i, out in enumerate(_stream(run_unit, payloads, a.jobs), 1):
        rows.extend(out)
        if i % 10 == 0:
            print(f"  {i}/{len(payloads)} draws ({len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    write_csv_atomic(df, root / a.out)
    if not df.empty:
        print(df.groupby("method")["pooled_eer"].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""U13: SGM on the Z, E0, F, and U tracks (NEW_TASKS 05 section 2.1).

The (track, draw) units are independent and every unit seeds itself from its own
draw index, so the rows are identical whether the driver runs with ``--jobs 1``
or ``--jobs 64``.
"""
from __future__ import annotations

import argparse
import contextlib
import multiprocessing as mp
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
from src.newtasks05 import enrol_mask, speaker_split, zero_scores, write_csv_atomic  # noqa: E402
from src.sgm import fit_1d, fit_sgm_lr, prepare_features  # noqa: E402

_G: Dict[str, Any] = {}

try:  # every pool worker must stay single-threaded or 24 workers x N BLAS
    from threadpoolctl import threadpool_limits  # threads thrash the box
except Exception:  # pragma: no cover
    threadpool_limits = None


@contextlib.contextmanager
def one_thread_blas():
    if threadpool_limits is None:  # pragma: no cover
        yield
        return
    with threadpool_limits(limits=1):
        yield


def _row(score, mask, method, track, draw, nlab, extra=None):
    return summarize(
        _G["y"][mask], np.asarray(score)[mask], _G["platform"][mask],
        extra={"protocol": "u13_sgm_v3", "region": "dev", "track": track,
               "draw": int(draw), "backbone": _G["backbone"], "spec": _G["spec"],
               "method": method, "mask": track, "k": int(nlab),
               "n_labeled": int(nlab), **(extra or {})},
    )


def run_unit(payload):
    with one_thread_blas():
        return _run_unit(payload)


def _run_unit(payload):
    track, draw = payload
    y, sp, pl, cell = _G["y"], _G["speaker"], _G["platform"], _G["cell"]
    s, X, Z = _G["zero"], _G["X"], _G["Z64"]
    rows: List[Dict[str, Any]] = []

    if track == "Z":
        f1 = fit_1d(s, sp, pl)
        rows.append(_row(f1.score, np.ones(len(y), bool), "SGM-1D", "Z", 0, 0,
                         {"polarity_flip": bool(f1.polarity_flip)}))
        fl = fit_sgm_lr(X, s, sp, pl, rounds=4, prepared=Z)
        rows.append(_row(fl.score, np.ones(len(y), bool), "SGM-LR", "Z", 0, 0,
                         {"polarity_flip": bool(fl.polarity_flip)}))
        return rows

    left, right = speaker_split(sp, draw, 0.6)
    if track == "E0":
        en = enrol_mask(y, sp, 5, draw)
        lab = np.full(len(y), -1, int)
        lab[en] = 0
        fit = fit_sgm_lr(X, s, sp, pl, lab, prepared=Z)
        rows.append(_row(fit.score, ~en, "SGM-E0", "E0", draw, 0,
                         {"n_enroll": int(en.sum()),
                          "polarity_flip": bool(fit.polarity_flip)}))
    elif track == "F":
        rng = np.random.default_rng(20261015 + int(draw))
        lab = np.full(len(y), -1, int)
        for c in np.unique(cell):
            cand = np.where(left & (cell == c))[0]
            take = min(10, cand.size // 2)
            if take > 0:
                pick = rng.choice(cand, take, replace=False)
                lab[pick] = y[pick]
        fit = fit_sgm_lr(X, s, sp, pl, lab, prepared=Z)
        rows.append(_row(fit.score, right, "SGM-LR", "F_strict", draw,
                         int((lab >= 0).sum()),
                         {"polarity_flip": bool(fit.polarity_flip)}))
    elif track == "U":
        lab = np.full(len(y), -1, int)
        lab[left] = y[left]
        fit = fit_sgm_lr(X, s, sp, pl, lab, prepared=Z)
        rows.append(_row(fit.score, right, "SGM-LR", "U", draw, int(left.sum()),
                         {"polarity_flip": bool(fit.polarity_flip)}))
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
    p = argparse.ArgumentParser(description="X3 U13 SGM")
    p.add_argument("--root", default=".")
    p.add_argument("--data")
    p.add_argument("--backbone", default="xls-r-2b")
    p.add_argument("--spec", default="S3")
    p.add_argument("--split", default=R.SPLIT_V3)
    p.add_argument("--draws", type=int, default=100)
    p.add_argument("--tracks", default="Z,E0,F,U")
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--out", default="results/u13_sgm_draws.csv")
    p.add_argument("--summary", default=None)
    a = p.parse_args(argv)

    root = Path(a.root)
    data = data_root(a.data)
    sub, X = R.load_region(root, data, a.backbone, a.spec, a.split)
    tracks = [t.strip().upper() for t in a.tracks.split(",") if t.strip()]
    _G.update({
        "y": sub["y"].to_numpy(int),
        "speaker": sub["speaker"].to_numpy(),
        "platform": sub["platform"].to_numpy(),
        "cell": sub["cell"].to_numpy(),
        "zero": zero_scores(data, a.backbone, sub["clip_id"].to_numpy()),
        "X": X,
        "backbone": a.backbone,
        "spec": a.spec,
    })
    _G["Z64"] = prepare_features(X, 64)
    print(f"{a.backbone}|{a.spec}: clips={len(sub)} "
          f"speakers={len(np.unique(_G['speaker']))} tracks={tracks} jobs={a.jobs}",
          flush=True)

    payloads: List[Any] = []
    for t in tracks:
        if t == "Z":
            payloads.append(("Z", 0))
        else:
            payloads.extend((t, d) for d in range(int(a.draws)))

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for i, out in enumerate(_stream(run_unit, payloads, a.jobs), 1):
        for r in out:
            if isinstance(r, dict) and "error" in r:
                errors.append(str(r["error"]))
            else:
                rows.append(r)
        if i % 25 == 0:
            print(f"  {i}/{len(payloads)} units ({len(rows)} rows, "
                  f"{len(errors)} errors)", flush=True)

    if errors:
        (root / "results" / "u13_errors.txt").write_text(
            "\n".join(errors) + "\n", encoding="utf-8")
        print(f"errors: {len(errors)}; first={errors[:3]}", flush=True)

    df = pd.DataFrame(rows)
    out = root / a.out
    write_csv_atomic(df, out)
    if not df.empty:
        default = "results/u13_sgm_summary.csv" if a.out == "results/u13_sgm_draws.csv" \
            else str(out.with_name(out.stem + "_summary.csv").relative_to(root))
        summary = root / (a.summary or default)
        (df.groupby(["track", "method"], dropna=False)
           .mean(numeric_only=True).reset_index()
           .to_csv(summary, index=False))
        print(df.groupby(["track", "method"])["pooled_eer"].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

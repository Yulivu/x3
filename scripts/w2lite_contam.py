#!/usr/bin/env python3
"""W2-lite 2.2: enrolment-contamination robustness (NEW_TASKS 07 section 2.2).

Real "known-genuine" references harvested from the web will contain some
deepfakes, and a plain 1-NN one-class score is the most fragile arm under that
contamination.  Setting: Track E0 (no labels at all), v3 dev.

* contamination ``c`` in {0, 10 %, 30 %}: that fraction of each speaker's
  enrolment set is replaced by *fake* clips of the same speaker, which are then
  removed from E (labels are used only to build the contamination);
* variants: ``o1`` 1-NN, ``o2`` trimmed-1NN (drop the single smallest distance),
  ``o3`` mean of the three smallest, ``oc`` centroid, ``omed`` median distance,
  ``z0f-o3`` (drop the 20 % most spoof-looking enrolment clips, then ``o3``);
* the headline robust variant is chosen on the FIRST 50 draws at c=10 % and
  compared against ``o1`` on the LAST 50 draws at c=30 % (selection and
  validation are kept apart);
* gate **G30**: at m=5, c=30 %, headline beats ``o1`` by >= 3.0 pooled with the
  (b) speaker-clustered interval clear of 0, else ``INCONCLUSIVE``/FAIL.
"""
from __future__ import annotations

import argparse
import json
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
    enrol_mask, l2_normalise, one_thread_blas, worker_slots, write_csv_atomic,
    zero_scores,
)

C_LIST = (0.0, 0.10, 0.30)
M = 5
N_DRAWS = 100
SELECT_DRAWS = 50
G30_MARGIN = 3.0
VARIANTS = ("o1", "o2", "o3", "oc", "omed", "z0f-o3")
_G: Dict[str, Any] = {}


def _dist(Xn, idx, ref):
    return 1.0 - Xn[idx] @ Xn[ref].T


def run_unit(payload):
    with one_thread_blas():
        return _run_unit(payload)


def _run_unit(payload):
    draw, c = payload
    y, sp, pl = _G["y"], _G["sp"], _G["pl"]
    Xn, s0 = _G["Xn"], _G["zero"]
    rng = np.random.default_rng(20261015 + int(draw))
    en = enrol_mask(y, sp, M, int(draw))
    # contaminate: replace a fraction of each speaker's enrolment with fakes
    if c > 0:
        for s in np.unique(sp):
            own = np.where(en & (sp == s))[0]
            if own.size == 0:
                continue
            nrep = max(1, int(round(c * own.size)))
            fakes = np.where((sp == s) & (y == 1) & ~en)[0]
            if fakes.size < nrep:
                continue
            take = rng.choice(fakes, size=nrep, replace=False)
            drop = rng.choice(own, size=nrep, replace=False)
            en[drop] = False
            en[take] = True
    ev = ~en
    scores: Dict[str, np.ndarray] = {k: np.full(len(y), np.nan) for k in VARIANTS}
    for s in np.unique(sp):
        idx = np.where(sp == s)[0]
        r = idx[en[idx]]
        if r.size == 0:
            continue
        d = _dist(Xn, idx, r)
        order = np.sort(d, axis=1)
        scores["o1"][idx] = order[:, 0]
        scores["o2"][idx] = order[:, 1] if order.shape[1] > 1 else order[:, 0]
        k = min(3, order.shape[1])
        scores["o3"][idx] = order[:, :k].mean(axis=1)
        scores["omed"][idx] = np.median(d, axis=1)
        cen = Xn[r].mean(axis=0)
        cen = cen / max(float(np.linalg.norm(cen)), 1e-12)
        scores["oc"][idx] = 1.0 - Xn[idx] @ cen
        if r.size >= 5:                       # drop the 20 % most spoof-looking refs
            keep = r[np.argsort(s0[r])[: max(1, int(0.8 * r.size))]]
            d2 = np.sort(_dist(Xn, idx, keep), axis=1)
            scores["z0f-o3"][idx] = d2[:, : min(3, d2.shape[1])].mean(axis=1)
        else:
            scores["z0f-o3"][idx] = scores["o3"][idx]
    rows: List[Dict[str, Any]] = []
    for name, v in scores.items():
        v = np.where(np.isnan(v), float(np.nanmedian(v)), v)
        rows.append(summarize(y[ev], v[ev], pl[ev], extra={
            "protocol": "w2lite_contam", "region": "dev", "setting": "TrackE0",
            "method": name, "m": M, "c": float(c), "draw": int(draw),
            "mask": "E0_all", "n_labeled": 0, "n_eval": int(ev.sum()),
            "n_enroll": int(en.sum()), "backbone": _G["backbone"],
            "spec": _G["spec"]}))
    if int(draw) < SELECT_DRAWS or c == 0.30:
        np.savez_compressed(
            _G["root"] / "scores" / f"w2c_{int(draw):04d}_{int(c*100):02d}.npz",
            label=y[ev], speaker=sp[ev].astype(str), platform=pl[ev].astype(str),
            **{k: v[ev] for k, v in scores.items()})
    return rows


def _macro_or_pooled(y, s, plat):
    return summarize(y, s, plat)["pooled_eer"]


def _cluster_interval(pairs, B_s=200, seed=20261015):
    rng = np.random.default_rng(seed)
    groups = [np.unique(p["speaker"]) for p in pairs]
    boots = np.empty(B_s)
    for b in range(B_s):
        vals = np.empty(len(pairs))
        for i, p in enumerate(pairs):
            gs = groups[i]
            pick = rng.integers(0, len(gs), size=len(gs))
            idx = np.concatenate([np.where(p["speaker"] == gs[j])[0] for j in pick])
            vals[i] = (_macro_or_pooled(p["y"][idx], p["sa"][idx], p["plat"][idx])
                       - _macro_or_pooled(p["y"][idx], p["sb"][idx], p["plat"][idx]))
        boots[b] = vals.mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def gate_g30(root: Path, df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"criterion": f"m={M}, c=30 %: headline robust arm beats "
                                       f"o1 by >= {G30_MARGIN} pooled, (b) clear of 0"}
    sel = df[(df["c"] == 0.10) & (df["draw"] < SELECT_DRAWS)]
    means = sel.groupby("method")["pooled_eer"].mean()
    robust = [m for m in VARIANTS if m != "o1"]
    robust = [m for m in robust if m in means.index]
    if not robust:
        out["verdict"] = "ROBUST_REF_FAIL"
        return out
    headline = min(robust, key=lambda m: means[m])
    out["headline"] = headline
    out["selection_table"] = {m: float(means[m]) for m in means.index}
    val = df[(df["c"] == 0.30) & (df["draw"] >= SELECT_DRAWS)]
    base = val[val["method"] == "o1"].groupby("draw")["pooled_eer"].mean()
    hd = val[val["method"] == headline].groupby("draw")["pooled_eer"].mean()
    delta = float((hd - base).mean()) if len(hd) and len(base) else float("nan")
    out["headline_minus_o1_c30"] = delta
    pairs = []
    for d in sorted(set(hd.index) & set(base.index)):
        f = root / "scores" / f"w2c_{int(d):04d}_30.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            pairs.append({"y": z["label"], "speaker": z["speaker"].astype(str),
                          "plat": z["platform"].astype(str),
                          "sa": z[headline].astype(float), "sb": z["o1"].astype(float)})
    if len(pairs) >= 5:
        lo, hi = _cluster_interval(pairs)
        out.update({"b_lo": lo, "b_hi": hi, "n_draws": len(pairs),
                    "b_half_width": float((hi - lo) / 2)})
        if delta <= -G30_MARGIN and not (lo <= 0 <= hi):
            out["verdict"] = "ROBUST_REF_PASS"
        elif delta <= -G30_MARGIN:
            out["verdict"] = "INCONCLUSIVE"
        else:
            out["verdict"] = "ROBUST_REF_FAIL"
    else:
        out["verdict"] = "INCONCLUSIVE"
        out["note"] = "score archives missing"
    # cost at c=0
    z0 = df[(df["c"] == 0.0)]
    m0 = z0.groupby("method")["pooled_eer"].mean()
    out["c0_cost"] = float(m0.get(headline, float("nan")) - m0.get("o1", float("nan")))
    return out


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
    ap = argparse.ArgumentParser(description="X3 W2-lite 2.2 contamination robustness")
    ap.add_argument("--root", default=".")
    ap.add_argument("--data")
    ap.add_argument("--backbones", default="xls-r-2b|S3,mms-1b|S3")
    ap.add_argument("--split", default=R.SPLIT_V3)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--jobs", type=int, default=worker_slots())
    ap.add_argument("--out", default="results/w2lite_contam.csv")
    a = ap.parse_args(argv)

    root = Path(a.root)
    data = data_root(a.data)
    rows: List[Dict[str, Any]] = []
    gates: Dict[str, Any] = {}
    for spec_pair in a.backbones.split(","):
        bb, spec = spec_pair.split("|")
        sub, X = R.load_region(root, data, bb, spec, a.split)
        _G.update({"y": sub["y"].to_numpy(int), "sp": sub["speaker"].to_numpy(),
                   "pl": sub["platform"].to_numpy(),
                   "zero": zero_scores(data, bb, sub["clip_id"].to_numpy()),
                   "Xn": l2_normalise(X), "root": root, "backbone": bb, "spec": spec})
        payloads = [(d, c) for c in C_LIST for d in range(int(a.draws))]
        local: List[Dict[str, Any]] = []
        for out in _stream(run_unit, payloads, a.jobs):
            local.extend(out)
        rows.extend(local)
        gates[bb] = gate_g30(root, pd.DataFrame(local))
        print(f"{bb}: " + json.dumps(gates[bb], default=str)[:200], flush=True)
    df = pd.DataFrame(rows)
    write_csv_atomic(df, root / a.out)
    (root / "results" / "w2lite_contam_gate.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True, default=str), encoding="utf-8")
    piv = (df.groupby(["backbone", "method", "c"])[["pooled_eer", "within_macro_eer"]]
           .mean().reset_index())
    print(piv.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

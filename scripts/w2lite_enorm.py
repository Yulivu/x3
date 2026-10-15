#!/usr/bin/env python3
"""W2-lite 2.3-2.5 + G28: label-free improvements on top of OC.

Track E0 (no labels at all, only enrolment), v3 dev, two backbones.

* ``E-norm-s``  scale-normalise the 1-NN distance by the speaker's enrolment
  scale ``sigma_s`` (leave-one-out nearest-neighbour distance inside R_s),
  shrunk towards the global median with kappa=5
* ``multi-OC``  standardise and average the two backbones' ``E-norm-s`` scores
* ``LCM``       two-component diagonal-covariance GMM over the four views
  ``[Z0_xls, Z0_mms, E-norm-s_xls, E-norm-s_mms]``; score = log likelihood ratio,
  polarity fixed by the larger sum of component means
* control      ``E0-fuse`` (standardised Z0 + OC, equal weight)

Gate **G28**: the best of 2.3/2.4/2.5 beats ``max-strength(OC_xls, OC_mms)`` on
both pooled and platform-macro by >= 1.0 with the (b) speaker-clustered interval
clear of 0 -> ``E0_METHOD_PASS``; point estimate over the bar but interval
containing 0 -> ``E0_METHOD_INCONCLUSIVE``; otherwise ``E0_METHOD_FAIL``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from src import regions as R  # noqa: E402
from src.data import data_root  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.newtasks05 import (  # noqa: E402
    enrol_mask, l2_normalise, oc_scores, one_thread_blas, worker_slots,
    write_csv_atomic, zero_scores,
)

M_LIST = (3, 5, 10)
N_DRAWS = 100
KAPPA = 5.0
G28_MARGIN = 1.0
SAVE = 30
_G: Dict[str, Any] = {}


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / max(float(x.std()), 1e-9)


def _o1(Xn: np.ndarray, sp: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """The SAME implementation U12R uses; a local re-derivation disagreed by up
    to 0.062 on the score scale and moved the OC EER from 15.1 to 28.8, so the
    shared helper is the single source of truth."""
    return oc_scores(Xn, sp, ref)


def _sigma(Xn: np.ndarray, sp: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    """Leave-one-out nearest-neighbour distance inside each enrolment set."""
    sig: Dict[str, float] = {}
    for s in np.unique(sp):
        r = np.where(ref & (sp == s))[0]
        if r.size >= 2:
            d = 1.0 - Xn[r] @ Xn[r].T
            np.fill_diagonal(d, np.inf)
            sig[s] = float(np.median(d.min(axis=1)))
        else:
            sig[s] = np.nan
    vals = np.array([v for v in sig.values() if np.isfinite(v) and v > 0])
    glob = float(np.median(vals)) if vals.size else 1.0
    return {s: (glob if not np.isfinite(v) else v) for s, v in sig.items()}, glob


def _enorm(o1: np.ndarray, sp: np.ndarray, ref: np.ndarray, m: int) -> np.ndarray:
    sig, glob = _sigma(_G["Xn0"], sp, ref)
    out = np.empty_like(o1)
    for s in np.unique(sp):
        idx = np.where(sp == s)[0]
        own = int(ref[idx].sum())
        if own < 3:
            sd = glob
        else:
            sd = (own * sig[s] + KAPPA * glob) / (own + KAPPA)
        out[idx] = o1[idx] / max(sd, 1e-6)
    return out


def run_unit(payload):
    with one_thread_blas():
        return _run_unit(payload)


def _run_unit(payload):
    m, d = payload
    y, sp, pl = _G["y"], _G["sp"], _G["pl"]
    en = enrol_mask(y, sp, int(m), int(d))
    ev = ~en
    scores: Dict[str, np.ndarray] = {}
    for name in ("xls", "mms"):
        Xn, s0 = _G["Xn"][name], _G["zero"][name]
        sc = _o1(Xn, sp, en)
        scores[f"OC_{name}"] = sc
        _G["Xn0"] = Xn
        scores[f"Enorm_{name}"] = _enorm(sc, sp, en, int(m))
        scores[f"Z0_{name}"] = s0
    scores["multi-OC"] = _z(scores["Enorm_xls"]) + _z(scores["Enorm_mms"])
    scores["E0-fuse"] = _z(scores["Z0_xls"]) + _z(scores["OC_xls"])
    views = np.column_stack([_z(scores[f"Z0_{k}"]) for k in ("xls", "mms")]
                            + [_z(scores[f"Enorm_{k}"]) for k in ("xls", "mms")])
    try:
        gm = GaussianMixture(n_components=2, covariance_type="diag",
                             max_iter=100, random_state=20261015).fit(views)
        pol = 1 if gm.means_.sum(axis=1)[1] >= gm.means_.sum(axis=1)[0] else 0
        post = np.clip(gm.predict_proba(views), 1e-12, 1.0)
        scores["LCM"] = np.log(post[:, pol]) - np.log(post[:, 1 - pol])
    except Exception:                                   # pragma: no cover
        scores["LCM"] = scores["multi-OC"].copy()

    rows: List[Dict[str, Any]] = []
    for name, v in scores.items():
        rows.append(summarize(y[ev], v[ev], pl[ev], extra={
            "protocol": "w2lite_enorm", "region": "dev", "setting": "TrackE0",
            "method": name, "m": int(m), "draw": int(d), "mask": "E0_all",
            "n_labeled": 0, "n_eval": int(ev.sum()), "n_enroll": int(en.sum())}))
    if int(d) < SAVE:
        np.savez_compressed(
            _G["root"] / "scores" / f"w2e_{int(m)}_{int(d):04d}.npz",
            label=y[ev], speaker=sp[ev].astype(str), platform=pl[ev].astype(str),
            **{k.replace("+", "p"): v[ev] for k, v in scores.items()})
    return rows


def _cluster(pairs, B_s=200, seed=20261015):
    rng = np.random.default_rng(seed)
    groups = [np.unique(p["speaker"]) for p in pairs]
    boots = np.empty(B_s)
    for b in range(B_s):
        vals = np.empty(len(pairs))
        for i, p in enumerate(pairs):
            gs = groups[i]
            pick = rng.integers(0, len(gs), size=len(gs))
            idx = np.concatenate([np.where(p["speaker"] == gs[j])[0] for j in pick])
            vals[i] = (summarize(p["y"][idx], p["sa"][idx], p["plat"][idx])["pooled_eer"]
                       - summarize(p["y"][idx], p["sb"][idx], p["plat"][idx])["pooled_eer"])
        boots[b] = vals.mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def gate_g28(root: Path, df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"criterion": "best of E-norm-s / LCM / multi-OC beats "
                                       "max-strength(OC_xls, OC_mms) by >= 1.0 on pooled "
                                       "and macro, (b) clear of 0"}
    best: Dict[str, Any] = {}
    for m, g in df.groupby("m"):
        means = g.groupby("method")[["pooled_eer", "within_macro_eer"]].mean()
        if not {"OC_xls", "OC_mms"} <= set(means.index):
            continue
        ref_name = min(("OC_xls", "OC_mms"), key=lambda k: means.loc[k, "pooled_eer"])
        cands = [c for c in ("E-norm_xls", "E-norm_mms", "multi-OC", "LCM")
                 if c in means.index]
        if not cands:
            continue
        pick = min(cands, key=lambda c: means.loc[c, "pooled_eer"])
        entry = {"best_method": pick, "reference": ref_name,
                 "pooled_delta": float(means.loc[pick, "pooled_eer"]
                                       - means.loc[ref_name, "pooled_eer"]),
                 "macro_delta": float(means.loc[pick, "within_macro_eer"]
                                      - means.loc[ref_name, "within_macro_eer"])}
        pairs = []
        for d in range(SAVE):
            f = root / "scores" / f"w2e_{int(m)}_{d:04d}.npz"
            if not f.exists():
                continue
            with np.load(f) as z:
                ka = pick.replace("+", "p")
                kb = ref_name.replace("+", "p")
                if ka in z.files and kb in z.files:
                    pairs.append({"y": z["label"], "speaker": z["speaker"].astype(str),
                                  "plat": z["platform"].astype(str),
                                  "sa": z[ka].astype(float), "sb": z[kb].astype(float)})
        if len(pairs) >= 5:
            lo, hi = _cluster(pairs)
            entry.update({"b_lo": lo, "b_hi": hi, "b_half_width": (hi - lo) / 2,
                          "n_draws": len(pairs)})
            over = entry["pooled_delta"] <= -G28_MARGIN and entry["macro_delta"] <= 0
            clear = not (lo <= 0 <= hi)
            entry["verdict"] = ("E0_METHOD_PASS" if over and clear else
                                "E0_METHOD_INCONCLUSIVE" if over else "E0_METHOD_FAIL")
        else:
            entry["verdict"] = "E0_METHOD_INCONCLUSIVE"
        best[f"m={m}"] = entry
    out["per_m"] = best
    verdicts = [v.get("verdict") for v in best.values()]
    out["verdict"] = ("E0_METHOD_PASS" if "E0_METHOD_PASS" in verdicts else
                      "E0_METHOD_INCONCLUSIVE" if "E0_METHOD_INCONCLUSIVE" in verdicts
                      else "E0_METHOD_FAIL")
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
    ap = argparse.ArgumentParser(description="X3 W2-lite 2.3-2.5 + G28")
    ap.add_argument("--root", default=".")
    ap.add_argument("--data")
    ap.add_argument("--split", default=R.SPLIT_V3)
    ap.add_argument("--ms", default=",".join(map(str, M_LIST)))
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--jobs", type=int, default=worker_slots())
    ap.add_argument("--out", default="results/w2lite_enorm.csv")
    a = ap.parse_args(argv)

    root = Path(a.root)
    data = data_root(a.data)
    Xns, zeros = {}, {}
    for name, bb in (("xls", "xls-r-2b"), ("mms", "mms-1b")):
        sub, X = R.load_region(root, data, bb, "S3", a.split)
        Xns[name] = l2_normalise(X)
        zeros[name] = zero_scores(data, bb, sub["clip_id"].to_numpy())
        print(f"loaded {bb} {X.shape}", flush=True)
    _G.update({"y": sub["y"].to_numpy(int), "sp": sub["speaker"].to_numpy(),
               "pl": sub["platform"].to_numpy(), "Xn": Xns, "zero": zeros,
               "root": root})
    ms = [int(x) for x in str(a.ms).split(",") if x.strip()]
    payloads = [(m, d) for m in ms for d in range(int(a.draws))]
    rows: List[Dict[str, Any]] = []
    for i, out in enumerate(_stream(run_unit, payloads, a.jobs), 1):
        rows.extend(out)
        if i % 25 == 0:
            print(f"  {i}/{len(payloads)} units ({len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    write_csv_atomic(df, root / a.out)
    piv = df.groupby(["m", "method"])[["pooled_eer", "within_macro_eer"]].mean().round(2)
    print(piv.to_string())
    gate = gate_g28(root, df)
    (root / "results" / "w2lite_enorm_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(gate, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

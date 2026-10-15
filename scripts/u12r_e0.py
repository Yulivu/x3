#!/usr/bin/env python3
"""U12R: Track E0 re-run with the OC sign fixed (NEW_TASKS 06 section 1).

Only the enrolment-only track is recomputed; U13/U14 are deliberately NOT
re-run (NEW_TASKS 06 section 0).  Every score is "larger = more spoof", so the
one-class score is the POSITIVE minimum cosine distance to the speaker's
enrolment set.

Arms (all label-free except the enrolment, which is known-real audio):
  Z0       the backbone's zero-shot logit
  OC       +min cosine distance to R_s
  E0-norm  z-scored zero-shot logit inside each speaker's R_s
  E0-fuse  equally weighted standardised Z0 + OC (GLOBAL mean/std, per the 06 spec)

Enrolment never enters the evaluation set and never counts toward k.
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
from src.newtasks05 import (  # noqa: E402
    enrol_mask, enrol_norm, l2_normalise, oc_scores, one_thread_blas,
    worker_slots, write_csv_atomic, zero_scores,
)
from src.paired import two_intervals  # noqa: E402

M_LIST = (3, 5, 10)
BACKBONES = (("xls-r-2b", "S3"), ("mms-1b", "S3"))
N_DRAWS = 100
SAVE_DRAWS = 30
METHODS = ("Z0", "OC", "E0-norm", "E0-fuse")
G20_MARGIN = 2.0

_G: Dict[str, Any] = {}


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / max(float(x.std()), 1e-9)


def run_unit(payload):
    with one_thread_blas():
        return _run_unit(payload)


def _run_unit(payload):
    key, m, draw = payload
    y, sp, pl = _G["y"], _G["speaker"], _G["platform"]
    s, Xn, meta = _G["zero"][key], _G["Xn"][key], _G["meta"]
    en = enrol_mask(y, sp, int(m), int(draw))
    ev = ~en
    oc = oc_scores(Xn, sp, en)
    enr = enrol_norm(s, sp, en)
    fuse = _z(s) + _z(oc)          # equal weight, global standardisation
    scores = {"Z0": s, "OC": oc, "E0-norm": enr, "E0-fuse": fuse}

    rows: List[Dict[str, Any]] = []
    for name, sc in scores.items():
        rows.append(summarize(
            y[ev], sc[ev], pl[ev],
            extra={"protocol": "u12r_e0_v3", "region": "dev", "track": "E0",
                   "backbone": key[0], "spec": key[1], "method": name,
                   "m": int(m), "draw": int(draw), "k": 0, "mask": "enrol_holdout",
                   "n_labeled": 0, "n_enroll": int(en.sum()),
                   "n_eval": int(ev.sum())},
        ))

    if int(draw) < SAVE_DRAWS:
        out_dir = _G["root"] / "scores"
        out_dir.mkdir(parents=True, exist_ok=True)
        ids = meta["clip_id"].to_numpy()[ev]
        lang = (meta["language"].to_numpy()[ev]
                if "language" in meta.columns else np.array([""] * int(ev.sum())))
        for name, sc in scores.items():
            np.savez_compressed(
                out_dir / f"u12r_{name}_{int(m)}_{key[0]}_{int(draw):04d}.npz",
                clip_id=ids, score=sc[ev], label=y[ev], speaker=sp[ev].astype(str),
                platform=pl[ev].astype(str), language=np.asarray(lang).astype(str))
    return rows


def _stream(fn, payloads, jobs):
    if jobs > 1:
        ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        with ctx.Pool(int(jobs)) as pool:
            yield from pool.imap_unordered(fn, payloads, chunksize=1)
    else:
        for p in payloads:
            yield fn(p)


def load_pair(root: Path, backbone: str, m: int, a: str, b: str) -> List[Dict[str, np.ndarray]]:
    """The saved first-30-draw archives, shaped for ``two_intervals``."""
    out: List[Dict[str, np.ndarray]] = []
    for d in range(SAVE_DRAWS):
        fa = root / "scores" / f"u12r_{a}_{m}_{backbone}_{d:04d}.npz"
        fb = root / "scores" / f"u12r_{b}_{m}_{backbone}_{d:04d}.npz"
        if not (fa.exists() and fb.exists()):
            continue
        with np.load(fa) as za, np.load(fb) as zb:
            out.append({"y": za["label"], "speaker": za["speaker"],
                        "sa": za["score"], "sb": zb["score"]})
    return out


def gate_g20(root: Path, df: pd.DataFrame) -> Dict[str, Any]:
    """E0-fuse vs the strongest of {Z0, OC, E0-norm}, pooled, (b) interval."""
    out: Dict[str, Any] = {"criterion": "E0-fuse beats max(Z0, OC, E0-norm) by >= "
                                       f"{G20_MARGIN} pooled, (b) interval clear of 0"}
    per: Dict[str, Any] = {}
    for (bb, m), grp in df.groupby(["backbone", "m"]):
        means = grp.groupby("method")["pooled_eer"].mean()
        if "E0-fuse" not in means.index:
            continue
        base = min(("Z0", "OC", "E0-norm"), key=lambda k: means.get(k, np.inf))
        entry: Dict[str, Any] = {
            "E0-fuse": float(means["E0-fuse"]), base: float(means[base]),
            "delta": float(means["E0-fuse"] - means[base]), "reference": base,
        }
        per_draw = load_pair(root, bb, int(m), "E0-fuse", base)
        if len(per_draw) >= 5:
            iv = two_intervals(per_draw)
            entry.update({k: iv[k] for k in ("b_lo", "b_hi", "a_lo", "a_hi", "n_draws")})
            entry["b_excludes_zero"] = bool(iv["b_excludes_zero"])
            entry["passes"] = bool(entry["delta"] <= -G20_MARGIN and iv["b_excludes_zero"])
        else:
            entry["passes"] = None
            entry["note"] = "per-clip archives missing"
        per[f"{bb}|m={m}"] = entry
    m5 = [v for k, v in per.items() if k.endswith("m=5")]
    out["per_setting"] = per
    out["verdict"] = ("ENROLL_ONLY_PASS" if m5 and all(v.get("passes") for v in m5)
                      else "ENROLL_ONLY_FAIL")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="X3 U12R Track E0 (OC sign fixed)")
    ap.add_argument("--root", default=".")
    ap.add_argument("--data")
    ap.add_argument("--backbones", default=",".join(f"{b}|{s}" for b, s in BACKBONES))
    ap.add_argument("--split", default=R.SPLIT_V3)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--ms", default=",".join(map(str, M_LIST)))
    ap.add_argument("--jobs", type=int, default=worker_slots())
    ap.add_argument("--out", default="results/u12r_e0.csv")
    a = ap.parse_args(argv)

    root = Path(a.root)
    data = data_root(a.data)
    ms = [int(x) for x in str(a.ms).split(",") if x.strip()]
    pairs = [tuple(x.split("|")) for x in a.backbones.split(",") if x.strip()]

    zeros, norms, metas = {}, {}, None
    for bb, spec in pairs:
        sub, X = R.load_region(root, data, bb, spec, a.split)
        Xn = l2_normalise(X)
        key = (bb, spec)
        zeros[key] = zero_scores(data, bb, sub["clip_id"].to_numpy())
        norms[key] = Xn
        if metas is None:
            metas = sub
            _G.update({"y": sub["y"].to_numpy(int), "speaker": sub["speaker"].to_numpy(),
                       "platform": sub["platform"].to_numpy(), "meta": sub,
                       "root": root})
        print(f"loaded {bb}|{spec} {X.shape}", flush=True)
    _G["zero"], _G["Xn"] = zeros, norms

    payloads = [(key, m, d) for key in zeros for m in ms for d in range(int(a.draws))]
    rows: List[Dict[str, Any]] = []
    for i, out in enumerate(_stream(run_unit, payloads, a.jobs), 1):
        rows.extend(out)
        if i % 50 == 0:
            print(f"  {i}/{len(payloads)} units ({len(rows)} rows)", flush=True)

    df = pd.DataFrame(rows)
    write_csv_atomic(df, root / a.out)
    # NEW_TASKS 07 section 0.1: how many speakers lose enrolment capacity, and
    # what the genuine/spoof balance of E becomes once enrolment is removed.
    cov: List[Dict[str, Any]] = []
    y, sp = _G["y"], _G["speaker"]
    total_real = float((y == 0).sum())
    for (bb, spec) in zeros:
        for m in ms:
            per_spk = {s: int(((sp == s) & (y == 0)).sum()) for s in np.unique(sp)}
            usable = {s: min(int(m), n // 2) for s, n in per_spk.items()
                      if min(int(m), n // 2) > 0}
            degraded = [s for s, n in per_spk.items() if n < int(m)]
            enrolled = float(sum(usable.values()))
            slots = float(len(per_spk) * int(m))
            cov.append({"backbone": bb, "spec": spec, "m": int(m),
                        "speakers_total": int(len(per_spk)),
                        "speakers_degraded_n_real_lt_m": len(degraded),
                        "speakers_usable": len(usable),
                        "speakers_excluded": int(len(per_spk)) - len(usable),
                        "mean_effective_m": (enrolled / len(usable)) if usable else 0.0,
                        "capacity_fraction": (enrolled / slots) if slots else 0.0,
                        "E_real_fraction": ((total_real - enrolled)
                                            / (len(y) - enrolled))})
    pd.DataFrame(cov).to_csv(root / "results" / "u12r_coverage.csv", index=False)
    (df.groupby(["backbone", "m", "method"])
       [["pooled_eer", "within_macro_eer", "worst_platform_eer", "auc"]]
       .mean().reset_index().to_csv(root / "results" / "u12r_summary.csv", index=False))
    gate = gate_g20(root, df)
    import json
    (root / "results" / "u12r_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(df.groupby(["backbone", "m", "method"])["pooled_eer"].mean().round(2).to_string())
    print()
    print(json.dumps(gate, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""U8 [URGENT] - ESN: speaker normalisation from enrolled genuine recordings.

Setting (NEW_TASKS 02 U8): v3 dev, ``strict_draw`` sampling, plus the same
``seen`` / ``unseen`` pair U1' uses so the m-curve can be read on both sides.
Every speaker - in ``S_L`` and in ``S_E`` - contributes ``m`` bona fide clips as its
enrollment set ``R_s`` (``min(m, floor(n_real/2))``), and those clips enter neither
``L`` nor ``E``.

Methods: ``B1`` (no enrollment), ``B1+enroll`` (the enrollment clips as extra
genuine labels), ``OC`` (nearest-neighbour cosine distance to ``R_s``), ``OC+B1``
(logit fusion, weight chosen in ``L``), ``ESN`` (``z~ = [z, z - mu_s]`` + LR) and
``ESN-P`` (per-platform ``mu`` when the enrollment has >= 2 clips of that platform).
``ESN`` at ``m=0`` must be identical to ``B1``, which the unit tests pin.

Every comparison carries the V-1 intervals; G17 (N5) is ESN-LR at ``m=5`` versus
``max(B1, B1+enroll, OC+B1)`` on ``strict_draw``: >= 2.0 pooled points better with
the speaker-clustered (b) interval clear of 0.

Writes ``results/u8_esn.csv``, ``results/u8_draws/``, ``results/u8_intervals.csv``
and ``results/u8_gate.json``.
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
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import u1_speaker_split as u1  # noqa: E402
import w6_baselines as w6  # noqa: E402
from src import regions as R  # noqa: E402
from src.baselines import fit_logreg_cv  # noqa: E402
from src.cv import select_config, speaker_grouped_folds, stratified_folds  # noqa: E402
from src.data import data_root  # noqa: E402
from src.draws import draw_labels, draw_seed  # noqa: E402
from src.esn import (  # noqa: E402
    choose_enrollment,
    esn_features,
    fuse,
    one_class_scores,
    speaker_means,
    standardise,
)
from src.metrics3 import eer, summarize  # noqa: E402
from src.paired import load_draw_scores, save_draw_scores, two_intervals  # noqa: E402

N_DRAWS = 100
M_LIST = (0, 3, 5, 10)
M_CELL = 10                                # k=290, the U1'/W6 budget
METHODS = ("B1", "B1+enroll", "OC", "OC+B1", "ESN", "ESN-P")
CONTROLS = ("B1", "B1+enroll", "OC+B1")
MASKS = ("seen", "unseen", "strict_draw")
FUSE_WEIGHTS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
STRICT_SEED = 777000
ENROLL_SEED = 20261015
GATE_M = 5

_G: Dict[str, Any] = {}

__all__ = ["esn_masks", "esn_labels", "fit_all", "run_one", "intervals_table",
           "gate_g17", "main"]


def esn_masks(speaker: np.ndarray, lab_pos: np.ndarray, enroll: np.ndarray
              ) -> Dict[str, np.ndarray]:
    """``seen`` / ``unseen`` over the clips that are neither labelled nor enrolled."""
    lab = np.zeros(len(speaker), bool)
    lab[lab_pos] = True
    lab_speakers = set(speaker[lab_pos].tolist())
    ev = np.where(~lab & ~enroll)[0]
    seen = np.array([s in lab_speakers for s in speaker[ev]], bool)
    return {"seen": ev[seen], "unseen": ev[~seen]}


def esn_labels(speaker: np.ndarray, cell: np.ndarray, enroll: np.ndarray, m: int,
               d: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(lab_pos, strict_lab, s_e)`` with the enrollment clips removed from the pool."""
    avail = ~enroll
    local, _ = draw_labels(cell[avail], M_CELL, draw_seed(d), cap_half=True)
    lab_pos = np.where(avail)[0][local]
    s_l, s_e = u1.split_strict(speaker, seed=STRICT_SEED + d)
    pool = np.where(s_l & avail)[0]
    strict = np.zeros(len(speaker), bool)
    rng = np.random.default_rng(STRICT_SEED + d + 5)
    for c in np.unique(cell):
        idx = pool[cell[pool] == c]
        if idx.size < 8:
            continue
        take = int(min(M_CELL, idx.size // 2))
        if take > 0:
            strict[rng.choice(idx, size=take, replace=False)] = True
    return lab_pos, np.where(strict)[0], s_e


def fit_all(Z, y, plat, speaker, lab_used, enroll, folds, Zt, ZtP, n
            ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Every U8 method for one draw / mask / m.  Scores cover all clips."""
    lab_used = np.asarray(lab_used)
    if lab_used.dtype == bool:
        lab_used = np.where(lab_used)[0]
    lab_mask = np.zeros(n, bool)
    lab_mask[lab_used] = True
    out: Dict[str, np.ndarray] = {}
    cv: Dict[str, Any] = {}

    m_b1, i_b1 = fit_logreg_cv(Z[lab_used], y[lab_used], folds=folds)
    out["B1"] = m_b1.decision_function(Z)
    cv["B1"] = {"C": i_b1["C"], "cv_loglik": i_b1["cv_loglik"]}

    # add the enrollment clips as extra genuine labels
    tr = np.concatenate([lab_used, np.where(enroll)[0]])
    ty = np.concatenate([y[lab_used], np.zeros(int(enroll.sum()), int)])
    m_en, i_en = fit_logreg_cv(Z[tr], ty, folds=np.concatenate(
        [folds, np.full(int(enroll.sum()), -1)]))
    out["B1+enroll"] = m_en.decision_function(Z)
    cv["B1+enroll"] = {"C": i_en["C"], "cv_loglik": i_en["cv_loglik"]}

    oc_defined = bool(enroll.any())
    if oc_defined:
        oc, info_oc = one_class_scores(Z, speaker, enroll)
        out["OC"] = oc
        cv["OC"] = info_oc
        z_oc = standardise(oc)
        z_b1 = standardise(out["B1"])
        C1 = float(i_b1["C"])

        def _eval_fuse(cfg, tr_mask, te_mask):
            from sklearn.linear_model import LogisticRegression

            m_ = LogisticRegression(C=C1, max_iter=5000).fit(Z[lab_used[tr_mask]],
                                                             y[lab_used[tr_mask]])
            return (standardise(m_.decision_function(Z))
                    + float(cfg) * z_oc)[lab_used[te_mask]]

        w, i_w = select_config(FUSE_WEIGHTS, _eval_fuse, y[lab_used], folds)
        out["OC+B1"] = z_b1 + float(w) * z_oc
        cv["OC+B1"] = {"w": float(w), "cv_loglik": i_w["cv_loglik"]}

    m_esn, i_esn = fit_logreg_cv(Zt[lab_used], y[lab_used], folds=folds)
    out["ESN"] = m_esn.decision_function(Zt)
    cv["ESN"] = {"C": i_esn["C"], "cv_loglik": i_esn["cv_loglik"]}
    m_p, i_p = fit_logreg_cv(ZtP[lab_used], y[lab_used], folds=folds)
    out["ESN-P"] = m_p.decision_function(ZtP)
    cv["ESN-P"] = {"C": i_p["C"], "cv_loglik": i_p["cv_loglik"]}
    return out, {"cv": cv, "oc_defined": oc_defined}


def run_one(payload) -> Tuple[int, int, List[Dict[str, Any]]]:
    d, m = payload
    Z, y, plat, speaker = _G["Z"], _G["y"], _G["plat"], _G["speaker"]
    cell = _G["cell"]
    n = len(y)
    rows: List[Dict[str, Any]] = []
    enroll = choose_enrollment(speaker, y, m, seed=ENROLL_SEED + 1000 * m + d)
    lab_pos, strict_lab, s_e = esn_labels(speaker, cell, enroll, m, d)
    masks = esn_masks(speaker, lab_pos, enroll)
    masks["strict_draw"] = np.where(s_e & ~enroll)[0]
    lab_sets = {"seen": lab_pos, "unseen": lab_pos, "strict_draw": strict_lab}

    means = speaker_means(Z, speaker, enroll, plat, m=m)
    Zt = esn_features(Z, speaker, means, plat, variant="plain")
    ZtP = esn_features(Z, speaker, means, plat, variant="platform")
    pop = np.array([s in means["speaker"] for s in speaker], bool)

    for mask_name in MASKS:
        ev = masks[mask_name]
        lab_used = lab_sets[mask_name]
        if ev.size < 50 or lab_used.size < 50 or len(np.unique(y[lab_used])) < 2:
            continue
        folds = (stratified_folds(y[lab_used]) if mask_name == "seen"
                 else speaker_grouped_folds(speaker[lab_used], y[lab_used]))
        if len(np.unique(folds)) < 2:
            continue
        try:
            scores, info = fit_all(Z, y, plat, speaker, lab_used, enroll, folds,
                                   Zt, ZtP, n)
        except Exception as exc:                        # pragma: no cover
            rows.append({"error": f"d={d} m={m} {mask_name}: {type(exc).__name__}: {exc}"})
            continue
        ev_scores = {k: v[ev] for k, v in scores.items()}
        save_draw_scores(_G["draw_root"] / f"m{m}" / mask_name, d, y[ev], ev_scores,
                         speaker[ev], extra={"m": int(m), "mask": mask_name,
                                             "cv": info["cv"]})
        for name in scores:
            for population, sel in (("all", np.ones(ev.size, bool)),
                                    ("enrolled_only", pop[ev])):
                if sel.sum() < 50 or len(np.unique(y[ev][sel])) < 2:
                    continue
                rows.append(summarize(y[ev][sel], ev_scores[name][sel], plat[ev][sel],
                                      extra={
                    "protocol": "dev_v3_esn", "region": "dev", "population": population,
                    "backbone": _G["backbone"], "spec": _G["spec"], "method": name,
                    "m": int(m), "k": int(lab_used.size), "draw": int(d),
                    "mask": mask_name,
                    "cv_folds": "stratified" if mask_name == "seen" else "speaker_group",
                    "n_labeled": int(lab_used.size), "n_eval": int(sel.sum()),
                    "n_enroll": int(enroll.sum()),
                    "n_enrolled_speakers": int(means["n_enrolled_speakers"]),
                    "oc_defined": bool(info["oc_defined"]),
                    "speaker_mean_is_global": bool(name in ("ESN", "ESN-P")
                                                   and (speaker[ev][~pop[ev]]).size > 0),
                    "cv_pick": json.dumps(info["cv"].get(name, {}), sort_keys=True),
                }))
    return d, m, rows


def _worker(payload):
    try:
        return run_one(payload)
    except Exception as exc:                            # pragma: no cover
        return payload[0], payload[1], [{"error": f"d={payload[0]}: {exc}"}]


def intervals_table(draw_root: Path, ms: Sequence[int] = M_LIST,
                    masks: Sequence[str] = MASKS) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for m in ms:
        for mask in masks:
            for a in ("ESN", "ESN-P"):
                for b in CONTROLS:
                    per = load_draw_scores(draw_root / f"m{m}" / mask, a, b)
                    if len(per) < 5:
                        continue
                    got = two_intervals(per)
                    rows.append({
                        "m": int(m), "mask": mask, "method_a": a, "method_b": b,
                        "n_draws": got["n_draws"], "delta": got["mean"],
                        "old_lo": got["old_lo"], "old_hi": got["old_hi"],
                        "a_lo": got["a_lo"], "a_hi": got["a_hi"],
                        "b_lo": got["b_lo"], "b_hi": got["b_hi"],
                        "b_contains_zero": got["b_contains_zero"],
                        "effect_points": abs(got["mean"]),
                        "meets_2pt_rule": bool(abs(got["mean"]) >= 2.0
                                               and got["b_excludes_zero"]),
                    })
    return pd.DataFrame(rows)


def gate_g17(df: pd.DataFrame, iv: pd.DataFrame, m_gate: int = GATE_M,
             population: str = "all") -> Dict[str, Any]:
    """G17/N5: ESN-LR at m=5 versus the best control on ``strict_draw``."""
    sel = df[(df["m"] == m_gate) & (df["mask"] == "strict_draw")
             & (df["population"] == population)]
    means = sel.groupby("method")["pooled_eer"].mean().to_dict()
    out: Dict[str, Any] = {"m": int(m_gate), "mask": "strict_draw",
                           "population": population,
                           "means": {k: float(v) for k, v in means.items()}}
    if "ESN" not in means:
        out["verdict"] = "ESN_NOT_RUN"
        return out
    available = [c for c in CONTROLS if c in means]
    if not available:
        out["verdict"] = "NO_CONTROL"
        return out
    best = min(available, key=lambda c: means[c])
    out["best_control"] = best
    out["delta_points"] = float(means["ESN"] - means[best])
    row = iv[(iv["m"] == m_gate) & (iv["mask"] == "strict_draw")
             & (iv["method_a"] == "ESN") & (iv["method_b"] == best)]
    if row.empty:
        out["verdict"] = "INTERVAL_MISSING"
        return out
    r = row.iloc[0]
    out["interval_b"] = [float(r["b_lo"]), float(r["b_hi"])]
    out["interval_old"] = [float(r["old_lo"]), float(r["old_hi"])]
    out["effect_ge_2"] = bool(out["delta_points"] <= -2.0)
    out["b_excludes_zero"] = bool(not r["b_contains_zero"])
    out["verdict"] = ("ESN_PASS" if out["effect_ge_2"] and out["b_excludes_zero"]
                      else "ESN_FAIL")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 U8 ESN")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--split", default=R.SPLIT_V3)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--ms", default=",".join(map(str, M_LIST)))
    parser.add_argument("--intervals-only", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    ms = [int(x) for x in str(args.ms).split(",") if x.strip()]
    draw_root = root / "results" / "u8_draws"
    out_csv = root / "results" / "u8_esn.csv"

    if not args.intervals_only:
        data_dir = data_root(args.data)
        sub, F = R.load_region(root, data_dir, args.backbone, args.spec, args.split)
        transform, Z = w6.fit_preprocess(F)
        _G.update({"Z": Z, "y": sub["y"].to_numpy(), "plat": sub["platform"].to_numpy(),
                   "cell": sub["cell"].to_numpy(),
                   "speaker": sub["speaker"].to_numpy(), "draw_root": draw_root,
                   "backbone": args.backbone, "spec": args.spec})
        print(f"v3 dev: clips={len(sub)} speakers={len(np.unique(_G['speaker']))} "
              f"Z={Z.shape}", flush=True)
        payloads = [(d, m) for m in ms for d in range(int(args.draws))]
        rows: List[Dict[str, Any]] = []
        errors: List[str] = []
        if args.jobs > 1:
            import multiprocessing as mp

            ctx = (mp.get_context("fork") if sys.platform != "win32"
                   else mp.get_context("spawn"))
            with ctx.Pool(args.jobs) as pool:
                for i, (_d, _m, out) in enumerate(
                        pool.imap_unordered(_worker, payloads, chunksize=1), 1):
                    for r in out:
                        if "error" in r:
                            errors.append(str(r["error"]))
                        else:
                            rows.append(r)
                    if i % 25 == 0:
                        print(f"  {i}/{len(payloads)} draw-m units ({len(rows)} rows, "
                              f"{len(errors)} errors)", flush=True)
        else:
            for pl in payloads:
                _d, _m, out = _worker(pl)
                for r in out:
                    if "error" in r:
                        errors.append(str(r["error"]))
                    else:
                        rows.append(r)
        if errors:
            print(f"errors: {len(errors)}; first: {errors[:3]}")
            (root / "results" / "u8_errors.txt").write_text(
                "\n".join(errors) + "\n", encoding="utf-8")
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.read_csv(out_csv)
    piv = df[df["population"] == "all"].pivot_table(
        index=["mask", "method"], columns="m", values="pooled_eer", aggfunc="mean")
    print("\npooled EER, all clips\n" + piv.round(3).to_string())
    iv = intervals_table(draw_root, ms)
    iv.to_csv(root / "results" / "u8_intervals.csv", index=False)
    print("\nintervals (delta = a - b; negative means a is better)\n"
          + iv.round(3).to_string(index=False))
    gate = gate_g17(df, iv)
    with open(root / "results" / "u8_gate.json", "w", encoding="utf-8") as fh:
        json.dump(gate, fh, indent=2, sort_keys=True)
    print("\n" + json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

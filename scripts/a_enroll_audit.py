#!/usr/bin/env python3
"""A: audit the U8 OC+B1 enrolment gain (NEW_TASKS 06 section 1).

The question is whether ``B1 14.45 -> OC+B1 5.70`` comes from *speaker-specific
known-real references*, or from a channel / platform shortcut, from same-source
segments (adjacent VAD cuts of one video), or simply from the extra genuine
labels that enrolment supplies.

Variants (same v3-dev draws, same seeds as U8: ``ENROLL_SEED + 1000*m + d``):

* **A0** the U8 setting itself, reported side by side (B1, B1+enroll, OC, OC+B1, ESN)
* **A1** shuffled enrolment: each speaker's R_s is replaced by ``m`` genuine clips
  drawn from OTHER speakers with the same platform composition
* **A2** cross-platform enrolment (the core test): for every (speaker, platform)
  group, R^×_{s,p} is drawn from that speaker's genuine clips on *other*
  platforms; ineligible groups are dropped from both L and E, and only
  within-platform metrics are read
* **A3** conservative same-source control: quantiles of the OC distance by label,
  then the same comparison after removing the 10 % lowest-distance *genuine*
  clips from E (this can only hurt the OC arm)

Outputs ``results/a_enroll_audit.csv``, ``results/a_variants.csv``,
``results/a_o_quantiles.csv``, ``results/a_gate.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

import u8_esn as u8  # noqa: E402  (reuse the frozen U8 recipe)
from src import regions as R  # noqa: E402
from src.baselines import fit_logreg_cv  # noqa: E402
from src.cv import speaker_grouped_folds  # noqa: E402
from src.data import data_root  # noqa: E402
from src.esn import choose_enrollment, one_class_scores, standardise  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.newtasks05 import one_thread_blas, worker_slots  # noqa: E402
from src.paired import two_intervals  # noqa: E402
from src.sgm import prepare_features  # noqa: E402

M_LIST = (5, 10)
N_DRAWS = 50
SAVE_DRAWS = 30
CONTROLS = ("B1", "B1+enroll", "OC", "OC+B1", "ESN")
G25_MARGIN = 2.0

_G: Dict[str, Any] = {}


def _shuffled_enrollment(speaker, y, plat, m, seed) -> np.ndarray:
    """A1: each speaker gets m genuine clips from other speakers, same platform mix."""
    rng = np.random.default_rng(seed)
    sp = np.asarray(speaker).astype(str)
    pl = np.asarray(plat).astype(str)
    y = np.asarray(y)
    out = np.zeros(len(sp), bool)
    real = np.where(y == 0)[0]
    for s in np.unique(sp):
        own = np.where(sp == s)[0]
        for p in np.unique(pl[own]):
            need = int(min(m, max(1, np.sum((pl[own] == p) & (y[own] == 0)))))
            pool = real[(sp[real] != s) & (pl[real] == p)]
            if pool.size == 0:
                pool = real[sp[real] != s]
            if pool.size == 0:
                continue
            out[rng.choice(pool, size=min(need, pool.size), replace=False)] = True
    return out


def _cross_platform_enrollment(speaker, y, plat, m, seed
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """A2: R^×_{s,p} from the speaker's genuine clips on OTHER platforms."""
    rng = np.random.default_rng(seed)
    sp = np.asarray(speaker).astype(str)
    pl = np.asarray(plat).astype(str)
    y = np.asarray(y)
    enrol = np.zeros(len(sp), bool)
    eligible = np.zeros(len(sp), bool)
    for s in np.unique(sp):
        for p in np.unique(pl[sp == s]):
            grp = (sp == s) & (pl == p)
            pool = np.where((sp == s) & (pl != p) & (y == 0))[0]
            if pool.size < m:
                continue                       # INELIGIBLE: dropped from L and E
            enrol[rng.choice(pool, size=int(m), replace=False)] = True
            eligible[grp] = True
    return enrol, eligible


def _score_set(Z, y, plat, speaker, lab_used, enrol, folds, tag):
    """B1 / B1+enroll / OC / OC+B1 for one draw, on an arbitrary enrolment mask."""
    from sklearn.linear_model import LogisticRegression

    out: Dict[str, np.ndarray] = {}
    m_b1, i_b1 = fit_logreg_cv(Z[lab_used], y[lab_used], folds=folds)
    out["B1"] = m_b1.decision_function(Z)
    en_idx = np.where(enrol)[0]
    if en_idx.size:
        tr = np.concatenate([lab_used, en_idx])
        ty = np.concatenate([y[lab_used], np.zeros(en_idx.size, int)])
        m_en, _ = fit_logreg_cv(Z[tr], ty,
                                folds=np.concatenate([folds, np.full(en_idx.size, -1)]))
        out["B1+enroll"] = m_en.decision_function(Z)
    else:
        out["B1+enroll"] = out["B1"].copy()
    oc, info = one_class_scores(Z, speaker, enrol)
    out["OC"] = oc
    z_oc, z_b1 = standardise(oc), standardise(out["B1"])
    C1 = float(i_b1["C"])
    best, best_ll = 0.0, -np.inf
    for w in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        ll = 0.0
        for tr, te in _fold_pairs(folds):
            mdl = LogisticRegression(C=C1, max_iter=5000).fit(
                Z[lab_used[tr]], y[lab_used[tr]])
            sc = standardise(mdl.decision_function(Z)) + w * z_oc
            p = np.clip(1.0 / (1.0 + np.exp(-sc[lab_used[te]])), 1e-9, 1 - 1e-9)
            ll += float(np.sum(y[lab_used[te]] * np.log(p)
                               + (1 - y[lab_used[te]]) * np.log(1 - p)))
        if ll > best_ll:
            best, best_ll = w, ll
    out["OC+B1"] = z_b1 + best * z_oc
    out["ESN"] = out["B1"].copy()          # filled by the caller when needed
    return out, {"w": float(best), "oc": info, "tag": tag}


def _fold_pairs(folds: np.ndarray):
    f = np.asarray(folds)
    for k in np.unique(f):
        tr = f != k
        te = f == k
        if tr.sum() > 20 and te.sum() > 5:
            yield tr, te


def run_unit(payload):
    with one_thread_blas():
        return _run_unit(payload)


def _run_unit(payload):
    bb, spec, m, d = payload
    Z, y, plat, speaker = _G["Z"][bb], _G["y"], _G["plat"], _G["speaker"]
    cell = _G["cell"]
    rows: List[Dict[str, Any]] = []
    variants: Dict[str, np.ndarray] = {}

    enrol0 = choose_enrollment(speaker, y, m, seed=u8.ENROLL_SEED + 1000 * m + d)
    variants["A0"] = enrol0
    variants["A1"] = _shuffled_enrollment(
        speaker, y, plat, m, seed=u8.ENROLL_SEED + 1000 * m + d + 77)
    enrol2, elig2 = _cross_platform_enrollment(
        speaker, y, plat, m, seed=u8.ENROLL_SEED + 1000 * m + d + 131)

    for tag, enrol in variants.items():
        lab_pos, strict_lab, s_e = u8.esn_labels(speaker, cell, enrol, m, d)
        masks = u8.esn_masks(speaker, lab_pos, enrol)
        masks["strict_draw"] = np.where(s_e & ~enrol)[0]
        ev, lab_used = masks["strict_draw"], strict_lab
        if ev.size < 50 or lab_used.size < 50 or len(np.unique(y[lab_used])) < 2:
            continue
        folds = speaker_grouped_folds(speaker[lab_used], y[lab_used])
        if len(np.unique(folds)) < 2:
            continue
        scores, info = _score_set(Z, y, plat, speaker, lab_used, enrol, folds, tag)
        o = scores["OC"]
        for name in ("B1", "B1+enroll", "OC", "OC+B1"):
            rows.append(summarize(y[ev], scores[name][ev], plat[ev], extra={
                "protocol": "a_enroll_audit", "region": "dev", "variant": tag,
                "backbone": bb, "spec": spec, "method": name, "m": int(m),
                "draw": int(d), "mask": "strict_draw", "k": int(lab_used.size),
                "n_labeled": int(lab_used.size), "n_eval": int(ev.size),
                "n_enroll": int(enrol.sum()), "w": info["w"]}))
        if tag == "A0" and d < SAVE_DRAWS:
            np.savez_compressed(
                _G["root"] / "scores" / f"u8r_{bb}_{m}_{d:04d}.npz",
                clip_id=_G["meta"]["clip_id"].to_numpy()[ev], label=y[ev],
                speaker=speaker[ev].astype(str), o=o[ev],
                **{n: scores[n][ev] for n in ("B1", "OC", "OC+B1", "B1+enroll")})

    # A2: cross-platform enrolment, within-platform view only
    if elig2.any():
        lab_pos, strict_lab, s_e = u8.esn_labels(speaker, cell, enrol2, m, d)
        masks = u8.esn_masks(speaker, lab_pos, enrol2)
        masks["strict_draw"] = np.where(s_e & ~enrol2)[0]
        ev = np.array([i for i in masks["strict_draw"] if elig2[i]], int)
        lab_used = np.array([i for i in strict_lab if elig2[i]], int)
        if ev.size >= 50 and lab_used.size >= 50 and len(np.unique(y[lab_used])) >= 2:
            folds = speaker_grouped_folds(speaker[lab_used], y[lab_used])
            if len(np.unique(folds)) >= 2:
                scores, info = _score_set(Z, y, plat, speaker, lab_used, enrol2,
                                          folds, "A2")
                sub = summarize(y[ev], scores["OC+B1"][ev], plat[ev], extra={
                    "protocol": "a_enroll_audit", "region": "dev", "variant": "A2",
                    "backbone": bb, "spec": spec, "method": "OC+B1", "m": int(m),
                    "mask": "strict_draw_crossplat", "k": int(lab_used.size),
                    "n_eval": int(ev.size), "n_enroll": int(enrol2.sum())})
                sub["coverage_clip"] = float(elig2.mean())
                sub["coverage_speakers"] = float(
                    len(np.unique(speaker[elig2])) / max(len(np.unique(speaker)), 1))
                rows.append(sub)
                base = summarize(y[ev], scores["B1"][ev], plat[ev], extra={
                    "protocol": "a_enroll_audit", "region": "dev", "variant": "A2",
                    "backbone": bb, "spec": spec, "method": "B1", "m": int(m),
                    "mask": "strict_draw_crossplat", "k": int(lab_used.size),
                    "n_eval": int(ev.size), "n_enroll": int(enrol2.sum())})
                base["coverage_clip"] = float(elig2.mean())
                base["coverage_speakers"] = float(
                    len(np.unique(speaker[elig2])) / max(len(np.unique(speaker)), 1))
                rows.append(base)
                if d < SAVE_DRAWS:
                    np.savez_compressed(
                        _G["root"] / "scores" / f"a2r_{bb}_{m}_{d:04d}.npz",
                        clip_id=_G["meta"]["clip_id"].to_numpy()[ev], label=y[ev],
                        speaker=speaker[ev].astype(str), platform=plat[ev].astype(str),
                        **{"B1": scores["B1"][ev], "OC+B1": scores["OC+B1"][ev]})

    return rows


def a3_same_source(root: Path, backbones, ms, save_draws: int = SAVE_DRAWS
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A3: OC-distance quantiles by label, and the conservative 10 %-trim test."""
    qrows: List[Dict[str, Any]] = []
    crows: List[Dict[str, Any]] = []
    for bb, _spec in backbones:
        for m in ms:
            base, trim = [], []
            for d in range(save_draws):
                path = root / "scores" / f"u8r_{bb}_{m}_{d:04d}.npz"
                if not path.exists():
                    continue
                with np.load(path) as z:
                    y, o, sp = z["label"], z["o"], z["speaker"]
                    b1, ocb1 = z["B1"], z["OC+B1"]
                for lab, name in ((1, "spoof"), (0, "genuine")):
                    sel = y == lab
                    if sel.sum() > 5:
                        q = np.percentile(o[sel], [10, 25, 50, 75, 90])
                        qrows.append({"backbone": bb, "m": m, "label": name,
                                      **{f"p{p}": float(v) for p, v in
                                         zip((10, 25, 50, 75, 90), q)}})
                # conservative: drop the 10 % smallest-o GENUINE clips from E
                gen = np.where(y == 0)[0]
                if gen.size > 20:
                    cut = np.percentile(o[gen], 10)
                    drop = gen[o[gen] <= cut]
                    keep = np.setdiff1d(np.arange(len(y)), drop)
                    if keep.size > 50 and len(np.unique(y[keep])) == 2:
                        base.append({"y": y[keep], "speaker": sp[keep],
                                     "sa": ocb1[keep], "sb": b1[keep]})
                        trim.append({"y": y[keep], "speaker": sp[keep],
                                     "sa": b1[keep], "sb": b1[keep]})
            if len(base) >= 5:
                iv = two_intervals(base)
                crows.append({"backbone": bb, "m": m, "test": "A3_trim10_genuine",
                              "n_draws": iv["n_draws"], "delta": iv["mean"],
                              "b_lo": iv["b_lo"], "b_hi": iv["b_hi"],
                              "b_excludes_zero": iv["b_excludes_zero"]})
    return pd.DataFrame(qrows), pd.DataFrame(crows)


def _macro(y, s, plat, min_per_class: int = 30) -> float:
    """Platform-macro EER, recomputed here without ``metrics3``."""
    from sklearn.metrics import roc_curve

    y = np.asarray(y); s = np.asarray(s); plat = np.asarray(plat)
    vals = []
    for p in np.unique(plat):
        sel = plat == p
        if min(int((y[sel] == 1).sum()), int((y[sel] == 0).sum())) < min_per_class:
            continue
        if len(np.unique(s[sel])) < 2:
            continue
        f, t, _ = roc_curve(y[sel], s[sel], pos_label=1, drop_intermediate=False)
        i = int(np.argmin(np.abs((1 - t) - f)))
        vals.append((f[i] + (1 - t[i])) / 2 * 100)
    return float(np.mean(vals)) if vals else float("nan")


def _cluster_interval(per_draw, metric, B_s: int = 200, seed: int = 20261015):
    """NEW_TASKS 06 V-1(b): resample evaluation SPEAKERS inside every draw."""
    rng = np.random.default_rng(seed)
    groups = [np.unique(d["speaker"]) for d in per_draw]
    boots = np.empty(int(B_s))
    for b in range(int(B_s)):
        vals = np.empty(len(per_draw))
        for i, d in enumerate(per_draw):
            gs = groups[i]
            pick = rng.integers(0, len(gs), size=len(gs))
            idx = np.concatenate([np.where(d["speaker"] == gs[j])[0] for j in pick])
            vals[i] = (metric(d["y"][idx], d["sa"][idx], d["plat"][idx])
                       - metric(d["y"][idx], d["sb"][idx], d["plat"][idx]))
        boots[b] = vals.mean()
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)),
            float(np.mean(boots)))


def _load_a2(root: Path, bb: str, m: int, a: str = "OC+B1", b: str = "B1"):
    out = []
    for d in range(SAVE_DRAWS):
        p = root / "scores" / f"a2r_{bb}_{m}_{d:04d}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            out.append({"y": z["label"], "speaker": z["speaker"].astype(str),
                        "plat": z["platform"].astype(str),
                        "sa": z[a].astype(float), "sb": z[b].astype(float)})
    return out


def gate_g25(root: Path, variants: pd.DataFrame, backbones, ms) -> Dict[str, Any]:
    out: Dict[str, Any] = {"criterion": "A2 cross-platform enrolment, m=5, "
                                       "within-platform macro, (b) interval clear of 0"}
    per: Dict[str, Any] = {}
    for bb, _spec in backbones:
        v = variants[(variants["backbone"] == bb) & (variants["m"] == 5)]
        if v.empty:
            continue
        a0 = v[v["variant"] == "A0"].groupby("method")["pooled_eer"].mean()
        a1 = v[v["variant"] == "A1"].groupby("method")["pooled_eer"].mean()
        entry: Dict[str, Any] = {}
        if {"B1", "OC+B1"} <= set(a0.index) and {"B1", "OC+B1"} <= set(a1.index):
            entry["gain_true"] = float(a0["B1"] - a0["OC+B1"])
            entry["gain_shuffled"] = float(a1["B1"] - a1["OC+B1"])
            entry["shuffled_ratio"] = (float(entry["gain_shuffled"] / entry["gain_true"])
                                       if entry["gain_true"] else float("nan"))
        a2 = variants[(variants["backbone"] == bb) & (variants["variant"] == "A2")
                      & (variants["method"] == "OC+B1")]
        if not a2.empty:
            entry["A2_macro"] = float(a2["within_macro_eer"].mean())
            entry["A2_coverage_clip"] = float(a2["coverage_clip"].mean())
            per_draw = _load_a2(root, bb, 5)
            if len(per_draw) >= 5:
                lo, hi, mean = _cluster_interval(per_draw, _macro)
                entry.update({"A2_macro_delta": mean, "b_lo": lo, "b_hi": hi,
                              "b_excludes_zero": bool(not (lo <= 0 <= hi)),
                              "n_draws": len(per_draw)})
                entry["A2_passes"] = bool(mean <= -G25_MARGIN
                                          and not (lo <= 0 <= hi))
                entry["A2_inconclusive"] = bool(mean <= -G25_MARGIN
                                                and (lo <= 0 <= hi))
                entry["b_half_width"] = float((hi - lo) / 2.0)
        per[bb] = entry
    out["per_backbone"] = per
    valid = [e for e in per.values() if e.get("A2_passes")]
    inconclusive = [e for e in per.values() if e.get("A2_inconclusive")]
    under = [bb for bb, e in per.items() if e.get("A2_coverage_clip", 1.0) < 0.30]
    if under and not valid and not inconclusive:
        out["verdict"] = "ENROLL_UNTESTABLE"
        out["underpowered"] = under
    elif len(valid) == len(per) and per:
        ratio = max(e.get("shuffled_ratio", 1.0) for e in per.values())
        out["verdict"] = "ENROLL_GENERIC" if ratio > 0.5 else "ENROLL_VALID"
    elif inconclusive:
        out["verdict"] = "ENROLL_INCONCLUSIVE"
    else:
        out["verdict"] = "ENROLL_SHORTCUT"
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
    ap = argparse.ArgumentParser(description="X3 A: enrolment audit")
    ap.add_argument("--root", default=".")
    ap.add_argument("--data")
    ap.add_argument("--backbones", default="xls-r-2b|S3,mms-1b|S3")
    ap.add_argument("--split", default=R.SPLIT_V3)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--ms", default=",".join(map(str, M_LIST)))
    ap.add_argument("--jobs", type=int, default=worker_slots())
    a = ap.parse_args(argv)

    root = Path(a.root)
    data = data_root(a.data)
    ms = [int(x) for x in str(a.ms).split(",") if x.strip()]
    pairs = [tuple(x.split("|")) for x in a.backbones.split(",") if x.strip()]
    Zs = {}
    meta = None
    for bb, spec in pairs:
        sub, X = R.load_region(root, data, bb, spec, a.split)
        Zs[bb] = prepare_features(X, 256)
        meta = sub if meta is None else meta
        print(f"loaded {bb}|{spec} -> Z {Zs[bb].shape}", flush=True)
    _G.update({"Z": Zs, "y": sub["y"].to_numpy(int), "plat": sub["platform"].to_numpy(),
               "speaker": sub["speaker"].to_numpy(), "cell": sub["cell"].to_numpy(),
               "meta": sub, "root": root})

    payloads = [(bb, spec, m, d) for bb, spec in pairs for m in ms
                for d in range(int(a.draws))]
    rows: List[Dict[str, Any]] = []
    for i, out in enumerate(_stream(run_unit, payloads, a.jobs), 1):
        rows.extend(out)
        if i % 25 == 0:
            print(f"  {i}/{len(payloads)} units ({len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(root / "results" / "a_enroll_audit.csv", index=False)
    piv = (df[df["variant"].isin(["A0", "A1"])]
           .groupby(["backbone", "m", "variant", "method"])
           [["pooled_eer", "within_macro_eer", "worst_platform_eer", "auc"]]
           .mean().reset_index())
    piv.to_csv(root / "results" / "a_variants.csv", index=False)
    print(piv.round(2).to_string(index=False))
    q, a3 = a3_same_source(root, pairs, ms)
    q.to_csv(root / "results" / "a_o_quantiles.csv", index=False)
    a3.to_csv(root / "results" / "a_a3_same_source.csv", index=False)
    if not q.empty:
        print("\nOC-distance quantiles by label\n" + q.round(4).to_string(index=False))
    if not a3.empty:
        print("\nA3 conservative same-source control\n" + a3.round(3).to_string(index=False))
    gate = gate_g25(root, df, pairs, ms)
    (root / "results" / "a_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("\n" + json.dumps(gate, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

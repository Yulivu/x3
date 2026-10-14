#!/usr/bin/env python3
"""U4 - M-ALP: active label propagation (NEW_TASKS 01 U4, revised by 追加 02).

What the revision changes, and what this script therefore does:

* evaluation masks are ``clips`` (the whole dev region, clip-wise held out) and
  ``strict_draw`` (``S_L`` / ``S_E`` speaker halves, ``E`` = all of ``S_E``) - the
  two masks V-2 allows a *selector* comparison to use;
* N1' (the learner's 0.5-point gate) is reported, not used as a novelty claim;
* hyper-parameters follow V-3 (speaker-grouped folds wherever ``strict_draw`` is
  reported).

Everything else is the U4 specification: the L-GRAPH learner (cosine kNN +
propagation + LR fusion, <= 16 configs), the ``kmeans`` warm start and four
``u * rho * c`` acquisition rounds, the ``SC`` / ``CC`` coverage variants, the
comparators (``U``, ``E``, ``K``, ``TypiClust-lite``, ``CASA``) and the
selector x learner ablation matrix.

Writes ``results/u4_malp.csv`` (per-draw rows), ``results/u4_draws/``,
``results/u4_intervals.csv``, ``results/u4_marginals.csv`` and
``results/malp_gate.json``.
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
from src.alloc import allocate  # noqa: E402
from src.baselines import fit_logreg_cv, knn_graph  # noqa: E402
from src.cv import speaker_grouped_folds, stratified_folds  # noqa: E402
from src.data import data_root  # noqa: E402
from src.malp import (  # noqa: E402
    BATCHES,
    N0,
    fuse,
    heat_start,
    hyperparam_grid,
    malp_scores,
    mass_normalise,
    propagate,
    select_batch,
    select_hyperparams,
    sparse_affinity,
)
from src.metrics3 import summarize  # noqa: E402
from src.paired import load_draw_scores, save_draw_scores, two_intervals  # noqa: E402

K_LIST = (30, 100, 290, 600)
DRAWS_MAIN = {30: 50, 100: 50, 290: 100, 600: 50}
DRAWS_ABLATION = 30
MASKS = ("clips", "strict_draw")
SELECTORS = ("U", "E", "K", "TypiClust", "CASA", "DWU", "SC-DWU", "CC-DWU", "full")
LEARNERS = ("LR", "LP", "LR+LP")
STRICT_SEED = 777000

_G: Dict[str, Any] = {}

__all__ = ["pool_and_eval", "run_selector", "learn", "run_one", "marginals",
           "gates", "main"]


def pool_and_eval(d: int, mask: str, Z, y, cell, speaker
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """``(pool, eval)`` for one draw and mask.

    ``clips``  - the pool is every dev clip; ``eval`` is a 25 % clip-wise holdout
    (so the selector never sees them, and every speaker is still represented).
    ``strict_draw`` - the pool is ``S_L`` and ``eval`` is all of ``S_E``.
    """
    n = len(y)
    if mask == "clips":
        rng = np.random.default_rng(1000 + d)
        ev = np.sort(rng.choice(n, size=int(0.25 * n), replace=False))
        return np.setdiff1d(np.arange(n), ev), ev
    s_l, s_e = u1.split_strict(speaker, seed=STRICT_SEED + d)
    return np.where(s_l)[0], np.where(s_e)[0]


def _folds(y_lab: np.ndarray, speaker_lab: np.ndarray, mask: str) -> np.ndarray:
    return (stratified_folds(y_lab) if mask == "clips"
            else speaker_grouped_folds(speaker_lab, y_lab))


def run_selector(name: str, Z, y, cell, speaker, pool: np.ndarray, k: int,
                 graph, cfg, mask: str, seed: int) -> np.ndarray:
    """One selection rule.  Returns the labelled indices (size <= k)."""
    k = int(min(k, pool.size))
    if name == "TypiClust":                       # one shot: cluster + typical clip
        return pool[heat_start(Z[pool], n0=k, seed=seed)]
    n0 = int(min(N0, k))
    picked = list(pool[heat_start(Z[pool], n0=n0, seed=seed)])
    batches = int(BATCHES) if k > n0 else 0
    batch = max(1, (k - len(picked)) // max(batches, 1)) if batches else 0
    rng = np.random.default_rng(seed)
    for _ in range(batches):
        if len(picked) >= k:
            break
        mask_lab = np.zeros(len(y), bool)
        mask_lab[np.asarray(picked, int)] = True
        lr, _ = fit_logreg_cv(Z[mask_lab], y[mask_lab])
        logit = lr.decision_function(Z)
        unlab = pool[~np.isin(pool, picked)]
        if name in ("DWU", "SC-DWU", "CC-DWU", "full"):
            sc = malp_scores(Z, y, mask_lab, graph, cfg, logit, speaker)
            variant = {"DWU": "D", "full": "full", "SC-DWU": "SC",
                       "CC-DWU": "CC"}[name]
            new = select_batch(Z, sc, graph[0], graph[1], np.asarray(picked, int),
                               unlab, batch, variant=variant, speaker=speaker,
                               cell=cell)
        else:
            q = 1.0 / (1.0 + np.exp(-logit))
            pos = {c: int(((cell == c) & mask_lab & (y == 1)).sum())
                   for c in np.unique(cell)}
            neg = {c: int(((cell == c) & mask_lab & (y == 0)).sum())
                   for c in np.unique(cell)}
            strat = {"U": "U", "E": "E", "K": "K", "CASA": "CASA"}[name]
            local = allocate(strat, Z_pool=Z, cell_pool=cell, unlab_idx=unlab,
                             q_pool=q, pos=pos, neg=neg, batch=batch, rng=rng)
            new = np.asarray(local, int)
        if new.size == 0:
            break
        picked.extend(new.tolist())
        picked = list(dict.fromkeys(picked))
    return np.asarray(picked[:k], int)


def learn(learner: str, Z, y, graph, labelled: np.ndarray, cfg) -> np.ndarray:
    """``LR`` (B1-style probe), ``LP`` (propagation alone) or ``LR+LP`` (the fusion)."""
    mask = np.zeros(len(y), bool)
    mask[np.asarray(labelled, int)] = True
    lr, _ = fit_logreg_cv(Z[mask], y[mask])
    logit = lr.decision_function(Z)
    if learner == "LR":
        return logit
    k_g, alpha, w_s, eta = cfg
    S = sparse_affinity(graph[0], graph[1], k_g)
    f = propagate(S, y[mask], mask, alpha=alpha)
    f = mass_normalise(f, float(np.mean(y[mask])))
    if learner == "LP":
        return f
    return fuse(logit, f, eta)


def run_one(payload) -> Tuple[int, str, int, List[Dict[str, Any]]]:
    d, mask, k = payload
    Z, y, cell, speaker = _G["Z"], _G["y"], _G["cell"], _G["speaker"]
    graph, plat = _G["graph"], _G["plat"]
    rows: List[Dict[str, Any]] = []
    pool, ev = pool_and_eval(d, mask, Z, y, cell, speaker)
    if ev.size < 50 or pool.size < k:
        return d, mask, k, rows
    # one V-3 selection of (k_g, alpha, w_s, eta) per draw, on the warm-start set
    warm = pool[heat_start(Z[pool], n0=min(N0, k), seed=d)]
    mask_warm = np.zeros(len(y), bool)
    mask_warm[warm] = True
    cfg, info_cfg = select_hyperparams(Z, y, mask_warm, graph,
                                       _folds(y[warm], speaker[warm], mask),
                                       speaker=speaker)
    saved: Dict[str, np.ndarray] = {}
    for selector in SELECTORS:
        picked = run_selector(selector, Z, y, cell, speaker, pool, k, graph, cfg,
                              mask, seed=d)
        if picked.size < 50:
            continue
        for learner in LEARNERS:
            sc = learn(learner, Z, y, graph, picked, cfg)
            name = f"{selector}|{learner}"
            saved[name] = sc[ev]
            rows.append(summarize(y[ev], sc[ev], plat[ev], extra={
                "protocol": "dev_v3_malp", "region": "dev", "mask": mask,
                "backbone": _G["backbone"], "spec": _G["spec"], "k": int(k),
                "draw": int(d), "selector": selector, "learner": learner,
                "method": name, "n_labeled": int(picked.size), "n_eval": int(ev.size),
                "cv_folds": "stratified" if mask == "clips" else "speaker_group",
                "malp_cfg": json.dumps(list(cfg)),
                "malp_cv_loglik": float(info_cfg["cv_loglik"]),
            }))
    if saved:
        save_draw_scores(_G["draw_root"] / mask / f"k{k}", d, y[ev], saved, speaker[ev],
                         extra={"mask": mask, "k": int(k)})
    return d, mask, k, rows


def _worker(payload):
    try:
        return run_one(payload)
    except Exception as exc:                            # pragma: no cover
        return payload[0], payload[1], payload[2], [{"error": f"{payload}: {exc}"}]


def _interval_row(job) -> Optional[Dict[str, Any]]:
    draw_root, mask, k, a, b = job
    per = load_draw_scores(Path(draw_root) / mask / f"k{k}", a, b)
    if len(per) < 5:
        return None
    got = two_intervals(per)
    return {"mask": mask, "k": int(k), "method_a": a, "method_b": b,
            "n_draws": got["n_draws"], "delta": got["mean"],
            "old_lo": got["old_lo"], "old_hi": got["old_hi"],
            "a_lo": got["a_lo"], "a_hi": got["a_hi"],
            "b_lo": got["b_lo"], "b_hi": got["b_hi"],
            "b_contains_zero": got["b_contains_zero"],
            "effect_points": abs(got["mean"]),
            "meets_2pt_rule": bool(abs(got["mean"]) >= 2.0 and got["b_excludes_zero"])}


def intervals_table(draw_root: Path, pairs: Sequence[Tuple[str, str]],
                    masks: Sequence[str] = MASKS, ks: Sequence[int] = (100, 290),
                    jobs: int = 1) -> pd.DataFrame:
    todo = [(str(draw_root), str(m), int(k), a, b)
            for m in masks for k in ks for a, b in pairs]
    rows: List[Dict[str, Any]] = []
    if jobs and int(jobs) > 1:
        import multiprocessing as mp

        ctx = (mp.get_context("fork") if sys.platform != "win32"
               else mp.get_context("spawn"))
        with ctx.Pool(int(jobs)) as pool:
            for r in pool.imap_unordered(_interval_row, todo, chunksize=1):
                if r is not None:
                    rows.append(r)
    else:
        for job in todo:
            r = _interval_row(job)
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)


def marginals(df: pd.DataFrame, learners: Sequence[str] = LEARNERS,
              selectors: Sequence[str] = SELECTORS) -> pd.DataFrame:
    """Factor marginals of the selector x learner matrix, per mask and budget."""
    rows: List[Dict[str, Any]] = []
    for (mask, k), part in df.groupby(["mask", "k"]):
        means = part.groupby(["selector", "learner"])["pooled_eer"].mean()
        for learner in learners:
            base = means.get(("U", learner), float("nan"))
            for sel in selectors:
                if sel == "U":
                    continue
                rows.append({"mask": mask, "k": int(k), "factor": "selector",
                             "level": sel, "reference": "U", "learner": learner,
                             "mean": float(means.get((sel, learner), np.nan)),
                             "effect": float(means.get((sel, learner), np.nan) - base)})
        for sel in selectors:
            base = means.get((sel, "LR"), float("nan"))
            for learner in learners:
                if learner == "LR":
                    continue
                rows.append({"mask": mask, "k": int(k), "factor": "learner",
                             "level": learner, "reference": "LR", "selector": sel,
                             "mean": float(means.get((sel, learner), np.nan)),
                             "effect": float(means.get((sel, learner), np.nan) - base)})
    return pd.DataFrame(rows)


def gates(df: pd.DataFrame, iv: pd.DataFrame) -> Dict[str, Any]:
    """G12-G15 (and what part of G16 the two masks can decide)."""
    out: Dict[str, Any] = {}
    if iv is None or iv.empty or "mask" not in getattr(iv, "columns", []):
        iv = pd.DataFrame(columns=["mask", "k", "method_a", "method_b", "delta",
                                   "b_lo", "b_hi", "b_contains_zero"])
    means = df.groupby(["mask", "k", "method"])["pooled_eer"].mean()

    def m(mask, k, method):
        return float(means.get((mask, int(k), method), float("nan")))

    # G12 / N1' - now informational (追加 02)
    best_lr = min((m("strict_draw", 290, f"{s}|LR+LP") for s in SELECTORS),
                  default=float("nan"))
    out["G12"] = {
        "status": "MALP_INFO",
        "rule": "N1' is informational after 追加 02; the 0.5-point rule is no longer a novelty gate",
        "best_selector_lr_lp_strict_draw_k290": best_lr,
        "b1_unseen_reference": 11.96,
    }
    # G13 / N2' - selector vs uniform and vs the best of {E, K, TypiClust} on strict_draw
    k = 100
    malp = m("strict_draw", k, "full|LR+LP")
    uni = m("strict_draw", k, "U|LR+LP")
    others = {s: m("strict_draw", k, f"{s}|LR+LP") for s in ("E", "K", "TypiClust")}
    best_other = min(others, key=lambda s: others[s]) if others else None
    row = iv[(iv["mask"] == "strict_draw") & (iv["k"] == k)
             & (iv["method_a"] == "full|LR+LP")]
    def _iv(partner):
        if row.empty:
            return None
        sub = row[row["method_b"] == partner]
        return None if sub.empty else sub.iloc[0].to_dict()
    vs_uni = _iv("U|LR+LP")
    vs_other = _iv(f"{best_other}|LR+LP") if best_other else None
    gain_uni = None if vs_uni is None else -float(vs_uni["delta"])
    gain_other = None if vs_other is None else -float(vs_other["delta"])
    status = "SEL_FAIL"
    if gain_uni is not None and gain_uni >= 3.0:
        status = "SEL_PASS" if (gain_other is not None and gain_other >= 0.5) \
            else "SEL_ONLY_VS_UNIFORM"
    out["G13"] = {"status": status, "mask": "strict_draw", "k": k,
                  "malp": malp, "uniform": uni, "gain_vs_uniform": gain_uni,
                  "best_other": best_other, "others": others,
                  "gain_vs_best_other": gain_other,
                  "interval_vs_uniform": None if vs_uni is None else
                  [float(vs_uni["b_lo"]), float(vs_uni["b_hi"])],
                  "interval_vs_best_other": None if vs_other is None else
                  [float(vs_other["b_lo"]), float(vs_other["b_hi"])]}
    # G14 / N3' - speaker coverage vs cell coverage at the same budget
    sc = m("strict_draw", k, "SC-DWU|LR+LP")
    cc = m("strict_draw", k, "CC-DWU|LR+LP")
    r = iv[(iv["mask"] == "strict_draw") & (iv["k"] == k)
           & (iv["method_a"] == "SC-DWU|LR+LP") & (iv["method_b"] == "CC-DWU|LR+LP")]
    delta = None if r.empty else float(r.iloc[0]["delta"])
    out["G14"] = {"status": ("SPEAKER_COVERAGE_PASS" if (delta is not None
                                                         and delta <= -1.0)
                             else "SPEAKER_COVERAGE_FAIL"),
                  "sc": sc, "cc": cc, "delta_sc_minus_cc": delta,
                  "interval": None if r.empty else [float(r.iloc[0]["b_lo"]),
                                                    float(r.iloc[0]["b_hi"])]}
    # G15 / N4' - label efficiency: k=100 against B1+U at k=290
    small = m("strict_draw", 100, "full|LR+LP")
    big = m("strict_draw", 290, "U|LR+LP")
    out["G15"] = {"status": ("LABEL_EFFICIENCY_PASS"
                             if np.isfinite(small) and np.isfinite(big)
                             and small - big <= 0.3
                             else "LABEL_EFFICIENCY_FAIL"),
                  "malp_k100": small, "b1u_k290": big,
                  "difference": None if not (np.isfinite(small) and np.isfinite(big))
                  else float(small - big)}
    # G16 / S-F' - only the comparators this run has; the second backbone is a note
    base = {s: m("strict_draw", 290, f"{s}|LR+LP") for s in ("U", "E", "K", "TypiClust")}
    malp290 = m("strict_draw", 290, "full|LR+LP")
    beats = {s: bool(np.isfinite(v) and malp290 < v) for s, v in base.items()}
    out["G16"] = {"status": "SOTA_F2_DEV_NO",
                  "reason": "S-F' needs every BASE_F' baseline on both masks and two "
                            "backbones; this run covers the selector comparators on "
                            "one backbone, so the gate stays negative by rule",
                  "malp_k290": malp290, "baselines": base, "beats": beats}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 U4 M-ALP")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--split", default=R.SPLIT_V3)
    parser.add_argument("--ks", default=",".join(map(str, K_LIST)))
    parser.add_argument("--masks", default=",".join(MASKS))
    parser.add_argument("--draws", type=int, default=0,
                        help="override the per-budget draw count (0 = use DRAWS_MAIN)")
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--interval-jobs", type=int, default=8)
    parser.add_argument("--intervals-only", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    draw_root = root / "results" / "u4_draws"
    out_csv = root / "results" / "u4_malp.csv"
    ks = [int(x) for x in str(args.ks).split(",") if x.strip()]
    masks = [m for m in str(args.masks).split(",") if m.strip()]

    if not args.intervals_only:
        data_dir = data_root(args.data)
        sub, F = R.load_region(root, data_dir, args.backbone, args.spec, args.split)
        transform, Z = w6.fit_preprocess(F)
        y = sub["y"].to_numpy()
        _G.update({"Z": Z, "y": y, "plat": sub["platform"].to_numpy(),
                   "cell": sub["cell"].to_numpy(),
                   "speaker": sub["speaker"].to_numpy(), "draw_root": draw_root,
                   "backbone": args.backbone, "spec": args.spec,
                   "graph": knn_graph(Z, k=max(30, 15))})
        print(f"v3 dev: clips={len(sub)} speakers={len(np.unique(_G['speaker']))} "
              f"Z={Z.shape}", flush=True)
        payloads = []
        for k in ks:
            n_draws = int(args.draws) if args.draws else DRAWS_MAIN.get(k, 30)
            for mask in masks:
                for d in range(n_draws):
                    payloads.append((d, mask, k))
        print(f"{len(payloads)} draw x mask x budget units", flush=True)
        rows: List[Dict[str, Any]] = []
        errors: List[str] = []
        if args.jobs > 1:
            import multiprocessing as mp

            ctx = (mp.get_context("fork") if sys.platform != "win32"
                   else mp.get_context("spawn"))
            with ctx.Pool(int(args.jobs)) as pool:
                for i, (_d, _m, _k, out) in enumerate(
                        pool.imap_unordered(_worker, payloads, chunksize=1), 1):
                    for r in out:
                        if "error" in r:
                            errors.append(str(r["error"]))
                        else:
                            rows.append(r)
                    if i % 25 == 0:
                        print(f"  {i}/{len(payloads)} units ({len(rows)} rows, "
                              f"{len(errors)} errors)", flush=True)
        else:
            for pl in payloads:
                _d, _m, _k, out = _worker(pl)
                for r in out:
                    if "error" in r:
                        errors.append(str(r["error"]))
                    else:
                        rows.append(r)
        if errors:
            print(f"errors: {len(errors)}; first: {errors[:3]}")
            (root / "results" / "u4_errors.txt").write_text(
                "\n".join(errors) + "\n", encoding="utf-8")
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    try:
        df = pd.read_csv(out_csv)
    except pd.errors.EmptyDataError:
        print("no U4 rows were produced", file=sys.stderr)
        return 1
    print("\nM-ALP main table (pooled EER, mean over draws)\n"
          + df.pivot_table(index=["mask", "k", "selector"], columns="learner",
                           values="pooled_eer", aggfunc="mean").round(3).to_string())
    mg = marginals(df)
    mg.to_csv(root / "results" / "u4_marginals.csv", index=False)
    print("\nlargest selector/learner effects (negative = better)\n"
          + mg.reindex(mg["effect"].abs().sort_values(ascending=False).index)
          .head(12).round(3).to_string(index=False))
    pairs = [("full|LR+LP", "U|LR+LP")]
    pairs += [(f"{s}|LR+LP", "full|LR+LP") for s in ("E", "K", "TypiClust", "CASA")]
    pairs += [("SC-DWU|LR+LP", "CC-DWU|LR+LP"), ("DWU|LR+LP", "full|LR+LP")]
    iv = intervals_table(draw_root, pairs, masks=masks, ks=[k for k in ks if k in (100, 290)],
                         jobs=args.interval_jobs)
    iv.to_csv(root / "results" / "u4_intervals.csv", index=False)
    if not iv.empty:
        print("\npaired intervals\n" + iv.round(3).to_string(index=False))
    gate = gates(df, iv)
    with open(root / "results" / "malp_gate.json", "w", encoding="utf-8") as fh:
        json.dump(gate, fh, indent=2, sort_keys=True, default=str)
    print("\n" + json.dumps(gate, indent=2, sort_keys=True, default=str)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

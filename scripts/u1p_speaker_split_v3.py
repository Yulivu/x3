#!/usr/bin/env python3
"""U1' [URGENT] - the U1 main table on the frozen v3 dev region (NEW_TASKS 02).

Three corrections against U1:

* **V-1** every comparison reports the old rule (2.5/97.5 quantiles of the per-draw
  differences) *and* the two new intervals: (a) bootstrap of the mean difference
  over the 100 draws, (b) speaker-clustered bootstrap.  The gate interval is (b).
* **V-3** hyper-parameters are chosen inside ``L`` with speaker-grouped folds where
  the number is reported for ``unseen`` / ``strict_draw``, and with the shared
  stratified folds where it is reported for ``seen``.  ``B3`` (sigma), ``B7``
  (tau), ``B9`` (k, alpha) and ``C1`` (MPP's sigma grid) are all re-selected;
  ``B1``'s ``C`` keeps the treatment it always had.
* the region is ``splits/speaker_split_v3.json`` instead of v2.

Masks: ``seen`` (evaluation clips whose speaker is labelled), ``unseen`` (the rest
of the per-cell draw) and ``strict_draw`` (speakers split into S_L / S_E; L only
from S_L, E = every clip of S_E).

Writes ``results/u1p_speaker_split_v3.csv`` (per-draw rows), the per-draw score
archives under ``results/u1p_draws/``, ``results/u1p_intervals.csv`` (old vs new
intervals), ``results/u1p_v2v3.csv`` (v2 -> v3 means) and
``results/u1p_verdicts.json``.
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
from src.baselines import (  # noqa: E402
    B3_SIGMAS,
    B7_TAUS,
    b3_platform_offset,
    b7_naive_pseudo_label,
    b9_label_propagation,
    fit_logreg_cv,
    knn_graph,
)
from src.cv import select_config, speaker_grouped_folds, stratified_folds  # noqa: E402
from src.data import data_root  # noqa: E402
from src.draws import draw_labels, draw_seed  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.mpp import fit_cv as mpp_fit_cv  # noqa: E402
from src.paired import load_draw_scores, save_draw_scores, two_intervals  # noqa: E402

N_DRAWS = 100
BUDGETS: Dict[int, int] = {290: 10, 100: 3}       # nominal k -> per-cell quota m
METHODS = ("B1", "B3", "B7", "B9", "C1")
B9_KS = (5, 10, 15)
B9_ALPHAS = (0.7, 0.9)
PAIRS = (("B9", "B1"), ("C1", "B1"), ("B3", "B1"), ("B7", "B1"), ("C1", "B9"))
MASKS = ("seen", "unseen", "strict_draw")
STRICT_SEED = 777000

_G: Dict[str, Any] = {}

__all__ = [
    "mask_sets", "strict_labels", "fit_methods", "run_one", "intervals_table",
    "verdicts", "main",
]


# --------------------------------------------------------------- draw plumbing
def mask_sets(speaker: np.ndarray, cell: np.ndarray, lab_pos: np.ndarray,
              m: int, seed: int) -> Dict[str, np.ndarray]:
    """The two clip-level evaluation masks of one draw (positions in the region)."""
    n = len(speaker)
    lab = np.zeros(n, bool)
    lab[lab_pos] = True
    lab_speakers = set(speaker[lab_pos].tolist())
    ev = np.setdiff1d(np.arange(n), lab_pos)
    seen = np.array([s in lab_speakers for s in speaker[ev]], bool)
    return {"seen": ev[seen], "unseen": ev[~seen]}


def strict_labels(speaker: np.ndarray, cell: np.ndarray, m: int, seed: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """S_L / S_E speaker split plus a per-cell quota drawn from S_L only."""
    s_l, s_e = u1.split_strict(speaker, seed=seed)
    pool = np.where(s_l)[0]
    lab = np.zeros(len(speaker), bool)
    rng = np.random.default_rng(seed + 5)
    for c in np.unique(cell):
        idx = pool[cell[pool] == c]
        if idx.size < 8:
            continue
        take = int(min(m, idx.size // 2))
        if take > 0:
            lab[rng.choice(idx, size=take, replace=False)] = True
    return lab, s_e


# ------------------------------------------------------------------ estimators
def fit_methods(Z, y, plat, lang, cell, speaker, lab_used, folds, graph,
                n_eval) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Fit B1/B3/B7/B9/C1 on ``L`` with the given folds; score every clip.

    ``lab_used`` may be an index array or a boolean mask over the region.
    """
    lab_used = np.asarray(lab_used)
    if lab_used.dtype == bool:
        lab_used = np.where(lab_used)[0]
    n = len(Z)
    lab_mask = np.zeros(n, bool)
    lab_mask[lab_used] = True
    Zl, yl = Z[lab_used], y[lab_used]
    pl, ll, cl = plat[lab_used], lang[lab_used], cell[lab_used]
    out: Dict[str, np.ndarray] = {}
    cv: Dict[str, Any] = {}

    model, info_b1 = fit_logreg_cv(Zl, yl, folds=folds)
    out["B1"] = model.decision_function(Z)
    cv["B1"] = {"C": info_b1["C"], "cv_loglik": info_b1["cv_loglik"]}
    C = float(info_b1["C"])

    def _base_logit(tr_mask):
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(C=C, max_iter=5000).fit(Zl[tr_mask], yl[tr_mask])
        return m.decision_function(Z)

    def _eval_b3(cfg, tr_mask, te_mask):
        sc = _base_logit(tr_mask)
        labm = np.zeros(n, bool)
        labm[lab_used[tr_mask]] = True
        s2, _ = b3_platform_offset(sc, plat, labm, y, sigma=float(cfg))
        return s2[lab_used[te_mask]]

    sigma, info_b3 = select_config(B3_SIGMAS, _eval_b3, yl, folds)
    s3, _ = b3_platform_offset(out["B1"], plat, lab_mask, y, sigma=float(sigma))
    out["B3"] = s3
    cv["B3"] = {"sigma": float(sigma), "cv_loglik": info_b3["cv_loglik"]}

    def _eval_b7(cfg, tr_mask, te_mask):
        s, _ = b7_naive_pseudo_label(Zl[tr_mask], yl[tr_mask], Zl[te_mask],
                                     tau=float(cfg), rounds=3)
        return s

    tau, info_b7 = select_config(B7_TAUS, _eval_b7, yl, folds)
    s7_unlab, _ = b7_naive_pseudo_label(Zl, yl, Z[~lab_mask], tau=float(tau), rounds=3)
    s7 = np.zeros(n)
    s7[~lab_mask] = s7_unlab
    out["B7"] = s7
    cv["B7"] = {"tau": float(tau), "cv_loglik": info_b7["cv_loglik"]}

    def _eval_b9(cfg, tr_mask, te_mask):
        kk, aa = cfg
        labm = np.zeros(n, bool)
        labm[lab_used[tr_mask]] = True
        s, _ = b9_label_propagation(Z, yl[tr_mask], labm, k=int(kk),
                                    alpha=float(aa), graph=graph)
        return s[lab_used[te_mask]]

    (kk, aa), info_b9 = select_config([(k, a) for k in B9_KS for a in B9_ALPHAS],
                                      _eval_b9, yl, folds)
    s9, _ = b9_label_propagation(Z, yl, lab_mask, k=int(kk), alpha=float(aa),
                                 graph=graph)
    out["B9"] = s9
    cv["B9"] = {"k": int(kk), "alpha": float(aa), "cv_loglik": info_b9["cv_loglik"]}

    c1, info_c1 = mpp_fit_cv(Zl, yl, pl, ll, cl, folds=folds)
    out["C1"] = c1.predict_logit(Z, plat, lang, cell)
    cv["C1"] = {"sigma_p": info_c1["sigma_p"], "sigma_gamma": info_c1["sigma_gamma"],
                "cv_loglik": info_c1["cv_loglik"]}
    return out, cv


# -------------------------------------------------------------------- one draw
def run_one(payload) -> Tuple[int, int, List[Dict[str, Any]]]:
    d, k_nominal, m = payload
    Z, y, plat, lang, cell, speaker = (_G["Z"], _G["y"], _G["plat"], _G["lang"],
                                       _G["cell"], _G["speaker"])
    graph = _G["graph"]
    n_eval = 0
    rows: List[Dict[str, Any]] = []
    if len(np.unique(speaker)) < 4:
        return d, k_nominal, rows

    lab_pos, _info = draw_labels(cell, m, draw_seed(d), cap_half=True)
    masks = mask_sets(speaker, cell, lab_pos, m, draw_seed(d))
    lab_strict, s_e = strict_labels(speaker, cell, m, STRICT_SEED + d)
    masks["strict_draw"] = np.where(s_e)[0]
    lab_sets = {"seen": lab_pos, "unseen": lab_pos,
                "strict_draw": np.where(lab_strict)[0]}

    for mask_name in MASKS:
        ev = masks[mask_name]
        lab_used = lab_sets[mask_name]
        if ev.size < 50 or lab_used.size < 50:
            continue
        if len(np.unique(y[lab_used])) < 2:
            continue
        folds = (stratified_folds(y[lab_used]) if mask_name == "seen"
                 else speaker_grouped_folds(speaker[lab_used], y[lab_used]))
        if len(np.unique(folds)) < 2:
            continue
        try:
            scores, cv = fit_methods(Z, y, plat, lang, cell, speaker, lab_used,
                                     folds, graph, ev.size)
        except Exception as exc:                       # pragma: no cover
            rows.append({"error": f"draw {d} k={k_nominal} {mask_name}: "
                                  f"{type(exc).__name__}: {exc}"})
            continue
        ev_scores = {k: v[ev] for k, v in scores.items()}
        outdir = (_G["draw_root"] / f"k{k_nominal}" / mask_name)
        save_draw_scores(outdir, d, y[ev], ev_scores, speaker[ev],
                         extra={"k": int(k_nominal), "mask": mask_name,
                                "n_labeled": int(lab_used.size), "cv": cv})
        for name in METHODS:
            rows.append(summarize(y[ev], ev_scores[name], plat[ev], extra={
                "protocol": "dev_v3", "region": "dev", "backbone": _G["backbone"],
                "spec": _G["spec"], "method": name, "k": int(k_nominal),
                "m_per_cell": int(m), "draw": int(d), "mask": mask_name,
                "cv_folds": "stratified" if mask_name == "seen" else "speaker_group",
                "n_labeled": int(lab_used.size), "n_eval": int(ev.size),
                "n_speakers_in_L": int(len(set(speaker[lab_used].tolist()))),
                "cv_pick": json.dumps(cv[name], sort_keys=True),
            }))
        n_eval += ev.size
    return d, k_nominal, rows


def _worker(payload):
    try:
        return run_one(payload)
    except Exception as exc:                           # pragma: no cover
        return payload[0], payload[1], [{"error": f"draw {payload[0]}: {exc}"}]


# ------------------------------------------------------------------- reporting
def intervals_table(draw_root: Path, budgets: Sequence[int] = tuple(BUDGETS),
                    pairs: Sequence[Tuple[str, str]] = PAIRS,
                    masks: Sequence[str] = MASKS, jobs: int = 1) -> pd.DataFrame:
    """Old rule vs (a) vs (b) for every pair, mask and budget."""
    todo = [(str(draw_root), int(k), str(mask), str(a), str(b))
            for k in budgets for mask in masks for a, b in pairs]
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


def _interval_row(job) -> Optional[Dict[str, Any]]:
    """One (budget, mask, pair) cell.  A module-level function so it can be pooled."""
    draw_root, k, mask, a, b = job
    per = load_draw_scores(Path(draw_root) / f"k{k}" / mask, a, b)
    if len(per) < 5:
        return None
    got = two_intervals(per)
    return {
        "k": int(k), "mask": mask, "method_a": a, "method_b": b,
        "n_draws": got["n_draws"], "delta": got["mean"],
        "old_lo": got["old_lo"], "old_hi": got["old_hi"],
        "old_contains_zero": got["old_contains_zero"],
        "a_lo": got["a_lo"], "a_hi": got["a_hi"],
        "b_lo": got["b_lo"], "b_hi": got["b_hi"],
        "b_contains_zero": got["b_contains_zero"],
        "effect_points": abs(got["mean"]),
        "meets_2pt_rule": bool(abs(got["mean"]) >= 2.0 and got["b_excludes_zero"]),
    }


def verdicts(df: pd.DataFrame, iv: pd.DataFrame) -> Dict[str, Any]:
    """The U1 verdicts, recomputed under the (b) rule."""
    flags: Dict[str, Any] = {}
    tab = df.groupby(["method", "mask"])["pooled_eer"].mean().to_dict()
    for method in sorted(df["method"].unique()):
        if (method, "seen") not in tab or (method, "unseen") not in tab:
            continue
        seen = float(tab[(method, "seen")])
        unseen = float(tab[(method, "unseen")])
        flags.setdefault("speaker_ratio", {})[method] = {
            "seen": seen, "unseen": unseen, "ratio": unseen / max(seen, 1e-9)}
    flags["SPEAKER_GAP"] = bool(any(v["ratio"] >= 1.5
                                    for v in flags.get("speaker_ratio", {}).values()))
    flags["SPEAKER_GAP_methods"] = sorted(
        m for m, v in flags.get("speaker_ratio", {}).items() if v["ratio"] >= 1.5)
    row = iv[(iv["method_a"] == "B9") & (iv["method_b"] == "B1")
             & (iv["mask"] == "unseen") & (iv["k"] == 290)]
    if not row.empty:
        r = row.iloc[0]
        flags["B9_minus_B1_unseen"] = float(r["delta"])
        flags["B9_minus_B1_unseen_ci_b"] = [float(r["b_lo"]), float(r["b_hi"])]
        flags["B9_minus_B1_unseen_ci_old"] = [float(r["old_lo"]), float(r["old_hi"])]
        flags["LP_GAIN_IS_SPEAKER_LOCAL"] = bool(not (r["b_hi"] < 0.0))
    row = iv[(iv["method_a"] == "C1") & (iv["method_b"] == "B1")
             & (iv["mask"] == "strict_draw") & (iv["k"] == 290)]
    if not row.empty:
        r = row.iloc[0]
        flags["G4_C1_minus_B1_strict_draw"] = {
            "delta": float(r["delta"]), "old": [float(r["old_lo"]), float(r["old_hi"])],
            "b": [float(r["b_lo"]), float(r["b_hi"])],
            "C1_beats_B1": bool(r["b_hi"] < 0.0)}
    return flags


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 U1' on the v3 dev region")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--split", default=R.SPLIT_V3)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--budgets", default="290,100")
    parser.add_argument("--interval-jobs", type=int, default=16)
    parser.add_argument("--intervals-only", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    budgets = [int(x) for x in str(args.budgets).split(",") if x.strip()]
    draw_root = root / "results" / "u1p_draws"
    out_csv = root / "results" / "u1p_speaker_split_v3.csv"

    if not args.intervals_only:
        data_dir = data_root(args.data)
        sub, F = R.load_region(root, data_dir, args.backbone, args.spec, args.split)
        transform, Z = w6.fit_preprocess(F)
        y = sub["y"].to_numpy()
        plat = sub["platform"].to_numpy()
        lang = sub["language"].to_numpy()
        cell = sub["cell"].to_numpy()
        speaker = sub["speaker"].to_numpy()
        print(f"v3 dev: clips={len(sub)} speakers={len(np.unique(speaker))} "
              f"Z={Z.shape} ({args.backbone}/{args.spec})", flush=True)
        print("building the kNN graph once for every draw ...", flush=True)
        graph = knn_graph(Z, k=max(B9_KS))
        _G.update({"Z": Z, "y": y, "plat": plat, "lang": lang, "cell": cell,
                   "speaker": speaker, "graph": graph, "draw_root": draw_root,
                   "backbone": args.backbone, "spec": args.spec})
        payloads = [(d, k, BUDGETS[k]) for d in range(int(args.draws)) for k in budgets]
        rows: List[Dict[str, Any]] = []
        errors: List[str] = []
        if args.jobs > 1:
            import multiprocessing as mp

            ctx = (mp.get_context("fork") if sys.platform != "win32"
                   else mp.get_context("spawn"))
            with ctx.Pool(args.jobs) as pool:
                for i, (_d, _k, out) in enumerate(
                        pool.imap_unordered(_worker, payloads, chunksize=1), 1):
                    for r in out:
                        if "error" in r:
                            errors.append(str(r["error"]))
                        else:
                            rows.append(r)
                    if i % 20 == 0:
                        print(f"  {i}/{len(payloads)} draw-budget units "
                              f"({len(rows)} rows, {len(errors)} errors)", flush=True)
        else:
            for pl in payloads:
                _d, _k, out = _worker(pl)
                for r in out:
                    if "error" in r:
                        errors.append(str(r["error"]))
                    else:
                        rows.append(r)
        if errors:
            print(f"errors: {len(errors)}; first: {errors[:3]}")
            (root / "results" / "u1p_errors.txt").write_text(
                "\n".join(errors) + "\n", encoding="utf-8")
        df = pd.DataFrame(rows)
        (root / "results").mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    df = pd.read_csv(out_csv)
    piv = df.groupby(["k", "method", "mask"])["pooled_eer"].mean().unstack()
    print("\npooled EER (mean over draws)\n" + piv.round(3).to_string())
    iv = intervals_table(draw_root, budgets, jobs=args.interval_jobs)
    iv.to_csv(root / "results" / "u1p_intervals.csv", index=False)
    print("\nintervals (delta = a - b; negative means a is better)\n"
          + iv.round(3).to_string(index=False))
    flags = verdicts(df, iv)
    with open(root / "results" / "u1p_verdicts.json", "w", encoding="utf-8") as fh:
        json.dump(flags, fh, indent=2, sort_keys=True)
    print("\n" + json.dumps(flags, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

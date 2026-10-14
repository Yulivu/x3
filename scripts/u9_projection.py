#!/usr/bin/env python3
"""U9 [URGENT] - nuisance-subspace projection of speaker / platform identity (N6).

Speaker and platform identity are metadata on every clip, genuine or fake, so the
directions that predict them can be removed before the probe is fitted.  The basis
is an LDA whose classes are the speaker ids; the platform basis is made orthogonal
to it, and ``(m_s, m_p)`` says how many directions of each are removed.

Two scopes, always labelled in the output:

* ``inductive`` - the basis is fitted inside ``L`` (on the labelled speakers only),
  so an evaluation speaker's identity can never influence the projection;
* ``transductive`` - the basis is fitted on every dev clip's metadata (no labels).

``(m_s, m_p)`` is chosen inside ``L`` with speaker-grouped folds (V-3) through the
15-point grid below; ``seen`` also gets its own CV-selected number so G18's
"seen no worse than +0.3" can be read.  The full grid is additionally reported on
``strict_draw`` (informational: choosing the best cell afterwards is optimistic,
which the report says out loud).

Writes ``results/u9_projection.csv``, ``results/u9_draws/``,
``results/u9_intervals.csv`` and ``results/u9_gate.json``.
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
import u1p_speaker_split_v3 as u1p  # noqa: E402
import w6_baselines as w6  # noqa: E402
from src import regions as R  # noqa: E402
from src.baselines import fit_logreg_cv  # noqa: E402
from src.cv import select_config, speaker_grouped_folds, stratified_folds  # noqa: E402
from src.data import data_root  # noqa: E402
from src.draws import draw_labels, draw_seed  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.paired import load_draw_scores, save_draw_scores, two_intervals  # noqa: E402
from src.proj import lda_directions, orthogonalise_against, residualise  # noqa: E402

N_DRAWS = 100
M_CELL = 10
MASKS = ("seen", "unseen", "strict_draw")
SCOPES = ("inductive", "transductive")
ABBR = {"inductive": "ind", "transductive": "tra"}
METHOD_NAMES = {"inductive": "PROJ-ind", "transductive": "PROJ-tra"}
# 15 configurations (hard rule 6): the two marginals plus joint cells
PROJ_GRID: Tuple[Tuple[int, int], ...] = (
    (0, 0), (4, 0), (8, 0), (16, 0), (32, 0), (64, 0),
    (0, 2), (0, 4), (0, 6),
    (8, 2), (16, 4), (32, 4), (64, 6), (16, 2), (32, 2),
)
MAX_MS, MAX_MP = 64, 6
STRICT_SEED = 777000

_G: Dict[str, Any] = {}

__all__ = ["full_bases", "slice_basis", "basis_for", "project_unit", "run_one",
           "intervals_table", "gate_g18", "main"]


def full_bases(Z: np.ndarray, speaker: np.ndarray, platform: np.ndarray,
               fit_idx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """``(speaker basis, platform basis)`` with the platform part made orthogonal.

    Computed once per fitting set and then sliced, because every LDA of >= 64
    classes is the same expensive fit no matter how many directions are kept.
    """
    idx = slice(None) if fit_idx is None else np.asarray(fit_idx)
    X = np.asarray(Z, float)[idx]
    spk = np.asarray(speaker)[idx]
    plat = np.asarray(platform)[idx]
    W_s = lda_directions(X, spk, MAX_MS)
    W_p = np.zeros((0, X.shape[1]))
    if len(np.unique(plat)) > 1:
        W_p = orthogonalise_against(lda_directions(X, plat, MAX_MP), W_s)
    return W_s, W_p


def slice_basis(full: Tuple[np.ndarray, np.ndarray], cfg: Tuple[int, int]) -> np.ndarray:
    """The first ``m_s`` speaker and ``m_p`` platform directions (both orthonormal)."""
    W_s, W_p = full
    m_s, m_p = int(cfg[0]), int(cfg[1])
    parts = []
    if m_s > 0 and W_s.size:
        parts.append(W_s[:m_s])
    if m_p > 0 and W_p.size:
        parts.append(W_p[:m_p])
    if not parts:
        return np.zeros((0, W_s.shape[1] if W_s.ndim == 2 else 0))
    return np.vstack(parts)


def basis_for(Z: np.ndarray, speaker: np.ndarray, platform: np.ndarray,
              cfg: Tuple[int, int], fit_idx: Optional[np.ndarray] = None,
              full: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """The joint ``(m_s, m_p)`` subspace for one configuration."""
    if full is None:
        full = full_bases(Z, speaker, platform, fit_idx)
    return slice_basis(full, cfg)


def project_unit(Z, y, speaker, platform, lab_used, ev, folds, cfg_grid=PROJ_GRID,
                 full_cache: Optional[Dict[str, Any]] = None
                 ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """B1 plus both projection scopes for one (draw, mask)."""
    n = len(y)
    lab_used = np.asarray(lab_used)
    lab_mask = np.zeros(n, bool)
    lab_mask[lab_used] = True
    model_b1, info = fit_logreg_cv(Z[lab_used], y[lab_used], folds=folds)
    out: Dict[str, np.ndarray] = {"B1": model_b1.decision_function(Z)}
    cv: Dict[str, Any] = {"B1": {"C": info["C"]}}
    C = float(info["C"])

    for scope in SCOPES:
        transductive = scope == "transductive"

        def _basis(cfg, tr_mask=None):
            if transductive:
                full = (full_cache or {}).get("all")
                if full is None:
                    full = full_bases(Z, speaker, platform, None)
                    if full_cache is not None:
                        full_cache["all"] = full
                return slice_basis(full, cfg)
            key = ("indL" if tr_mask is None
                   else ("indF", bytes(np.asarray(tr_mask, bool).tobytes())))
            full = (full_cache or {}).get(key)
            if full is None:
                fit_idx = lab_used if tr_mask is None else lab_used[tr_mask]
                full = full_bases(Z, speaker, platform, fit_idx)
                if full_cache is not None:
                    full_cache[key] = full
            return slice_basis(full, cfg)

        def _evaluate(cfg, tr_mask, te_mask):
            from sklearn.linear_model import LogisticRegression

            W = _basis(cfg, tr_mask)
            Xtr = residualise(Z[lab_used[tr_mask]], W)
            Xte = residualise(Z[lab_used[te_mask]], W)
            m = LogisticRegression(C=C, max_iter=5000).fit(Xtr, y[lab_used[tr_mask]])
            return m.decision_function(Xte)

        cfg, info_cv = select_config(list(cfg_grid), _evaluate, y[lab_used], folds)
        W = _basis(cfg)
        X = residualise(Z, W)
        m = None
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(C=C, max_iter=5000).fit(X[lab_used], y[lab_used])
        out[METHOD_NAMES[scope]] = m.decision_function(X)
        cv[METHOD_NAMES[scope]] = {"m_s": int(cfg[0]), "m_p": int(cfg[1]),
                                   "dim_removed": int(W.shape[0]),
                                   "cv_loglik": info_cv["cv_loglik"]}
    return out, cv


def run_one(payload) -> Tuple[int, List[Dict[str, Any]]]:
    d = payload[0]
    Z, y, plat, speaker = _G["Z"], _G["y"], _G["plat"], _G["speaker"]
    cell = _G["cell"]
    n = len(y)
    rows: List[Dict[str, Any]] = []

    lab_pos, _info = draw_labels(cell, M_CELL, draw_seed(d), cap_half=True)
    masks = u1p.mask_sets(speaker, cell, lab_pos, M_CELL, draw_seed(d))
    lab_strict, s_e = u1p.strict_labels(speaker, cell, M_CELL, STRICT_SEED + d)
    masks["strict_draw"] = np.where(s_e)[0]
    lab_sets = {"seen": lab_pos, "unseen": lab_pos, "strict_draw": np.where(lab_strict)[0]}
    # the transductive basis is a metadata-only LDA over the whole region, so one
    # process computes it once and every draw/mask reuses it
    if "full_all" not in _G:
        _G["full_all"] = full_bases(Z, speaker, plat, None)
    cache: Dict[str, Any] = {"all": _G["full_all"]}

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
            scores, cv = project_unit(Z, y, speaker, plat, lab_used, ev, folds,
                                      full_cache=cache)
        except Exception as exc:                        # pragma: no cover
            rows.append({"error": f"d={d} {mask_name}: {type(exc).__name__}: {exc}"})
            continue
        ev_scores = {k: v[ev] for k, v in scores.items()}
        save_draw_scores(_G["draw_root"] / mask_name, d, y[ev], ev_scores, speaker[ev],
                         extra={"mask": mask_name, "cv": cv})
        for name, sc in ev_scores.items():
            rows.append(summarize(y[ev], sc, plat[ev], extra={
                "protocol": "dev_v3_proj", "region": "dev", "backbone": _G["backbone"],
                "spec": _G["spec"], "method": name, "k": int(lab_used.size),
                "draw": int(d), "mask": mask_name,
                "cv_folds": "stratified" if mask_name == "seen" else "speaker_group",
                "n_labeled": int(lab_used.size), "n_eval": int(ev.size),
                "m_s": int(cv[name].get("m_s", 0)), "m_p": int(cv[name].get("m_p", 0)),
                "dim_removed": int(cv[name].get("dim_removed", 0)),
                "cv_pick": json.dumps(cv[name], sort_keys=True),
            }))

        # the full grid on strict_draw only: reporting the curve, not selecting on it
        if mask_name == "strict_draw":
            if "indL" not in cache:
                cache["indL"] = full_bases(Z, speaker, plat, lab_used)
            for scope in SCOPES:
                full = cache["all"] if scope == "transductive" else cache["indL"]
                for (m_s, m_p) in PROJ_GRID:
                    W = slice_basis(full, (m_s, m_p))
                    X = residualise(Z, W)
                    from sklearn.linear_model import LogisticRegression

                    mm = LogisticRegression(C=float(cv["B1"]["C"]), max_iter=5000).fit(
                        X[lab_used], y[lab_used])
                    sc = mm.decision_function(X)[ev]
                    rows.append(summarize(y[ev], sc, plat[ev], extra={
                        "protocol": "dev_v3_proj", "region": "dev",
                        "backbone": _G["backbone"], "spec": _G["spec"],
                        "method": f"GRID-{ABBR[scope]}", "k": int(lab_used.size),
                        "draw": int(d), "mask": mask_name, "scope": scope,
                        "m_s": int(m_s), "m_p": int(m_p), "dim_removed": int(W.shape[0]),
                        "cv_folds": "speaker_group", "n_labeled": int(lab_used.size),
                        "n_eval": int(ev.size), "cv_pick": "{}",
                    }))
    return d, rows


def _worker(payload):
    try:
        return run_one(payload)
    except Exception as exc:                            # pragma: no cover
        return payload[0], [{"error": f"d={payload[0]}: {exc}"}]


def _interval_row(job) -> Optional[Dict[str, Any]]:
    draw_root, mask, a, b = job
    per = load_draw_scores(Path(draw_root) / mask, a, b)
    if len(per) < 5:
        return None
    got = two_intervals(per)
    return {"mask": mask, "method_a": a, "method_b": b, "n_draws": got["n_draws"],
            "delta": got["mean"], "old_lo": got["old_lo"], "old_hi": got["old_hi"],
            "a_lo": got["a_lo"], "a_hi": got["a_hi"],
            "b_lo": got["b_lo"], "b_hi": got["b_hi"],
            "b_contains_zero": got["b_contains_zero"],
            "effect_points": abs(got["mean"]),
            "meets_2pt_rule": bool(abs(got["mean"]) >= 2.0 and got["b_excludes_zero"])}


def intervals_table(draw_root: Path, masks: Sequence[str] = MASKS,
                    jobs: int = 1) -> pd.DataFrame:
    todo = [(str(draw_root), str(mask), name, "B1")
            for mask in masks for name in METHOD_NAMES.values()]
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


def gate_g18(df: pd.DataFrame, iv: pd.DataFrame, grid: pd.DataFrame,
             seen_tolerance: float = 0.3, population_mask: str = "strict_draw"
             ) -> Dict[str, Any]:
    """G18/N6 on the CV-selected variants, plus the (optimistic) best-grid cell."""
    means = df.groupby(["mask", "method"])["pooled_eer"].mean().to_dict()
    out: Dict[str, Any] = {"cv_selected": {f"{k[0]}|{k[1]}": float(v)
                                           for k, v in means.items()}}
    best = None
    for name in METHOD_NAMES.values():
        strict = means.get((population_mask, name))
        seen = means.get(("seen", name))
        if strict is None:
            continue
        row = iv[(iv["mask"] == population_mask) & (iv["method_a"] == name)
                 & (iv["method_b"] == "B1")]
        if row.empty:
            continue
        r = row.iloc[0]
        cand = {"method": name, "strict_draw": float(strict),
                "b1_strict_draw": float(means.get((population_mask, "B1"), float("nan"))),
                "delta_vs_b1": float(strict - means[(population_mask, "B1")]),
                "seen": None if seen is None else float(seen),
                "seen_delta": (None if seen is None
                               else float(seen - means.get(("seen", "B1"), float("nan")))),
                "interval_b": [float(r["b_lo"]), float(r["b_hi"])],
                "b_excludes_zero": bool(not r["b_contains_zero"])}
        if best is None or cand["strict_draw"] < best["strict_draw"]:
            best = cand
    if best is None:
        out["verdict"] = "PROJ_NOT_RUN"
        return out
    out.update(best)
    out["effect_ge_2"] = bool(out["delta_vs_b1"] <= -2.0)
    out["seen_within_tolerance"] = bool(out["seen_delta"] is not None
                                        and out["seen_delta"] <= seen_tolerance)
    out["verdict"] = ("PROJ_PASS" if out["effect_ge_2"] and out["b_excludes_zero"]
                      and out["seen_within_tolerance"] else "PROJ_FAIL")
    if not grid.empty:
        g = grid.groupby(["scope", "m_s", "m_p"])["pooled_eer"].mean().reset_index()
        g = g.sort_values("pooled_eer").head(5)
        out["best_grid_cells"] = g.to_dict(orient="records")
        out["best_grid_note"] = ("choosing the best cell after seeing strict_draw is "
                                 "optimistic; the CV-selected number above is the "
                                 "pre-registered comparison")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 U9 projection")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--split", default=R.SPLIT_V3)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--interval-jobs", type=int, default=8)
    parser.add_argument("--intervals-only", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    draw_root = root / "results" / "u9_draws"
    out_csv = root / "results" / "u9_projection.csv"

    if not args.intervals_only:
        data_dir = data_root(args.data)
        sub, F = R.load_region(root, data_dir, args.backbone, args.spec, args.split)
        transform, Z = w6.fit_preprocess(F)
        _G.update({"Z": Z, "y": sub["y"].to_numpy(), "plat": sub["platform"].to_numpy(),
                   "cell": sub["cell"].to_numpy(),
                   "speaker": sub["speaker"].to_numpy(), "draw_root": draw_root,
                   "backbone": args.backbone, "spec": args.spec})
        print(f"v3 dev: clips={len(sub)} speakers={len(np.unique(_G['speaker']))}", flush=True)
        payloads = [(d,) for d in range(int(args.draws))]
        rows: List[Dict[str, Any]] = []
        errors: List[str] = []
        if args.jobs > 1:
            import multiprocessing as mp

            ctx = (mp.get_context("fork") if sys.platform != "win32"
                   else mp.get_context("spawn"))
            with ctx.Pool(args.jobs) as pool:
                for i, (_d, out) in enumerate(
                        pool.imap_unordered(_worker, payloads, chunksize=1), 1):
                    for r in out:
                        if "error" in r:
                            errors.append(str(r["error"]))
                        else:
                            rows.append(r)
                    if i % 10 == 0:
                        print(f"  {i}/{len(payloads)} draws ({len(rows)} rows, "
                              f"{len(errors)} errors)", flush=True)
        else:
            for pl in payloads:
                _d, out = _worker(pl)
                for r in out:
                    if "error" in r:
                        errors.append(str(r["error"]))
                    else:
                        rows.append(r)
        if errors:
            print(f"errors: {len(errors)}; first: {errors[:3]}")
            (root / "results" / "u9_errors.txt").write_text(
                "\n".join(errors) + "\n", encoding="utf-8")
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.read_csv(out_csv)
    picked = df[df["method"].isin(list(METHOD_NAMES.values()))]
    print("\nCV-selected projection, pooled EER\n"
          + picked.pivot_table(index="method", columns="mask", values="pooled_eer",
                               aggfunc="mean").round(3).to_string())
    b1 = df[df["method"] == "B1"]
    print("\nB1\n" + b1.pivot_table(index="method", columns="mask", values="pooled_eer",
                                    aggfunc="mean").round(3).to_string())
    grid = df[df["method"].str.startswith("GRID-", na=False)]
    if not grid.empty:
        best = (grid.groupby(["scope", "m_s", "m_p"])["pooled_eer"].mean()
                .reset_index().sort_values("pooled_eer").head(10))
        print("\nbest grid cells on strict_draw (informational)\n" + best.round(3).to_string(index=False))
    iv = intervals_table(draw_root, jobs=args.interval_jobs)
    iv.to_csv(root / "results" / "u9_intervals.csv", index=False)
    print("\nintervals\n" + iv.round(3).to_string(index=False))
    gate = gate_g18(df, iv, grid)
    with open(root / "results" / "u9_gate.json", "w", encoding="utf-8") as fh:
        json.dump(gate, fh, indent=2, sort_keys=True)
    print("\n" + json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

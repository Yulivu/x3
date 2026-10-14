#!/usr/bin/env python3
"""U10 - backbone-side cheap gains: spec S5 vs S3, and the cross-backbone ensemble.

Both halves are CPU work on the cached feature blocks:

* **S5 vs S3** - S5 is S3 plus the zero-shot logit column.  Same probe, same folds,
  same draws, so the difference is the extra column and nothing else.
* **linear ensemble** - with three backbones extracted (xls-r-2b, xls-r-1b, mms-1b)
  each member's score is Platt-calibrated on ``L`` and the calibrated logits are
  averaged.  Calibration uses only ``L``; the evaluation clips never enter a fitted
  parameter.

Every number is V-3 compliant: speaker-grouped folds on ``strict_draw``, the shared
stratified folds on ``seen``, ``C`` chosen inside ``L``.

Writes ``results/u10_fusion.csv``, ``results/u10_draws/`` and
``results/u10_intervals.csv``.
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
from src.cv import speaker_grouped_folds, stratified_folds  # noqa: E402
from src.data import data_root  # noqa: E402
from src.draws import draw_labels, draw_seed  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.paired import load_draw_scores, two_intervals  # noqa: E402

N_DRAWS = 100
M_CELL = 10
BACKBONES = ("xls-r-2b", "xls-r-1b", "mms-1b")
SPECS = ("S3", "S5")
MASKS = ("seen", "strict_draw")
STRICT_SEED = 777000

_G: Dict[str, Any] = {}

__all__ = ["platt", "fuse_members", "run_one", "fuse_draw", "intervals_table", "main"]


def platt(score: Sequence[float], y: Sequence[float], l2: float = 1e-3
          ) -> Tuple[float, float]:
    """Two-parameter calibration ``a * score + b`` fitted on labelled clips only."""
    from sklearn.linear_model import LogisticRegression

    s = np.asarray(score, float).reshape(-1, 1)
    yy = np.asarray(y, int)
    if len(np.unique(yy)) < 2 or s.shape[0] < 4:
        return 1.0, 0.0
    lr = LogisticRegression(C=1.0 / max(l2, 1e-6), max_iter=2000).fit(s, yy)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def fuse_members(members: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]
                 ) -> np.ndarray:
    """Average of the Platt-calibrated logits; ``members`` = (eval, labelled, y)."""
    parts = []
    for s_eval, s_lab, y_lab in members:
        a, b = platt(s_lab, y_lab)
        parts.append(a * np.asarray(s_eval, float) + b)
    if not parts:
        raise ValueError("no members to fuse")
    return np.mean(parts, axis=0)


def _masks_for(d: int, Z, y, cell, speaker) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    lab_pos, _ = draw_labels(cell, M_CELL, draw_seed(d), cap_half=True)
    masks = u1p.mask_sets(speaker, cell, lab_pos, M_CELL, draw_seed(d))
    lab_strict, s_e = u1p.strict_labels(speaker, cell, M_CELL, STRICT_SEED + d)
    masks["strict_draw"] = np.where(s_e)[0]
    return {"seen": (lab_pos, masks["seen"]),
            "strict_draw": (np.where(lab_strict)[0], masks["strict_draw"])}


def run_one(payload) -> Tuple[int, str, List[Dict[str, Any]]]:
    d, mask_name = payload
    Z, y, plat, cell, speaker = (_G["Z"], _G["y"], _G["plat"], _G["cell"], _G["speaker"])
    rows: List[Dict[str, Any]] = []
    lab_used, ev = _G["masks"][d][mask_name]
    if ev.size < 50 or lab_used.size < 50 or len(np.unique(y[lab_used])) < 2:
        return d, mask_name, rows
    folds = (stratified_folds(y[lab_used]) if mask_name == "seen"
             else speaker_grouped_folds(speaker[lab_used], y[lab_used]))
    model, info = fit_logreg_cv(Z[lab_used], y[lab_used], folds=folds)
    scores_all = model.decision_function(Z)
    scores = scores_all[ev]
    member = f"{_G['backbone']}|{_G['spec']}"
    out_path = _G["draw_root"] / member / mask_name / f"draw{d:04d}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                        y=y[ev].astype(np.int8), speaker=np.asarray(speaker[ev]).astype(str),
                        plat=np.asarray(plat[ev]).astype(str),
                        s_eval=scores, s_lab=scores_all[lab_used],
                        y_lab=y[lab_used].astype(np.int8),
                        meta=np.array(json.dumps({"C": info["C"]})))
    rows.append(summarize(y[ev], scores, plat[ev], extra={
        "protocol": "dev_v3_fusion", "region": "dev", "backbone": _G["backbone"],
        "spec": _G["spec"], "method": member, "k": int(lab_used.size), "draw": int(d),
        "mask": mask_name, "cv_folds": "stratified" if mask_name == "seen" else "speaker_group",
        "n_labeled": int(lab_used.size), "n_eval": int(ev.size), "C": info["C"]}))
    return d, mask_name, rows


def _worker(payload):
    try:
        return run_one(payload)
    except Exception as exc:                            # pragma: no cover
        return payload[0], payload[1], [{"error": f"d={payload[0]} {payload[1]}: {exc}"}]


def load_member(draw_root: Path, member: str, mask: str, draw: int
                ) -> Optional[Dict[str, np.ndarray]]:
    path = Path(draw_root) / member / mask / f"draw{draw:04d}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return {"y": z["y"], "speaker": z["speaker"], "s_eval": z["s_eval"],
                "plat": z["plat"] if "plat" in z.files else None,
                "s_lab": z["s_lab"], "y_lab": z["y_lab"]}


def fuse_draw(draw_root: Path, members: Sequence[str], mask: str, draw: int
              ) -> Optional[Dict[str, np.ndarray]]:
    loaded = [load_member(draw_root, m, mask, draw) for m in members]
    if any(m is None for m in loaded):
        return None
    fused = fuse_members([(m["s_eval"], m["s_lab"], m["y_lab"]) for m in loaded])
    return {"y": loaded[0]["y"], "speaker": loaded[0]["speaker"], "sa": fused}


def variant_defs(backbones: Sequence[str] = BACKBONES,
                 specs: Sequence[str] = SPECS) -> List[Tuple[str, List[str]]]:
    """``(name, members)``: every single probe, then the ensembles."""
    singles = [f"{b}|{s}" for b in backbones for s in specs]
    out: List[Tuple[str, List[str]]] = [(m, [m]) for m in singles]
    for spec in specs:
        out.append((f"ENS{len(backbones)}-{spec}",
                    [f"{b}|{spec}" for b in backbones]))
    out.append((f"ENS{len(backbones)}-mix", singles))
    return out


def fused_rows(draw_root: Path, backbones: Sequence[str] = BACKBONES,
               specs: Sequence[str] = SPECS, masks: Sequence[str] = MASKS,
               draws: int = N_DRAWS) -> pd.DataFrame:
    """Per-draw pooled EER and AUC for every single probe and every ensemble."""
    from src.metrics3 import auc, eer

    rows: List[Dict[str, Any]] = []
    for mask in masks:
        for d in range(int(draws)):
            for name, members in variant_defs(backbones, specs):
                loaded = [load_member(draw_root, m, mask, d) for m in members]
                if any(x is None for x in loaded):
                    continue
                fused = fuse_members([(x["s_eval"], x["s_lab"], x["y_lab"])
                                      for x in loaded])
                y = loaded[0]["y"]
                rows.append({"variant": name, "members": ",".join(members),
                             "mask": mask, "draw": int(d),
                             "pooled_eer": float(eer(y, fused)),
                             "auc": float(auc(y, fused)), "n_eval": int(len(fused)),
                             "n_members": len(members)})
    return pd.DataFrame(rows)


def _interval_row(job) -> Optional[Dict[str, Any]]:
    draw_root, mask, a, b = job
    per = load_draw_scores(Path(draw_root) / "fused" / mask, a, b)
    if len(per) < 5:
        return None
    got = two_intervals(per)
    return {"mask": mask, "method_a": a, "method_b": b, "n_draws": got["n_draws"],
            "delta": got["mean"], "old_lo": got["old_lo"], "old_hi": got["old_hi"],
            "b_lo": got["b_lo"], "b_hi": got["b_hi"],
            "b_contains_zero": got["b_contains_zero"],
            "effect_points": abs(got["mean"]),
            "meets_2pt_rule": bool(abs(got["mean"]) >= 2.0 and got["b_excludes_zero"])}


def intervals_table(draw_root: Path, pairs: Sequence[Tuple[str, str]],
                    masks: Sequence[str] = MASKS, jobs: int = 1) -> pd.DataFrame:
    todo = [(str(draw_root), str(mask), a, b) for mask in masks for a, b in pairs]
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


def persist_fused(draw_root: Path, backbones: Sequence[str] = BACKBONES,
                  specs: Sequence[str] = SPECS, masks: Sequence[str] = MASKS,
                  draws: int = N_DRAWS) -> List[str]:
    """Write singles + ensembles in the U1'-style draw archive for the intervals."""
    from src.paired import save_draw_scores

    written: List[str] = []
    for mask in masks:
        for d in range(int(draws)):
            scores: Dict[str, np.ndarray] = {}
            y = speaker = None
            for name, members in variant_defs(backbones, specs):
                loaded = [load_member(draw_root, m, mask, d) for m in members]
                if any(x is None for x in loaded):
                    continue
                y, speaker = loaded[0]["y"], loaded[0]["speaker"]
                scores[name] = (loaded[0]["s_eval"] if len(members) == 1
                                else fuse_members([(x["s_eval"], x["s_lab"], x["y_lab"])
                                                   for x in loaded]))
            if not scores:
                continue
            save_draw_scores(draw_root / "fused" / mask, d, y, scores, speaker)
            written.append(f"{mask}:{d}")
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 U10 fusion")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbones", default=",".join(BACKBONES))
    parser.add_argument("--specs", default=",".join(SPECS))
    parser.add_argument("--split", default=R.SPLIT_V3)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--interval-jobs", type=int, default=8)
    parser.add_argument("--members-only", action="store_true",
                        help="only fit the single (backbone, spec) probes")
    parser.add_argument("--fuse-only", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    data_dir = data_root(args.data)
    draw_root = root / "results" / "u10_draws"
    backbones = [b for b in str(args.backbones).split(",") if b.strip()]
    specs = [s for s in str(args.specs).split(",") if s.strip()]

    if not args.fuse_only:
        rows: List[Dict[str, Any]] = []
        errors: List[str] = []
        for backbone in backbones:
            for spec in specs:
                sub, F = R.load_region(root, data_dir, backbone, spec, args.split)
                transform, Z = w6.fit_preprocess(F)
                y = sub["y"].to_numpy()
                cell = sub["cell"].to_numpy()
                speaker = sub["speaker"].to_numpy()
                _G.update({"Z": Z, "y": y, "plat": sub["platform"].to_numpy(),
                           "cell": cell, "speaker": speaker,
                           "draw_root": draw_root, "backbone": backbone, "spec": spec,
                           "masks": {d: _masks_for(d, Z, y, cell, speaker)
                                     for d in range(int(args.draws))}})
                print(f"{backbone}/{spec}: clips={len(sub)} Z={Z.shape} "
                      f"zero_col={spec == 'S5'}", flush=True)
                payloads = [(d, m) for d in range(int(args.draws)) for m in MASKS]
                if args.jobs > 1:
                    import multiprocessing as mp

                    ctx = (mp.get_context("fork") if sys.platform != "win32"
                           else mp.get_context("spawn"))
                    with ctx.Pool(int(args.jobs)) as pool:
                        for _d, _m, out in pool.imap_unordered(_worker, payloads,
                                                               chunksize=1):
                            for r in out:
                                if "error" in r:
                                    errors.append(str(r["error"]))
                                else:
                                    rows.append(r)
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
        pd.DataFrame(rows).to_csv(root / "results" / "u10_fusion.csv", index=False)
        if args.members_only:
            return 0

    try:
        df = pd.read_csv(root / "results" / "u10_fusion.csv")
    except pd.errors.EmptyDataError:
        print("no member rows were produced; nothing to fuse", file=sys.stderr)
        return 1
    if df.empty:
        print("no member rows were produced; nothing to fuse", file=sys.stderr)
        return 1
    print("\nmember pooled EER\n"
          + df.pivot_table(index="method", columns="mask", values="pooled_eer",
                           aggfunc="mean").round(3).to_string())
    written = persist_fused(draw_root, backbones, specs)
    print(f"\nfused archives: {len(written)} draw/mask entries")
    rows = fused_rows(draw_root, backbones, specs)
    if not rows.empty:
        rows.to_csv(root / "results" / "u10_variants.csv", index=False)
        print("\nensembles and the S5/S3 contrast (mean over draws)\n"
              + rows.pivot_table(index="variant", columns="mask", values="pooled_eer",
                                 aggfunc="mean").round(3).to_string())
    pairs: List[Tuple[str, str]] = [(f"ENS{len(backbones)}-{specs[-1]}",
                                     f"ENS{len(backbones)}-{specs[0]}")]
    pairs += [(f"{b}|S5", f"{b}|S3") for b in backbones]
    for spec in specs:
        for b in backbones:
            pairs.append((f"ENS{len(backbones)}-{spec}", f"{b}|{spec}"))
    iv = intervals_table(draw_root, pairs, jobs=args.interval_jobs)
    iv.to_csv(root / "results" / "u10_intervals.csv", index=False)
    print("\nintervals (delta = a - b; negative means a is better)\n"
          + iv.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

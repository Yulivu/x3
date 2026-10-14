#!/usr/bin/env python3
"""U1 [URGENT] - seen / unseen / strict_draw speaker decomposition (NEW_TASKS 01).

Every published X3 number so far used the ``clips`` mask, in which most evaluation
clips belong to a speaker that also carries a label.  U1 splits that apart:

* ``seen``        - evaluation clips whose speaker appears in ``L``
* ``unseen``      - evaluation clips whose speaker never appears in ``L``
  (per-platform EER is reported as NaN when support is short; pooled is always valid)
* ``strict_draw`` - a different SAMPLING protocol: the dev speakers are split into
  S_L and S_E, ``L`` is drawn only from S_L and ``E`` is all of S_E

100 draws per protocol.  Verdicts:
``SPEAKER_GAP`` sets when a method's unseen pooled is >= 1.5x its seen pooled;
``LP_GAIN_IS_SPEAKER_LOCAL`` sets when B9 is not better than B1 on unseen.

Writes ``results/u1_speaker_split.csv`` and ``results/u1_verdicts.csv``.
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

import w6_baselines as w6  # noqa: E402
from src.baselines import (  # noqa: E402
    b2_line_a,
    b3_platform_offset,
    b7_naive_pseudo_label,
    b8_fixmatch_lite,
    b9_label_propagation,
    fit_logreg_cv,
)
from src.data import data_root  # noqa: E402
from src.draws import draw_labels, draw_seed  # noqa: E402
from src.metrics3 import eer, summarize  # noqa: E402
from src.mpp import MPP  # noqa: E402

N_DRAWS = 100
K_PER_CELL = 10
METHODS = ("B1", "B2", "B3", "B7", "B8", "B9", "C1")

__all__ = ["split_strict", "run_draw", "verdicts", "main"]


def split_strict(speaker: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Random half/half split of dev SPEAKERS into (S_L, S_E)."""
    rng = np.random.default_rng(seed)
    speakers = np.unique(speaker)
    perm = rng.permutation(len(speakers))
    half = len(speakers) // 2
    s_l = set(speakers[perm[:half]].tolist())
    mask_l = np.array([s in s_l for s in speaker])
    return mask_l, ~mask_l


def run_draw(payload) -> List[Dict[str, Any]]:
    (d, Z, strong, y, plat, lang, cell, speaker, k_per_cell) = payload
    rows: List[Dict[str, Any]] = []

    # ---- protocol 1: the usual per-cell draw, split by seen / unseen speaker ----
    lab_pos, _ = draw_labels(cell, k_per_cell, draw_seed(d), cap_half=True)
    lab = np.zeros(len(y), bool)
    lab[lab_pos] = True
    if lab.sum() < 50:
        return rows
    lab_speakers = set(speaker[lab_pos].tolist())
    ev_all = np.setdiff1d(np.arange(len(y)), lab_pos)
    seen = np.array([s in lab_speakers for s in speaker[ev_all]])
    masks = {"seen": ev_all[seen], "unseen": ev_all[~seen]}

    # ---- protocol 2: strict_draw, labels only from S_L ----
    s_l_mask, s_e_mask = split_strict(speaker, seed=777000 + d)
    pool_pos = np.where(s_l_mask)[0]
    strict_lab = np.zeros(len(y), bool)
    rng = np.random.default_rng(draw_seed(d) + 5)
    for c in np.unique(cell):
        idx = pool_pos[cell[pool_pos] == c]
        if idx.size < 8:
            continue
        take = min(k_per_cell, idx.size // 2)
        strict_lab[rng.choice(idx, size=take, replace=False)] = True
    if strict_lab.sum() >= 50:
        masks["strict_draw"] = np.where(s_e_mask)[0]
    lab_for_fit = lab
    strict_lab_for_fit = strict_lab

    for mask_name, ev in masks.items():
        lab_used = strict_lab_for_fit if mask_name == "strict_draw" else lab_for_fit
        if ev.size < 50 or lab_used.sum() < 50:
            continue
        Zl, yl, pl, ll, cl = Z[lab_used], y[lab_used], plat[lab_used], lang[lab_used], cell[lab_used]
        scores: Dict[str, np.ndarray] = {}
        lr, _ = fit_logreg_cv(Zl, yl)
        base = lr.decision_function(Z)
        scores["B1"] = base
        s2, _ = b2_line_a(base, cell, plat, lab_used, y, variant="bb")
        scores["B2"] = s2
        s3, _ = b3_platform_offset(base, plat, lab_used, y, sigma=1.0)
        scores["B3"] = s3
        s7, _ = b7_naive_pseudo_label(Zl, yl, Z[~lab_used], tau=0.9, rounds=3)
        sc7 = np.zeros(len(y)); sc7[~lab_used] = s7
        scores["B7"] = sc7
        if strong is not None:
            s8, _ = b8_fixmatch_lite(Zl, yl, Z[~lab_used], strong[~lab_used],
                                     tau=0.9, rounds=3)
            sc8 = np.zeros(len(y)); sc8[~lab_used] = s8
            scores["B8"] = sc8
        s9, _ = b9_label_propagation(Z, yl, lab_used, k=15, alpha=0.9)
        scores["B9"] = s9
        c1 = MPP(sigma_p=0.5, sigma_gamma=0.5, lam=1.0).fit(Zl, yl, pl, ll, cl)
        scores["C1"] = c1.predict_logit(Z, plat, lang, cell)
        for name, sc in scores.items():
            rows.append(summarize(y[ev], sc[ev], plat[ev], extra={
                "protocol": "dev_v2", "region": "dev", "backbone": w6.BACKBONE,
                "spec": w6.SPEC, "method": name, "k": int(k_per_cell * 29), "draw": int(d),
                "mask": mask_name, "n_labeled": int(lab_used.sum()), "n_eval": int(ev.size),
                "n_speakers_in_L": int(len(set(speaker[lab_used].tolist()))),
            }))
    return rows


def _worker(payload):
    try:
        return payload[0], run_draw(payload)
    except Exception as exc:  # pragma: no cover
        return payload[0], [{"error": f"{type(exc).__name__}: {exc}"}]


def verdicts(df: pd.DataFrame) -> pd.DataFrame:
    """SPEAKER_GAP and LP_GAIN_IS_SPEAKER_LOCAL, on the paired draws."""
    clips = df[df["mask"].isin(["seen", "unseen"])]
    rows = []
    for method in sorted(clips["method"].unique()):
        a = clips[(clips["method"] == method) & (clips["mask"] == "seen")].set_index("draw")["pooled_eer"]
        b = clips[(clips["method"] == method) & (clips["mask"] == "unseen")].set_index("draw")["pooled_eer"]
        common = a.index.intersection(b.index)
        if len(common) < 5:
            continue
        seen_m = float(a.loc[common].astype(float).mean())
        unseen_m = float(b.loc[common].astype(float).mean())
        rows.append({"method": method, "seen_pooled": seen_m, "unseen_pooled": unseen_m,
                     "ratio": unseen_m / max(seen_m, 1e-9),
                     "n_draws": int(len(common))})
    out = pd.DataFrame(rows)
    flags = {}
    if not out.empty:
        gap = out[out["ratio"] >= 1.5]["method"].tolist()
        flags["SPEAKER_GAP"] = bool(gap)
        flags["SPEAKER_GAP_methods"] = gap
        b1 = out[out["method"] == "B1"]
        b9 = out[out["method"] == "B9"]
        if not b1.empty and not b9.empty:
            d = float(b9["unseen_pooled"].iloc[0] - b1["unseen_pooled"].iloc[0])
            flags["LP_GAIN_IS_SPEAKER_LOCAL"] = bool(d >= -1e-9)
            flags["B9_minus_B1_unseen"] = d
    out.attrs["flags"] = flags
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 U1 speaker decomposition")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    data_dir = data_root(args.data)
    sub, F = w6.load_spec(root, data_dir, args.backbone, args.spec)
    transform, Z = w6.fit_preprocess(F)
    strong = None
    p = data_dir / "x3" / "views" / "famA1" / args.backbone / "spec_feats.npy"
    if p.exists():
        strong = transform(np.asarray(np.load(p), np.float32))
    y = sub["y"].to_numpy()
    plat = sub["platform"].to_numpy()
    lang = sub["language"].to_numpy()
    cell = sub["cell"].to_numpy()
    speaker = sub["speaker"].to_numpy()
    print(f"dev clips={len(sub)} speakers={len(np.unique(speaker))} "
          f"strong_view={'yes' if strong is not None else 'no'}")
    payloads = [(d, Z, strong, y, plat, lang, cell, speaker, K_PER_CELL)
                for d in range(args.draws)]
    rows, errors = [], []
    if args.jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        with ctx.Pool(args.jobs) as pool:
            for d, out in pool.imap_unordered(_worker, payloads, chunksize=1):
                for r in out:
                    if "error" in r:
                        errors.append(f"draw {d}: {r['error']}")
                    else:
                        rows.append(r)
                if d % 20 == 0:
                    print(f"  draw {d} ({len(rows)} rows)", flush=True)
    else:
        for pl in payloads:
            d, out = _worker(pl)
            for r in out:
                if "error" in r:
                    errors.append(f"draw {d}: {r['error']}")
                else:
                    rows.append(r)
    if errors:
        print("errors:", errors[:5], "...", len(errors))
    df = pd.DataFrame(rows)
    (root / "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "results" / "u1_speaker_split.csv", index=False)
    piv = df.groupby(["method", "mask"])["pooled_eer"].mean().unstack().round(3)
    print(piv.to_string())
    v = verdicts(df)
    print()
    print(v.round(3).to_string(index=False))
    with open(root / "results" / "u1_verdicts.json", "w", encoding="utf-8") as fh:
        json.dump({k: (v2 if not isinstance(v2, list) else v2)
                   for k, v2 in v.attrs.get("flags", {}).items()}, fh, indent=2)
    print(json.dumps(v.attrs.get("flags", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

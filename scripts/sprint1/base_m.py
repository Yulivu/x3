#!/usr/bin/env python3
"""S1-B: BASE_M - the information-equal opponent set (NEW_TASKS 09 section 1).

Every method sees exactly what the arena will see: the source platforms' labels, one
target-platform genuine library of size N (with the query's own speaker masked) and
the unlabelled target stream.  Target labels enter only through the metric.

Implemented from 09 section 1 items 1-3:

1. ``Z0`` and ``Z0-anchor`` (the zero-shot score's rank inside the library's own
   zero-shot scores);
2. the reference-distance family: ``OC-1NN``, ``OC-3NN``, ``OC-centroid``,
   ``OC-top5``, ``E-norm-s``, shrunk ``Mahalanobis`` (source-estimated covariance),
   ``OCSVM``, ``GMM``, ``LOF``;
3. supervised: ``B1`` (source labels, C by leave-one-speaker CV), ``B1+refs`` (the
   target library joins training as extra labelled genuine clips), ``supKNN`` and the
   U8-style ``OC+B1`` fusion.

The library is split into a **proper** half (the reference every score uses) and a
**calibration** half (held-out genuine clips that supply the 95 %/99 % operating
points), and the transfer threshold is the median of the *other* platforms' 95 %
points - so no metric ever reads the target's own labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sprint1 import config as cfg  # noqa: E402
from scripts.sprint1 import harness as h  # noqa: E402

MIN_PER_CLASS = 30
KAPPA = 5.0
C_GRID = (1e-2, 1e-1, 1.0, 10.0)
METHODS = ("Z0", "Z0-anchor", "OC-1NN", "OC-3NN", "OC-centroid", "OC-top5",
           "E-norm-s", "Mahalanobis", "OCSVM", "GMM", "LOF",
           "B1", "B1+refs", "supKNN", "OC+B1")
#: 09 section 3: Z0 is the baseline to beat, never "the strongest opponent"; the
#: equal-budget few-shot upper bound (item 6) is excluded from the arena entirely.
STRONGEST_EXCLUDE = ("Z0",)


def standardise(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float)
    mu, sd = float(np.mean(s)), float(np.std(s))
    return (s - mu) / (sd if sd > 1e-12 else 1.0)


def smallest_k_distances(Zq: np.ndarray, Zl: np.ndarray, q_speaker: Sequence[Any],
                         l_speaker: Sequence[Any], k: int,
                         device: str = "cpu") -> np.ndarray:
    """``(n_q, k)`` smallest cosine distances, skipping the query's own speaker."""
    import torch

    Zq = np.asarray(Zq, np.float32)
    Zl = np.asarray(Zl, np.float32)
    q_sp = np.asarray(q_speaker).astype(str)
    l_sp = np.asarray(l_speaker).astype(str)
    out = np.full((Zq.shape[0], int(k)), np.nan, np.float32)
    if Zl.shape[0] == 0 or Zq.shape[0] == 0:
        return out
    Zn = h.l2n(Zq, device)
    Zl_t = h.l2n(Zl, device)
    for s in np.unique(q_sp):
        qi = np.where(q_sp == s)[0]
        keep = np.where(l_sp != s)[0]
        if keep.size == 0:
            continue
        ref = Zl_t[torch.as_tensor(keep, device=device)]
        d = 1.0 - Zn[torch.as_tensor(qi, device=device)] @ ref.T
        kk = int(min(int(k), d.shape[1]))
        out[qi, :kk] = torch.topk(d, kk, largest=False).values.cpu().numpy()
    return out


def e_norm_s(o1: np.ndarray, Z_ref: np.ndarray, ref_speaker: Sequence[Any],
             sigma_global: float, kappa: float = KAPPA) -> Tuple[np.ndarray, float]:
    """``o1 / sigma~_B``: the leave-one-out library scale shrunk to the global scale."""
    ref_sp = np.asarray(ref_speaker).astype(str)
    loo = smallest_k_distances(Z_ref, Z_ref, ref_sp, ref_sp, 1)[:, 0]
    loo = loo[np.isfinite(loo)]
    sigma_b = float(np.median(loo)) if loo.size else float(sigma_global)
    m = float(np.asarray(Z_ref).shape[0])
    sigma = (m * sigma_b + kappa * float(sigma_global)) / (m + kappa) if m else float(sigma_global)
    sigma = sigma if sigma > 1e-12 else 1.0
    return np.asarray(o1, float) / sigma, float(sigma)


def source_scale(Zs: np.ndarray, s_speaker: Sequence[Any]) -> float:
    """``sigma_glob``: median 1-NN distance inside the source platforms."""
    nn = smallest_k_distances(Zs, Zs, s_speaker, s_speaker, 1)[:, 0]
    nn = nn[np.isfinite(nn)]
    return float(np.median(nn)) if nn.size else 1.0


def mahalanobis_precision(Zs: np.ndarray, s_speaker: Sequence[Any]) -> np.ndarray:
    """Shrunk precision from de-meaned source speakers (09 section 1 item 2)."""
    from sklearn.covariance import LedoitWolf

    sp = np.asarray(s_speaker).astype(str)
    parts = [Zs[sp == s] - Zs[sp == s].mean(axis=0, keepdims=True)
             for s in np.unique(sp) if int((sp == s).sum()) >= 2]
    X = np.concatenate(parts, axis=0) if parts else np.asarray(Zs, float)
    return LedoitWolf().fit(X).precision_


def fit_b1(Zs: np.ndarray, y_s: np.ndarray, sp_s: Sequence[Any]) -> Tuple[Any, float]:
    """Logistic head on the source; ``C`` by leave-one-speaker CV (09 section 1)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    best_c, best_loss = C_GRID[0], np.inf
    n_splits = int(min(4, len(set(map(str, sp_s)))))
    if n_splits >= 2:
        for C in C_GRID:
            losses = []
            for tr, te in GroupKFold(n_splits=n_splits).split(
                    Zs, y_s, groups=list(map(str, sp_s))):
                lr = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
                lr.fit(Zs[tr], y_s[tr])
                m = (2 * y_s[te] - 1) * lr.decision_function(Zs[te])
                losses.append(float(np.mean(np.log1p(np.exp(-np.clip(m, -50, 50))))))
            loss = float(np.mean(losses))
            if loss < best_loss:
                best_c, best_loss = C, loss
    lr = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced").fit(Zs, y_s)
    return lr, float(best_c)


def score_methods(Zq: np.ndarray, z0q: np.ndarray, sp_q: Sequence[Any],
                  ref: Sequence[int], Zt_all: np.ndarray, z0t_all: np.ndarray,
                  sp_t_all: Sequence[Any], Zs: np.ndarray, y_s: np.ndarray,
                  cov_prec: Optional[np.ndarray], sigma_global: float,
                  b1_logit_q: Optional[np.ndarray], device: str = "cpu"
                  ) -> Dict[str, np.ndarray]:
    """Score queries ``Zq`` against the library's proper half (``ref``)."""
    import torch

    ref = np.asarray(ref, int)
    Z_ref = np.asarray(Zt_all, float)[ref]
    sp_ref = np.asarray(sp_t_all)[ref]
    out: Dict[str, np.ndarray] = {"Z0": np.asarray(z0q, float)}

    small = smallest_k_distances(Zq, Z_ref, sp_q, sp_ref, 5, device)
    o1 = small[:, 0].astype(float)
    out["OC-1NN"] = o1
    out["OC-3NN"] = np.nanmean(small[:, :3], axis=1)
    out["OC-top5"] = np.nanmean(small[:, : small.shape[1]], axis=1)

    Zn = h.l2n(Zq, device)
    sp_q_s = np.asarray(sp_q).astype(str)
    cent = np.full(Zn.shape[0], np.nan, np.float32)
    for s in np.unique(sp_q_s):
        qi = np.where(sp_q_s == s)[0]
        keep = np.where(sp_ref != s)[0]
        if keep.size == 0:
            continue
        c = h.l2n(Z_ref[keep], device).mean(dim=0)
        c = c / c.norm().clamp_min(1e-12)
        cent[qi] = (1.0 - Zn[torch.as_tensor(qi, device=device)] @ c).cpu().numpy()
    out["OC-centroid"] = cent

    out["E-norm-s"] = e_norm_s(o1, Z_ref, sp_ref, sigma_global)[0]
    if cov_prec is not None:
        diff = np.asarray(Zq, float) - Z_ref.mean(axis=0, keepdims=True)
        out["Mahalanobis"] = np.sqrt(np.abs(np.einsum("ij,jk,ik->i", diff, cov_prec, diff)))

    ref_z0 = np.asarray(z0t_all, float)[ref]
    out["Z0-anchor"] = np.asarray([float((ref_z0 <= v).mean()) for v in np.asarray(z0q, float)])

    try:
        from sklearn.svm import OneClassSVM

        out["OCSVM"] = -OneClassSVM(nu=0.1, gamma="scale").fit(Z_ref).score_samples(Zq)
    except Exception:                                             # noqa: BLE001
        pass
    try:
        from sklearn.mixture import GaussianMixture

        out["GMM"] = -GaussianMixture(n_components=2, covariance_type="diag",
                                      random_state=cfg.DRAW_BASE, max_iter=100
                                      ).fit(Z_ref).score_samples(Zq)
    except Exception:                                             # noqa: BLE001
        pass
    try:
        from sklearn.neighbors import LocalOutlierFactor

        k = int(min(20, max(2, len(ref) - 1)))
        out["LOF"] = -LocalOutlierFactor(n_neighbors=k, novelty=True
                                         ).fit(Z_ref).score_samples(Zq)
    except Exception:                                             # noqa: BLE001
        pass

    if b1_logit_q is not None:
        out["B1"] = np.asarray(b1_logit_q, float)
        try:
            from sklearn.linear_model import LogisticRegression

            x = np.vstack([Zs, Z_ref])
            yv = np.concatenate([np.asarray(y_s, int), np.zeros(len(ref), int)])
            lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(x, yv)
            out["B1+refs"] = lr.decision_function(Zq)
            sim = (h.l2n(Zq, "cpu") @ h.l2n(x, "cpu").T).cpu().numpy()
            knn = np.argsort(-sim, axis=1)[:, :min(5, sim.shape[1])]
            out["supKNN"] = np.asarray([yv[row].mean() for row in knn])
        except Exception:                                         # noqa: BLE001
            pass
        out["OC+B1"] = standardise(o1) + standardise(np.asarray(b1_logit_q, float))
    return out


def _tpr_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> float:
    pos = y == 1
    if pos.sum() == 0 or not np.isfinite(thr):
        return float("nan")
    return float((s[pos] >= thr).mean() * 100)


def _hter(y: np.ndarray, s: np.ndarray, thr: float) -> float:
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0 or not np.isfinite(thr):
        return float("nan")
    return float(((s[pos] < thr).mean() + (s[neg] >= thr).mean()) / 2 * 100)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X3 sprint 1 S1-B BASE_M")
    ap.add_argument("--out-dir", default="sprint1/results")
    ap.add_argument("--tables", default="sprint1/tables")
    ap.add_argument("--pca", default="sprint1/pca")
    ap.add_argument("--backbones", default=",".join(cfg.BACKBONES))
    ap.add_argument("--spec", default="S3")
    ap.add_argument("--sizes", default=",".join(str(s) for s in cfg.LIBRARY_SIZES))
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--save-scores-count", type=int, default=30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", default=None)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else cfg.REPO_ROOT
    data_root = cfg.data_root(args.data)
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = json.loads((root / args.tables / "p1_folds.json").read_text(encoding="utf-8"))
    audio_index = json.loads((root / "sprint1/audio/dev_index.json").read_text(encoding="utf-8"))
    if audio_index.get("side") != "dev":
        print("REFUSED: only dev data may be used before FREEZE_OK")
        return 3
    sizes = [int(s) for s in str(args.sizes).split(",") if s.strip()]
    backbones = [b.strip() for b in str(args.backbones).split(",") if b.strip()]
    draws = int(args.draws)
    device = h.pick_device(args.device)
    h.set_threads()

    from src.data import load_mlitw
    from src.newtasks05 import zero_scores

    meta = load_mlitw(data_root)
    y_all = meta["y"].to_numpy()
    sp_all = meta["speaker"].astype(str).to_numpy()
    dev_rows = np.asarray(audio_index["rows"], int)
    pos_of = {int(r): i for i, r in enumerate(dev_rows)}
    all_clip_ids = np.arange(len(meta))

    rows_out: List[Dict[str, Any]] = []
    t0 = time.time()
    for bb in backbones:
        # NOTE: spec S3 is [mean, std] and has NO zero-shot column (only S5 does), so
        # the zero-shot score is read from the backbone's scores.csv by clip id.
        z0_dev = np.asarray(zero_scores(data_root, bb, all_clip_ids), float)[dev_rows]
        for p, target in tables["targets"].items():
            pca_path = root / args.pca / f"pca_{bb}_{p}.npz"
            if not pca_path.exists():
                print(f"missing {pca_path}; run P-3 first")
                return 2
            npz = np.load(pca_path)
            t_rows = np.asarray(npz["target_rows"], int)
            s_rows = np.asarray(npz["source_rows"], int)
            Zt = np.asarray(npz["Z_target"], float)
            Zs = np.asarray(npz["Z_source"], float)
            z0t = z0_dev[[pos_of[int(r)] for r in t_rows]]
            y_t, sp_t = y_all[t_rows], sp_all[t_rows]
            y_s, sp_s = y_all[s_rows], sp_all[s_rows]
            cov_prec = mahalanobis_precision(Zs, sp_s)
            sigma_global = source_scale(Zs, sp_s)
            lr_src, c_chosen = fit_b1(Zs, y_s, sp_s)
            b1_target = lr_src.decision_function(Zt)
            t_index = {int(r): i for i, r in enumerate(t_rows)}
            for N in sizes:
                for d in range(draws):
                    entry = (target["library_by_draw"].get(str(d)) or {}).get(str(N))
                    if not entry:
                        continue
                    lib = [t_index[int(r)] for r in entry["rows"] if int(r) in t_index]
                    proper = [t_index[int(r)] for r in entry["proper"] if int(r) in t_index]
                    calib = [t_index[int(r)] for r in entry["calibration"] if int(r) in t_index]
                    lib_global = set(int(r) for r in entry["rows"])
                    ev = np.array([i for i, r in enumerate(t_rows) if int(r) not in lib_global], int)
                    if ev.size == 0 or not proper:
                        continue
                    sc_ev = score_methods(Zt[ev], z0t[ev], sp_t[ev], proper, Zt, z0t, sp_t,
                                          Zs, y_s, cov_prec, sigma_global, b1_target[ev], device)
                    sc_ca = (score_methods(Zt[calib], z0t[calib], sp_t[calib], proper, Zt, z0t,
                                           sp_t, Zs, y_s, cov_prec, sigma_global,
                                           b1_target[calib], device) if calib else {})
                    row: Dict[str, Any] = {"backbone": bb, "target": p, "N": int(N),
                                           "draw": int(d), "n_eval": int(ev.size),
                                           "effective_n": int(entry.get("effective_n", len(lib))),
                                           "n_proper": len(proper), "n_calibration": len(calib),
                                           "c_b1": c_chosen,
                                           "n_real": int((y_t[ev] == 0).sum()),
                                           "n_fake": int((y_t[ev] == 1).sum())}
                    for m in METHODS:
                        if m not in sc_ev:
                            continue
                        row[f"eer_{m}"] = h.eer_t(sc_ev[m], y_t[ev])
                        row[f"auc_{m}"] = h.auc_t(sc_ev[m], y_t[ev])
                        thr95 = (float(np.percentile(sc_ca[m], 95))
                                 if m in sc_ca and len(sc_ca[m]) else float("nan"))
                        thr99 = (float(np.percentile(sc_ca[m], 99))
                                 if m in sc_ca and len(sc_ca[m]) else float("nan"))
                        row[f"thr95_{m}"] = thr95
                        row[f"tpr5_{m}"] = _tpr_at_threshold(y_t[ev], sc_ev[m], thr95)
                        row[f"tpr1_{m}"] = _tpr_at_threshold(y_t[ev], sc_ev[m], thr99)
                    if d < args.save_scores_count:
                        arch = root / "sprint1/scores" / f"base_m_{bb}_{p}_N{N}_{d}.npz"
                        arch.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(arch, rows=t_rows[ev], y=y_t[ev], speaker=sp_t[ev],
                                            **{f"s_{m}": sc_ev[m] for m in METHODS if m in sc_ev})
                    rows_out.append(row)
            print(f"{bb} {p}: done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows_out)
    df.to_csv(out_dir / "base_m.csv", index=False)

    # transfer thresholds: median of the *other* platforms' calibration 95 % point
    hter_rows = []
    for (bb, N, d), grp in df.groupby(["backbone", "N", "draw"]):
        for m in METHODS:
            col = f"thr95_{m}"
            if col not in grp:
                continue
            for _, row in grp.iterrows():
                others = grp[grp["target"] != row["target"]][col].dropna()
                thr = float(np.median(others)) if len(others) else float("nan")
                hter_rows.append({"backbone": bb, "N": N, "draw": d,
                                  "target": row["target"], "method": m, "thr_transfer": thr})
    hter = pd.DataFrame(hter_rows)

    agg_rows: List[Dict[str, Any]] = []
    for bb in backbones:
        for N in sizes:
            sub = df[(df["backbone"] == bb) & (df["N"] == N)]
            if sub.empty:
                continue
            for m in METHODS:
                col = f"eer_{m}"
                if col not in sub:
                    continue
                per_draw, n_plat = [], []
                for _d, g in sub.groupby("draw"):
                    elig = g[(g["n_real"] >= MIN_PER_CLASS) & (g["n_fake"] >= MIN_PER_CLASS)]
                    if len(elig):
                        per_draw.append(float(elig[col].mean()))
                        n_plat.append(int(len(elig)))
                agg_rows.append({
                    "backbone": bb, "N": int(N), "method": m,
                    "macro_eer_mean": float(np.mean(per_draw)) if per_draw else float("nan"),
                    "macro_eer_sd": float(np.std(per_draw)) if per_draw else float("nan"),
                    "pooled_eer_mean": float(sub[col].mean()),
                    "auc_mean": float(sub[f"auc_{m}"].mean()) if f"auc_{m}" in sub else float("nan"),
                    "tpr5_mean": float(sub[f"tpr5_{m}"].mean()) if f"tpr5_{m}" in sub else float("nan"),
                    "tpr1_mean": float(sub[f"tpr1_{m}"].mean()) if f"tpr1_{m}" in sub else float("nan"),
                    "n_draws_scored": len(per_draw),
                    "n_platforms_mean": float(np.mean(n_plat)) if n_plat else float("nan"),
                })
    agg = pd.DataFrame(agg_rows)

    strongest = []
    for (bb, N), g in agg.groupby(["backbone", "N"]):
        cand = g[~g["method"].isin(STRONGEST_EXCLUDE)].dropna(subset=["macro_eer_mean"])
        if cand.empty:
            continue
        best = cand.sort_values("macro_eer_mean").iloc[0]
        z0 = g[g["method"] == "Z0"]["macro_eer_mean"]
        strongest.append({"backbone": bb, "N": int(N), "strongest": best["method"],
                          "macro_eer_mean": float(best["macro_eer_mean"]),
                          "z0_macro_eer": float(z0.iloc[0]) if len(z0) else float("nan")})
    gate = pd.DataFrame(strongest)
    agg.to_csv(out_dir / "base_m_agg.csv", index=False)
    gate.to_csv(out_dir / "base_m_gate.csv", index=False)
    hter.to_csv(out_dir / "base_m_thresholds.csv", index=False)
    with (out_dir / "base_m_summary.md").open("w", encoding="utf-8") as fh:
        fh.write("# BASE_M (P1, dev) - macro EER, headline N=40\n\n")
        if not agg.empty:
            piv = agg[agg["N"] == 40].pivot_table(index="method", columns="backbone",
                                                  values="macro_eer_mean")
            fh.write(piv.round(2).to_string() + "\n\n")
        fh.write("## strongest opponent per backbone (Z0 excluded)\n\n")
        fh.write(gate.round(2).to_string(index=False) + "\n")
    print(json.dumps({"rows": len(df), "seconds": round(time.time() - t0, 1),
                      "strongest": gate.to_dict("records")}, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

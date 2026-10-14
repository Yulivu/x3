"""Gate rules from TASKBOOK section 8, expressed as pure functions over the
``results/`` CSVs.

Everything here is read-only: it inspects files, applies the pre-registered rule
and returns a verdict.  A gate whose input file does not exist returns
``{"status": "NOT_RUN"}`` - it never guesses and never falls back to a weaker rule.

Schemas consumed (produced by the W-scripts, TASKBOOK section 9):

* ``results/zeroshot_mlitw.csv``    backbone, pooled_eer, within_macro_eer, fidelity
* ``results/reconciliation.csv``    scope, metric, headline, rerun, diff
* ``results/w4_stability.csv``      metric, mean3, mean100, diff
* ``results/backbone_rank.csv``     backbone, spec, metric, mean, paired_mean, paired_lo, paired_hi
* ``results/w6_summary.csv``        backbone, spec, method, metric, mean, sd
* ``results/w6_paired.csv``         backbone, spec, method, other, metric, mean, lo, hi, n
* ``results/w6_draws_*.csv``        per-draw three-view rows (method, k, draw, mask, ...)
* ``results/w6_collapse.csv``       cell, true_rate, pl_rate
* ``results/w7_curve.csv``          method, k, pooled_eer, ... (+ paired columns)
* ``results/w8_zlabel.csv``         method, pooled_eer, within_macro_eer
* ``results/w11_past.csv``          method, other, metric, mean, lo, hi, n
* ``results/w12_casa.csv``          method, other, metric, k, mean, lo, hi, n
* ``results/w15_itw.csv``           method, other, metric, mean, lo, hi, n
* ``results/w16_mechanism.csv``     backbone, spec, method, metric, value
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

NOT_RUN = "NOT_RUN"
DECISION_ORDER = tuple(f"G{i}" for i in range(12))
BASE_F_DEFAULT = ("B1", "B2", "B3", "B4", "B7", "B8", "B9", "B10", "B11")
C1_CANDIDATES = ("B1", "B2", "B3", "B11")
VIEWS = ("pooled_eer", "within_macro_eer", "worst_platform_eer")

C_F_POOL_GAIN = 1.0
C_F_MACRO_GAIN = 0.7
PAST_MACRO_GAIN = 1.5
PAST_POOL_GAIN = 1.0
CASA_POOL_GAIN = 1.0
SF_MARGIN_TOLERANCE = 0.3
SF_MIN_VIEWS_WIN = 2
Z_POOLED_MAX = 24.65
Z_GAIN_VS_ZEROSHOT = 2.0
ITW_TOLERANCE = 0.3
DECONF_AUC_GAP = 0.03
REPRO_TOLERANCE = 0.5
STABILITY_TOLERANCE = 2.0
COLLAPSE_DEVIATION = 0.2
COLLAPSE_SHARE = 0.3

__all__ = [
    "NOT_RUN", "DECISION_ORDER", "BASE_F_DEFAULT", "VIEWS",
    "read_csv", "paired_diff", "sota_f", "gate_g0",
    "gate_g1", "gate_g2", "gate_g3", "gate_g4", "gate_g5", "gate_g6",
    "gate_g7", "gate_g8", "gate_g9", "gate_g10", "gate_g11", "load_draws",
]


# ------------------------------------------------------------------ plumbing
def read_csv(root: Path, name: str) -> Optional[pd.DataFrame]:
    path = Path(root) / "results" / name
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_draws(root: Path) -> Optional[pd.DataFrame]:
    frames = []
    for pattern in ("w6_draws_*.csv", "wf_draws_*.csv"):
        for p in sorted((Path(root) / "results").glob(pattern)):
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if "region" not in df.columns:
        df["region"] = "dev"
    return df


def paired_diff(
    df: pd.DataFrame, a: str, b: str, metric: str, **filters: Any
) -> Optional[Dict[str, Any]]:
    """Draw-paired difference ``a - b`` (negative means ``a`` is better)."""
    if df is None or not {"method", "draw", metric}.issubset(df.columns):
        return None
    sub = df
    for key, value in filters.items():
        if value is None:
            continue
        if key not in sub.columns:
            return None
        sub = sub[sub[key] == value]
    if sub.empty:
        return None
    wa = sub[sub["method"] == a].set_index("draw")[metric]
    wb = sub[sub["method"] == b].set_index("draw")[metric]
    common = wa.index.intersection(wb.index)
    if len(common) < 3:
        return None
    diff = (wa.loc[common].astype(float) - wb.loc[common].astype(float)).dropna().values
    if diff.size < 3:
        return None
    return {
        "mean": float(diff.mean()),
        "lo": float(np.percentile(diff, 2.5)),
        "hi": float(np.percentile(diff, 97.5)),
        "wins": int((diff < 0).sum()),
        "n": int(diff.size),
    }


def _paired_from_table(
    root: Path, name: str, a: str, b: str, metric: str, **filters: Any
) -> Optional[Dict[str, Any]]:
    """Read a pre-computed paired table (method/other/metric/mean/lo/hi)."""
    df = read_csv(root, name)
    if df is None or not {"method", "other", "metric", "mean"}.issubset(df.columns):
        return None
    sub = df[(df["method"] == a) & (df["other"] == b) & (df["metric"] == metric)]
    for key, value in filters.items():
        if value is not None and key in sub.columns:
            sub = sub[sub[key] == value]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return {
        "mean": float(row["mean"]),
        "lo": float(row["lo"]) if "lo" in row else float("nan"),
        "hi": float(row["hi"]) if "hi" in row else float("nan"),
        "n": int(row["n"]) if "n" in row else -1,
    }


def _ci_excludes_zero(p: Optional[Dict[str, Any]]) -> bool:
    return bool(p is not None and p["hi"] < 0)


# --------------------------------------------------------------------- gates
def gate_g0(root: Path) -> Dict[str, Any]:
    df = read_csv(root, "zeroshot_mlitw.csv")
    if df is None or "fidelity" not in df.columns:
        return {"status": NOT_RUN, "why": "results/zeroshot_mlitw.csv missing or has no fidelity column"}
    n = int(df["fidelity"].astype(str).str.lower().isin(["pass", "ok", "true", "1"]).sum())
    status = "pass" if n >= 3 else ("amber" if n >= 1 else "fail")
    return {"status": status, "backbones_with_fidelity": n, "n_rows": int(len(df))}


def gate_g1(root: Path) -> Dict[str, Any]:
    df = read_csv(root, "zeroshot_mlitw.csv")
    if df is None or not {"pooled_eer", "within_macro_eer"}.issubset(df.columns):
        return {"status": NOT_RUN, "why": "zero-shot table missing"}
    if "kind" in df.columns:
        df = df[df["kind"] != "metadata"]
    best_pooled = float(df["pooled_eer"].min())
    best_macro = float(df["within_macro_eer"].min())
    ok = best_pooled > 15.0 or best_macro > 20.0
    return {
        "status": "HEADROOM_OK" if ok else "HEADROOM_SMALL",
        "best_pooled": best_pooled,
        "best_within_macro": best_macro,
    }


def gate_g2(root: Path) -> Dict[str, Any]:
    rec = read_csv(root, "reconciliation.csv")
    stab = read_csv(root, "w4_stability.csv")
    if rec is None or "diff" not in rec.columns:
        return {"status": NOT_RUN, "stability": NOT_RUN, "why": "results/reconciliation.csv missing"}
    # The stored reference is a mean over 24 (seed, rep) cells, so the 100-draw row
    # is the like-for-like comparison; the 3-draw prefix stays in the file as a
    # diagnostic but is far too noisy to gate on.
    if "scope" in rec.columns and (rec["scope"] == "100draw").any():
        rec = rec[rec["scope"] == "100draw"]
    worst = float(np.nanmax(np.abs(rec["diff"].astype(float))))
    repro = "REPRODUCED" if worst <= REPRO_TOLERANCE else "NOT_REPRODUCED"
    if stab is None or "diff" not in stab.columns:
        stable = NOT_RUN
    else:
        gap = float(np.nanmax(np.abs(stab["diff"].astype(float))))
        stable = "STABLE" if gap <= STABILITY_TOLERANCE else "OPTIMISTIC"
    return {"status": repro, "worst_reproduction_gap": worst, "stability": stable}


def gate_g3(root: Path, select: bool = False) -> Dict[str, Any]:
    df = read_csv(root, "backbone_rank.csv")
    if df is None or not {"backbone", "spec", "metric", "mean"}.issubset(df.columns):
        return {"status": NOT_RUN, "why": "results/backbone_rank.csv missing"}
    macro = df[df["metric"] == "within_macro_eer"].copy()
    if macro.empty:
        return {"status": NOT_RUN, "why": "backbone_rank has no within_macro_eer rows"}
    macro = macro.sort_values("mean")
    best = macro.iloc[0]
    second = macro.iloc[1] if len(macro) > 1 else None
    switch = False
    if {"paired_mean", "paired_hi"}.issubset(macro.columns):
        switch = float(best["paired_mean"]) <= -0.5 and float(best["paired_hi"]) < 0
    status = f"{best['backbone']}|{best['spec']}" if switch else "KEEP_ORIGINAL"
    chosen = {
        "status": status,
        "backbone": str(best["backbone"]),
        "spec": str(best["spec"]),
        "within_macro": float(best["mean"]),
        "switched": bool(switch),
        "second": (
            {"backbone": str(second["backbone"]), "spec": str(second["spec"]),
             "within_macro": float(second["mean"])}
            if second is not None else None
        ),
    }
    if select:
        (Path(root) / "chosen_backbone.json").write_text(
            json.dumps(chosen, indent=2, sort_keys=True), encoding="utf-8"
        )
    return chosen


def _chosen_pairs(chosen: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not chosen or chosen.get("status") in (NOT_RUN, "KEEP_ORIGINAL"):
        return []
    out = [{"backbone": chosen["backbone"], "spec": chosen["spec"]}]
    if chosen.get("second"):
        out.append({"backbone": chosen["second"]["backbone"], "spec": chosen["second"]["spec"]})
    return out


def best_of(df: pd.DataFrame, candidates: Sequence[str], metric: str) -> Optional[str]:
    sub = df[df["method"].isin(candidates) & (df["metric"] == metric)]
    if sub.empty:
        return None
    return str(sub.sort_values("mean").iloc[0]["method"])


def gate_g4(root: Path, region: str = "dev", chosen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    summary = read_csv(root, "w6_summary.csv")
    if summary is None or not {"method", "metric", "mean"}.issubset(summary.columns):
        return {"status": NOT_RUN, "why": "results/w6_summary.csv missing"}
    best = best_of(summary, C1_CANDIDATES, "within_macro_eer")
    if best is None:
        return {"status": NOT_RUN, "why": "no C1 candidate rows in w6_summary.csv"}
    draws = load_draws(root)
    p_pool = paired_diff(draws, "C1", best, "pooled_eer", region=region)
    p_macro = paired_diff(draws, "C1", best, "within_macro_eer", region=region)
    pool_ok = _ci_excludes_zero(p_pool) and p_pool["mean"] <= -C_F_POOL_GAIN
    macro_ok = _ci_excludes_zero(p_macro) and p_macro["mean"] <= -C_F_MACRO_GAIN
    if p_pool is None and p_macro is None:
        return {"status": NOT_RUN, "why": "no per-draw W6 rows to pair", "best_baseline": best}
    return {
        "status": "C1_PASS" if (pool_ok or macro_ok) else "C1_FAIL",
        "best_baseline": best,
        "pooled": p_pool,
        "within_macro": p_macro,
    }


def gate_g5(root: Path) -> Dict[str, Any]:
    df = read_csv(root, "w6_collapse.csv")
    if df is None or not {"true_rate", "pl_rate"}.issubset(df.columns):
        return {"status": NOT_RUN, "why": "results/w6_collapse.csv missing"}
    ext = df[(df["true_rate"] >= 0.8) | (df["true_rate"] <= 0.1)]
    if ext.empty:
        return {"status": NOT_RUN, "why": "no extreme-prior cells in w6_collapse.csv"}
    dev = (ext["pl_rate"] - ext["true_rate"]).abs() > COLLAPSE_DEVIATION
    share = float(dev.mean())
    return {
        "status": "PL_COLLAPSE" if share >= COLLAPSE_SHARE else "PL_OK",
        "share_of_extreme_cells": share,
        "n_extreme_cells": int(len(ext)),
    }


def gate_g6(root: Path, region: str = "dev", chosen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p_macro = _paired_from_table(root, "w11_past.csv", "PAST", "C1", "within_macro_eer")
    p_pool = _paired_from_table(root, "w11_past.csv", "PAST", "C1", "pooled_eer")
    if p_macro is None and p_pool is None:
        return {"status": NOT_RUN, "why": "results/w11_past.csv missing"}
    macro_ok = bool(p_macro and p_macro["mean"] <= -PAST_MACRO_GAIN and p_macro["hi"] < 0)
    pool_ok = bool(p_pool and p_pool["mean"] <= -PAST_POOL_GAIN and p_pool["hi"] < 0)
    return {
        "status": "PAST_PASS" if (macro_ok or pool_ok) else "PAST_FAIL",
        "within_macro": p_macro,
        "pooled": p_pool,
    }


def gate_g7(root: Path, region: str = "dev", chosen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    df = read_csv(root, "w12_casa.csv")
    if df is None or not {"method", "other", "metric", "mean"}.issubset(df.columns):
        return {"status": NOT_RUN, "why": "results/w12_casa.csv missing"}
    sub = df[(df["method"] == "CASA") & (df["other"] == "U") & (df["metric"] == "pooled_eer")]
    if "k" in sub.columns:
        sub = sub[sub["k"] == 100]
    if sub.empty:
        return {"status": NOT_RUN, "why": "no CASA-vs-U row at k=100"}
    row = sub.iloc[0]
    ok = float(row["mean"]) <= -CASA_POOL_GAIN and float(row.get("hi", np.nan)) < 0
    return {
        "status": "CASA_PASS" if ok else "CASA_FAIL",
        "pooled": {"mean": float(row["mean"]), "hi": float(row.get("hi", np.nan)), "n": int(row.get("n", -1))},
    }


def gate_g8(root: Path) -> Dict[str, Any]:
    z = read_csv(root, "w8_zlabel.csv")
    zs = read_csv(root, "zeroshot_mlitw.csv")
    if z is None or not {"method", "pooled_eer", "within_macro_eer"}.issubset(z.columns):
        return {"status": NOT_RUN, "why": "results/w8_zlabel.csv missing"}
    if zs is None or "pooled_eer" not in zs.columns:
        return {"status": NOT_RUN, "why": "zero-shot table missing"}
    best_zero_pool = float(zs["pooled_eer"].min())
    best_zero_macro = float(zs["within_macro_eer"].min())
    candidates = []
    for method in ("Z2", "Z3"):
        row = z[z["method"] == method]
        if row.empty:
            continue
        r = row.iloc[0]
        candidates.append((method, float(r["pooled_eer"]), float(r["within_macro_eer"])))
    passing = [
        c for c in candidates
        if c[1] <= Z_POOLED_MAX and c[1] <= best_zero_pool - Z_GAIN_VS_ZEROSHOT and c[2] <= best_zero_macro
    ]
    if passing:
        return {
            "status": "Z_PASS",
            "method": passing[0][0],
            "pooled": passing[0][1],
            "best_zero_shot_pooled": best_zero_pool,
        }
    best = min(candidates, key=lambda c: c[1]) if candidates else None
    return {
        "status": "Z_FAIL",
        "method": best[0] if best else None,
        "pooled": best[1] if best else None,
        "best_zero_shot_pooled": best_zero_pool,
    }


def gate_g9(root: Path) -> Dict[str, Any]:
    c1_b1 = _paired_from_table(root, "w15_itw.csv", "C1", "B1", "within_macro_eer")
    past_c1 = _paired_from_table(root, "w15_itw.csv", "PAST", "C1", "within_macro_eer")
    if c1_b1 is None and past_c1 is None:
        return {"status": NOT_RUN, "why": "results/w15_itw.csv missing"}
    c1_ok = bool(c1_b1 and c1_b1["mean"] <= ITW_TOLERANCE)
    past_ok = bool(past_c1 and past_c1["mean"] <= 0.0)
    return {
        "status": "ITW_REPLICATES" if (c1_ok and past_ok) else "ITW_MIXED",
        "c1_vs_b1": c1_b1,
        "past_vs_c1": past_c1,
    }


def gate_g10(root: Path, region: str = "dev", chosen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    df = read_csv(root, "w16_mechanism.csv")
    if df is None or not {"method", "metric", "value"}.issubset(df.columns):
        return {"status": NOT_RUN, "why": "results/w16_mechanism.csv missing"}

    def val(method: str, metric: str) -> Optional[float]:
        sub = df[(df["method"] == method) & (df["metric"] == metric)]
        return float(sub.iloc[0]["value"]) if not sub.empty else None

    auc_b1, auc_c1 = val("B1", "platform_auc"), val("C1", "platform_auc")
    mac_b1, mac_c1 = val("B1", "within_macro_eer"), val("C1", "within_macro_eer")
    if None in (auc_b1, auc_c1, mac_b1, mac_c1):
        return {"status": NOT_RUN, "why": "w16_mechanism.csv is missing platform_auc or within_macro_eer rows"}
    ok = (auc_c1 <= auc_b1 - DECONF_AUC_GAP) and (mac_c1 <= mac_b1)
    return {
        "status": "DECONF_OK" if ok else "DECONF_NOT_SHOWN",
        "platform_auc": {"B1": auc_b1, "C1": auc_c1},
        "within_macro": {"B1": mac_b1, "C1": mac_c1},
    }


def sota_f(
    draws: pd.DataFrame,
    backbone: str,
    spec: str,
    base_f: Sequence[str] = BASE_F_DEFAULT,
    region: str = "dev",
    k: int = 290,
) -> Dict[str, Any]:
    """S-F for one (backbone, spec): wins on >=2 views for EVERY baseline."""
    detail: Dict[str, Any] = {}
    all_ok = True
    for b in base_f:
        wins, third_ok, per = 0, True, {}
        for view in VIEWS:
            p = paired_diff(
                draws, "C1", b, view, region=region, k=k, backbone=backbone, spec=spec
            )
            per[view] = p
            if p is None:
                third_ok = False
                continue
            if p["hi"] < 0 and p["mean"] < 0:
                wins += 1
            elif p["mean"] > SF_MARGIN_TOLERANCE:
                third_ok = False
        ok = wins >= SF_MIN_VIEWS_WIN and third_ok
        detail[b] = {"wins": wins, "ok": ok, "views": per}
        all_ok = all_ok and ok
    return {"ok": bool(all_ok), "per_baseline": detail}


def gate_g11(root: Path, region: str = "dev", chosen: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    draws = load_draws(root)
    pairs = _chosen_pairs(chosen)
    if draws is None:
        return {"status": NOT_RUN, "why": "no per-draw rows under results/"}
    if not pairs:
        return {"status": NOT_RUN, "why": "no chosen (backbone, spec) yet - run G3 with --select"}
    results = {}
    ok = True
    for pair in pairs[:2]:
        r = sota_f(draws, pair["backbone"], pair["spec"], region=region)
        results[f"{pair['backbone']}|{pair['spec']}"] = r
        ok = ok and r["ok"]
    return {
        "status": "SOTA_F_DEV_YES" if ok else "SOTA_F_DEV_NO",
        "per_backbone": {k: v["ok"] for k, v in results.items()},
        "detail": results,
    }

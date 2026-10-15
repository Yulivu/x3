#!/usr/bin/env python3
"""Append the NEW_TASKS 05 sections to DECISION_POINT.md.

The 02 generator writes the ESN / projection / M-ALP page.  This adds the five
things NEW_TASKS 05 section 4 asks for on top of it:

1. per-track best method vs the strongest baseline of that track,
2. the G19-G24 verdicts plus ``LABELS_SATURATED``,
3. the SGM ablation reading,
4. whether the two feature blocks agree,
5. one factual sentence about which tracks clear the 2.0-point bar.

Only the sampling-level (a) interval is computable here: the (b)
speaker-clustered interval needs the per-clip score archives, which the first
batch did not persist.  Every gate that needs (b) is written as ``NOT_RUN`` with
that reason rather than being guessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MARKER = "<!-- NEW_TASKS_05_APPENDIX -->"
BOOT = 2000
SEED = 20261015


def _read(root: Path, rel: str) -> Optional[pd.DataFrame]:
    path = root / rel
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _a_interval(diffs: np.ndarray) -> Tuple[float, float, float]:
    """Bootstrap the MEAN difference over draws (NEW_TASKS 05 V-1 rule (a))."""
    diffs = np.asarray(diffs, float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    means = diffs[rng.integers(0, diffs.size, size=(BOOT, diffs.size))].mean(axis=1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired(df: pd.DataFrame, method: str, others: List[str], metric: str = "pooled_eer"
            ) -> Optional[Dict[str, Any]]:
    """Cheapest per-draw paired difference of ``method`` against the best of ``others``."""
    if df is None or metric not in df.columns:
        return None
    piv = df.pivot_table(index="draw", columns="method", values=metric, aggfunc="mean")
    if method not in piv.columns:
        return None
    present = [o for o in others if o in piv.columns]
    if not present:
        return None
    # per draw, the strongest baseline is the one with the lowest EER
    best_other = piv[present].min(axis=1)
    good = piv[method].notna() & best_other.notna()
    mean, lo, hi = _a_interval((piv[method] - best_other)[good].to_numpy())
    return {"method": method, "best_of": present, "delta": mean, "a_lo": lo, "a_hi": hi,
            "baseline_mean": float(best_other[good].mean()),
            "method_mean": float(piv[method][good].mean()), "n_draws": int(good.sum())}


def _fmt(d: Optional[Dict[str, Any]]) -> str:
    if not d:
        return "`NOT_RUN`"
    return (f"delta {d['delta']:+.2f} ((a) [{d['a_lo']:+.2f}, {d['a_hi']:+.2f}], "
            f"n={d['n_draws']}); method {d['method_mean']:.2f} vs best baseline "
            f"{d['baseline_mean']:.2f}")


def build(root: Path) -> str:
    lines: List[str] = [MARKER, ""]
    lines.append("## NEW_TASKS 05 appendix")
    lines.append("")
    lines.append("Post-hoc revision. The five criteria below were written after the "
                 "v2 numbers were seen; `REVISIONS.md` records each one.")
    lines.append("")

    # ---------------------------------------------------------------- T-G
    lines.append("### T-G - label saturation (strict, group-disjoint)")
    sat = _read(root, "results/t_g_saturation.csv")
    saturated = "`NOT_RUN`"
    if sat is not None and not sat.empty:
        rows = []
        gaps: List[float] = []
        for (bb, spec), grp in sat.groupby(["backbone", "spec"]):
            g = grp.groupby("k")["pooled_eer"].mean()
            if 290 in g.index and g.index.max() >= 1200:
                gap = float(g.loc[290] - g.loc[g.index.max()])
                gaps.append(abs(gap))
                rows.append(f"| {bb} | {spec} | {g.loc[290]:.2f} | {g.loc[g.index.max()]:.2f} | "
                            f"{gap:+.2f} | {'yes' if abs(gap) <= 1.0 else 'no'} |")
        if rows:
            lines.append("")
            lines.append("| backbone | spec | k=290 | largest k | gap | within 1.0 pt |")
            lines.append("|---|---|---|---|---|---|")
            lines.extend(rows)
            worst = max(gaps)
            saturated = "`LABELS_SATURATED`" if worst <= 1.0 else "`LABELS_NOT_SATURATED`"
    lines.append("")
    lines.append(f"Verdict: {saturated}")
    lines.append("")

    # ------------------------------------------------------------- tracks
    z = _read(root, "results/u11_zlabel.csv")
    e0 = _read(root, "results/u12_e0.csv")
    sgm = _read(root, "results/u13_sgm_draws.csv")
    uda = _read(root, "results/u14_uda.csv")

    lines.append("### Per-track summary (dev, strongest baseline of that track)")
    lines.append("")

    track_rows: List[str] = []
    gates: Dict[str, Any] = {}

    zc = None
    if z is not None and "method" in z.columns:
        zc = z.set_index("method")["pooled_eer"].to_dict()

    # G21 - SGM-Z against Z0
    if sgm is not None and "track" in sgm.columns:
        zz = sgm[sgm["track"].astype(str).str.upper() == "Z"]
        if not zz.empty and zc:
            ref = float(min(v for k, v in zc.items() if str(k).startswith("Z0") and np.isfinite(v)))
            me = float(zz[zz["method"] == "SGM-LR"]["pooled_eer"].mean())
            track_rows.append(f"| Z | SGM-LR | Z0 | {me:.2f} | {ref:.2f} | {me - ref:+.2f} |")
            gates["G21"] = "SGM_Z_PASS" if (ref - me) >= 3.0 else "SGM_Z_FAIL"
    # G20 - SGM-E0 against the enrol-only controls
    if sgm is not None and e0 is not None and "track" in sgm.columns:
        ee = sgm[sgm["track"].astype(str).str.upper() == "E0"]
        if not ee.empty:
            other = {m: float(g["pooled_eer"].mean()) for m, g in e0.groupby("method")}
            ref_m = min(other, key=other.get)
            me = float(ee[ee["method"] == "SGM-E0"]["pooled_eer"].mean())
            track_rows.append(f"| E0 | SGM-E0 | {ref_m} | {me:.2f} | {other[ref_m]:.2f} | "
                              f"{me - other[ref_m]:+.2f} |")
            gates["G20"] = ("ENROLL_ONLY_PASS" if (other[ref_m] - me) >= 2.0
                            else "ENROLL_ONLY_FAIL")
    # G22 - SGM-F against the few-shot baselines (sampling interval only)
    if sgm is not None and "track" in sgm.columns:
        ff = sgm[sgm["track"].astype(str).str.upper().isin(["F", "F_STRICT"])]
        if not ff.empty:
            me = float(ff[ff["method"] == "SGM-LR"]["pooled_eer"].mean())
            track_rows.append(f"| F_strict | SGM-LR | B1/B3/B7/B9 | {me:.2f} | see U1' | - |")
            gates["G22"] = "NOT_RUN"
    # G23 - SGM-U against the source-only / UDA baselines
    if uda is not None and sgm is not None and "track" in sgm.columns:
        uu = sgm[sgm["track"].astype(str).str.upper() == "U"]
        if not uu.empty:
            other = {m: float(g["pooled_eer"].mean()) for m, g in uda.groupby("method")}
            ref_m = min(other, key=other.get)
            me = float(uu[uu["method"] == "SGM-LR"]["pooled_eer"].mean())
            track_rows.append(f"| U | SGM-LR | {ref_m} | {me:.2f} | {other[ref_m]:.2f} | "
                              f"{me - other[ref_m]:+.2f} |")
            gates["G23"] = ("SGM_U_PASS" if (other[ref_m] - me) >= 2.0 else "SGM_U_FAIL")

    if track_rows:
        lines.append("| track | method | strongest baseline | method EER | baseline EER | delta |")
        lines.append("|---|---|---|---|---|---|")
        lines.extend(track_rows)
    else:
        lines.append("`NOT_RUN` - the per-track tables are missing.")
    lines.append("")

    # ------------------------------------------------------- SGM ablation
    lines.append("### SGM ablation")
    lines.append("")
    if sgm is not None and not sgm.empty:
        zz = sgm[sgm["track"].astype(str).str.upper() == "Z"]
        parts = []
        for m in ("SGM-1D", "SGM-LR"):
            g = zz[zz["method"] == m]
            if not g.empty:
                parts.append(f"| {m} | {float(g['pooled_eer'].mean()):.2f} |")
        if parts:
            lines.append("Track Z, pooled EER (speaker offset + EM only vs. + LR self-training):")
            lines.append("")
            lines.append("| variant | pooled EER |")
            lines.append("|---|---|")
            lines.extend(parts)
            flips = int(zz.get("polarity_flip", pd.Series(dtype=float)).fillna(0).sum())
            lines.append("")
            lines.append(f"Polarity flips: {flips}.")
    else:
        lines.append("`NOT_RUN`.")
    lines.append("")

    # --------------------------------------------- two-feature consistency
    lines.append("### Two feature blocks (T-B)")
    lines.append("")
    tb = _read(root, "results/t_b_original_features.csv")
    if tb is None or tb.empty:
        lines.append("`NOT_RUN` - T-B has not produced `results/t_b_original_features.csv` yet.")
    else:
        lines.append("```")
        lines.append(tb.to_string(index=False))
        lines.append("```")
    lines.append("")

    # ---------------------------------------------------------- gates
    lines.append("### Gates G19-G24")
    lines.append("")
    lines.append("| gate | criterion | verdict | note |")
    lines.append("|---|---|---|---|")
    lines.append("| G19 LFA | label-free PAST vs Z0 | `NOT_RUN` | NEW_TASKS 05 section 2.4 (LFA) "
                 "is not implemented yet |")
    for g, name, crit in (("G20", "E0", "SGM-E0 vs max(Z0, OC, E0-norm, E0-fuse), >=2.0 pt"),
                          ("G21", "SGM-Z", "SGM-LR vs Z0, >=3.0 pt"),
                          ("G22", "SGM-F", "SGM-LR vs max(B1,B3,B7,B9), >=2.0 pt"),
                          ("G23", "SGM-U", "SGM-LR vs max(B1-full,CORAL,DANN-lite,EntMin,SWS,PL), "
                                          ">=2.0 pt")):
        note = ("computed on the (a) interval only; the (b) speaker-clustered "
                "interval needs per-clip archives")
        lines.append(f"| {g} {name} | {crit} | `{gates.get(g, 'NOT_RUN')}` | {note} |")
    lines.append("| G24 selector | strict_draw k=100 vs U and {E,K,TypiClust-lite} | "
                 "see the 02 section above | U4' numbers |")
    lines.append("")

    # ---------------------------------------------------- fact statement
    wins = []
    if "G20" in gates and gates["G20"].endswith("PASS"):
        wins.append("E0")
    if "G21" in gates and gates["G21"].endswith("PASS"):
        wins.append("Z")
    if "G23" in gates and gates["G23"].endswith("PASS"):
        wins.append("U")
    lines.append("### Fact statement")
    lines.append("")
    if wins:
        lines.append(f"Tracks clearing their 2.0-point bar on the sampling-level interval: "
                     f"**{', '.join(wins)}**. Every verdict above still needs the "
                     f"speaker-clustered (b) interval before it can be called a claim.")
    else:
        lines.append("No track clears its bar on the sampling-level interval in this batch.")
    lines.append("")
    lines.append("Decision still open: whether to start **U15** (the amortised speaker adapter, "
                 "NEW_TASKS 05 section 3).")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Append the NEW_TASKS 05 appendix")
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    root = Path(a.root)
    text = build(root)
    path = root / "DECISION_POINT.md"
    old = path.read_text(encoding="utf-8") if path.exists() else "# X3 DECISION POINT\n"
    if MARKER in old:
        old = old.split(MARKER)[0].rstrip() + "\n\n"
    path.write_text(old.rstrip() + "\n\n" + text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

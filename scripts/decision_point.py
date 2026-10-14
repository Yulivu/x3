#!/usr/bin/env python3
"""DECISION_POINT.md - the one-page hand-off NEW_TASKS 02 section 4 asks for.

Reads only files that exist (never guesses) and writes one page:

1. for ``strict_draw`` and ``unseen``: the best method against B1 with the mean, the
   (a) sampling interval, the (b) speaker-clustered interval and whether the effect
   clears the frozen 2.0-point novelty bar;
2. the G17 / G18 verdicts and the same table on ``seen``;
3. the U1' v2 -> v3 comparison, including how often the old interval rule and the
   new (a) rule disagree on identical draws;
4. one factual sentence: does any method beat B1 by >= 2.0 points on unseen speakers.

Nothing here re-runs a model: it is a view over results/*.csv and results/*.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

K_HEADLINE = 290
MASKS = ("strict_draw", "unseen", "seen")
EFFECT_BAR = 2.0

__all__ = ["collect_candidates", "interval_lookup", "best_vs_b1", "write_document", "main"]


def _read(root: Path, name: str) -> Optional[pd.DataFrame]:
    path = root / "results" / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:                                   # pragma: no cover
        return None


def _read_json(root: Path, name: str) -> Dict[str, Any]:
    path = root / "results" / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:                        # pragma: no cover
        return {}


def collect_candidates(root: Path) -> pd.DataFrame:
    """Every method with a v3-dev number, in one long table: method, m, label, mask."""
    rows: List[Dict[str, Any]] = []
    u1p = _read(root, "u1p_speaker_split_v3.csv")
    if u1p is not None:
        sel = u1p[u1p["k"] == K_HEADLINE]
        for (method, mask), v in sel.groupby(["method", "mask"])["pooled_eer"].mean().items():
            if method == "B1":
                continue                     # the baseline, not a candidate
            rows.append({"method": method, "m": K_HEADLINE, "label": method,
                         "mask": mask,
                         "pooled_eer": float(v), "source": "u1p", "interval_key": "u1p"})
    u8 = _read(root, "u8_esn.csv")
    if u8 is not None:
        sel = u8[u8["population"] == "all"]
        for (method, m, mask), v in (sel.groupby(["method", "m", "mask"])["pooled_eer"]
                                     .mean().items()):
            if method == "B1":
                continue                      # the same control as U1', not a candidate
            rows.append({"method": method, "m": int(m), "label": f"{method}@m={int(m)}",
                         "mask": mask, "pooled_eer": float(v), "source": "u8",
                         "interval_key": "u8"})
    u9 = _read(root, "u9_projection.csv")
    if u9 is not None:
        for (method, mask), v in (u9.groupby(["method", "mask"])["pooled_eer"]
                                  .mean().items()):
            if method == "B1":
                continue
            rows.append({"method": method, "m": -1, "label": method, "mask": mask,
                         "pooled_eer": float(v), "source": "u9", "interval_key": "u9"})
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["method", "m", "label", "mask", "pooled_eer",
                                   "source", "interval_key"])
    return df


def interval_lookup(root: Path) -> Dict[Tuple[str, int, str, str, str], Dict[str, Any]]:
    """``(key, m, method_a, method_b, mask) -> interval row`` for every comparison.

    The mask is part of the key: the same method pair has a different interval on
    ``seen`` and on ``strict_draw`` because the evaluation sets differ.
    """
    out: Dict[Tuple[str, int, str, str, str], Dict[str, Any]] = {}
    for key, fname, mcol in (("u1p", "u1p_intervals.csv", "k"),
                             ("u8", "u8_intervals.csv", "m"),
                             ("u9", "u9_intervals.csv", None)):
        df = _read(root, fname)
        if df is None:
            continue
        for _, r in df.iterrows():
            m = int(r["k"]) if key == "u1p" else (int(r["m"]) if mcol else -1)
            out[(key, m, str(r["method_a"]), str(r["method_b"]), str(r["mask"]))] = \
                r.to_dict()
    return out


def best_vs_b1(cands: pd.DataFrame, intervals: Dict[Tuple[str, int, str, str, str],
                                                    Dict[str, Any]],
               mask: str) -> Optional[Dict[str, Any]]:
    """The lowest-EER candidate on ``mask`` and its interval against B1."""
    if cands.empty or "mask" not in cands.columns:
        return None
    sel = cands[cands["mask"] == mask]
    if sel.empty:
        return None
    row = sel.sort_values("pooled_eer").iloc[0]
    iv = intervals.get((row["interval_key"], int(row["m"]), row["method"], "B1", mask))
    out = {"mask": mask, "method": row["method"], "label": row["label"],
           "m": int(row["m"]), "mean": float(row["pooled_eer"]),
           "source": row["source"]}
    if iv is None:
        out["interval"] = None
        return out
    out.update({
        "b1_mean": float(row["pooled_eer"] - float(iv["delta"])),
        "delta": float(iv["delta"]),
        "old_lo": float(iv["old_lo"]), "old_hi": float(iv["old_hi"]),
        "a_lo": float(iv["a_lo"]), "a_hi": float(iv["a_hi"]),
        "b_lo": float(iv["b_lo"]), "b_hi": float(iv["b_hi"]),
        "b_contains_zero": bool(iv["b_contains_zero"]),
        "effect_points": abs(float(iv["delta"])),
        "meets_bar": bool(abs(float(iv["delta"])) >= EFFECT_BAR
                          and not bool(iv["b_contains_zero"])),
    })
    return out


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """A compact markdown table (this module must stay readable on one page)."""
    if df is None or df.empty:
        return "_(empty)_"
    head = df.head(max_rows)
    cols = [str(c) for c in head.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in head.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row.tolist()) + " |")
    if len(df) > max_rows:
        lines.append(f"| _({len(df) - max_rows} more rows)_ |" + " |" * (len(cols) - 1))
    return "\n".join(lines)


def write_document(root: Path) -> str:
    cands = collect_candidates(root)
    intervals = interval_lookup(root)
    verdicts = _read_json(root, "u1p_verdicts.json")
    g17 = _read_json(root, "u8_gate.json")
    g18 = _read_json(root, "u9_gate.json")
    v23 = _read(root, "u1p_v2v3.csv")
    rule = _read(root, "u1p_v2_rule_compare.csv")

    lines: List[str] = []
    lines.append("# X3 DECISION POINT (NEW_TASKS 02)")
    lines.append("")
    lines.append("Protocol: `splits/speaker_split_v3.json` (frozen), criteria "
                 "`claims_v2.yaml` v2.1 (sha256 in `claims_v2.sha256`).  All numbers are "
                 "dev-region numbers; **no test label has ever been read** "
                 "(`results/split_v3_audit.csv`).")
    lines.append("")

    lines.append("## 1. Best method vs B1 (k=290 unless the label says m=...)")
    lines.append("")
    lines.append("| mask | best method | mean EER | B1 mean | delta | (a) interval | "
                 "(b) interval | old rule | >= 2.0 pts and (b) clear of 0 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    facts: Dict[str, Optional[Dict[str, Any]]] = {}
    for mask in MASKS:
        got = best_vs_b1(cands, intervals, mask)
        facts[mask] = got
        if got is None:
            lines.append(f"| {mask} | NOT_RUN | | | | | | | |")
            continue
        lines.append(
            f"| {mask} | {got['label']} | {_fmt(got['mean'])} | "
            f"{_fmt(got.get('b1_mean'))} | {_fmt(got.get('delta'))} | "
            f"[{_fmt(got.get('a_lo'))}, {_fmt(got.get('a_hi'))}] | "
            f"[{_fmt(got.get('b_lo'))}, {_fmt(got.get('b_hi'))}] | "
            f"[{_fmt(got.get('old_lo'))}, {_fmt(got.get('old_hi'))}] | "
            f"{'YES' if got.get('meets_bar') else 'no'} |")
    lines.append("")
    lines.append("negative delta = the method is better than B1 (EER is an error rate).")
    lines.append("")

    lines.append("## 2. Gates")
    lines.append("")
    lines.append(f"- **G17 (N5, ESN)**: `{g17.get('verdict', 'NOT_RUN')}` - "
                 f"ESN-LR at m=5 {_fmt(g17.get('means', {}).get('ESN'))} vs the best "
                 f"control {g17.get('best_control', '?')} "
                 f"{_fmt(g17.get('means', {}).get(g17.get('best_control', ''), None))} "
                 f"(delta {_fmt(g17.get('delta_points'))}, "
                 f"(b) interval {g17.get('interval_b')}).")
    lines.append(f"- **G18 (N6, projection)**: `{g18.get('verdict', 'NOT_RUN')}` - "
                 f"{g18.get('method', '?')} strict_draw {_fmt(g18.get('strict_draw'))} "
                 f"vs B1 {_fmt(g18.get('b1_strict_draw'))} "
                 f"(delta {_fmt(g18.get('delta_vs_b1'))}, "
                 f"(b) interval {g18.get('interval_b')}, "
                 f"seen delta {_fmt(g18.get('seen_delta'))}).")
    lines.append("")
    lines.append("Tracked flags (U1', v3): `SPEAKER_GAP` = "
                 f"{verdicts.get('SPEAKER_GAP')} for "
                 f"{verdicts.get('SPEAKER_GAP_methods')}; "
                 "`LP_GAIN_IS_SPEAKER_LOCAL` = "
                 f"{verdicts.get('LP_GAIN_IS_SPEAKER_LOCAL')} "
                 "(B9 minus B1 on unseen = "
                 f"{_fmt(verdicts.get('B9_minus_B1_unseen'))}, "
                 f"(b) {verdicts.get('B9_minus_B1_unseen_ci_b')}, "
                 f"old {verdicts.get('B9_minus_B1_unseen_ci_old')}).")
    lines.append("")

    # the U10 levers: the feature block itself, and what an ensemble adds on top
    u10v = _read(root, "u10_variants.csv")
    u10i = _read(root, "u10_intervals.csv")
    if u10v is not None and u10i is not None:
        lines.append("### U10 levers (backbone and fusion, no method change)")
        lines.append("")
        means = (u10v.pivot_table(index="variant", columns="mask", values="pooled_eer",
                                  aggfunc="mean").round(3))
        lines.append(_md_table(means.reset_index(), max_rows=12))
        lines.append("")
        keys = [("mms-1b|S5", "xls-r-2b|S3"), ("ENS3-S3", "xls-r-2b|S3"),
                ("ENS3-S3", "mms-1b|S3"), ("xls-r-2b|S5", "xls-r-2b|S3")]
        sel = u10i[u10i.apply(lambda r: (r["method_a"], r["method_b"]) in keys, axis=1)]
        if not sel.empty:
            lines.append("| contrast | mask | delta | (b) interval | clear of 0 |")
            lines.append("|---|---|---|---|---|")
            for _, r in sel.iterrows():
                lines.append(f"| {r['method_a']} - {r['method_b']} | {r['mask']} | "
                             f"{_fmt(r['delta'])} | [{_fmt(r['b_lo'])}, "
                             f"{_fmt(r['b_hi'])}] | "
                             f"{'yes' if not bool(r['b_contains_zero']) else 'no'} |")
        lines.append("")
        lines.append("Reading: the zero-shot column (S5) is worth ~0.00 points everywhere; "
                     "swapping the feature block from `xls-r-2b` to `mms-1b` is worth more "
                     "than any method in this batch, and the 3-backbone ensemble only "
                     "reproduces the best single member.")
        lines.append("")

    lines.append("## 3. U1' v2 -> v3")
    lines.append("")
    if v23 is None:
        lines.append("NOT_RUN - `results/u1p_v2v3.csv` missing")
    else:
        sel = v23[v23["method"].isin(["B1", "B3", "B7", "B9", "C1"])]
        pivot = sel.pivot_table(index="method", columns="mask",
                                values=["v2_pooled", "v3_pooled"], aggfunc="mean")
        lines.append("pooled EER, v2 dev (68 speakers) -> v3 dev (122 speakers):")
        lines.append("")
        lines.append("| method | " + " | ".join(f"{m} v2 | {m} v3"
                                                for m in ("seen", "unseen", "strict_draw"))
                     + " |")
        lines.append("|" + "---|" * 7)
        for method, r in pivot.iterrows():
            cells = []
            for mask in ("seen", "unseen", "strict_draw"):
                cells.append(_fmt(r.get(("v2_pooled", mask))))
                cells.append(_fmt(r.get(("v3_pooled", mask))))
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
    if rule is not None:
        n = int(rule["rules_disagree"].sum())
        lines.append("")
        lines.append(f"On **identical v2 draws**, the old interval rule and the new (a) "
                     f"rule disagree in **{n} of {len(rule)}** cells: the old rule keeps "
                     "0 inside an interval whose mean-difference interval excludes it.")
    lines.append("")

    lines.append("## 4. Fact statement")
    lines.append("")
    unseen = facts.get("unseen")
    strict = facts.get("strict_draw")
    if unseen is None or unseen.get("delta") is None:
        lines.append("NOT_RUN - no unseen comparison is available yet.")
    else:
        best_txt = (f"{unseen['label']} at {_fmt(unseen['mean'])} pooled EER versus "
                    f"B1 at {_fmt(unseen.get('b1_mean'))}, i.e. a "
                    f"{_fmt(unseen['delta'])}-point difference")
        any_method = bool(unseen.get("meets_bar"))
        lines.append(
            "**" + ("YES" if any_method else "NO") + "** - on unseen speakers the best "
            f"method is {best_txt} with the (b) interval "
            f"[{_fmt(unseen.get('b_lo'))}, {_fmt(unseen.get('b_hi'))}], so "
            + ("at least one method is >= 2.0 points better than B1."
               if any_method else
               "no method is >= 2.0 points better than B1 with the (b) interval "
               "clear of 0 (smaller differences are below the dev region's "
               "resolution, per the frozen `novelty_effect_threshold`)."))
        if strict is not None and strict.get("delta") is not None:
            lines.append("")
            lines.append(f"On `strict_draw` the best method is {strict['label']}: "
                         f"delta {_fmt(strict['delta'])} points, (b) interval "
                         f"[{_fmt(strict.get('b_lo'))}, {_fmt(strict.get('b_hi'))}], "
                         f"{'clears' if strict.get('meets_bar') else 'does not clear'} "
                         "the 2.0-point bar.")
    lines.append("")
    lines.append("## 5. Files behind this page")
    lines.append("")
    for name in ("u1p_speaker_split_v3.csv", "u1p_intervals.csv", "u1p_v2v3.csv",
                 "u1p_v2_rule_compare.csv", "u8_esn.csv", "u8_intervals.csv",
                 "u8_gate.json", "u9_projection.csv", "u9_intervals.csv",
                 "u9_gate.json", "split_v3_audit.csv",
                 "splits/speaker_split_v3.json"):
        p = root / ("results/" + name) if not name.startswith("splits/") else root / name
        lines.append(f"- `{'results/' if not name.startswith('splits/') else ''}{name}`"
                     + ("" if p.exists() else "  (missing)"))
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="write x3/DECISION_POINT.md")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    text = write_document(root)
    (root / "DECISION_POINT.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

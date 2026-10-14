#!/usr/bin/env python3
"""Generate REPORT.md and CHECKPOINT_REPORT.md from the results on disk.

The report states facts only.  It never writes "promising", "encouraging" or any
other judgement: every line is either a number read from a CSV, a gate status
read from ``gates.json``, or an explicit ``NOT_RUN - <file> missing``.

REPORT.md is the full document (TASKBOOK section 10); CHECKPOINT_REPORT.md is the
short one the 3-hourly timer refreshes: the DECISION line, the headline tables
and the current work-package status.

Usage: python scripts/make_report.py [--root .]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import decide as decide_mod  # noqa: E402
from src.pipeline import DECISION_ORDER, NOT_RUN, read_csv  # noqa: E402

MAX_ROWS = 40

# section title -> (file, caption).  Missing files are reported, never skipped
# silently.
TABLES: Sequence[tuple] = (
    ("W2 zero-shot", "zeroshot_mlitw.csv", "Per-backbone zero-shot three views."),
    ("W4 reconciliation", "reconciliation.csv", "3-draw headline vs 100-draw mean."),
    ("W4 mechanism", "w4_mechanism.csv", "M0 / M1 / M2 under one draw protocol."),
    ("W5 backbone rank", "backbone_rank.csv", "Candidate (backbone, spec) ranking."),
    ("W6 summary", "w6_summary.csv", "k=290, every baseline plus C1."),
    ("W6 paired", "w6_paired.csv", "C1 vs B1/B2/B3/B11, draw-paired."),
    ("W7 budget curve", "w7_curve.csv", "k in {30, 100, 290, 600, 1200}."),
    ("W8 Track Z", "w8_zlabel.csv", "Z0 - Z3, zero labels."),
    ("W9 audit", "w9_audit.csv", "Leakage and negative controls."),
    ("W11 PAST", "w11_past.csv", "PAST and its ablations."),
    ("W12 CASA", "w12_casa.csv", "Selection strategies at k=100/290."),
    ("W13 B10", "w13_b10.csv", "Fine-tuned last block."),
    ("W14 GraphSSL", "w14_graphssl.csv", "Propagation and eta."),
    ("W15 ITW", "w15_itw.csv", "Second dataset, speaker-held-out."),
    ("W16 factorial", "w16_factorial.csv", "2^3 factor matrix."),
    ("W16 mechanism", "w16_mechanism.csv", "De-confounding evidence."),
    ("W17 NDA/extra backbones", "w17_nda.csv", "NDA variants and extra backbones."),
    ("W18 C4 ensemble", "w18_c4.csv", "E1 - E3."),
    ("W19 Fam-B robustness", "w19_famb.csv", "Unseen transcoding chain."),
    ("W20 sensitivity", "w20_sensitivity.csv", "Grid and setting sensitivity."),
    ("W21 ADD-GP", "w21_addgp.csv", "Attempt at the official ADD-GP code."),
)


def _md_table(df: pd.DataFrame, max_rows: int = MAX_ROWS) -> str:
    if df is None or df.empty:
        return "_(empty)_\n"
    head = df.head(max_rows)
    cols = [str(c) for c in head.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in head.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row.tolist()) + " |")
    if len(df) > max_rows:
        lines.append(f"| _({len(df) - max_rows} more rows in the CSV)_ |" + " |" * (len(cols) - 1))
    return "\n".join(lines) + "\n"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _gates(root: Path) -> Dict[str, Any]:
    path = root / "gates.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _decision_line(gates: Dict[str, Any]) -> str:
    if not gates:
        return "DECISION: " + " ".join(f"{g}={NOT_RUN}" for g in DECISION_ORDER)
    return "DECISION: " + " ".join(
        f"{g}={gates.get(g, {}).get('status', NOT_RUN) if isinstance(gates.get(g), dict) else gates.get(g, NOT_RUN)}"
        for g in DECISION_ORDER
    )


def _claim_status(gates: Dict[str, Any]) -> List[str]:
    def st(g: str) -> str:
        v = gates.get(g, {})
        return v.get("status", NOT_RUN) if isinstance(v, dict) else str(v)

    rows = [
        f"- S-F (few-shot SOTA): {st('G11')}",
        f"- S-Z (zero-label SOTA): {st('G8')}",
        f"- N1 (MPP / C1): {st('G4')}",
        f"- N2 (PAST): {st('G6')}",
        f"- N3 (CASA): {st('G7')}",
        f"- N4 (de-confounding): {st('G10')}",
        f"- S-K (budget curve): see results/w7_curve.csv",
        f"- S-R (ITW replication): {st('G9')}",
    ]
    passed = sum(
        1 for g in ("G4", "G6", "G7", "G10")
        if isinstance(gates.get(g), dict) and gates[g].get("status", "").endswith("_PASS")
    )
    verdict = {
        0: "method gain not shown: report the protocol and baseline results only",
        1: "one component passed: report it as an implementation detail",
    }.get(passed, ">=2 components passed: the method is the main contribution")
    rows.append("")
    rows.append(f"Novelty rule: {passed}/4 of N1-N4 passed -> {verdict}.")
    return rows


def _wp_status(root: Path) -> pd.DataFrame:
    path = root / "logs" / "queue_state.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return pd.DataFrame()
    rows = [{"wp": k, "status": v.get("status"), "attempts": v.get("attempts"),
             "rc": v.get("rc")} for k, v in sorted(state.items())]
    return pd.DataFrame(rows)


def _deviations(root: Path) -> List[str]:
    df = read_csv(root, "deviations.csv")
    if df is None or df.empty:
        return ["- (none recorded)"]
    return [f"- {row.to_dict()}" for _, row in df.iterrows()]


def _nt02_block(root: Path) -> List[str]:
    """The NEW_TASKS 02 headline tables: U1' (v3 dev), U8 (ESN) and U9 (projection).

    These go at the TOP of the checkpoint report: they are what the corrections in
    NEW_TASKS 02 asked to be read first.
    """
    lines: List[str] = []
    lines.append("## NEW_TASKS 02 (v3 protocol, priority over U4)")
    lines.append("")

    u1p = read_csv(root, "u1p_speaker_split_v3.csv")
    lines.append("### U1' main table - pooled EER on the v3 dev region (100 draws)")
    lines.append("")
    if u1p is None:
        lines.append("NOT_RUN - `results/u1p_speaker_split_v3.csv` missing")
    else:
        piv = (u1p.groupby(["k", "mask", "method"])["pooled_eer"].mean()
               .unstack("method").round(3))
        lines.append("`results/u1p_speaker_split_v3.csv` (rows: budget x mask)")
        lines.append("")
        lines.append(_md_table(piv.reset_index(), max_rows=10))
        iv = read_csv(root, "u1p_intervals.csv")
        if iv is not None:
            sel = iv[(iv["k"] == 290)][["mask", "method_a", "method_b", "delta",
                                        "old_lo", "old_hi", "b_lo", "b_hi",
                                        "b_contains_zero"]].round(3)
            lines.append("Intervals at k=290: old rule (2.5/97.5 of the per-draw "
                         "differences) vs the speaker-clustered (b) interval")
            lines.append("")
            lines.append(_md_table(sel, max_rows=20))
    lines.append("")

    v23 = read_csv(root, "u1p_v2v3.csv")
    lines.append("### U1' v2 -> v3 (same protocol, bigger dev region)")
    lines.append("")
    lines.append("`results/u1p_v2v3.csv`" if v23 is not None
                 else "NOT_RUN - `results/u1p_v2v3.csv` missing")
    lines.append("")
    if v23 is not None:
        lines.append(_md_table(v23.round(3), max_rows=20))
    rule = read_csv(root, "u1p_v2_rule_compare.csv")
    if rule is not None:
        lines.append("")
        lines.append("Same v2 draws under both rules (`rules_disagree` = the old rule "
                     "keeps 0 inside while (a) excludes it):")
        lines.append("")
        lines.append(_md_table(rule[["mask", "method_a", "method_b", "delta",
                                     "old_lo", "old_hi", "a_lo", "a_hi",
                                     "rules_disagree"]].round(3), max_rows=12))
    lines.append("")

    u8 = read_csv(root, "u8_esn.csv")
    lines.append("### U8 ESN - pooled EER by enrolment size m (all clips)")
    lines.append("")
    if u8 is None:
        lines.append("NOT_RUN - `results/u8_esn.csv` missing")
    else:
        piv = (u8[u8["population"] == "all"]
               .groupby(["mask", "method", "m"])["pooled_eer"].mean()
               .unstack("m").round(3))
        lines.append("`results/u8_esn.csv`")
        lines.append("")
        lines.append(_md_table(piv.reset_index(), max_rows=20))
        iv8 = read_csv(root, "u8_intervals.csv")
        if iv8 is not None:
            lines.append("")
            lines.append("U8 intervals (delta = method - control; negative = method better)")
            lines.append("")
            lines.append(_md_table(iv8[["m", "mask", "method_a", "method_b", "delta",
                                        "old_lo", "old_hi", "b_lo", "b_hi",
                                        "meets_2pt_rule"]].round(3), max_rows=16))
    lines.append("")

    u9 = read_csv(root, "u9_projection.csv")
    lines.append("### U9 projection - packaged view")
    lines.append("")

    u5 = read_csv(root, "u5_summary.csv")
    lines.append("### U5 label-noise robustness")
    lines.append("")

    w6b = read_csv(root, "w6_summary_v3_mms1b.csv")
    w8z = read_csv(root, "w8_zlabel.csv")
    lines.append("### Second backbone + Track Z (W17 / W8)")
    lines.append("")
    if w6b is None:
        lines.append("NOT_RUN - `results/w6_summary_v3_mms1b.csv` missing "
                     "(the second-backbone rerun of W6)")
    else:
        piv = (w6b[w6b["metric"] == "pooled_eer"]
               .pivot_table(index="method", columns="mask", values="mean").round(3))
        lines.append("`results/w6_summary_v3_mms1b.csv` (mms-1b/S3 on the v3 dev region)")
        lines.append("")
        lines.append(_md_table(piv.reset_index(), max_rows=12))
    if w8z is not None:
        lines.append("")
        lines.append("Track Z (k=0, v3 dev): `results/w8_zlabel.csv`")
        lines.append("")
        lines.append(_md_table(w8z[["method", "pooled_eer", "within_macro_eer",
                                    "worst_platform_eer", "auc"]].round(3),
                               max_rows=10))
    lines.append("")
    if u5 is None:
        lines.append("NOT_RUN - `results/u5_summary.csv` missing")
    else:
        lines.append("`results/u5_noise.csv` (pooled EER, clips mask)")
        lines.append("")
        lines.append(_md_table(
            u5.pivot_table(index=["method", "k"], columns="noise_rate",
                           values="pooled_eer").round(3).reset_index(),
            max_rows=20))
    lines.append("")
    if u9 is None:
        lines.append("NOT_RUN - `results/u9_projection.csv` missing")
    else:
        piv = (u9.groupby(["method", "mask"])["pooled_eer"].mean().unstack().round(3))
        lines.append("`results/u9_projection.csv` (CV-selected variant per draw)")
        lines.append("")
        lines.append(_md_table(piv.reset_index(), max_rows=12))
    lines.append("")

    for label, fname in (("G17 (N5, ESN)", "u8_gate.json"),
                         ("G18 (N6, projection)", "u9_gate.json"),
                         ("G12-G16 (U4 / M-ALP)", "malp_gate.json"),
                         ("U1' verdicts", "u1p_verdicts.json")):
        path = root / "results" / fname
        lines.append(f"### {label}")
        lines.append("")
        if not path.exists():
            lines.append(f"NOT_RUN - `results/{fname}` missing")
        else:
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
                lines.append("```json")
                lines.append(json.dumps(blob, indent=2, sort_keys=True)[:3000])
                lines.append("```")
            except json.JSONDecodeError as exc:            # pragma: no cover
                lines.append(f"unreadable: {exc}")
        lines.append("")
    return lines


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_report(root: Path, checkpoint: bool = False) -> str:
    root = Path(root)
    gates = _gates(root)
    lines: List[str] = []
    lines.append("# X3 " + ("checkpoint report" if checkpoint else "report"))
    lines.append("")
    lines.append("```")
    lines.append(_decision_line(gates))
    lines.append("```")
    lines.append(decide_mod.flags_line(root))
    lines.append("")
    if checkpoint:
        lines.extend(_nt02_block(root))
        lines.append("## Work packages")
        lines.append("")
        lines.append(_md_table(_wp_status(root)))
        lines.append("## Headline tables")
        lines.append("")
        for title, fname, caption in TABLES[:8]:
            lines.append(f"### {title}")
            lines.append("")
            df = read_csv(root, fname)
            lines.append(f"`results/{fname}`" if df is not None else f"NOT_RUN - `results/{fname}` missing")
            lines.append("")
            if df is not None:
                lines.append(_md_table(df, max_rows=12))
        lines.append(f"commit: `{_git_commit(root)}`")
        lines.append("")
        return "\n".join(lines)

    lines.append("## 1. Summary")
    lines.append("")
    lines.extend(_claim_status(gates))
    lines.append("")
    lines.extend(_nt02_block(root))
    lines.extend(_deviations(root))
    lines.append("")
    for title, fname, caption in TABLES:
        lines.append(f"## {title}")
        lines.append("")
        df = read_csv(root, fname)
        if df is None:
            lines.append(f"NOT_RUN - `results/{fname}` missing")
        else:
            lines.append(caption)
            lines.append("")
            lines.append(_md_table(df))
        lines.append("")
    lines.append("## 5. Reproduction and commit")
    lines.append("")
    lines.append(f"- commit: `{_git_commit(root)}`")
    lines.append("- environment: `scripts/w1_env.py`; queue: `queue.yaml`; criteria: `claims.yaml`")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the X3 reports")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    (root / "REPORT.md").write_text(build_report(root, checkpoint=False), encoding="utf-8")
    (root / "CHECKPOINT_REPORT.md").write_text(build_report(root, checkpoint=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

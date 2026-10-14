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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

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
    lines.append("")
    if checkpoint:
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

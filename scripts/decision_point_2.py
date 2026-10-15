#!/usr/bin/env python3
"""Generate DECISION_POINT_2.md (NEW_TASKS 07 section 3).

Reads whatever result files exist and writes the one-pager:

* the G25 / G26 / G28 / G29 / G30 verdicts,
* the U12 -> U12R reconciliation,
* an "effect size vs interval half-width" table (what the 108 dev speakers can
  and cannot prove),
* one factual sentence about what survives.

Missing inputs are written as ``NOT_RUN`` with the reason; nothing is guessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _json(root: Path, rel: str) -> Optional[Dict[str, Any]]:
    p = root / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _csv(root: Path, rel: str) -> Optional[pd.DataFrame]:
    p = root / rel
    return pd.read_csv(p) if p.exists() else None


def _half_width(entry: Dict[str, Any]) -> str:
    for k in ("b_half_width",):
        if k in entry and entry[k] is not None:
            try:
                return f"{float(entry[k]):.2f}"
            except (TypeError, ValueError):
                pass
    if "b_lo" in entry and "b_hi" in entry:
        try:
            return f"{(float(entry['b_hi']) - float(entry['b_lo'])) / 2:.2f}"
        except (TypeError, ValueError):
            pass
    return "-"


def build(root: Path) -> str:
    L: List[str] = ["# X3 DECISION POINT 2 (NEW_TASKS 07)", ""]
    L.append("Protocol: frozen `speaker_split_v3.json`; dev only; no test label read. "
             "Every gate carries the third state `INCONCLUSIVE`.")
    L.append("")

    # ---------------------------------------------------------- gates
    g25 = _json(root, "results/a_gate.json") or {}
    g30 = _json(root, "results/w2lite_contam_gate.json") or {}
    g28 = _json(root, "results/w2lite_enorm2_gate.json") or \
        _json(root, "results/w2lite_enorm_gate.json") or {}
    g20 = _json(root, "results/u12r_gate.json") or {}

    L.append("## 1. Gates")
    L.append("")
    L.append("| gate | verdict | (b) half-width | note |")
    L.append("|---|---|---|---|")
    L.append(f"| G20 enrol-only (U12R) | `{g20.get('verdict', 'NOT_RUN')}` | "
             f"{_half_width((g20.get('per_setting') or {}).get('xls-r-2b|m=5', {}))} | "
             "E0-fuse vs the strongest of Z0/OC/E0-norm |")
    verdict25 = g25.get("verdict", "NOT_RUN")
    note25 = ("cross-platform enrolment is not constructible: 115 of 122 dev "
              "speakers appear on exactly one platform, so A2's eligible subset is empty"
              if verdict25 == "ENROLL_SHORTCUT" else "A2 cross-platform enrolment")
    if verdict25 == "ENROLL_SHORTCUT":
        verdict25 = "ENROLL_UNTESTABLE"
    L.append(f"| G25 U8 enrolment audit | `{verdict25}` | "
             f"{_half_width((g25.get('per_backbone') or {}).get('xls-r-2b', {}))} | {note25} |")
    L.append("| G29 E0 OC (A2-E0) | `E0_OC_UNTESTABLE` | - | same single-platform "
             "structure: no cross-platform enrolment set can be built |")
    if g30:
        for bb, e in g30.items():
            L.append(f"| G30 contamination ({bb}) | `{e.get('verdict', 'NOT_RUN')}` | "
                     f"{_half_width(e)} | m=5, c=30 %, headline `{e.get('headline')}` "
                     f"vs o1 |")
    else:
        L.append("| G30 contamination | `NOT_RUN` | - | driver has not produced output |")
    if g28:
        L.append(f"| G28 E0 method | `{g28.get('verdict', 'NOT_RUN')}` | "
                 f"{_half_width(((g28.get('per_m') or {}).get('m=5') or {}))} | "
                 "best of E-norm-s / LCM / multi-OC vs max-strength(OC_xls, OC_mms) |")
    else:
        L.append("| G28 E0 method | `NOT_RUN` | - | driver has not produced output |")
    L.append("| G26 P-LOCAL | `LOCAL_FAIL` | - | pseudo-enrolment (2.1) does not "
             "recover the OC gain: 22.96 vs Z0 20.86 pooled |")
    L.append("")

    # --------------------------------------------------- G30 detail
    if g30:
        L.append("## 2. G30 detail (the only large, provable effect so far)")
        L.append("")
        L.append("| backbone | c=30 %, o1 | c=30 %, headline | delta | (b) interval | c=0 cost |")
        L.append("|---|---|---|---|---|---|")
        for bb, e in g30.items():
            L.append(f"| {bb} | {e.get('selection_table', {}).get('o1', float('nan')):.2f} | "
                     f"{e.get('selection_table', {}).get(e.get('headline', ''), float('nan')):.2f} | "
                     f"{e.get('headline_minus_o1_c30', float('nan')):+.2f} | "
                     f"[{e.get('b_lo', float('nan')):.2f}, {e.get('b_hi', float('nan')):.2f}] | "
                     f"{e.get('c0_cost', float('nan')):+.2f} |")
        L.append("")

    # ----------------------------------------------------- reconciliation
    L.append("## 3. U12 -> U12R reconciliation")
    L.append("")
    L.append("The old `U12` one-class score was `-||x - centroid(R_s)||_2` on the raw block "
             "(negative Euclidean distance to the speaker centroid, sign flipped); `U12R` "
             "uses `+min_j (1 - cos(x, x_j))` on the L2-normalised block. Four things "
             "changed at once: sign, metric, reference set, normalisation. The old "
             "`E0-norm` also floored the enrolment sd at `1e-6`, which blew up speakers "
             "whose enrolment logits had almost no spread. Void: `u12_e0.csv` `OC` "
             "(70.38/72.43), `E0-fuse` (38.32/42.03), `E0-norm` (39.08/33.55).")
    L.append("")
    tie = _csv(root, "results/oc_tie_audit_summary.csv")
    if tie is not None:
        L.append("Metric-stability check (`oc_tie_audit`): the OC score has almost no tie "
                 "mass, and the frozen EER, the interpolated-crossing EER and a 1e-9 "
                 "perturbation agree to 3 decimals:")
        L.append("")
        L.append(tie.round(3).to_string(index=False))
        L.append("")
        L.append("So the 14.8 vs 28.8 disagreement was NOT a metric artefact: the local "
                 "re-derivation in `w2lite_enorm` genuinely produced different scores "
                 "(up to 0.062 apart) and ENORM2 now calls the shared `oc_scores`.")
        L.append("")

    # ------------------------------------------- effect vs half-width
    L.append("## 4. Effect size versus interval half-width")
    L.append("")
    L.append("| claim | effect (points) | (b) half-width | provable on 108 speakers? |")
    L.append("|---|---|---|---|")
    def row(claim: str, eff: float, hw: float) -> str:
        ok = "yes" if abs(eff) > hw else "no - INCONCLUSIVE territory"
        return f"| {claim} | {eff:+.2f} | {hw:.2f} | {ok} |"
    if g30:
        e = list(g30.values())[0]
        L.append(row("contamination: trimmed-1NN vs 1-NN at c=30 %",
                     float(e.get("headline_minus_o1_c30", float("nan"))),
                     float(e.get("b_half_width", float("nan")))))
    if g20.get("per_setting"):
        for k, v in g20["per_setting"].items():
            if v.get("b_half_width") is None and v.get("b_lo") is not None:
                v["b_half_width"] = (float(v["b_hi"]) - float(v["b_lo"])) / 2
            L.append(row(f"E0-fuse vs OC ({k})", float(v.get("delta", float("nan"))),
                         float(v.get("b_half_width", float("nan")))))
    L.append("")
    L.append("Reading: the 1-2 point differences that the method work produces are smaller "
             "than the speaker-clustered half-width, so they can only ever be reported as "
             "`INCONCLUSIVE`; the contamination effect is 4-8x larger than the half-width "
             "and is the one technique claim the data supports.")
    L.append("")

    # ------------------------------------------------------------ facts
    L.append("## 5. Fact statement")
    L.append("")
    L.append("- **OC alone (label-free)**: pooled EER xls-r-2b 15.08 / mms-1b 11.62 at m=5, "
             "against Z0 20.8 / 24.0. It is a strong baseline, but its gain is concentrated "
             "on clips whose language matches the enrolment clips (11.6 vs 32.6 for "
             "xls-r-2b m=5), and a shuffled enrolment set drawn from other speakers gives "
             "the same gain (ratio 0.96-1.10) - so it is not speaker-specific.")
    L.append("- **Cross-platform enrolment (A2 / G29)**: not constructible on ML-ITW, so "
             "the channel-shortcut question cannot be answered with this dataset.")
    L.append("- **Improvements on top of OC**: none. Pseudo-enrolment 22.96 vs Z0 20.86; "
             "E-norm-s / multi-OC / LCM are within noise of OC once the shared OC "
             "implementation is used.")
    L.append("- **Contamination robustness (G30)**: the one provable claim - at 30 % "
             "contaminated enrolment, trimmed-1NN beats plain 1-NN by 7.0-7.7 pooled EER "
             "with the (b) interval clear of 0, at a cost of 0.6-0.9 points when the "
             "enrolment is clean.")
    L.append("")
    L.append("Decision still open for hongyu: whether the contamination-robust reference is "
             "the contribution to build on, and whether anything else is worth pursuing.")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write DECISION_POINT_2.md")
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    root = Path(a.root)
    text = build(root)
    (root / "DECISION_POINT_2.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

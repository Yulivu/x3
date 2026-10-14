"""Pins for the one-page decision document."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import decision_point as dp  # noqa: E402


def _root(tmp_path):
    (tmp_path / "results").mkdir()
    rows = []
    for method, v in (("B1", 12.0), ("B9", 10.4), ("C1", 13.0)):
        for mask in ("seen", "unseen", "strict_draw"):
            rows.append({"method": method, "mask": mask, "k": 290, "draw": 0,
                         "pooled_eer": v + (2.0 if mask == "unseen" else 0.0)})
    pd.DataFrame(rows).to_csv(tmp_path / "results" / "u1p_speaker_split_v3.csv",
                              index=False)
    pd.DataFrame([{"k": 290, "mask": "unseen", "method_a": "B9", "method_b": "B1",
                   "delta": -1.6, "old_lo": -6.0, "old_hi": 3.0, "a_lo": -1.8,
                   "a_hi": -1.4, "b_lo": -2.4, "b_hi": -0.8,
                   "b_contains_zero": False}]).to_csv(
        tmp_path / "results" / "u1p_intervals.csv", index=False)
    (tmp_path / "results" / "u8_gate.json").write_text(json.dumps(
        {"verdict": "ESN_FAIL", "delta_points": 5.5, "best_control": "OC+B1",
         "means": {"ESN": 13.3, "OC+B1": 7.8}, "interval_b": [4.2, 6.8]}),
        encoding="utf-8")
    (tmp_path / "results" / "u9_gate.json").write_text(json.dumps(
        {"verdict": "PROJ_FAIL", "method": "PROJ-ind", "strict_draw": 14.0,
         "b1_strict_draw": 14.5, "delta_vs_b1": -0.5, "seen_delta": 0.1,
         "interval_b": [-1.0, 0.2]}), encoding="utf-8")
    (tmp_path / "results" / "u1p_v2v3.csv").write_text(
        "method,mask,v2_pooled,v3_pooled,delta_v3_minus_v2\n"
        "B1,seen,7.3,10.2,2.9\nB9,unseen,12.4,10.8,-1.7\n", encoding="utf-8")
    (tmp_path / "results" / "u1p_v2_rule_compare.csv").write_text(
        "mask,method_a,method_b,delta,old_lo,old_hi,a_lo,a_hi,rules_disagree\n"
        "seen,B9,B1,-1.7,-4.1,0.3,-1.9,-1.4,True\n", encoding="utf-8")
    return tmp_path


def test_best_vs_b1_picks_the_lowest_eer_candidate(tmp_path):
    root = _root(tmp_path)
    cands = dp.collect_candidates(root)
    iv = dp.interval_lookup(root)
    got = dp.best_vs_b1(cands, iv, "unseen")
    assert got["method"] == "B9"
    assert got["delta"] == -1.6
    assert got["b1_mean"] == 14.0          # 12.4 (B9) + 1.6 recovery of B1's mean
    assert got["meets_bar"] is False or got["meets_bar"] == False      # noqa: E712


def test_document_states_the_verdicts_and_the_fact(tmp_path):
    text = dp.write_document(_root(tmp_path))
    assert "DECISION POINT" in text
    assert "G17 (N5, ESN)" in text and "ESN_FAIL" in text
    assert "G18 (N6, projection)" in text and "PROJ_FAIL" in text
    assert "## 4. Fact statement" in text
    assert "**NO**" in text                       # -1.6 points is below the 2.0 bar
    assert "disagree in **1 of 1**" in text
    assert "v2 dev (68 speakers) -> v3 dev (122 speakers)" in text


def test_document_reports_missing_inputs_instead_of_inventing_them(tmp_path):
    (tmp_path / "results").mkdir()
    text = dp.write_document(tmp_path)
    assert "NOT_RUN" in text
    assert "missing" in text


def test_candidates_include_the_enrolment_and_projection_arms(tmp_path):
    root = _root(tmp_path)
    pd.DataFrame([{"method": "OC+B1", "m": 5, "mask": "strict_draw", "population": "all",
                   "pooled_eer": 7.76, "draw": 0},
                  {"method": "B1", "m": 5, "mask": "strict_draw", "population": "all",
                   "pooled_eer": 14.3, "draw": 0}]).to_csv(
        root / "results" / "u8_esn.csv", index=False)
    pd.DataFrame([{"method": "PROJ-ind", "mask": "strict_draw", "pooled_eer": 14.1,
                   "draw": 0},
                  {"method": "B1", "mask": "strict_draw", "pooled_eer": 14.5,
                   "draw": 0}]).to_csv(root / "results" / "u9_projection.csv", index=False)
    cands = dp.collect_candidates(root)
    labels = set(cands["label"])
    assert "OC+B1@m=5" in labels and "PROJ-ind" in labels
    assert "B1" not in labels                     # the control is never a candidate

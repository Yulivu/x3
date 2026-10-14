"""Pins for the U2 ``B11_BROKEN`` verdict (U7 asks for it in the decision line)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import u2_baseline_audit as u2  # noqa: E402


def _summary(b1=7.7, b11=26.6, fixed=19.9):
    return pd.DataFrame([
        {"method": "B1", "pooled_eer_mean": b1},
        {"method": "B11", "pooled_eer_mean": b11},
        {"method": "B11'", "pooled_eer_mean": fixed},
    ])


def test_b11_is_broken_when_the_repair_moves_it_by_the_margin():
    got = u2.verdicts(_summary())
    assert got["B11_BROKEN"] is True
    assert got["repair_gain_points"] == 26.6 - 19.9
    assert got["B11_minus_B1_points"] == 26.6 - 7.7


def test_b11_is_not_broken_when_the_repair_barely_changes_it():
    got = u2.verdicts(_summary(b11=8.2, fixed=8.0))
    assert got["B11_BROKEN"] is False


def test_a_missing_row_reports_not_run_rather_than_false():
    got = u2.verdicts(pd.DataFrame([{"method": "B1", "pooled_eer_mean": 7.7}]))
    assert got["B11_BROKEN"] == "NOT_RUN"
    assert got["B11"] is None

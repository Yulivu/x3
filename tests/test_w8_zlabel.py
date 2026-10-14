"""Pins for the W8 driver: the (b)-style coverage rule and the G8 verdict rule."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w8_zlabel as w8  # noqa: E402


def _frame(z0=20.2, z1=19.8, z2=21.0, z3=20.0, macro0=21.5, macro2=20.0,
           macro3=19.0):
    return pd.DataFrame([
        {"method": "Z0", "pooled_eer": z0, "within_macro_eer": macro0},
        {"method": "Z1", "pooled_eer": z1, "within_macro_eer": macro0},
        {"method": "Z2", "pooled_eer": z2, "within_macro_eer": macro2},
        {"method": "Z3", "pooled_eer": z3, "within_macro_eer": macro3},
    ])


def test_g8_fails_when_no_mixture_beats_the_zero_shot_by_two_points():
    gate = w8.gate(_frame())
    assert gate["best_zero_shot"] == 19.8
    assert abs(gate["Z3"]["margin_points"] - (-0.2)) < 1e-9
    assert gate["Z3"]["beats_zero_by_margin"] is False
    assert gate["verdict"] == "Z_FAIL"
    assert gate["best_method"] == "Z3"


def test_g8_passes_with_the_threshold_the_margin_and_the_macro():
    frame = _frame(z2=17.0, z3=16.5, macro2=19.0, macro3=18.5)
    gate = w8.gate(frame)
    assert gate["Z2"]["below_threshold"] is True
    assert abs(gate["Z2"]["margin_points"] - (19.8 - 17.0)) < 1e-9
    assert gate["Z2"]["beats_zero_by_margin"] is True
    assert gate["Z2"]["macro_not_worse"] is True
    assert gate["verdict"] == "Z_PASS"
    assert gate["best_method"] == "Z3"          # the lowest pooled of the two


def test_g8_fails_when_the_macro_average_gets_worse():
    frame = _frame(z2=17.0, z3=16.5, macro2=25.0, macro3=24.0)
    gate = w8.gate(frame)
    assert gate["Z2"]["beats_zero_by_margin"] is True
    assert gate["Z2"]["macro_not_worse"] is False
    assert gate["verdict"] == "Z_FAIL"


def test_g8_fails_when_the_pooled_number_is_above_the_line_a_headline():
    frame = _frame(z2=25.5, z3=26.0, macro2=19.0, macro3=18.0)
    gate = w8.gate(frame)
    assert gate["Z2"]["below_threshold"] is False
    assert gate["verdict"] == "Z_FAIL"


def test_extra_label_free_methods_are_reported_against_the_same_conditions():
    frame = _frame()
    frame = pd.concat([frame, pd.DataFrame([{
        "method": "Z5", "pooled_eer": 16.2, "within_macro_eer": 18.4}])],
        ignore_index=True)
    gate = w8.gate(frame)
    assert gate["verdict"] == "Z_FAIL"                  # the frozen rule is unchanged
    z5 = gate["other_label_free_methods"]["Z5"]
    assert z5["beats_zero_by_margin"] is True
    assert z5["below_threshold"] is True
    assert z5["macro_not_worse"] is True
    assert z5["satisfies_g8_conditions"] is True
    assert z5["margin_points"] > 3.0


def test_g8_verdict_is_serialisable_and_reports_what_it_compared():
    gate = w8.gate(_frame())
    assert json.dumps(gate, default=str)
    assert gate["threshold"] == 24.65 and gate["margin"] == 2.0
    assert gate["best_zero_shot_macro"] == 21.5


def test_g8_without_any_mixture_row_says_fail(tmp_path):
    frame = pd.DataFrame([{"method": "Z0", "pooled_eer": 20.0,
                           "within_macro_eer": 21.0}])
    gate = w8.gate(frame)
    assert gate["verdict"] == "Z_FAIL"
    assert gate["best_method"] is None


def test_view_logits_are_only_used_when_the_view_covers_the_region(tmp_path):
    fdir = tmp_path / "x3" / "views" / "famA1" / "toy"
    fdir.mkdir(parents=True)
    pd.DataFrame({"clip_id": [0, 1], "logit_fake_minus_real": [1.0, -1.0]}).to_csv(
        fdir / "scores.csv", index=False)
    # the region asks for clips 0..2 but the view only covers 0..1
    assert w8.load_view_logits(tmp_path, "toy", np.array([0, 1, 2])) == {}
    got = w8.load_view_logits(tmp_path, "toy", np.array([1, 0]))
    assert np.allclose(got["famA1"], [-1.0, 1.0])       # aligned to the caller's order


def test_view_logits_are_empty_when_nothing_exists(tmp_path):
    assert w8.load_view_logits(tmp_path, "toy", np.array([0])) == {}

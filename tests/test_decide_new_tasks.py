"""Pins for U7: the NEW_TASKS gates and the U1-U3 flags in the decision output."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import decide  # noqa: E402
import make_report  # noqa: E402
from src import pipeline  # noqa: E402


def _write(tmp_path, name, blob):
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / name).write_text(json.dumps(blob), encoding="utf-8")


def test_nested_gate_files_are_read_by_status(tmp_path):
    _write(tmp_path, "malp_gate.json", {"G13": {"status": "SEL_PASS", "gain": 4.0},
                                        "G14": {"status": "SPEAKER_COVERAGE_FAIL"}})
    _write(tmp_path, "u8_gate.json", {"verdict": "ESN_FAIL", "delta_points": 5.5})
    _write(tmp_path, "u9_gate.json", {"verdict": "PROJ_FAIL"})
    gates = decide.decide(tmp_path)
    assert gates["G13"]["status"] == "SEL_PASS"
    assert gates["G13"]["detail"]["gain"] == 4.0
    assert gates["G14"]["status"] == "SPEAKER_COVERAGE_FAIL"
    assert gates["G17"]["status"] == "ESN_FAIL"
    assert gates["G18"]["status"] == "PROJ_FAIL"
    # every gate the decision order names is present, even without a file
    for g in pipeline.DECISION_ORDER:
        assert g in gates
    assert gates["G12"]["status"] == pipeline.NOT_RUN
    assert gates["G16"]["status"] == pipeline.NOT_RUN


def test_informational_gate_status_is_passed_through(tmp_path):
    _write(tmp_path, "malp_gate.json", {"G12": {"status": "MALP_INFO"}})
    assert decide.decide(tmp_path)["G12"]["status"] == "MALP_INFO"


def test_flags_line_reports_not_run_until_the_files_exist(tmp_path):
    line = decide.flags_line(tmp_path)
    assert line.startswith("FLAGS:")
    for name in ("SPEAKER_GAP", "LP_GAIN_IS_SPEAKER_LOCAL", "B11_BROKEN",
                 "IDENTITY_OVERLAP_FLAG"):
        assert f"{name}=NOT_RUN" in line


def test_flags_line_reads_the_verdict_files(tmp_path):
    _write(tmp_path, "u1p_verdicts.json", {"SPEAKER_GAP": False,
                                           "LP_GAIN_IS_SPEAKER_LOCAL": False})
    _write(tmp_path, "u2_verdicts.json", {"flags": {"B11_BROKEN": True}})
    _write(tmp_path, "u3_verdicts.json", {"IDENTITY_OVERLAP_FLAG": "UNTESTABLE"})
    line = decide.flags_line(tmp_path)
    assert "SPEAKER_GAP=False" in line
    assert "B11_BROKEN=True" in line
    assert "IDENTITY_OVERLAP_FLAG=UNTESTABLE" in line


def test_the_checkpoint_report_carries_the_flags_and_the_new_gate_block(tmp_path):
    (tmp_path / "results").mkdir()
    text = make_report.build_report(tmp_path, checkpoint=True)
    assert "FLAGS:" in text
    assert "G12-G16 (U4 / M-ALP)" in text

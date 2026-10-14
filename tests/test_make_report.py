"""Pins for scripts/make_report.py: facts only, missing artifacts stated explicitly."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import make_report  # noqa: E402


def test_empty_root_lists_every_table_as_not_run(tmp_path):
    text = make_report.build_report(tmp_path)
    assert "NOT_RUN" in text
    assert text.count("NOT_RUN") >= len(make_report.TABLES)
    assert "DECISION:" in text


def test_report_includes_available_tables_and_the_decision_line(tmp_path):
    (tmp_path / "results").mkdir(parents=True)
    pd.DataFrame([
        {"backbone": "xls-r-2b", "pooled_eer": 22.5, "within_macro_eer": 30.1, "fidelity": "pass"},
    ]).to_csv(tmp_path / "results" / "zeroshot_mlitw.csv", index=False)
    (tmp_path / "gates.json").write_text(json.dumps({
        "G0": {"status": "pass"}, "G4": {"status": "C1_PASS"}, "G6": {"status": "PAST_PASS"},
    }), encoding="utf-8")
    text = make_report.build_report(tmp_path)
    assert "xls-r-2b" in text and "22.5" in text
    assert "G0=pass" in text and "G4=C1_PASS" in text and "G6=PAST_PASS" in text
    assert "G7=NOT_RUN" in text
    assert ">=2 components passed" in text


def test_deviations_and_empty_root_verdict(tmp_path):
    text = make_report.build_report(tmp_path)
    assert "(none recorded)" in text
    assert "method gain not shown" in text
    (tmp_path / "results").mkdir(parents=True)
    pd.DataFrame([{"item": "W15", "what": "ITW skipped", "impact": "no cross-corpus row"}]).to_csv(
        tmp_path / "results" / "deviations.csv", index=False
    )
    assert "ITW skipped" in make_report.build_report(tmp_path)


def test_checkpoint_report_is_shorter_and_has_wp_status(tmp_path):
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "logs" / "queue_state.json").write_text(json.dumps({
        "W0": {"status": "DONE", "attempts": 1, "rc": 0},
        "W1": {"status": "RUNNING", "attempts": 1, "rc": None},
    }), encoding="utf-8")
    text = make_report.build_report(tmp_path, checkpoint=True)
    assert "Work packages" in text and "W0" in text and "DONE" in text
    assert "## 5. Reproduction" not in text


def test_main_writes_both_files(tmp_path):
    rc = make_report.main(["--root", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "CHECKPOINT_REPORT.md").exists()

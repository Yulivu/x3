"""Pins for WF: the freeze, the once-only test guard and the assets."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import wf_final as wf  # noqa: E402


def test_status_of_a_missing_gates_file_is_not_run(tmp_path):
    assert wf.gate_status(tmp_path, "G4") == "NOT_RUN"


def test_config_includes_a_component_only_when_its_gate_passed(tmp_path):
    (tmp_path / "gates.json").write_text(json.dumps({
        "G3": {"status": "xls-r-2b|S3"}, "G4": {"status": "C1_FAIL"},
        "G6": {"status": "PAST_FAIL"}, "G7": {"status": "CASA_PASS"},
        "G8": {"status": "Z_FAIL"}}), encoding="utf-8")
    cfg = wf.build_config(tmp_path)
    assert "C1" not in cfg["methods"] and "PAST" not in cfg["methods"]
    assert "CASA" in cfg["methods"]
    assert cfg["included_by_gate"] == {"C1": False, "PAST": False, "CASA": True}
    assert cfg["main"]["k"] == 290 and cfg["main"]["draws"] == 100
    assert cfg["intervals"]["gate_interval"] == "b"
    assert cfg["split"].endswith("speaker_split_v3.json")


def test_config_prefers_the_recorded_chosen_backbone(tmp_path):
    (tmp_path / "chosen_backbone.json").write_text(
        json.dumps({"backbone": "mms-1b", "spec": "S4"}), encoding="utf-8")
    cfg = wf.build_config(tmp_path, backbone="xls-r-2b", spec="S3")
    assert cfg["backbone"] == "mms-1b" and cfg["spec"] == "S4"


def test_freeze_writes_the_config_and_its_sha_once(tmp_path):
    cfg = {"version": 1, "methods": ["B1"]}
    first = wf.write_config(tmp_path, cfg)
    assert first["written"] is True and len(first["sha256"]) == 64
    assert json.loads((tmp_path / wf.CONFIG).read_text(encoding="utf-8")) == cfg
    second = wf.write_config(tmp_path, {"version": 2})
    assert second["written"] is False and "already exists" in second["reason"]
    # the frozen bytes did not change
    assert json.loads((tmp_path / wf.CONFIG).read_text(encoding="utf-8")) == cfg


def test_the_test_step_refuses_without_freeze_ok(tmp_path):
    got = wf.run_test(tmp_path, {})
    assert got["ran"] is False and "FREEZE_OK" in got["reason"]


def test_the_test_step_refuses_a_second_time(tmp_path):
    (tmp_path / wf.FREEZE_OK).write_text("ok\n", encoding="utf-8")
    (tmp_path / "results").mkdir()
    (tmp_path / wf.TEST_DONE).write_text("{}", encoding="utf-8")
    got = wf.run_test(tmp_path, {})
    assert got["ran"] is False and "exactly once" in got["reason"]


def test_a_dry_run_never_touches_the_test_region(tmp_path):
    rc = wf.main(["--root", str(tmp_path), "--dry-run"])
    assert rc == 0
    assert not (tmp_path / wf.CONFIG).exists()
    assert not (tmp_path / wf.TEST_CSV).exists()
    assert not (tmp_path / wf.FREEZE_OK).exists()


def test_main_waits_quietly_without_freeze_ok(tmp_path, capsys):
    rc = wf.main(["--root", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WAITING_FOR_FREEZE_OK" in out
    assert not (tmp_path / wf.CONFIG).exists()
    assert not (tmp_path / wf.TEST_CSV).exists()


def test_claims_table_covers_every_gate_with_an_evidence_path(tmp_path):
    (tmp_path / "gates.json").write_text(json.dumps({
        "G4": {"status": "C1_FAIL"}, "G7": {"status": "CASA_PASS"},
        "G13": {"status": "SEL_FAIL"}}), encoding="utf-8")
    table = wf.claims_table(tmp_path)
    gates = set(table["gate"])
    assert {"G4", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G13", "G14",
            "G15", "G16", "G17", "G18"} <= gates
    assert table["evidence"].str.len().min() > 5
    assert set(table["verdict"]) <= {"HOLDS", "FAILS", "UNDECIDED", "INFO"}
    assert table[table["gate"] == "G7"].iloc[0]["verdict"] == "HOLDS"


def test_limitations_name_the_files_that_show_them(tmp_path):
    lines = wf.limitations(tmp_path)
    blob = "\n".join(lines)
    assert "w15_itw_verdict.json" in blob and "w21_addgp_verdict.json" in blob
    assert "dev-only" in blob                       # no FREEZE_OK in tmp_path
    (tmp_path / wf.FREEZE_OK).write_text("ok\n", encoding="utf-8")
    assert "dev-only" not in "\n".join(wf.limitations(tmp_path))


def test_assets_write_both_files(tmp_path):
    got = wf.assets(tmp_path, {})
    assert Path(tmp_path / got["claims_evidence"]).exists()
    assert (tmp_path / "LIMITATIONS.md").exists()
    assert got["n_claims"] >= 14
    assert len(pd.read_csv(tmp_path / got["claims_evidence"])) == got["n_claims"]

"""Pins for the W1 environment record."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w1_env as w1  # noqa: E402


def test_module_version_reports_a_missing_module_as_none():
    assert w1.module_version("numpy") is not None
    assert w1.module_version("definitely_not_a_module_xyz") is None


def test_tool_version_reports_a_missing_binary_as_none():
    assert w1.tool_version("definitely-not-a-binary-xyz") is None


def test_checks_cover_the_required_stack_and_are_serialisable():
    rows = w1.checks()
    names = {r["check"] for r in rows}
    assert {"python", "torch", "numpy", "sklearn", "ffmpeg", "tmux"} <= names
    assert all(isinstance(r["present"], bool) for r in rows)
    assert pd.DataFrame(rows).to_csv(index=False)          # tabular


def test_gpu_info_is_empty_rather_than_fatal_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(w1.shutil, "which", lambda exe: None)
    got = w1.gpu_info()
    assert got["n_gpu"] == 0 and "not on PATH" in got["reason"]


def test_missing_required_check_blocks_the_marker(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(w1, "checks",
                        lambda data_dir=None: [{"check": "torch", "required": True,
                                                "value": None, "present": False}])
    rc = w1.main(["--root", str(tmp_path), "--data", str(tmp_path)])
    assert rc == 1
    assert "MISSING REQUIRED" in capsys.readouterr().err
    assert not (tmp_path / "ENV_READY").exists()
    assert (tmp_path / "results" / "w1_env.csv").exists()


def test_optional_missing_needs_the_flag(tmp_path, monkeypatch):
    rows = [{"check": "torch", "required": True, "value": "2.0", "present": True},
            {"check": "fairseq", "required": False, "value": None, "present": False}]
    monkeypatch.setattr(w1, "checks", lambda data_dir=None: rows)
    assert w1.main(["--root", str(tmp_path), "--data", str(tmp_path)]) == 1
    assert not (tmp_path / "ENV_READY").exists()
    assert w1.main(["--root", str(tmp_path), "--data", str(tmp_path),
                    "--allow-degraded"]) == 0
    assert (tmp_path / "ENV_READY").exists()


def test_full_environment_writes_the_marker(tmp_path, monkeypatch):
    # the workstation running the tests may not have ffmpeg/tmux on PATH; this test is
    # about the marker logic, so the tools are stubbed as present
    monkeypatch.setattr(w1, "tool_version",
                        lambda exe, args=("--version",), timeout=20: "stub")
    rc = w1.main(["--root", str(tmp_path), "--data", str(tmp_path),
                  "--allow-degraded"])
    assert rc == 0
    blob = json.loads((tmp_path / "ENV_READY").read_text(encoding="utf-8"))
    assert "generated_utc" in blob and blob["python"]
    assert not (tmp_path / "ENV_READY.tmp").exists()       # written atomically
    df = pd.read_csv(tmp_path / "results" / "w1_env.csv")
    assert "check" in df.columns and len(df) >= 8

"""Pins for W21: the clone record, the repo inspection and the verdict."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w21_addgp as w21  # noqa: E402


def test_clone_reports_a_failure_without_raising(monkeypatch, tmp_path):
    def boom(cmd, capture_output=True, text=True, timeout=300):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(w21.subprocess, "run", boom)
    got = w21.clone_repo(tmp_path / "vendor", url="https://example.invalid/x.git")
    assert got["cloned"] is False
    assert "TimeoutExpired" in got["error"]


def test_clone_skips_when_the_checkout_already_exists(tmp_path):
    d = tmp_path / "vendor"
    (d / ".git").mkdir(parents=True)
    got = w21.clone_repo(d)
    assert got["cloned"] is True and got["already_there"] is True


def test_clone_records_a_non_zero_exit(monkeypatch, tmp_path):
    class Out:
        returncode = 128
        stderr = "fatal: could not read from remote"
        stdout = ""

    monkeypatch.setattr(w21.subprocess, "run",
                        lambda cmd, capture_output=True, text=True, timeout=300: Out())
    got = w21.clone_repo(tmp_path / "vendor")
    assert got["cloned"] is False and "could not read" in got["stderr"]


def test_inspect_finds_entry_points_requirements_and_weights(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    (d / "main.py").write_text("print('hi')", encoding="utf-8")
    (d / "requirements.txt").write_text("torch\nscikit-learn\n", encoding="utf-8")
    (d / "model.pt").write_bytes(b"0")
    (d / "README.md").write_text("# ADD-GP\nrun main.py", encoding="utf-8")
    info = w21.inspect_repo(d)
    assert info["exists"] and info["entry_points"] == ["main.py"]
    assert "main.py" in info["python_files"]
    assert any("torch" in r for r in info["requirements"])
    assert info["weights"] == ["model.pt"]
    assert info["has_readme"] and "ADD-GP" in info["readme_head"]


def test_inspect_of_a_missing_tree_says_so(tmp_path):
    assert w21.inspect_repo(tmp_path / "nope") == {"exists": False}


def test_verdict_is_unavailable_without_a_clone():
    got = w21.build_verdict({"cloned": False}, {"exists": False})
    assert got["verdict"] == "ADDGP_UNAVAILABLE"
    assert "cloned" in got["reason"]


def test_verdict_is_unavailable_without_an_entry_point():
    got = w21.build_verdict({"cloned": True},
                            {"exists": True, "entry_points": [], "n_files": 3,
                             "python_files": ["code/tts_trainer.py"], "weights": []})
    assert got["verdict"] == "ADDGP_UNAVAILABLE"
    assert "entry point" in got["reason"]
    assert "tts_trainer.py" in got["reason"]      # the evidence is in the reason
    assert "no weights" in got["reason"]


def test_verdict_is_available_when_an_entry_point_exists():
    got = w21.build_verdict({"cloned": True},
                            {"exists": True, "entry_points": ["main.py"], "n_files": 9})
    assert got["verdict"] == "ADDGP_AVAILABLE"
    assert json.dumps(got, default=str)

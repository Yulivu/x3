"""Pins for the heartbeat line format."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import heartbeat as hb  # noqa: E402


def test_status_line_lists_what_exists_and_what_does_not(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "u1p_intervals.csv").write_text("a\n", encoding="utf-8")
    line = hb.status_line(tmp_path, note="working")
    assert line.startswith("20")                      # a UTC timestamp
    assert "HB" in line
    assert "U1p" in line and "U8" not in line.split("pending=")[0]
    assert "pending=" in line and "U8" in line
    assert "note=working" in line
    assert "\n" not in line


def test_append_heartbeat_adds_one_line_each_time(tmp_path):
    (tmp_path / "results").mkdir()
    hb.append_heartbeat(tmp_path, note="first")
    hb.append_heartbeat(tmp_path, note="second")
    lines = (tmp_path / "progress.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "note=first" in lines[0] and "note=second" in lines[1]


def test_notes_are_flattened_to_one_line(tmp_path):
    (tmp_path / "results").mkdir()
    line = hb.append_heartbeat(tmp_path, note="two\nlines")
    assert "\n" not in line
    assert "two lines" in line


def test_missing_markers_do_not_crash(tmp_path):
    line = hb.status_line(tmp_path)
    assert "pending=" in line and "done=-" in line

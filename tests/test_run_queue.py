"""End-to-end pins for run_queue.py.

Covers dependency ordering, lane mutual exclusion, retry-once-then-FAILED,
BLOCKED propagation, resume-by-marker, PAUSE, MISSING blocking ALL_DONE and the
non-gating P3 backlog.
"""

import sys
import time
from pathlib import Path

import pytest
import yaml

import run_queue
from run_queue import Orchestrator, cmd_missing, load_queue, parse_lanes

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _helper(path: Path, out: str, rc: int = 0) -> None:
    _write(
        path,
        "import os, pathlib, sys\n"
        "os.makedirs('out', exist_ok=True)\n"
        f"if {rc} == 0:\n"
        f"    pathlib.Path({out!r}).write_text('ok')\n"
        f"sys.exit({rc})\n",
    )


def _make_queue(tmp: Path):
    _helper(tmp / "h_a.py", "out/a.txt", 0)
    _helper(tmp / "h_b.py", "out/b.txt", 0)
    _helper(tmp / "h_c.py", "out/c.txt", 0)
    _helper(tmp / "h_d.py", "out/d.txt", 1)  # always fails
    _helper(tmp / "h_e.py", "out/e.txt", 0)
    wps = [
        {"id": "A", "tier": "P0", "lane": "C", "deps": [], "cmd": f'"{PY}" h_a.py', "outputs": ["out/a.txt"]},
        {"id": "B", "tier": "P0", "lane": "G", "deps": ["A"], "cmd": f'"{PY}" h_b.py', "outputs": ["out/b.txt"]},
        {"id": "C", "tier": "P0", "lane": "C", "deps": ["A"], "cmd": f'"{PY}" h_c.py', "outputs": ["out/c.txt"]},
        {"id": "D", "tier": "P0", "lane": "G", "deps": ["A"], "cmd": f'"{PY}" h_d.py', "outputs": ["out/d.txt"]},
        {"id": "E", "tier": "P0", "lane": "C", "deps": ["D"], "cmd": f'"{PY}" h_e.py', "outputs": ["out/e.txt"]},
    ]
    (tmp / "queue.yaml").write_text(yaml.safe_dump({"wps": wps}), encoding="utf-8")
    return load_queue(tmp / "queue.yaml")


def _run_to_terminal(root: Path, wps, timeout: float = 120.0) -> Orchestrator:
    orch = Orchestrator(root, wps, poll_sec=0.05, git_sync=False, allow_shutdown=False, once_mode=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        orch.run_once()
        if orch.all_terminal():
            return orch
        time.sleep(0.05)
    raise AssertionError("queue did not reach a terminal state in time")


def test_parse_lanes():
    assert parse_lanes("C") == {"C"}
    assert parse_lanes("G+C") == {"G", "C"}
    assert parse_lanes("C+G") == {"G", "C"}
    with pytest.raises(ValueError):
        parse_lanes("X")


def test_cmd_missing(tmp_path: Path):
    assert cmd_missing("python scripts/nope.py", tmp_path) == "scripts/nope.py"
    _write(tmp_path / "scripts" / "ok.py", "print('x')")
    assert cmd_missing("python scripts/ok.py", tmp_path) is None
    assert cmd_missing("/root/miniconda3/bin/python scripts/nope.py", tmp_path) == "scripts/nope.py"


def test_dependency_lane_retry_blocked(tmp_path: Path):
    wps = _make_queue(tmp_path)
    orch = _run_to_terminal(tmp_path, wps)

    assert (tmp_path / "out" / "a.txt").exists()
    assert (tmp_path / "out" / "b.txt").exists()
    assert (tmp_path / "out" / "c.txt").exists()
    assert not (tmp_path / "out" / "d.txt").exists()
    assert not (tmp_path / "out" / "e.txt").exists()

    assert orch.state["D"]["status"] == "FAILED"
    assert orch.state["D"]["attempts"] == 2, "failure must be retried exactly once"
    assert orch.state["E"]["status"] == "BLOCKED"
    assert orch.state["E"]["blocked_by"] == "D"

    # B never starts before A finishes
    assert orch.state["B"]["started"][0] >= orch.state["A"]["ended"][0]
    # two lane-C work packages never overlap
    a0, a1 = orch.state["A"]["started"][0], orch.state["A"]["ended"][0]
    c0, c1 = orch.state["C"]["started"][0], orch.state["C"]["ended"][0]
    assert not (a0 < c1 and c0 < a1), "lane C ran two work packages concurrently"
    # markers + ALL_DONE
    assert (tmp_path / "logs" / "done" / "A.DONE").exists()
    assert (tmp_path / "ALL_DONE").exists()
    assert (tmp_path / "progress.log").exists()


def test_resume_skips_finished(tmp_path: Path):
    wps = _make_queue(tmp_path)
    _run_to_terminal(tmp_path, wps)
    first = {k: dict(v) for k, v in _run_to_terminal(tmp_path, wps).state.items()}
    assert first["A"]["attempts"] == 1, "finished work package must not be re-run"
    assert first["B"]["attempts"] == 1
    assert first["C"]["attempts"] == 1


def test_pause_stops_new_starts(tmp_path: Path):
    wps = _make_queue(tmp_path)
    (tmp_path / "PAUSE").write_text("", encoding="utf-8")
    orch = Orchestrator(tmp_path, wps, poll_sec=0.05, git_sync=False, allow_shutdown=False, once_mode=True)
    orch.run_once()
    assert not orch.running
    assert not (tmp_path / "out" / "a.txt").exists()
    for wid in ("A", "B", "C", "D", "E"):
        assert orch.state[wid]["status"] == "PENDING"


def test_missing_command_blocks_all_done(tmp_path: Path):
    wps = [
        {
            "id": "X",
            "tier": "P0",
            "lane": "C",
            "deps": [],
            "cmd": f'"{PY}" h_missing.py',
            "outputs": ["out/x.txt"],
        }
    ]
    orch = Orchestrator(tmp_path, wps, poll_sec=0.05, git_sync=False, allow_shutdown=False, once_mode=True)
    orch.run_once()
    assert orch.state["X"]["status"] == "MISSING"
    assert not orch.all_terminal()
    assert not (tmp_path / "ALL_DONE").exists()


def test_non_gating_backlog_does_not_block_all_done(tmp_path: Path):
    _helper(tmp_path / "h_main.py", "out/main.txt", 0)
    _helper(tmp_path / "h_backlog.py", "out/backlog.txt", 0)
    wps = [
        {"id": "M", "tier": "P0", "lane": "C", "deps": [], "cmd": f'"{PY}" h_main.py', "outputs": ["out/main.txt"]},
        {
            "id": "BG",
            "tier": "P3",
            "lane": "C",
            "deps": [],
            "gating": False,
            "cmd": f'"{PY}" h_backlog.py',
            "outputs": ["out/backlog.txt"],
        },
    ]
    orch = Orchestrator(tmp_path, wps, poll_sec=0.05, git_sync=False, allow_shutdown=False, once_mode=True)
    deadline = time.time() + 60
    while time.time() < deadline and not (tmp_path / "ALL_DONE").exists():
        orch.run_once()
        time.sleep(0.05)
    assert (tmp_path / "ALL_DONE").exists()
    assert orch.gating_ids() == ["M"]


def test_shipped_queue_is_valid(capsys):
    wps = load_queue(ROOT / "queue.yaml")
    ids = [wp["id"] for wp in wps]
    assert ids[0] == "W0" and ids[-1] == "WF"
    assert len(wps) == 24  # W0..W22 plus WF
    by_id = {wp["id"]: wp for wp in wps}
    # dependencies and lanes
    assert by_id["WF"]["deps"][0] == "FREEZE_OK"
    assert by_id["W2"]["deps"] == ["W1"]
    assert by_id["W22"]["gating"] is False
    assert parse_lanes(by_id["W10"]["lane"]) == {"G", "C"}
    # dry-run must not crash and must report the missing scripts (they are authored per WP)
    rc = run_queue.main(["--dry-run", "--root", str(ROOT)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "W0" in out and "WF" in out


def test_claims_and_configs_parse():
    claims = yaml.safe_load((ROOT / "claims.yaml").read_text(encoding="utf-8"))
    configs = yaml.safe_load((ROOT / "configs.yaml").read_text(encoding="utf-8"))
    assert claims["seed_base"] == 20261015
    assert claims["base_f"] == ["B1", "B2", "B3", "B4", "B7", "B8", "B9", "B10", "B11"]
    assert set(claims["claims"]) == {"S-F", "S-Z", "S-K", "S-R", "N1", "N2", "N3", "N4"}
    assert configs["cv"]["max_configs_per_method"] == 16
    grid = configs["methods"]["C1"]["grid"]
    combos = len(grid["sigma_p"]) * len(grid["sigma_gamma"])
    assert combos == configs["cv"]["max_configs_per_method"]

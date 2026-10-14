"""Pins for the W22 backlog driver: the quota map, the masks and the status table."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w22_backlog as w22  # noqa: E402


def test_the_k_grid_is_monotone_and_covers_the_taskbook_budgets():
    ks = sorted(w22.K_PER_CELL)
    assert ks == [30, 60, 100, 200, 290, 600]
    m = [w22.K_PER_CELL[k] for k in ks]
    assert m == sorted(m) and m[0] >= 1        # monotone per-cell quota
    assert w22.K_PER_CELL[290] == 10           # the headline budget is unchanged


def test_masks_for_returns_two_disjoint_splits_per_draw():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, size=n)
    speaker = np.array([f"s{i // 10:02d}" for i in range(n)])
    Z = rng.normal(size=(n, 4))
    got = w22.masks_for(0, Z, y, np.array(["c|en"] * n), speaker)
    assert set(got) == {"clips", "strict_draw"}
    for name, (lab, ev) in got.items():
        assert lab.size > 0 and ev.size > 0
        assert set(lab.tolist()) & set(ev.tolist()) == set(), name
    lab, ev = got["clips"]
    assert lab.size + ev.size == n                       # a partition of the region
    slab, sev = got["strict_draw"]
    assert set(speaker[slab].tolist()) & set(speaker[sev].tolist()) == set()


def test_backlog_table_marks_what_was_done_and_what_was_not():
    table = w22.backlog_table(pd.DataFrame([{"x": 1}]), pd.DataFrame([{"x": 1}]),
                              Path("."))
    got = dict(zip(table["item"], table["status"]))
    assert any(k.startswith("2.") for k in got)
    assert got[[k for k in got if k.startswith("2.")][0]] == "DONE"
    assert got[[k for k in got if k.startswith("4.")][0]] == "DONE"
    assert got[[k for k in got if k.startswith("3.")][0]] == "DONE"   # Z4 lives in W8
    assert got[[k for k in got if k.startswith("6.")][0]] == "NOT_RUN"
    assert got[[k for k in got if k.startswith("5.")][0]] == "PARTIAL"
    assert len(table) == 6


def test_backlog_table_reports_the_empty_runs():
    table = w22.backlog_table(pd.DataFrame(), pd.DataFrame(), Path("."))
    got = dict(zip(table["item"], table["status"]))
    assert got[[k for k in got if k.startswith("2.")][0]] == "NOT_RUN"
    assert got[[k for k in got if k.startswith("4.")][0]] == "NOT_RUN"
    assert json.dumps(table.to_dict(orient="records"), default=str)


def test_delegation_uses_a_suffix_so_nothing_is_overwritten():
    """The command has to write _d200 files, never the stored 100-draw tables."""
    import subprocess

    captured = {}

    class Out:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake(cmd, capture_output=True, text=True, timeout=None):
        captured["cmd"] = cmd
        return Out()

    orig = w22.subprocess.run
    w22.subprocess.run = fake
    try:
        got = w22.delegate_heavy(Path("/tmp/x"), "xls-r-2b", "S3", "speaker_split_v3.json",
                                 draws=200, jobs=8)
    finally:
        w22.subprocess.run = orig
    assert got["ran"] is True
    cmd = captured["cmd"]
    assert "--suffix" in cmd and "_d200" in cmd
    assert cmd[cmd.index("--draws") + 1] == "200"

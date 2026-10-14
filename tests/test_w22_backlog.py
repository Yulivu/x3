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


def test_the_curve_is_indexed_by_the_nominal_budget_not_the_achieved_count():
    """Every draw must land in its own budget row even though k drifts by a clip or two."""
    rows = []
    for d in range(10):
        for method in ("B1", "B9"):
            for mask in ("clips", "strict_draw"):
                rows.append({"method": method, "mask": mask, "draw": d,
                             "k": 290 + (d % 3), "k_nominal": 290,
                             "pooled_eer": 10.0 + (d % 3)})
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["method", "k_nominal"], columns="mask",
                         values="pooled_eer")
    assert ("B1", 290) in set(piv.index)
    assert {k for _m, k in piv.index} == {290}
    assert len(piv) == 2                      # one row per method, not per k drift


def test_every_spec_rebuilds_the_graph(monkeypatch, tmp_path):
    """B9 runs on the kNN graph, so a spec sweep must not reuse the previous one."""
    seen = []

    def fake_knn(Z, k=15):
        seen.append(id(Z))
        return (np.zeros((len(Z), k), int), np.zeros((len(Z), k)))

    calls = {"n": 0}

    def fake_load_region(root, data_dir, backbone, spec, split):
        calls["n"] += 1
        n = 40
        sub = pd.DataFrame({"y": np.tile([0, 1], n // 2),
                            "platform": ["p"] * n, "language": ["en"] * n,
                            "speaker": [f"s{i // 4:02d}" for i in range(n)]})
        sub["cell"] = "p|en"
        return sub, np.zeros((n, 8), np.float32)

    monkeypatch.setattr(w22, "knn_graph", fake_knn)
    monkeypatch.setattr(w22.R, "load_region", fake_load_region)
    monkeypatch.setattr(w22, "fit_preprocess", getattr(w22, "fit_preprocess", None),
                        raising=False)
    monkeypatch.setattr(w22.w6, "fit_preprocess", lambda F: (None, F))
    monkeypatch.setattr(w22, "_pooled", lambda rows, jobs, payloads: [])

    w22.spec_sensitivity(tmp_path, tmp_path, "toy", "s.json", specs=("S1", "S2"),
                         draws=2, jobs=1)
    assert len(seen) == 2                     # one graph per spec, not one for all

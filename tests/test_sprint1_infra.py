"""Infra contracts for the NEW_TASKS 10 sprint (packages, scheduler, tables).

These tests are pure-Python on purpose: they must run on the Windows worktree so
the sprint code is validated before it ever reaches the HPC.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sprint1 import config as cfg  # noqa: E402
from scripts.sprint1 import sampling_tables as stab  # noqa: E402
from scripts.sprint1 import state as st  # noqa: E402
from scripts.sprint1 import audio_memmap as am  # noqa: E402
from scripts.sprint1 import download_itw as itw  # noqa: E402
from scripts.sprint1 import fold_pca as fp  # noqa: E402
from scripts.sprint1 import inknn_graph as ik  # noqa: E402
from scripts.sprint1 import recode_feats as rf  # noqa: E402
from scripts.sprint1 import counterfactual as cf  # noqa: E402
from scripts.sprint1 import extra_feats as ef  # noqa: E402
from scripts.sprint1 import x_e_audit_list as xe  # noqa: E402
from scripts.sprint1 import base_m as bm  # noqa: E402
from scripts.sprint1.scheduler import Sprint, cmd_missing, load_packages  # noqa: E402


# --- package table contract --------------------------------------------------
def test_package_table_contract() -> None:
    packages, backlog = load_packages(ROOT / "packages.yaml")
    assert packages, "packages.yaml must declare the sprint packages"
    ids = [p["id"] for p in packages + backlog]
    assert len(ids) == len(set(ids)), "package ids must be unique"
    known = set(ids)
    for pkg in packages + backlog:
        assert pkg["lane"] in cfg.LANES, pkg
        assert pkg["cmd"], pkg
        assert pkg["artifacts"], pkg
        assert float(pkg["eta_min"]) > 0, pkg
        for dep in pkg.get("deps") or []:
            assert dep in known or dep in ("FREEZE_OK", "PAUSE", "SHUTDOWN_OK"), (pkg["id"], dep)
    # dependency graph is acyclic (the file order itself need not be topological)
    graph = {p["id"]: [d for d in (p.get("deps") or []) if d in known]
             for p in packages + backlog}
    visiting, finished = set(), set()

    def visit(node: str) -> None:
        if node in finished:
            return
        assert node not in visiting, f"dependency cycle at {node}"
        visiting.add(node)
        for dep in graph[node]:
            visit(dep)
        visiting.discard(node)
        finished.add(node)

    for node in graph:
        visit(node)


def test_lane_budgets_match_section_1() -> None:
    assert cfg.LANE_THREADS["A"] == 5
    assert cfg.LANE_THREADS["B"] == 3
    assert cfg.LANE_NICE["B"] == 10
    assert sum(cfg.LANE_THREADS.values()) <= cfg.CPU_BUDGET
    assert cfg.VRAM_BUDGET_GB <= cfg.VRAM_TOTAL_GB
    from src.newtasks05 import worker_slots

    assert 1 <= worker_slots() <= cfg.CPU_BUDGET


def test_lane_env_caps_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    env = cfg.lane_env("A")
    for key in cfg.THREAD_ENV_KEYS:
        assert env[key] == str(cfg.LANE_THREADS["A"])
    assert env["X3_LANE"] == "A"
    assert cfg.lane_prefix("B")[:2] == ["nice", "-n"]
    assert cfg.lane_prefix("A") == []


# --- state/resume contract ---------------------------------------------------
def test_state_artifacts_and_dependencies(tmp_path: Path) -> None:
    state: dict = {}
    art = tmp_path / "out.csv"
    st.mark(state, "A1", "DONE")
    st.record_artifacts(state, "A1", ["out.csv"], tmp_path)
    assert state["A1"]["artifacts"]["out.csv"]["bytes"] == 0
    ok, missing = st.artifacts_ok(["out.csv"], tmp_path)
    assert not ok and missing == ["out.csv"]
    art.write_text("x", encoding="utf-8")
    ok, missing = st.artifacts_ok(["out.csv"], tmp_path)
    assert ok and not missing

    pkg = {"id": "A2", "deps": ["A1"]}
    assert st.deps_satisfied(pkg, state)[0]
    st.mark(state, "A1", "FAILED")
    assert not st.deps_satisfied(pkg, state)[0]
    assert st.blocked_by_failed(pkg, state) == "A1"
    # FREEZE_OK-style dependencies are control files, not packages
    assert st.deps_satisfied({"id": "X", "deps": ["FREEZE_OK"]}, {})[0]


def test_state_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = {"P": {"status": "DONE", "attempts": 1}}
    st.save_state(path, state)
    assert st.load_state(path) == state
    path.write_text("{not json", encoding="utf-8")
    assert st.load_state(path) == {}


def test_all_complete_blocks_missing_and_blocked() -> None:
    state = {"A": {"status": "DONE"}, "B": {"status": "MISSING"}}
    assert st.all_terminal(state, ["A", "B"]) is True
    assert st.all_complete(state, ["A", "B"]) is False, "MISSING must block ALL_DONE"
    state["B"] = {"status": "BLOCKED"}
    assert st.all_complete(state, ["A", "B"]) is False
    state["B"] = {"status": "TIMEOUT"}
    assert st.all_complete(state, ["A", "B"]) is True


# --- scheduler ---------------------------------------------------------------
def _write_pkg_table(tmp_path: Path, packages: list) -> Path:
    path = tmp_path / "packages.yaml"
    path.write_text(yaml.safe_dump({"packages": packages, "backlog": []}), encoding="utf-8")
    return path


def test_cmd_missing_detects_modules(tmp_path: Path) -> None:
    assert cmd_missing("python -m scripts.sprint1.preflight --out x", ROOT) is None
    assert cmd_missing("python -m scripts.sprint1.nope", tmp_path) == "scripts.sprint1.nope"
    assert cmd_missing("python scripts/nope.py", tmp_path) == "scripts/nope.py"


def test_scheduler_marks_missing_and_done(tmp_path: Path) -> None:
    (tmp_path / "helper.py").write_text(
        "import pathlib, sys\npathlib.Path('art.txt').write_text('ok')\n", encoding="utf-8")
    packages = [
        {"id": "S1", "lane": "A", "deps": [], "eta_min": 5, "priority": "P0",
         "cmd": f"{sys.executable} helper.py", "artifacts": ["art.txt"]},
        {"id": "S2", "lane": "A", "deps": ["S1"], "eta_min": 5, "priority": "P0",
         "cmd": "python -m scripts.sprint1.definitely_not_written",
         "artifacts": ["other.txt"]},
    ]
    table = _write_pkg_table(tmp_path, packages)
    sprint = Sprint(table, tmp_path / "state.json", poll=1, once=True, root=tmp_path,
                    verbose=False)
    sprint.loop()
    sprint.drain(timeout=120)
    sprint.loop()          # second pass: S1 is DONE, so S2 becomes runnable
    sprint.drain(timeout=120)
    state = st.load_state(tmp_path / "state.json")
    assert state["S1"]["status"] == "DONE"
    assert state["S1"]["artifacts"]["art.txt"]["bytes"] > 0
    assert state["S2"]["status"] == "MISSING"


def test_scheduler_fails_on_missing_artifact(tmp_path: Path) -> None:
    (tmp_path / "liar.py").write_text("print('nothing written')\n", encoding="utf-8")
    packages = [{"id": "L1", "lane": "B", "deps": [], "eta_min": 5, "priority": "P0",
                 "cmd": f"{sys.executable} liar.py", "artifacts": ["never.txt"]}]
    table = _write_pkg_table(tmp_path, packages)
    sprint = Sprint(table, tmp_path / "state.json", poll=1, once=True, root=tmp_path,
                    verbose=False)
    sprint.loop()
    sprint.drain(timeout=120)
    state = st.load_state(tmp_path / "state.json")
    assert state["L1"]["status"] == "FAILED"
    assert "artifacts missing" in state["L1"]["note"]


def test_scheduler_timebox_marks_timeout(tmp_path: Path) -> None:
    (tmp_path / "slow.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    packages = [{"id": "T1", "lane": "B", "deps": [], "eta_min": 0.02, "priority": "P0",
                 "cmd": f"{sys.executable} slow.py", "artifacts": ["t.txt"]}]
    table = _write_pkg_table(tmp_path, packages)
    sprint = Sprint(table, tmp_path / "state.json", poll=1, once=True, root=tmp_path,
                    verbose=False)
    sprint.loop()
    assert sprint.running, "the package should have been started"
    sprint.running["T1"]["t0"] -= 3600.0     # pretend the time box has expired
    sprint.timebox()
    state = st.load_state(tmp_path / "state.json")
    assert state["T1"]["status"] == "TIMEOUT"


def test_scheduler_respects_pause_and_freeze(tmp_path: Path) -> None:
    packages = [
        {"id": "F1", "lane": "A", "deps": [], "eta_min": 5, "priority": "P0",
         "needs_freeze": True, "cmd": f"{sys.executable} -c pass", "artifacts": ["f.txt"]},
    ]
    table = _write_pkg_table(tmp_path, packages)
    sprint = Sprint(table, tmp_path / "state.json", poll=1, once=True, root=tmp_path,
                    verbose=False)
    monkeypatch_control = ROOT / "control"
    monkeypatch_control.mkdir(exist_ok=True)
    try:
        (monkeypatch_control / "PAUSE").write_text("", encoding="utf-8")
        assert sprint.ready("A") is None
        (monkeypatch_control / "PAUSE").unlink()
        assert sprint.ready("A") is None, "needs_freeze must wait for FREEZE_OK"
        (monkeypatch_control / "FREEZE_OK").write_text("", encoding="utf-8")
        assert (sprint.ready("A") or {}).get("id") == "F1"
    finally:
        for name in ("PAUSE", "FREEZE_OK"):
            p = monkeypatch_control / name
            if p.exists():
                p.unlink()


def test_scheduler_honours_an_external_state_reset(tmp_path: Path) -> None:
    """state.json is authoritative, so `state_tool --set R1 TODO` re-opens the package
    even while the scheduler that retired it is still running."""
    (tmp_path / "job.py").write_text(
        "import pathlib, sys\n"
        "if not pathlib.Path('flag').exists():\n"
        "    sys.exit(1)\n"
        "pathlib.Path('art.txt').write_text('ok')\n", encoding="utf-8")
    packages = [{"id": "R1", "lane": "A", "deps": [], "eta_min": 5, "priority": "P0",
                 "cmd": f"{sys.executable} job.py", "artifacts": ["art.txt"]}]
    table = _write_pkg_table(tmp_path, packages)
    sprint = Sprint(table, tmp_path / "state.json", poll=1, once=True, root=tmp_path,
                    verbose=False)
    sprint.loop()
    sprint.drain(timeout=60)
    assert st.load_state(tmp_path / "state.json")["R1"]["status"] == "FAILED"

    state = st.load_state(tmp_path / "state.json")
    st.mark(state, "R1", "TODO", note="operator re-opened it")
    st.save_state(tmp_path / "state.json", state)
    (tmp_path / "flag").write_text("go", encoding="utf-8")
    sprint.loop()
    sprint.drain(timeout=60)
    assert st.load_state(tmp_path / "state.json")["R1"]["status"] == "DONE"


def test_reclaim_orphans_reopens_a_restart_leftover(tmp_path: Path) -> None:
    packages = [{"id": "O1", "lane": "A", "deps": [], "eta_min": 5, "priority": "P0",
                 "cmd": f"{sys.executable} -c pass", "artifacts": ["o.txt"]}]
    table = _write_pkg_table(tmp_path, packages)
    state_path = tmp_path / "state.json"
    state: dict = {}
    st.mark(state, "O1", "RUNNING", os_pid=999999, note="lane A pid 999999")
    st.save_state(state_path, state)
    sprint = Sprint(table, state_path, poll=1, once=True, root=tmp_path, verbose=False)
    sprint.sync_from_disk()
    assert sprint.retired == {}, "RUNNING is not terminal"
    assert sprint.reclaim_orphans() == ["O1"]
    assert st.load_state(state_path)["O1"]["status"] == "TODO"


# --- sampling tables ---------------------------------------------------------
def synthetic_meta(n_platforms: int = 4, speakers_per_platform: int = 6,
                   clips_per_speaker: int = 12) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    for pi in range(n_platforms):
        for si in range(speakers_per_platform):
            speaker = f"p{pi}s{si}"
            for ci in range(clips_per_speaker):
                rows.append({"speaker": speaker, "language": "en",
                             "platform": f"plat{pi}", "label": "spoof" if ci % 4 == 3 else "bonafide"})
    meta = pd.DataFrame(rows)
    meta["clip_id"] = np.arange(len(meta))
    meta["y"] = (meta["label"] == "spoof").astype(int)
    meta["cell"] = meta["platform"] + "|" + meta["language"]
    dev_rows = np.arange(len(meta))
    return meta, dev_rows


def test_p1_is_speaker_disjoint_and_deterministic() -> None:
    meta, dev = synthetic_meta()
    a = stab.build_p1(meta, dev, draws=3, sizes=(10, 40), boot=5)
    b = stab.build_p1(meta, dev, draws=3, sizes=(10, 40), boot=5)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    y = meta["y"].to_numpy()
    sp = meta["speaker"].to_numpy()
    plat = meta["platform"].to_numpy()
    for p, block in a["targets"].items():
        target_speakers = set(sp[block["target_rows"]])
        source_speakers = {sp[r] for r in block["source_rows"]}
        assert not (target_speakers & source_speakers), "P1 source must exclude p's speakers"
        assert all(plat[r] != p for r in block["source_rows"])
        lib = block["library_pool_rows"]
        assert lib, "each platform needs a genuine library pool"
        assert all(plat[r] == p and y[r] == 0 for r in lib)
        assert {sp[r] for r in lib} <= set(block["library_pool_speakers"])
        assert set(block["library_pool_speakers"]) <= target_speakers
        for draw, sizes in block["library_by_draw"].items():
            for n, lib_entry in sizes.items():
                rows = lib_entry["rows"]
                assert len(rows) <= int(n)
                assert len(set(rows)) == len(rows)
                assert lib_entry["effective_n"] == len(rows)
                # 09 section 0: at most ceil(N/4) clips per speaker
                cap = max(1, -(-int(n) // 4))
                counts: dict = {}
                for r in rows:
                    counts[sp[r]] = counts.get(sp[r], 0) + 1
                assert max(counts.values(), default=0) <= cap
                assert set(lib_entry["proper"]) | set(lib_entry["calibration"]) == set(rows)
        assert len(block["bootstrap_speaker_index"]) == 5
        assert all(len(r) == len(target_speakers) for r in block["bootstrap_speaker_index"])
        folds = a["lopo_folds"][p]
        assert folds and all(len(fold) <= len(block["source_rows"]) for fold in folds.values())


def test_p1_macro_eligibility_uses_thirty_per_class() -> None:
    meta, dev = synthetic_meta()
    out = stab.build_p1(meta, dev, draws=2, sizes=(10,), boot=3)
    for block in out["targets"].values():
        expect = block["min_real"] >= 30 and block["min_fake"] >= 30
        assert block["macro_eligible"] is expect
        assert block["min_real"] >= 0 and block["min_fake"] >= 0


def test_assert_no_test_rows_blocks_leaks() -> None:
    stab.assert_no_test_rows({"source_rows": [1, 2, 3]}, {9})
    with pytest.raises(ValueError):
        stab.assert_no_test_rows({"target_rows": [1, 9]}, {9})


def test_sample_library_caps_clips_per_speaker() -> None:
    rows = list(range(40))
    speakers = [f"s{i % 2}" for i in range(40)]      # only two speakers, 20 clips each
    gen = np.random.default_rng(11)
    lib = stab.sample_library(rows, speakers, 20, gen)
    cap = max(1, -(-20 // 4))                        # ceil(20/4) = 5
    # two speakers x 5 = the whole library the cap allows; effective N < N is reported
    assert len(lib) == 10
    counts: dict = {}
    for r in lib:
        counts[speakers[r]] = counts.get(speakers[r], 0) + 1
    assert max(counts.values()) <= cap
    # a pool too small for N returns everything it has, with the cap still respected
    small = stab.sample_library(rows[:6], speakers[:6], 40, np.random.default_rng(12))
    assert len(small) <= 6
    assert len(set(small)) == len(small)


# --- S1-B BASE_M -------------------------------------------------------------
def test_smallest_k_distances_masks_the_query_speaker() -> None:
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(12, 4)).astype(np.float32)
    speaker = np.array(["a", "a", "b", "b", "c", "c"] * 2)
    d = bm.smallest_k_distances(Z, Z, speaker, speaker, 2)
    assert d.shape == (12, 2)
    # a query must never take a reference of its own speaker: distances are larger
    # than the 0 it would get from itself
    assert np.all(d[:, 0] > 1e-6)
    # the query's own vector is in the pool, so a missing mask would give ~0
    d_norm = bm.smallest_k_distances(Z, Z[:1], np.array(["a"]), np.array(["a"]), 1)
    assert np.all(np.isnan(d_norm[:, 0]))


def test_e_norm_s_shrinks_towards_the_global_scale() -> None:
    rng = np.random.default_rng(6)
    Z = rng.normal(size=(20, 4)).astype(np.float32)
    speaker = np.array([f"s{i % 4}" for i in range(20)])
    o1 = np.abs(rng.normal(size=5)).astype(np.float32)
    scaled, sigma = bm.e_norm_s(o1, Z, speaker, sigma_global=1.0)
    assert sigma > 0
    assert np.allclose(scaled, o1 / sigma)
    # a library with a tiny spread cannot collapse the scale to zero
    Z_tight = np.ones((8, 4), np.float32) + 1e-9 * rng.normal(size=(8, 4)).astype(np.float32)
    _, sigma_tight = bm.e_norm_s(o1, Z_tight, np.array(["s"] * 8), sigma_global=2.0)
    assert sigma_tight > 0 and np.isfinite(sigma_tight)


def test_fit_b1_picks_a_grid_value() -> None:
    rng = np.random.default_rng(7)
    Z = rng.normal(size=(200, 6)).astype(np.float32)
    y = (rng.random(200) < 0.5).astype(int)
    sp = np.array([f"s{i % 10}" for i in range(200)])
    lr, C = bm.fit_b1(Z, y, sp)
    assert C in bm.C_GRID
    assert lr.decision_function(Z[:5]).shape == (5,)


def test_threshold_helpers() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.5, 0.9])
    assert bm._tpr_at_threshold(y, s, 0.6) == pytest.approx(50.0)
    assert bm._tpr_at_threshold(y, s, 0.5) == pytest.approx(100.0)
    assert bm._hter(y, s, 0.6) == pytest.approx(25.0)
    assert np.isnan(bm._tpr_at_threshold(y, s, float("nan")))


def test_base_m_end_to_end_on_synthetic_inputs(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    # 09 section 0: a platform enters the macro average only with >=30 clips per
    # class, so the fixture has to clear that bar
    n_platforms, speakers_per_platform, clips_per_speaker = 2, 6, 14
    rows = []
    for pi in range(n_platforms):
        for si in range(speakers_per_platform):
            for ci in range(clips_per_speaker):
                rows.append({"speaker": f"p{pi}s{si}", "language": "en",
                             "platform": f"plat{pi}",
                             "label": "spoof" if ci % 2 else "bonafide"})
    meta = pd.DataFrame(rows)
    meta["clip_id"] = np.arange(len(meta))
    meta["y"] = (meta["label"] == "spoof").astype(int)
    meta["cell"] = meta["platform"] + "|" + meta["language"]
    n = len(meta)
    rng = np.random.default_rng(8)
    block = rng.normal(size=(n, 5)).astype(np.float32)
    block[:, -1] = (meta["y"].to_numpy() * 0.6 + rng.normal(size=n) * 0.4)

    monkeypatch.setattr("src.data.load_mlitw", lambda *a, **k: meta)
    monkeypatch.setattr("src.regions.load_spec_block",
                        lambda data_dir, bb, spec, rows=None, want_zero=True:
                        block if rows is None else block[np.asarray(rows, int)])
    # Z0 must come from the backbone's scores.csv (spec S3 carries no zero column)
    monkeypatch.setattr("src.newtasks05.zero_scores",
                        lambda data_dir, bb, clip_id: (3.0 * meta["y"].to_numpy() - 1.0))

    _write_fake_audio_index(tmp_path, n)          # dev rows = 0..n-1, side dev
    dev_rows = np.arange(n)
    tables = stab.build_p1(meta, dev_rows, draws=2, sizes=(10,), boot=3)
    (tmp_path / "sprint1" / "tables").mkdir(parents=True, exist_ok=True)
    (tmp_path / "sprint1" / "tables" / "p1_folds.json").write_text(
        json.dumps(tables), encoding="utf-8")
    pca_dir = tmp_path / "sprint1" / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)
    Z = rng.normal(size=(n, 4)).astype(np.float32)
    for p, t in tables["targets"].items():
        t_rows = np.asarray(t["target_rows"], int)
        s_rows = np.asarray(t["source_rows"], int)
        np.savez_compressed(pca_dir / f"pca_syn_{p}.npz",
                            target_rows=t_rows, source_rows=s_rows,
                            Z_target=Z[t_rows], Z_source=Z[s_rows])

    rc = bm.main(["--root", str(tmp_path), "--backbone" if False else "--backbones", "syn",
                  "--sizes", "10", "--draws", "2", "--save-scores-count", "0"])
    assert rc == 0
    out = pd.read_csv(tmp_path / "sprint1" / "results" / "base_m.csv")
    assert len(out) > 0
    agg = pd.read_csv(tmp_path / "sprint1" / "results" / "base_m_agg.csv")
    methods = set(agg["method"])
    for m in ("Z0", "OC-1NN", "E-norm-s", "B1", "B1+refs", "OC+B1"):
        assert m in methods
    z0_auc = float(agg[agg["method"] == "Z0"]["auc_mean"].iloc[0])
    assert z0_auc > 0.9, "Z0 must be the scores.csv column, not the S3 block's last column"
    gate = pd.read_csv(tmp_path / "sprint1" / "results" / "base_m_gate.csv")
    assert "strongest" in gate.columns and gate["strongest"].iloc[0] != "Z0"


def test_p2_fallback_uses_u8_sampler() -> None:
    meta, dev = synthetic_meta()
    out = stab.build_p2(meta, dev, draws=3, sizes=(10, 40), boot=4)
    y = meta["y"].to_numpy()
    assert len(out["draws_by_index"]) == 3
    for block in out["draws_by_index"].values():
        per_speaker = block["library_per_speaker"]["5"]
        assert all(y[r] == 0 for r in per_speaker), "enrolment libraries must be genuine"
        assert all(y[r] == 0 for r in block["library_by_size"]["10"]["budget"])
        assert len(block["bootstrap_speaker_index"]) == 4
        assert all(len(r) == len(block["eval_speakers"])
                   for r in block["bootstrap_speaker_index"])


# --- memmap helpers ----------------------------------------------------------
def test_prepare_waveform_tiles_and_truncates() -> None:
    x = np.linspace(-1, 1, 1000, dtype=np.float32)
    padded = am.prepare_waveform(x, 64600)
    assert padded.shape == (64600,)
    assert not np.allclose(padded[:1000], 0)
    long = np.ones(100000, dtype=np.float32)
    assert am.prepare_waveform(long, 64600).shape == (64600,)
    ints = am.to_int16(np.array([-2.0, 0.0, 1.0], np.float32))
    assert ints.dtype == np.int16 and ints[0] == -32768 and ints[2] == 32767


def test_keep_positions_filters_by_global_index() -> None:
    # a split row list must select exactly those global clips, not the first N clips
    rows = [3, 5, 9, 10]
    assert am.keep_positions(rows, 12) == [3, 5, 9, 10]
    assert am.keep_positions([0, 1], 12) == [0, 1]
    assert am.keep_positions([11], 12) == [11]
    assert am.keep_positions([], 5) == []


def test_test_side_is_refused_without_freeze(tmp_path: Path) -> None:
    rc = am.main(["--side", "test", "--root", str(tmp_path)])
    assert rc == 3, "reading test clips before FREEZE_OK must be refused"


def test_preflight_runs_and_reports_blocking(tmp_path: Path) -> None:
    from scripts.sprint1 import preflight as pf

    data = tmp_path / "data"
    data.mkdir()
    rc = pf.main(["--root", str(tmp_path), "--out", "preflight.md",
                  "--json", "logs/env.json", "--data", str(data)])
    assert rc == 0
    env = json.loads((tmp_path / "logs" / "env.json").read_text(encoding="utf-8"))
    assert env["blocking"], "an empty data root must be reported as blocking"
    assert (tmp_path / "preflight.md").exists()
    assert env["budget"]["cpu_budget"] == cfg.CPU_BUDGET


# --- PCA / graph / download --------------------------------------------------
def test_standardise_uses_only_the_given_rows() -> None:
    train = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    mu, sd = fp.standardise(train)
    assert np.allclose(mu, [2.0, 3.0])
    assert np.allclose(sd, [1.0, 1.0])
    mu2, _ = fp.standardise(np.array([[10.0, 10.0]], np.float32))
    assert np.allclose(mu2, [10.0, 10.0]), "the fit must follow the rows handed in"


def test_knn_excludes_self_and_matches_bruteforce() -> None:
    rng = np.random.default_rng(7)
    Z = rng.normal(size=(60, 8)).astype(np.float32)
    idx, sim = ik.knn(Z, 5)
    assert idx.shape == (60, 5) and sim.shape == (60, 5)
    for i, row in enumerate(idx):
        assert i not in row
    assert np.all(np.diff(sim, axis=1) <= 1e-6), "neighbours must be sorted by similarity"
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    S = Zn @ Zn.T
    np.fill_diagonal(S, -np.inf)
    assert np.array_equal(idx[:, 0], S.argmax(axis=1).astype(np.int32))
    assert np.allclose(sim[:, 0], S.max(axis=1), atol=1e-5)


def test_itw_disk_guard_blocks_without_downloading(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "itw"
    out.mkdir()
    monkeypatch.setattr(cfg, "SYSTEM_FREE_MIN_GB", 10 ** 9)
    status = tmp_path / "itw_status.json"
    rc = itw.main(["--out", str(out), "--status", str(status)])
    assert rc == 0
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED" and payload["reason"] == "disk"
    assert "attempts" not in payload


def test_harness_selftest_passes_on_this_machine(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from scripts.sprint1 import harness_selftest as hs

    rc = hs.main(["--root", str(tmp_path), "--out", "sprint1/harness_report.json",
                  "--clips", "600", "--dim", "16"])
    assert rc == 0, "the harness must agree with src.metrics3 before S1-B uses it"
    report = json.loads((tmp_path / "sprint1" / "harness_report.json").read_text(encoding="utf-8"))
    assert report["ok"] and report["n_checks"] >= 6


# --- G-A: codec-normalised features ------------------------------------------
def test_recode_needs_the_memmap_and_refuses_non_dev(tmp_path: Path) -> None:
    assert rf.main(["--backbone", "xls-r-2b", "--root", str(tmp_path)]) == 2
    audio = tmp_path / "sprint1" / "audio"
    audio.mkdir(parents=True)
    mm = np.memmap(audio / "fake.mm", dtype=np.int16, mode="w+", shape=(2, 100))
    mm[:] = 0
    mm.flush()
    del mm
    (audio / "dev_index.json").write_text(json.dumps({
        "side": "test", "path": str(audio / "fake.mm"), "dtype": "int16",
        "shape": [2, 100], "rows": [0, 1]}), encoding="utf-8")
    assert rf.main(["--backbone", "xls-r-2b", "--root", str(tmp_path)]) == 3


def test_transcode_roundtrip_when_ffmpeg_is_present(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is only on the compute machine")
    tone = (np.sin(2 * np.pi * 220 * np.arange(32000) / 16000) * 3000).astype(np.int16)
    out = rf.transcode_one(tone, "64k", tmp_path)
    assert out.ndim == 1 and out.size > 0
    assert np.abs(out).max() <= 1.0


# --- G-B / G-C / X-E ---------------------------------------------------------
def test_handcrafted_descriptors_are_finite_and_bounded() -> None:
    x = np.sin(2 * np.pi * 220 * np.arange(32000) / 16000).astype(np.float32)
    d = ef.handcrafted(x)
    assert d.shape == (len(ef.DESC_NAMES),)
    assert np.isfinite(d).all()
    assert d[ef.DESC_NAMES.index("duration_s")] == pytest.approx(2.0, abs=0.01)
    assert d[ef.DESC_NAMES.index("zcr")] > 0


def test_layer_indices_stay_in_range() -> None:
    idx = ef.layer_indices(48)
    assert idx == sorted(set(idx))
    assert all(1 <= v <= 48 for v in idx)
    assert idx == ef.layer_indices(48)


def _write_fake_audio_index(root: Path, n: int = 24) -> None:
    audio = root / "sprint1" / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(audio / "dev_i16.mm", dtype=np.int16, mode="w+", shape=(n, 1600))
    rng = np.random.default_rng(3)
    for i in range(n):
        mm[i] = (np.sin(2 * np.pi * (100 + 10 * i) * np.arange(1600) / 16000) * 8000).astype(np.int16)
    mm.flush()
    del mm
    (audio / "dev_index.json").write_text(json.dumps({
        "side": "dev", "path": str(audio / "dev_i16.mm"), "dtype": "int16",
        "shape": [n, 1600], "rows": list(range(n)), "clip_id": list(range(n)),
        "meta": {"speaker": [f"s{i % 4}" for i in range(n)],
                 "platform": [["youtube", "douyin"][i % 2] for i in range(n)],
                 "language": ["english"] * n,
                 "label": [["bonafide", "spoof"][i % 2] for i in range(n)],
                 "y": [i % 2 for i in range(n)],
                 "file": ["eval-00000-of-00004.parquet"] * n},
    }), encoding="utf-8")


def test_g_b_and_g_c_need_p1_and_refuse_test(tmp_path: Path) -> None:
    assert ef.main(["--root", str(tmp_path), "--limit", "2"]) == 2
    assert cf.main(["--root", str(tmp_path)]) == 2
    assert xe.main(["--root", str(tmp_path)]) == 2
    _write_fake_audio_index(tmp_path, 8)
    index = json.loads((tmp_path / "sprint1/audio/dev_index.json").read_text(encoding="utf-8"))
    index["side"] = "test"
    (tmp_path / "sprint1/audio/dev_index.json").write_text(json.dumps(index), encoding="utf-8")
    assert cf.main(["--root", str(tmp_path)]) == 3
    assert xe.main(["--root", str(tmp_path)]) == 3


def test_extra_feats_handcrafted_block_on_a_tiny_memmap(tmp_path: Path) -> None:
    _write_fake_audio_index(tmp_path, 6)
    rc = ef.main(["--root", str(tmp_path), "--limit", "6", "--backbones", "",
                  "--max-minutes", "0.05"])
    assert rc == 0
    npz = np.load(tmp_path / "sprint1/feats/extra_layers_dev.npz")
    assert npz["handcrafted"].shape == (6, len(ef.DESC_NAMES))
    assert np.isfinite(npz["handcrafted"]).all()
    index = json.loads((tmp_path / "sprint1/feats/extra_index.json").read_text(encoding="utf-8"))
    assert index["status"]["handcrafted"]["ok"] is True


def test_x_e_writes_a_stratified_list(tmp_path: Path) -> None:
    _write_fake_audio_index(tmp_path, 24)
    rc = xe.main(["--root", str(tmp_path), "--total", "8"])
    assert rc == 0
    csv = pd.read_csv(tmp_path / "paper_tables/audit_list.csv")
    assert len(csv) >= 4
    assert {"clip_id", "platform", "label", "wav", "row"} <= set(csv.columns)
    assert (tmp_path / "paper_tables/audit_audio").glob("*.wav")

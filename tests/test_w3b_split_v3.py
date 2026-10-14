"""Pins for the protocol-v3 split builder (NEW_TASKS 02 section 1)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w3b_split_v3 as w3b  # noqa: E402


def _meta():
    """12 speakers, 2 platforms, each speaker 20 clips with a majority label."""
    rows = []
    for i in range(12):
        speaker = f"s{i:02d}"
        majority = i % 2
        platform = "p0" if i < 6 else "p1"
        n_real = 15 if majority == 0 else 5
        for j in range(20):
            rows.append({"speaker": speaker, "platform": platform,
                         "language": "en", "label": "spoof" if j >= n_real else "bona fide"})
    meta = pd.DataFrame(rows)
    meta["y"] = (meta["label"] == "spoof").astype(int)
    return meta


def _v2():
    dev = [f"s{i:02d}" for i in (0, 1, 2, 6, 7, 8)]
    test = [f"s{i:02d}" for i in (3, 4, 5, 9, 10, 11)]
    return {"dev_speakers": dev, "test_speakers": test}


def test_platform_class_cell_counts():
    counts = w3b.speaker_cell_counts(_meta())
    assert counts["s00"] == {("p0", 0): 15, ("p0", 1): 5}
    assert counts["s01"] == {("p0", 0): 5, ("p0", 1): 15}


def test_speaker_majority_and_its_tie_rule():
    maj = w3b.speaker_majority(_meta())
    assert maj["s00"] == 0 and maj["s01"] == 1
    tie = pd.DataFrame({"speaker": ["t", "t"], "platform": ["p", "p"],
                        "language": ["en", "en"],
                        "label": ["bona fide", "spoof"]})
    tie["y"] = (tie["label"] == "spoof").astype(int)
    assert w3b.speaker_majority(tie)["t"] == 0        # ties fall to bona fide


def test_largest_remainder_is_proportional_and_exact():
    assert w3b.largest_remainder({0: 50, 1: 50}, 10) == {0: 5, 1: 5}
    assert w3b.largest_remainder({0: 90, 1: 10}, 10) == {0: 9, 1: 1}
    got = w3b.largest_remainder({0: 1, 1: 2}, 2)
    assert sum(got.values()) == 2
    assert got == {0: 1, 1: 1}
    assert w3b.largest_remainder({0: 3, 1: 0}, 2) == {0: 2, 1: 0}


def test_stratified_order_keeps_the_label_proportions():
    speakers = [f"r{i}" for i in range(40)] + [f"f{i}" for i in range(20)]
    majority = {s: (1 if s.startswith("f") else 0) for s in speakers}
    import numpy as np

    order = w3b.stratified_order(speakers, majority, 20, np.random.default_rng(0))
    assert len(order) == 60 and len(set(order)) == 60        # a full permutation
    head = order[:20]
    n_fake = sum(1 for s in head if s.startswith("f"))
    assert abs(n_fake - 20 / 60 * 20) <= 1
    assert len(set(head)) == 20                       # no speaker twice


def test_moves_come_from_v2_test_and_keep_the_test_floor():
    meta = _meta()
    out = w3b.try_move(meta, _v2()["dev_speakers"], _v2()["test_speakers"], 2,
                       seed=1, dev_min=5, test_min=10)
    assert out is not None
    assert len(out["moved"]) == 2
    assert set(out["moved"]) <= set(_v2()["test_speakers"])
    assert set(out["dev_speakers"]) == set(_v2()["dev_speakers"]) | set(out["moved"])
    assert set(out["test_speakers"]) == set(_v2()["test_speakers"]) - set(out["moved"])
    assert min(out["test_counts"].values()) >= 10
    assert min(out["dev_counts"].values()) >= 5
    assert out["n_dev"] == len(set(out["dev_speakers"])) * 20
    assert out["n_dev"] + sum(out["test_counts"].values()) == out["n_total"] == 240


def test_move_is_refused_when_the_test_floor_cannot_be_held():
    meta = _meta()
    # v2 test already holds only 25 clips of one cell, so a floor of 30 is impossible
    assert w3b.try_move(meta, _v2()["dev_speakers"], _v2()["test_speakers"], 1,
                        seed=1, dev_min=5, test_min=30) is None
    # asking for more moves than speakers
    assert w3b.try_move(meta, _v2()["dev_speakers"], _v2()["test_speakers"], 99,
                        seed=1, dev_min=5, test_min=10) is None


def test_final_counts_conserve_every_clip():
    meta = _meta()
    counts = w3b.speaker_cell_counts(meta)
    dev = {("p0", 0): 20, ("p0", 1): 20}
    test = {("p0", 0): 20, ("p0", 1): 20}
    d, t = w3b.final_counts(dev, test, counts, ["s00"])
    assert d[("p0", 0)] == 35 and t[("p0", 0)] == 5
    assert sum(d.values()) + sum(t.values()) == sum(dev.values()) + sum(test.values())


def test_make_split_v3_is_deterministic_and_reports_its_size():
    meta = _meta()
    a = w3b.make_split_v3(meta, _v2(), n_move=2, dev_min=5, test_min=10, seed0=7)
    b = w3b.make_split_v3(meta, _v2(), n_move=2, dev_min=5, test_min=10, seed0=7)
    assert a["moved"] == b["moved"]
    assert a["n_dev_speakers"] == 8 and a["n_test_speakers"] == 4
    assert 0.0 < a["dev_frac"] < 1.0
    assert a["moved_strata"]["real_majority"] + a["moved_strata"]["fake_majority"] == 2


def test_audit_flags_a_result_file_that_names_the_test_region(tmp_path):
    (tmp_path / "results").mkdir()
    pd.DataFrame({"region": ["dev", "dev"]}).to_csv(tmp_path / "results" / "ok.csv", index=False)
    pd.DataFrame({"region": ["test"]}).to_csv(tmp_path / "results" / "bad.csv", index=False)
    audit = w3b.audit_test_usage(tmp_path)
    row = audit[audit["check"] == "result_files_region_column"].iloc[0]
    assert row["ok"] is False or row["ok"] == False           # noqa: E712
    assert "bad.csv" in row["detail"]


def test_audit_passes_on_a_clean_tree(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "w6_baselines.py").write_text("x = 1\n", encoding="utf-8")
    pd.DataFrame({"region": ["dev"]}).to_csv(tmp_path / "results" / "ok.csv", index=False)
    (tmp_path / "scripts" / "sneaky.py").write_text(
        "side = 'test'\n", encoding="utf-8")
    audit = w3b.audit_test_usage(tmp_path)
    rows = dict(zip(audit["check"], audit["ok"]))
    assert rows["result_files_region_column"] is True or rows["result_files_region_column"] == True  # noqa: E712
    assert not (rows["source_paths_touching_test_side"] is True
                or rows["source_paths_touching_test_side"] == True)  # noqa: E712

"""Pins for the W12 driver and the paired table decide.py reads."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w12_casa  # noqa: E402


def _pool(seed=0, d=16, n_cells=6, per=120):
    rng = np.random.default_rng(seed)
    Z, y, plat, lang, cell, spk = [], [], [], [], [], []
    for c in range(n_cells):
        rate = (0.85, 0.5, 0.15)[c % 3]
        yy = (rng.random(per) < rate).astype(int)
        Z.append(rng.normal(size=(per, d)) + (2 * yy[:, None] - 1) * 0.7)
        y.append(yy)
        plat += [f"p{c % 3}"] * per
        lang += [f"l{c}"] * per
        cell += [f"p{c % 3}|l{c}"] * per
        spk += [f"s{c}_{i // 4}" for i in range(per)]
    return (np.vstack(Z), np.concatenate(y), np.array(plat), np.array(lang),
            np.array(cell), np.array(spk))


def test_run_cell_labels_exactly_k_and_scores_the_rest():
    Z, y, plat, lang, cell, spk = _pool()
    rows = w12_casa.run_cell(("U", 100, 0, Z, y, plat, lang, cell, spk))
    assert rows, "no rows produced"
    for r in rows:
        assert r["n_labeled"] == 100
        assert r["k"] == 100
        assert r["method"] == "U"
        assert r["n_eval"] > 100
        assert 0.0 <= r["pooled_eer"] <= 100.0
    assert {r["mask"] for r in rows} <= {"clips", "strict"}


def test_paired_table_shape_matches_gate_g7():
    Z, y, plat, lang, cell, spk = _pool()
    rows = []
    for strategy in ("U", "CASA"):
        for d in range(4):
            rows += w12_casa.run_cell((strategy, 100, d, Z, y, plat, lang, cell, spk))
    df = pd.DataFrame(rows)
    paired = w12_casa.paired_table(df)
    assert {"method", "other", "metric", "k", "mean", "lo", "hi", "n"}.issubset(paired.columns)
    assert set(paired["method"]) == {"CASA"}
    assert set(paired["other"]) == {"U"}
    assert set(paired["metric"]) == {"pooled_eer", "within_macro_eer"}
    assert (paired["k"] == 100).all()
    assert (paired.dropna(subset=["lo", "hi"])["hi"] >=
            paired.dropna(subset=["lo", "hi"])["lo"]).all()


def test_casa_can_be_better_than_uniform_when_one_cell_is_starved():
    """CASA's V_c term should help when a small cell is under-sampled by uniform."""
    rng = np.random.default_rng(3)
    d = 12
    Z, y, plat, lang, cell, spk = [], [], [], [], [], []
    sizes = {"p0|l0": 900, "p0|l1": 120, "p1|l0": 900, "p1|l1": 120}
    for name, n in sizes.items():
        rate = 0.5
        yy = (rng.random(n) < rate).astype(int)
        Z.append(rng.normal(size=(n, d)) + (2 * yy[:, None] - 1) * 0.9)
        y.append(yy)
        plat += [name.split("|")[0]] * n
        lang += [name.split("|")[1]] * n
        cell += [name] * n
        spk += [f"{name}_{i // 20}" for i in range(n)]
    Z, y = np.vstack(Z), np.concatenate(y)
    plat, lang, cell, spk = map(np.array, (plat, lang, cell, spk))
    rows = []
    for strategy in ("U", "CASA"):
        for dd in range(6):
            rows += w12_casa.run_cell((strategy, 200, dd, Z, y, plat, lang, cell, spk))
    df = pd.DataFrame(rows)
    paired = w12_casa.paired_table(df)
    assert not paired.empty
    # both directions are recorded with the right sign
    assert (paired["lo"] <= paired["mean"]).all() and (paired["mean"] <= paired["hi"]).all()
def test_w12_can_target_a_backbone_and_a_suffix(monkeypatch, tmp_path):
    """W22 item 5 needs CASA on the second backbone without touching the stored table."""
    import w12_casa as w12

    called = {}

    def fake_load_spec(root, data_dir, backbone, spec, split_name=None):
        called.update({"backbone": backbone, "spec": spec, "split": split_name})
        n = 200
        sub = pd.DataFrame({"y": np.tile([0, 1], n // 2),
                            "platform": ["p"] * n, "language": ["en"] * n,
                            "speaker": [f"s{i // 4:02d}" for i in range(n)]})
        sub["cell"] = "p|en"
        return sub, np.zeros((n, 6), np.float32)

    monkeypatch.setattr(w12.w6, "load_spec", fake_load_spec)

    rc = w12.main(["--root", str(tmp_path), "--data", str(tmp_path), "--draws", "1",
                   "--jobs", "1", "--backbone", "mms-1b", "--spec", "S3",
                   "--split", "speaker_split_v3.json", "--suffix", "_mms1b"])
    assert rc == 0
    assert called == {"backbone": "mms-1b", "spec": "S3",
                      "split": "speaker_split_v3.json"}
    # the rows must record the backbone the run actually used
    assert w12.w6.BACKBONE == "mms-1b" and w12.w6.SPEC == "S3"
    out = tmp_path / "results"
    assert (out / "w12_draws_mms1b.csv").exists()
    assert (out / "w12_casa_mms1b.csv").exists()

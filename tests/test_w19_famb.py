"""Pins for the W19 unseen-chain driver."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w19_famb  # noqa: E402


def _world(seed=0, n_cells=8, per=140, dim=16):
    rng = np.random.default_rng(seed)
    Z, y, plat, lang, cell, spk = [], [], [], [], [], []
    for c in range(n_cells):
        rate = (0.85, 0.5, 0.15)[c % 3]
        yy = (rng.random(per) < rate).astype(int)
        Z.append(rng.normal(size=(per, dim)) + (2 * yy[:, None] - 1) * 0.8)
        y.append(yy)
        plat += [f"p{c % 3}"] * per
        lang += [f"l{c}"] * per
        cell += [f"p{c % 3}|l{c}"] * per
        spk += [f"s{c}_{i // 4}" for i in range(per)]
    Z = np.vstack(Z)
    rng2 = np.random.default_rng(seed + 1)
    # the unseen chain degrades the features by a controlled amount
    tiles = {
        "famA1": Z + rng2.normal(0, 0.02, Z.shape).astype(np.float32),
        "famA2": Z + rng2.normal(0, 0.03, Z.shape).astype(np.float32),
        "famB1": Z + rng2.normal(0, 0.60, Z.shape).astype(np.float32),
    }
    return Z.astype(np.float32), tiles, np.concatenate(y), np.array(plat), np.array(lang), \
        np.array(cell), np.array(spk)


def test_run_one_reports_clean_and_shifted_rows_for_every_method():
    Z, tiles, y, plat, lang, cell, spk = _world()
    rows = w19_famb.run_one((0, Z, tiles, y, plat, lang, cell, spk, 10))
    assert rows
    df = pd.DataFrame(rows)
    assert set(df["method"]) == set(w19_famb.METHODS)
    assert set(df["mask"]) == {"clean", "famb"}
    assert (df["n_eval"] > 100).all()
    assert df["pooled_eer"].between(0, 100).all()
    # no method may IMPROVE on an unseen chain, and most must degrade
    piv = df.pivot_table(index="method", columns="mask", values="pooled_eer")
    assert (piv["famb"] >= piv["clean"] - 0.5).all(), piv
    assert (piv["famb"] > piv["clean"]).sum() >= 2, piv


def test_load_tiles_validates_row_count(tmp_path):
    (tmp_path / "x3" / "views" / "famB1" / "b").mkdir(parents=True)
    np.save(tmp_path / "x3" / "views" / "famB1" / "b" / "spec_feats.npy",
            np.zeros((5, 4), np.float16))
    ident = lambda X: np.asarray(X, np.float32)  # noqa: E731
    out = w19_famb.load_tiles(tmp_path, "b", ["famB1"], ident, 5)
    assert out["famB1"].shape == (5, 4)
    try:
        w19_famb.load_tiles(tmp_path, "b", ["famB1"], ident, 6)
    except ValueError as exc:
        assert "expected 6" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a row-count mismatch must be refused")


def test_load_tiles_refuses_a_missing_view(tmp_path):
    ident = lambda X: np.asarray(X, np.float32)  # noqa: E731
    try:
        w19_famb.load_tiles(tmp_path, "b", ["nope"], ident, 1)
    except FileNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a missing view must be refused, not skipped")

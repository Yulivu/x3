"""Pins for the W18 C4 / TTA driver."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w18_c4  # noqa: E402


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
    Z = np.vstack(Z).astype(np.float32)
    rng2 = np.random.default_rng(seed + 1)
    tiles = {
        "famA1": Z + rng2.normal(0, 0.05, Z.shape).astype(np.float32),
        "famA2": Z + rng2.normal(0, 0.05, Z.shape).astype(np.float32),
    }
    return Z, tiles, np.concatenate(y), np.array(plat), np.array(lang), np.array(cell), \
        np.array(spk)


def test_run_one_emits_all_four_arms_on_both_masks():
    Z, tiles, y, plat, lang, cell, spk = _world()
    rows = w18_c4.run_one((0, Z, tiles, y, plat, lang, cell, spk, 10))
    assert rows
    df = pd.DataFrame(rows)
    assert set(df["method"]) == set(w18_c4.ARMS)
    assert set(df["mask"]) == {"clips", "strict"}
    assert set(df[df["method"].str.endswith("+TTA")]["arm"]) == {"E2"}
    assert (df[df["method"] == "C1"]["arm"] == "none").all()
    assert df["pooled_eer"].between(0, 100).all()


def test_paired_table_pairs_each_tta_arm_with_its_own_base():
    Z, tiles, y, plat, lang, cell, spk = _world(seed=2)
    rows = []
    for d in range(4):
        rows += w18_c4.run_one((d, Z, tiles, y, plat, lang, cell, spk, 10))
    tbl = w18_c4.paired(pd.DataFrame(rows))
    assert not tbl.empty
    assert set(zip(tbl["method"], tbl["other"])) == {("C1+TTA", "C1"), ("PAST+TTA", "PAST")}
    assert (tbl["hi"] >= tbl["lo"]).all()
    assert (tbl["arm"] == "E2").all()

"""Pins for the W11 driver and the PAST ablations."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w11_past  # noqa: E402
from src.past import PAST  # noqa: E402


def _pool(seed=0, d=14, n_cells=6, per=140):
    rng = np.random.default_rng(seed)
    Z, y, plat, lang, cell, spk = [], [], [], [], [], []
    for c in range(n_cells):
        rate = (0.88, 0.5, 0.12)[c % 3]
        yy = (rng.random(per) < rate).astype(int)
        Z.append(rng.normal(size=(per, d)) + (2 * yy[:, None] - 1) * 0.7)
        y.append(yy)
        plat += [f"p{c % 3}"] * per
        lang += [f"l{c}"] * per
        cell += [f"p{c % 3}|l{c}"] * per
        spk += [f"s{c}_{i // 4}" for i in range(per)]
    return (np.vstack(Z), np.concatenate(y), np.array(plat), np.array(lang),
            np.array(cell), np.array(spk))


def test_past_ablations_toggle_the_documented_pieces():
    Z, y, plat, lang, cell, spk = _pool()
    rng = np.random.default_rng(1)
    lab = np.zeros(len(y), bool)
    for c in np.unique(cell):
        idx = np.where(cell == c)[0]
        lab[rng.choice(idx, 10, replace=False)] = True
    args = (Z[lab], y[lab], plat[lab], lang[lab], cell[lab],
            Z[~lab], plat[~lab], lang[~lab], cell[~lab])
    base = PAST(rounds=2).fit(*args)
    no_em = PAST(rounds=2, use_em=False).fit(*args)
    no_q = PAST(rounds=2, use_quota=False).fit(*args)
    one = PAST(rounds=1).fit(*args)
    def sc(m):
        return m.predict_logit(Z, plat, lang, cell)

    assert not np.allclose(sc(base), sc(no_em))
    assert not np.allclose(sc(base), sc(no_q))
    assert len(one.history_) == 1 and len(base.history_) == 2
    for m in (base, no_em, no_q, one):
        s = sc(m)
        assert s.shape == (len(y),) and np.isfinite(s).all()


def test_run_one_and_paired_table():
    Z, y, plat, lang, cell, spk = _pool(seed=4)
    rows = w11_past.run_one((0, Z, y, plat, lang, cell, spk, 10))
    assert rows
    df = pd.DataFrame(rows)
    assert set(df["method"]) == set(w11_past.VARIANTS)
    assert set(df["mask"]) <= {"clips", "strict"}
    assert (df["views"] == "none").all()

    rows = []
    for d in range(4):
        rows += w11_past.run_one((d, Z, y, plat, lang, cell, spk, 10))
    paired = w11_past.paired_table(pd.DataFrame(rows))
    assert set(paired["method"]) == {"PAST"}
    assert "C1" in set(paired["other"])
    assert {"mean", "lo", "hi", "n", "metric"}.issubset(paired.columns)
    finite = paired.dropna(subset=["lo", "hi"])
    assert (finite["hi"] >= finite["lo"]).all()

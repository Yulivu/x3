"""Pins for the W7 budget-curve driver."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w6_baselines as w6  # noqa: E402
import w7_curve  # noqa: E402


def _pool(seed=0, d=14, n_cells=6, per=160):
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


def test_run_one_draw_supports_past_and_the_budget_knob():
    Z, y, plat, lang, cell, spk = _pool()
    zero = y * 0.8 + np.random.default_rng(1).normal(0, 1.0, len(y))
    for m in (1, 3, 10):
        rows = pd.DataFrame(w6.run_one_draw((0, Z, y, plat, lang, cell, spk, zero,
                                             ("B1", "C1", "PAST"), m)))
        assert not rows.empty, f"m={m} produced nothing"
        assert set(rows["method"]) == {"B1", "C1", "PAST"}
        assert (rows["n_labeled"] > 0).all()
    small = pd.DataFrame(w6.run_one_draw((0, Z, y, plat, lang, cell, spk, zero, ("B1",), 1)))
    big = pd.DataFrame(w6.run_one_draw((0, Z, y, plat, lang, cell, spk, zero, ("B1",), 40)))
    assert big["n_labeled"].iloc[0] > small["n_labeled"].iloc[0]


def test_past_rows_are_marked_as_the_no_view_variant():
    Z, y, plat, lang, cell, spk = _pool(seed=2)
    zero = np.zeros(len(y))
    rows = pd.DataFrame(w6.run_one_draw((0, Z, y, plat, lang, cell, spk, zero, ("PAST",), 10)))
    assert set(rows["method"]) == {"PAST"}
    assert (rows["n_eval"] > 100).all()
    assert {"pooled_eer", "within_macro_eer", "auc"}.issubset(rows.columns)

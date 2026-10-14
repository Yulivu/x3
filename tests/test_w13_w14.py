"""Pins for the W13 (B10 reduced) and W14 (GraphSSL) drivers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w13_ft_last  # noqa: E402
import w14_graphssl  # noqa: E402


def _pool(seed=0, d=14, n_cells=8, per=140):
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


def test_b10_rows_are_labelled_reduced_and_carry_the_lr():
    Z, y, plat, lang, cell, spk = _pool()
    rows = pd.DataFrame(w13_ft_last.run_one((0, Z, y, plat, lang, cell, spk, 10)))
    assert not rows.empty
    assert set(rows["method"]) == {"B10"}
    assert set(rows["variant"]) == {"B10_REDUCED"}
    assert rows["lr"].iloc[0] in (1e-5, 3e-5)
    assert rows["note"].str.contains("B10_REDUCED").all()
    assert {"pooled_eer", "within_macro_eer", "auc"}.issubset(rows.columns)


def test_graphssl_rows_are_marked_transductive_and_record_eta():
    Z, y, plat, lang, cell, spk = _pool(seed=3)
    rows = pd.DataFrame(w14_graphssl.run_one((0, Z, y, plat, lang, cell, spk, 10)))
    assert not rows.empty
    assert set(rows["method"]) == {"B9", "GraphSSL"}
    assert rows["transductive"].all()
    eta = rows[rows["method"] == "GraphSSL"]["eta"].iloc[0]
    assert eta in (0.25, 0.5, 1.0)
    assert rows["pooled_eer"].between(0, 100).all()


def test_both_drivers_share_the_w6_draw_protocol():
    """S-F pairs by (method, draw), so the draw column must be the same seed family."""
    Z, y, plat, lang, cell, spk = _pool(seed=5)
    a = pd.DataFrame(w13_ft_last.run_one((0, Z, y, plat, lang, cell, spk, 10)))
    b = pd.DataFrame(w14_graphssl.run_one((0, Z, y, plat, lang, cell, spk, 10)))
    for df in (a, b):
        assert df["protocol"].unique().tolist() == ["dev_v2"]
        assert df["region"].unique().tolist() == ["dev"]
        assert df["draw"].unique().tolist() == [0]
        assert (df["k"] == 290).all()

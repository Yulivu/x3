"""Pins for the W16 factor matrix and the G10 mechanism measurement."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w16_factorial  # noqa: E402


def _pool(seed=0, d=14, n_cells=6, per=140):
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


def test_platform_macro_auc_bounds():
    plat = np.array(["a"] * 100 + ["b"] * 100)
    perfect = np.concatenate([np.zeros(100), np.ones(100)])
    assert w16_factorial.platform_macro_auc(perfect, plat) == 1.0
    rng = np.random.default_rng(0)
    assert 0.3 < w16_factorial.platform_macro_auc(rng.normal(size=200), plat) < 0.7


def test_run_config_emits_all_eight_factor_settings():
    Z, y, plat, lang, cell, spk = _pool()
    rows = []
    for c1 in (0, 1):
        for past in (0, 1):
            for casa in (0, 1):
                rows += w16_factorial.run_config((c1, past, casa, 100, 0, Z, y, plat, lang, cell, spk))
    df = pd.DataFrame(rows)
    assert df["method"].nunique() == 8
    assert set(df["mask"]) == {"clips", "strict"}
    assert (df["n_labeled"] == 100).all()
    assert df["method"].str.contains("C1=").all()


def test_marginals_report_sign_and_labels():
    Z, y, plat, lang, cell, spk = _pool(seed=3)
    rows = []
    for c1 in (0, 1):
        for past in (0, 1):
            for casa in (0, 1):
                for d in range(3):
                    rows += w16_factorial.run_config((c1, past, casa, 290, d, Z, y, plat, lang, cell, spk))
    marg = w16_factorial.marginals(pd.DataFrame(rows))
    assert set(marg[marg["kind"] == "marginal"]["factor"]) == {"c1", "past", "casa"}
    assert set(marg[marg["kind"] == "interaction"]["factor"]) == {"c1xpast", "c1xcasa", "pastxcasa"}
    assert np.isfinite(marg["effect"]).all()


def test_mechanism_rows_have_both_methods_and_metrics():
    Z, y, plat, lang, cell, spk = _pool(seed=5)
    rows = []
    for d in range(2):
        rows += w16_factorial.mechanism((d, Z, y, plat, lang, cell, spk, 290))
    df = pd.DataFrame(rows)
    assert set(df["method"]) == {"B1", "C1"}
    assert set(df["metric"]) == {"platform_auc", "within_macro_eer"}
    assert df["value"].between(0, 100.01).all()

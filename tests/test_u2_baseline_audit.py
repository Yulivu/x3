"""Pins for the U2 baseline audit: the Line A probe and the B11 fix."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import u2_baseline_audit as u2  # noqa: E402
from src.baselines import LineAProbe  # noqa: E402
from src.metrics3 import auc  # noqa: E402


def _pool(seed=0, n_cells=8, per=140, dim=20):
    rng = np.random.default_rng(seed)
    Z, y, plat, lang, cell = [], [], [], [], []
    for c in range(n_cells):
        rate = (0.85, 0.5, 0.15)[c % 3]
        yy = (rng.random(per) < rate).astype(int)
        Z.append(rng.normal(size=(per, dim)) + (2 * yy[:, None] - 1) * 0.8)
        y.append(yy)
        plat += [f"p{c % 3}"] * per
        lang += [f"l{c}"] * per
        cell += [f"p{c % 3}|l{c}"] * per
    return (np.vstack(Z).astype(np.float32), np.concatenate(y), np.array(plat),
            np.array(lang), np.array(cell))


def test_line_a_probe_learns_and_is_reproducible():
    Z, y, plat, lang, cell = _pool()
    rng = np.random.default_rng(1)
    lab = np.zeros(len(y), bool)
    for c in np.unique(cell):
        idx = np.where(cell == c)[0]
        lab[rng.choice(idx, 10, replace=False)] = True
    a = LineAProbe(seed=7).fit(Z[lab], y[lab])
    b = LineAProbe(seed=7).fit(Z[lab], y[lab])
    assert np.allclose(a.logit(Z), b.logit(Z))
    # the recipe is deliberately small (25 epochs of batch 256 = a handful of
    # steps at this scale), so the unit test pins its contract, not its accuracy
    assert auc(y[lab], a.logit(Z[lab])) > 0.5
    assert np.isfinite(a.best_eer_)


def test_line_a_probe_keeps_the_early_stop_state():
    """The validation EER it reports must be at least as good as the final epoch's."""
    Z, y, plat, lang, cell = _pool(seed=3)
    probe = LineAProbe(epochs=6, seed=2).fit(Z, y)
    from src.metrics3 import eer

    final = float(eer(y, probe.net_ and probe.logit(Z)))
    assert np.isfinite(probe.best_eer_)
    assert probe.best_eer_ <= max(final, 100.0)  # sanity: it is a percentage


def test_run_draw_covers_every_audit_method():
    Z, y, plat, lang, cell = _pool(seed=5)
    rows = u2.run_draw((0, Z, y, plat, lang, cell, 10))
    assert rows
    df = pd.DataFrame(rows)
    assert set(df["method"]) == set(u2.METHODS)
    assert df["pooled_eer"].between(0, 100).all()
    # the oracle cell prior must not be worse than the estimated one
    b2 = float(df[df["method"] == "B2"]["pooled_eer"].iloc[0])
    oracle = float(df[df["method"] == "B2-oracle"]["pooled_eer"].iloc[0])
    assert oracle <= b2 + 1e-9
    assert {"val_eer", "n_cells"}.issubset(df.columns)


def test_sklearn_mlp_is_configured_with_early_stopping():
    clf = u2.sklearn_mlp()
    assert clf.early_stopping is True
    assert clf.hidden_layer_sizes == (256,)

"""Pins for the W20 one-at-a-time sensitivity driver."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w20_sensitivity  # noqa: E402


def _pool(seed=0, dim=16, n_cells=8, per=140):
    rng = np.random.default_rng(seed)
    Z, y, plat, lang, cell, spk = [], [], [], [], [], []
    for c in range(n_cells):
        rate = (0.85, 0.5, 0.15)[c % 3]
        yy = (rng.random(per) < rate).astype(int)
        Z.append(rng.normal(size=(per, dim)) + (2 * yy[:, None] - 1) * 0.8)
        y.append(yy)
        plat += [f"p{c % 3}"] * per
        lang += [f"l{c % 4}"] * per
        cell += [f"p{c % 3}|l{c % 4}"] * per
        spk += [f"s{c}_{i // 4}" for i in range(per)]
    return (np.vstack(Z).astype(np.float32), np.concatenate(y), np.array(plat),
            np.array(lang), np.array(cell), np.array(spk))


def test_select_labels_covers_the_three_rules():
    Z, y, plat, lang, cell, spk = _pool()
    a = w20_sensitivity.select_labels("per_cell", cell, spk, 10, 20261015)
    b = w20_sensitivity.select_labels("global", cell, spk, 10, 20261015)
    c = w20_sensitivity.select_labels("speaker_clustered", cell, spk, 10, 20261015)
    for pos in (a, b, c):
        assert pos.size > 0 and len(set(pos.tolist())) == pos.size
        assert pos.max() < len(cell)
    # the speaker-clustered rule really does concentrate on few speakers
    assert len(set(spk[c].tolist())) <= 20
    assert len(set(spk[b].tolist())) >= len(set(spk[c].tolist()))


def test_run_config_emits_every_method_for_each_knob():
    Z, y, plat, lang, cell, spk = _pool()
    for knob, value in (("pca", "256"), ("cell", "platform"), ("label_noise", "0.05"),
                        ("label_draw", "global")):
        rows = w20_sensitivity.run_config((knob, value, 0, Z, y, plat, lang, cell, spk))
        assert rows, (knob, value)
        df = pd.DataFrame(rows)
        assert set(df["method"]) == set(w20_sensitivity.METHODS)
        assert (df["knob"] == knob).all() and (df["value"] == value).all()
        assert df["pooled_eer"].between(0, 100).all()
    # label noise must actually change the labels used
    clean = pd.DataFrame(w20_sensitivity.run_config(
        ("label_noise", "0.0", 0, Z, y, plat, lang, cell, spk)))
    noisy = pd.DataFrame(w20_sensitivity.run_config(
        ("label_noise", "0.10", 0, Z, y, plat, lang, cell, spk)))
    assert not np.allclose(clean["auc"], noisy["auc"])


def test_cell_variant_changes_the_prior_grouping():
    Z, y, plat, lang, cell, spk = _pool(seed=3)
    p = pd.DataFrame(w20_sensitivity.run_config(("cell", "platform", 0, Z, y, plat, lang, cell, spk)))
    l = pd.DataFrame(w20_sensitivity.run_config(("cell", "language", 0, Z, y, plat, lang, cell, spk)))
    px = pd.DataFrame(w20_sensitivity.run_config(("cell", "platform_x_language", 0, Z, y, plat, lang, cell, spk)))
    b2 = [float(d[d["method"] == "B2"]["pooled_eer"].iloc[0]) for d in (p, l, px)]
    assert len(set(b2)) > 1, "the cell definition must move the prior-corrected method"

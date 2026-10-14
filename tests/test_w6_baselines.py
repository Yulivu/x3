"""Pins for the W6 driver (synthetic arrays; the real run happens on the box)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import w6_baselines  # noqa: E402


def _pool(seed=0, d=16, n_cells=6, per=90):
    rng = np.random.default_rng(seed)
    Z, y, plat, lang, cell, spk = [], [], [], [], [], []
    for c in range(n_cells):
        rate = (0.9, 0.5, 0.1)[c % 3]
        yy = (rng.random(per) < rate).astype(int)
        Z.append(rng.normal(size=(per, d)) + (2 * yy[:, None] - 1) * 0.7)
        y.append(yy)
        plat += [f"p{c % 3}"] * per
        lang += [f"l{c}"] * per
        cell += [f"p{c % 3}|l{c}"] * per
        spk += [f"s{c}_{i // 3}" for i in range(per)]
    return (np.vstack(Z), np.concatenate(y), np.array(plat), np.array(lang),
            np.array(cell), np.array(spk))


def test_one_draw_produces_both_masks_and_the_documented_schema():
    Z, y, plat, lang, cell, spk = _pool()
    zero = y * 0.8 + np.random.default_rng(1).normal(0, 1.0, len(y))
    payload = (0, Z, y, plat, lang, cell, spk, zero, ("B1", "B2", "C1", "B7"), 10)
    rows = pd.DataFrame(w6_baselines.run_one_draw(payload))
    assert not rows.empty
    for col in ("protocol", "region", "backbone", "spec", "method", "k", "draw",
                "mask", "n_labeled", "n_eval", "pooled_eer", "within_macro_eer",
                "worst_platform_eer", "auc"):
        assert col in rows.columns, col
    main = rows[rows["mask"].isin(["clips", "strict"])]
    assert set(main["method"]) == {"B1", "B2", "C1", "B7"}
    assert set(main["mask"]) == {"clips", "strict"}
    assert main["n_labeled"].iloc[0] == 60
    assert main["protocol"].unique().tolist() == ["dev_v2"]
    assert main["spec"].unique().tolist() == ["S1-lineA"]
    # regression: an inductive probe must score EVERY clip, not only the labelled
    # rows (scoring only L makes the evaluation set constant and EER exactly 50)
    for method in ("B1", "B11"):
        sub = main[(main["method"] == method) & (main["mask"] == "clips")]
        if not sub.empty:
            assert sub["auc"].iloc[0] > 0.55, f"{method} scored a constant on the evaluation set"


def test_collapse_rows_are_emitted_for_b7():
    Z, y, plat, lang, cell, spk = _pool(seed=4)
    zero = y * 0.5 + np.random.default_rng(2).normal(0, 1.0, len(y))
    rows = pd.DataFrame(w6_baselines.run_one_draw(
        (0, Z, y, plat, lang, cell, spk, zero, ("B1", "B7"), 10)))
    coll = rows[rows["mask"] == "collapse"]
    assert not coll.empty
    assert {"cell", "true_rate", "pl_rate", "n_pseudo"}.issubset(coll.columns)
    assert ((coll["true_rate"] >= 0) & (coll["true_rate"] <= 1)).all()


def test_summarize_draws_builds_summary_paired_and_collapse_tables():
    Z, y, plat, lang, cell, spk = _pool(seed=7)
    zero = y * 0.8 + np.random.default_rng(3).normal(0, 1.0, len(y))
    rows = []
    for d in range(4):
        rows += w6_baselines.run_one_draw((d, Z, y, plat, lang, cell, spk, zero,
                                           ("B1", "C1", "B7"), 10))
    df = pd.DataFrame(rows)
    summary, paired_tbl, collapse = w6_baselines.summarize_draws(df)
    assert {"backbone", "spec", "method", "mask", "pooled_eer_mean"}.issubset(summary.columns)
    assert set(summary["method"]) == {"B1", "C1", "B7"}
    assert not paired_tbl.empty
    assert set(paired_tbl["other"]) == {"B1"}
    finite = paired_tbl.dropna(subset=["lo", "hi"])
    assert (finite["hi"] >= finite["lo"]).all()
    assert not collapse.empty

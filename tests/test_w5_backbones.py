"""Pins for the W5 spec search and the G3 ranking table."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w5_backbones  # noqa: E402
from src import pipeline  # noqa: E402


def _block(n=60, L=8, D=6, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    mean = rng.normal(size=(n, L + 1, D)).astype(np.float32)
    if informative:
        mean[:, -1, :] += (2 * y[:, None] - 1) * 1.5
    std = np.abs(rng.normal(size=(n, 8, D))).astype(np.float32)
    zero = (2 * y - 1) * 1.0 + rng.normal(0, 1.0, n)
    return mean, std, zero.astype(np.float32), y


def test_spec_menu_shapes_and_channels():
    mean, std, zero, _y = _block()
    specs = w5_backbones.specs_for(mean, std, zero, n_layers=8)
    assert set(specs) == {"S1", "S2", "S3", "S4", "S5"}
    assert specs["S1"].shape == (60, 5 * 6)          # five layers concatenated
    assert specs["S2"].shape == (60, 2 * 6)          # mean + std of the last state
    assert specs["S3"].shape == (60, 2 * 6)
    assert specs["S4"].shape == (60, 3 * 6)
    assert specs["S5"].shape == (60, 2 * 6 + 1)      # S3 plus the zero-shot logit
    for name, F in specs.items():
        assert np.isfinite(F).all(), name


def test_rank_produces_the_gate_g3_columns_and_a_paired_diff():
    rng = np.random.default_rng(0)
    rows = []
    for key, centre in (("xlsr300m-linea/S1-lineA", 15.0), ("xls-r-2b/S3", 13.5)):
        backbone, spec = key.split("/", 1)
        for d in range(6):
            for metric, base in (("pooled_eer", centre - 2), ("within_macro_eer", centre),
                                 ("worst_platform_eer", centre + 5), ("auc", 0.9)):
                rows.append({"backbone": backbone, "spec": spec, "method": "M2", "mask": "clips",
                             "draw": d, "metric": metric, metric: base + rng.normal(0, 0.2)})
    df = pd.DataFrame(rows)
    table = w5_backbones.rank(df)
    assert {"backbone", "spec", "metric", "mean", "sd", "paired_mean", "paired_lo",
            "paired_hi"}.issubset(table.columns)
    chal = table[(table["backbone"] == "xls-r-2b") & (table["metric"] == "within_macro_eer")].iloc[0]
    assert chal["paired_mean"] < -1.0
    assert chal["paired_hi"] < 0
    # the incumbent has no paired row
    inc = table[(table["spec"] == "S1-lineA") & (table["metric"] == "within_macro_eer")].iloc[0]
    assert np.isnan(inc["paired_mean"])


def test_gate_g3_switches_only_on_a_clear_paired_win(tmp_path):
    (tmp_path / "results").mkdir(parents=True)
    base = [{"backbone": "xlsr300m-linea", "spec": "S1-lineA", "metric": "within_macro_eer",
             "mean": 14.56, "paired_mean": np.nan, "paired_hi": np.nan},
            {"backbone": "xls-r-2b", "spec": "S3", "metric": "within_macro_eer",
             "mean": 13.2, "paired_mean": -1.4, "paired_hi": -0.6}]
    pd.DataFrame(base).to_csv(tmp_path / "results" / "backbone_rank.csv", index=False)
    chosen = pipeline.gate_g3(tmp_path, select=True)
    assert chosen["status"] == "xls-r-2b|S3" and chosen["switched"] is True
    assert (tmp_path / "chosen_backbone.json").exists()
    # a gain inside the tolerance keeps the incumbent
    base[1].update({"paired_mean": -0.2, "paired_hi": 0.3})
    pd.DataFrame(base).to_csv(tmp_path / "results" / "backbone_rank.csv", index=False)
    assert pipeline.gate_g3(tmp_path)["status"] == "KEEP_ORIGINAL"

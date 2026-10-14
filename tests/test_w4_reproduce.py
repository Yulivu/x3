"""Pins for the W4 reproduction driver (synthetic manifest, no box data needed)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import w4_reproduce  # noqa: E402
from src.data import add_cell  # noqa: E402


def _fake_meta(seed=0, n_cells=6, per=200):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cells):
        plat = f"p{c % 3}"
        rate = (0.9, 0.5, 0.1)[c % 3]
        for i in range(per):
            rows.append({
                "speaker": f"s{c}_{i // 25}",
                "language": f"l{c % 2}",
                "platform": plat,
                "label": "spoof" if rng.random() < rate else "bona-fide",
            })
    df = pd.DataFrame(rows)
    df["y"] = (df["label"] == "spoof").astype(int)
    df["clip_id"] = np.arange(len(df))
    return add_cell(df)


def test_run_draws_produces_the_documented_schema():
    meta = _fake_meta()
    rng = np.random.default_rng(1)
    scores = {sd: meta["y"].to_numpy() * 0.8 + rng.normal(0, 1.0, len(meta)) for sd in (0, 1, 2)}
    draws = w4_reproduce.run_draws(meta, scores, n_draws=4, k_per_cell=10)
    for col in ("protocol", "region", "backbone", "spec", "method", "k", "draw", "mask",
                "n_labeled", "n_eval", "pooled_eer", "within_macro_eer",
                "worst_platform_eer", "auc", "hter_pooled"):
        assert col in draws.columns, col
    assert set(draws["method"]) == {"M0", "M1", "M2"}
    assert set(draws["mask"]) == {"clips", "strict"}
    assert draws["protocol"].unique().tolist() == ["v1_full"]
    assert draws["draw"].nunique() == 4
    assert (draws["n_labeled"] > 0).all()
    # M2 includes the prior, so on prior-confounded data it beats M0
    piv = draws[draws["mask"] == "clips"].groupby("method")["pooled_eer"].mean()
    assert piv["M2"] < piv["M0"]


def test_mechanism_reconciliation_and_mask_tables():
    meta = _fake_meta(seed=3)
    rng = np.random.default_rng(2)
    scores = {sd: meta["y"].to_numpy() * 0.7 + rng.normal(0, 1.0, len(meta)) for sd in (0, 1, 2)}
    draws = w4_reproduce.run_draws(meta, scores, n_draws=6, k_per_cell=10)
    mech = w4_reproduce.mechanism_table(draws)
    assert {"method", "mask", "pooled_eer_mean", "pooled_eer_std"}.issubset(mech.columns)
    assert len(mech) == 6  # 3 methods x 2 masks

    rec = w4_reproduce.reconciliation(draws)
    assert set(rec["scope"]) == {"3draw", "100draw"}
    assert set(rec["metric"]) == set(w4_reproduce.STORED_FSCVAR)
    assert np.allclose(rec["diff"], rec["rerun"] - rec["headline"])
    assert set(rec["reference"]) == {"fsc_var_results.csv (bb, k=10)"}
    # the two masks must never be averaged together
    assert set(rec["mask"]) == {"clips"}

    ident = w4_reproduce.mask_identity(draws)
    assert {"method", "n_draws", "n_identical", "share_identical"}.issubset(ident.columns)
    assert (ident["share_identical"] >= 0).all()


def test_linea_routes_separates_the_two_methods():
    routes = w4_reproduce.linea_routes()
    assert set(routes["route"]) == {"A-adaptation", "B-prior-correction"}
    adapt = routes[routes["route"] == "A-adaptation"].iloc[0]
    prior = routes[routes["route"] == "B-prior-correction"].iloc[0]
    assert adapt["prior_correction"] is False or adapt["prior_correction"] == False  # noqa: E712
    assert prior["prior_correction"] is True or prior["prior_correction"] == True  # noqa: E712
    # the adaptation route is the 13.x headline; the prior route is the 24.x one
    assert abs(adapt["strict_pooled"] - 13.64) < 0.01
    assert abs(prior["clips_pooled"] - 24.65) < 0.01

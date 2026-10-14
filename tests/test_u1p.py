"""Pins for U1' (v3 dev, V-1 intervals, V-3 folds)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import u1p_speaker_split_v3 as u1p  # noqa: E402
from src.baselines import knn_graph  # noqa: E402
from src.paired import save_draw_scores  # noqa: E402


def _region(n_speakers=24, per_speaker=12, seed=0):
    rng = np.random.default_rng(seed)
    speaker = np.repeat(np.arange(n_speakers), per_speaker)
    platform = np.array(["p0", "p1"])[speaker % 2]
    language = np.array(["en", "zh"])[(speaker // 2) % 2]
    cell = np.array([f"{p}|{l}" for p, l in zip(platform, language)])
    y = rng.integers(0, 2, size=len(speaker))
    return speaker, platform, language, cell, y


def test_mask_sets_are_a_partition_of_the_unlabelled_clips():
    speaker, platform, language, cell, y = _region()
    lab = np.array([0, 1, 12, 13, 24, 25])
    got = u1p.mask_sets(speaker, cell, lab, m=2, seed=1)
    ev = np.concatenate([got["seen"], got["unseen"]])
    assert len(ev) == len(speaker) - len(lab)
    assert not set(ev.tolist()) & set(lab.tolist())
    assert set(got["seen"].tolist()) & set(got["unseen"].tolist()) == set()
    lab_spk = set(speaker[lab].tolist())
    assert set(speaker[got["seen"]].tolist()) == lab_spk
    assert not set(speaker[got["unseen"]].tolist()) & lab_spk


def test_strict_labels_only_come_from_s_l_and_honour_the_quota():
    speaker, platform, language, cell, y = _region()
    lab, s_e = u1p.strict_labels(speaker, cell, m=3, seed=777000)
    s_l = set(np.unique(speaker)) - set(speaker[s_e].tolist())
    assert lab.sum() > 0
    assert set(speaker[lab].tolist()) <= s_l            # never from S_E
    assert set(speaker[lab].tolist()) & set(speaker[s_e].tolist()) == set()
    counts = pd.Series(cell[lab]).value_counts()
    assert counts.max() <= 3
    assert len(s_e) > 0


def test_strict_labels_are_reproducible_and_move_with_the_seed():
    speaker, platform, language, cell, y = _region()
    a, ea = u1p.strict_labels(speaker, cell, m=3, seed=100)
    b, eb = u1p.strict_labels(speaker, cell, m=3, seed=100)
    assert np.array_equal(a, b) and np.array_equal(ea, eb)
    c, ec = u1p.strict_labels(speaker, cell, m=3, seed=101)
    assert not np.array_equal(ea, ec)


def _toy_fit_inputs(n_speakers=20, per_speaker=10, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    speaker = np.repeat(np.arange(n_speakers), per_speaker)
    platform = np.array(["p0", "p1"])[speaker % 2]
    language = np.array(["en", "zh"])[(speaker // 2) % 2]
    cell = np.array([f"{p}|{l}" for p, l in zip(platform, language)])
    # speaker offsets make the grouped folds matter
    off = rng.normal(size=n_speakers)[speaker]
    y = rng.integers(0, 2, size=len(speaker))
    Z = rng.normal(size=(len(speaker), dim)).astype(np.float32)
    Z[:, 0] += 1.2 * y + 0.4 * off
    lab = rng.choice(len(speaker), size=80, replace=False)
    return Z, y, platform, language, cell, speaker, lab


def test_fit_methods_returns_one_finite_score_per_clip():
    from src.cv import speaker_grouped_folds

    Z, y, plat, lang, cell, speaker, lab = _toy_fit_inputs()
    lab_mask = np.zeros(len(y), bool)
    lab_mask[lab] = True
    folds = speaker_grouped_folds(speaker[lab_mask], y[lab_mask])
    graph = knn_graph(Z, k=10)
    scores, cv = u1p.fit_methods(Z, y, plat, lang, cell, speaker, lab_mask, folds,
                                 graph, int((~lab_mask).sum()))
    assert set(scores) == set(u1p.METHODS)
    for name, sc in scores.items():
        assert sc.shape == (len(y),)
        assert np.all(np.isfinite(sc)), name
    for name in u1p.METHODS:
        assert "cv_loglik" in cv[name]
    # every method separates the toy signal better than chance
    for name, sc in scores.items():
        assert np.mean(sc[y == 1]) > np.mean(sc[y == 0]), name


def test_intervals_table_pairs_the_persisted_draws(tmp_path):
    rng = np.random.default_rng(2)
    speech = np.repeat(np.arange(30), 4)
    y = np.tile([0, 1], 60)
    for d in range(7):
        sa = y * 1.0 + rng.normal(0, 1, size=len(y))
        sb = sa + 1.5 * y                      # "b" separates better -> a is worse
        save_draw_scores(tmp_path / "k290" / "strict_draw", d, y,
                         {"B9": sa, "B1": sb}, speech)
    iv = u1p.intervals_table(tmp_path, budgets=[290],
                             pairs=[("B9", "B1")], masks=["strict_draw"])
    assert len(iv) == 1
    row = iv.iloc[0]
    assert row["n_draws"] == 7
    assert row["method_a"] == "B9" and row["method_b"] == "B1"
    assert row["delta"] > 0.0                  # EER(B9) - EER(B1) > 0
    assert row["old_lo"] <= row["old_hi"] and row["b_lo"] <= row["b_hi"]
    assert bool(row["meets_2pt_rule"]) == (row["effect_points"] >= 2.0
                                           and not row["b_contains_zero"])


def test_intervals_table_needs_enough_draws(tmp_path):
    y = np.tile([0, 1], 20)
    for d in range(3):
        save_draw_scores(tmp_path / "k100" / "seen", d, y,
                         {"B9": y * 1.0, "B1": y * 0.5}, np.repeat(np.arange(10), 4))
    empty = u1p.intervals_table(tmp_path, budgets=[100], pairs=[("B9", "B1")],
                                masks=["seen"])
    assert empty.empty                              # 3 draws is not a verdict


def test_verdicts_read_the_b_interval_and_the_speaker_ratio():
    df = pd.DataFrame({
        "method": ["B1"] * 4 + ["B9"] * 4,
        "mask": ["seen", "unseen", "strict_draw", "strict_draw"] * 2,
        "pooled_eer": [7.0, 13.0, 15.0, 15.0, 6.0, 12.4, 14.0, 14.0],
        "k": [290] * 8,
    })
    iv = pd.DataFrame([
        {"k": 290, "mask": "unseen", "method_a": "B9", "method_b": "B1",
         "delta": -0.6, "old_lo": -4.0, "old_hi": 2.8, "b_lo": -1.2, "b_hi": -0.1,
         "b_contains_zero": False},
        {"k": 290, "mask": "strict_draw", "method_a": "C1", "method_b": "B1",
         "delta": -1.4, "old_lo": -6.0, "old_hi": 1.0, "b_lo": -3.5, "b_hi": 0.4,
         "b_contains_zero": True},
    ])
    flags = u1p.verdicts(df, iv)
    assert flags["SPEAKER_GAP"] is True
    assert "B1" in flags["SPEAKER_GAP_methods"] and "B9" in flags["SPEAKER_GAP_methods"]
    # B9 does beat B1 on unseen under the (b) rule -> the gain is not speaker-local
    assert flags["LP_GAIN_IS_SPEAKER_LOCAL"] is False
    assert flags["G4_C1_minus_B1_strict_draw"]["C1_beats_B1"] is False
    assert json.dumps(flags)                       # the flags must be serialisable


def test_verdicts_separate_a_real_but_small_gain_from_a_resolved_one():
    df = pd.DataFrame({"method": ["B1"] * 2 + ["B9"] * 2,
                       "mask": ["seen", "unseen"] * 2,
                       "pooled_eer": [7.0, 13.0, 6.0, 12.4], "k": [290] * 4})
    small = pd.DataFrame([{"k": 290, "mask": "unseen", "method_a": "B9",
                           "method_b": "B1", "delta": -0.6, "old_lo": -4.0,
                           "old_hi": 2.8, "b_lo": -1.2, "b_hi": -0.1,
                           "b_contains_zero": False}])
    flags = u1p.verdicts(df, small)
    assert flags["LP_GAIN_UNSEEN_BELOW_RESOLUTION"] is True
    assert flags["LP_GAIN_UNSEEN_MEETS_2PT_BAR"] is False
    big = small.copy()
    big.loc[0, "delta"] = -3.0
    big.loc[0, "b_hi"] = -2.2
    flags = u1p.verdicts(df, big)
    assert flags["LP_GAIN_UNSEEN_MEETS_2PT_BAR"] is True
    assert flags["LP_GAIN_UNSEEN_BELOW_RESOLUTION"] is False

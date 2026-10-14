"""Pins for the U8 driver: masks, the method block, the intervals and G17."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import u8_esn as u8  # noqa: E402
from src.cv import speaker_grouped_folds  # noqa: E402
from src.esn import choose_enrollment, esn_features, speaker_means  # noqa: E402
from src.paired import save_draw_scores  # noqa: E402


def _region(n_speakers=40, per_class=10, dim=6, n_platforms=6, seed=0):
    """Small, but big enough for the driver's floors: >= 50 labelled clips per mask
    and a non-empty ``unseen`` mask, which needs more speakers than cells."""
    rng = np.random.default_rng(seed)
    speaker, y, plat, lang = [], [], [], []
    for s in range(n_speakers):
        for label in (0, 1):
            for _ in range(per_class):
                speaker.append(f"s{s:02d}")
                y.append(label)
                plat.append(f"p{s % n_platforms}")
                lang.append("en" if s % 2 else "zh")
    speaker = np.asarray(speaker)
    y = np.asarray(y)
    plat = np.asarray(plat)
    cell = np.array([f"{p}|{l}" for p, l in zip(plat, lang)])
    Z = rng.normal(size=(len(y), dim)).astype(np.float32)
    Z[y == 1, 0] += 1.5
    Z += np.repeat(np.arange(n_speakers), 2 * per_class)[:, None] * 0.1
    return Z, y, speaker, plat, cell


def test_esn_masks_exclude_both_labelled_and_enrolled_clips():
    Z, y, speaker, plat, cell = _region()
    enroll = choose_enrollment(speaker, y, 3)
    lab_pos = np.where((y == 1) & ~enroll)[0][:40]
    masks = u8.esn_masks(speaker, lab_pos, enroll)
    ev = np.concatenate([masks["seen"], masks["unseen"]])
    assert not set(ev.tolist()) & set(lab_pos.tolist())
    assert not enroll[ev].any()
    assert set(masks["seen"].tolist()) & set(masks["unseen"].tolist()) == set()


def test_esn_labels_never_draw_a_label_from_an_enrolled_clip():
    Z, y, speaker, plat, cell = _region()
    enroll = choose_enrollment(speaker, y, 5)
    lab_pos, strict_lab, s_e = u8.esn_labels(speaker, cell, enroll, 5, d=0)
    assert lab_pos.size > 0 and strict_lab.size > 0
    assert not enroll[lab_pos].any()
    assert not enroll[strict_lab].any()
    # the strict labels come from S_L only
    s_l = set(np.unique(speaker)) - set(speaker[s_e].tolist())
    assert set(speaker[strict_lab].tolist()) <= s_l


def test_fit_all_returns_every_method_and_matches_b1_at_m_zero():
    Z, y, speaker, plat, cell = _region()
    n = len(y)
    m = 5
    enroll = choose_enrollment(speaker, y, m)
    lab_mask = np.zeros(n, bool)
    lab_mask[::4] = True
    lab_mask &= ~enroll
    folds = speaker_grouped_folds(speaker[lab_mask], y[lab_mask])
    means = speaker_means(Z, speaker, enroll, plat, m=m)
    Zt = esn_features(Z, speaker, means, plat, variant="plain")
    ZtP = esn_features(Z, speaker, means, plat, variant="platform")
    scores, info = u8.fit_all(Z, y, plat, speaker, np.where(lab_mask)[0], enroll,
                              folds, Zt, ZtP, n)
    assert set(scores) == set(u8.METHODS)
    for name, sc in scores.items():
        assert sc.shape == (n,) and np.all(np.isfinite(sc)), name
    assert info["oc_defined"] is True
    # m=0: ESN must be B1 (byte-identical design matrix, same C)
    zero = np.zeros(n, bool)
    means0 = speaker_means(Z, speaker, zero, plat, m=0)
    Zt0 = esn_features(Z, speaker, means0, plat, variant="plain")
    assert np.array_equal(Zt0, Z)
    s0, _ = u8.fit_all(Z, y, plat, speaker, np.where(lab_mask)[0], zero, folds,
                       Zt0, Zt0, n)
    assert np.array_equal(s0["ESN"], s0["B1"])
    assert np.array_equal(s0["ESN-P"], s0["B1"])
    assert set(s0) == {"B1", "B1+enroll", "ESN", "ESN-P"}      # OC needs enrolment


def test_run_one_writes_draws_for_every_mask_and_m():
    import tempfile

    Z, y, speaker, plat, cell = _region()
    with tempfile.TemporaryDirectory() as td:
        u8._G.update({"Z": Z, "y": y, "plat": plat, "cell": cell, "speaker": speaker,
                      "draw_root": Path(td), "backbone": "toy", "spec": "S3"})
        d, m, rows = u8.run_one((0, 5))
        assert not any("error" in r for r in rows)
        assert {r["mask"] for r in rows} == {"seen", "unseen", "strict_draw"}
        assert "all" in {r["population"] for r in rows}
        assert {r["population"] for r in rows} <= {"all", "enrolled_only"}
        assert (Path(td) / "m5" / "strict_draw" / "draw0000.npz").exists()
        assert all(r["region"] == "dev" for r in rows)
    u8._G.clear()


def test_intervals_table_is_parallel_safe_and_reports_the_old_rule(tmp_path):
    rng = np.random.default_rng(3)
    speech = np.repeat(np.arange(20), 6)
    y = np.tile([0, 1, 1], 40)
    for d in range(8):
        base = y * 1.0 + rng.normal(0, 1, size=len(y))
        scores = {"B1": base, "ESN": base + 1.2 * y, "ESN-P": base + 1.0 * y,
                  "B1+enroll": base - 0.1 * y, "OC": base * 0.3, "OC+B1": base}
        save_draw_scores(tmp_path / "m5" / "strict_draw", d, y, scores, speech)
    iv = u8.intervals_table(tmp_path, ms=[5], masks=["strict_draw"], jobs=2)
    assert set(iv["method_b"]) == set(u8.CONTROLS)
    assert (iv["method_a"] == "ESN").any()
    row = iv[(iv["method_a"] == "ESN") & (iv["method_b"] == "B1")].iloc[0]
    assert row["n_draws"] == 8
    assert row["delta"] < 0.0                       # ESN is the better (lower) method
    assert row["old_hi"] >= row["old_lo"]
    assert row["b_hi"] >= row["b_lo"]


def _gate_frames(esn=10.0, b1=13.0, oc=12.0, enroll=13.5, delta=-3.0, b_hi=-1.0,
                 b_contains_zero=False):
    df = pd.DataFrame({
        "m": [5] * 4, "mask": ["strict_draw"] * 4, "population": ["all"] * 4,
        "method": ["ESN", "B1", "OC+B1", "B1+enroll"],
        "pooled_eer": [esn, b1, oc, enroll],
    })
    iv = pd.DataFrame([{
        "m": 5, "mask": "strict_draw", "method_a": "ESN", "method_b": "OC+B1",
        "delta": delta, "old_lo": -6.0, "old_hi": 0.5, "b_lo": -5.0, "b_hi": b_hi,
        "b_contains_zero": b_contains_zero,
    }])
    return df, iv


def test_g17_passes_only_with_both_the_2_point_effect_and_a_clear_interval():
    df, iv = _gate_frames()
    gate = u8.gate_g17(df, iv)
    assert gate["best_control"] == "OC+B1"          # the strongest control
    assert gate["delta_points"] == -2.0
    assert gate["effect_ge_2"] is True
    assert gate["b_excludes_zero"] is True
    assert gate["verdict"] == "ESN_PASS"
    assert json.dumps(gate)                          # serialisable


def test_g17_fails_on_a_small_effect_or_an_interval_that_touches_zero():
    df, iv = _gate_frames(delta=-1.0, b_hi=0.4)
    iv.loc[0, "b_contains_zero"] = True
    gate = u8.gate_g17(df, iv)
    assert gate["verdict"] == "ESN_FAIL"
    df2, iv2 = _gate_frames()
    iv2.loc[0, "b_contains_zero"] = True
    assert u8.gate_g17(df2, iv2)["verdict"] == "ESN_FAIL"


def test_g17_reports_when_esn_was_not_run():
    df = pd.DataFrame({"m": [5], "mask": ["strict_draw"], "population": ["all"],
                       "method": ["B1"], "pooled_eer": [13.0]})
    assert u8.gate_g17(df, pd.DataFrame())[ "verdict"] == "ESN_NOT_RUN"

"""Pins for the U4 driver: the two allowed masks, the selector loop, the marginal
table and the G12-G15 verdict logic."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import u4_malp as u4  # noqa: E402
from src.baselines import knn_graph  # noqa: E402


def _toy(n_speakers=24, per_speaker=20, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    speaker, y, cell, plat = [], [], [], []
    for s in range(n_speakers):
        label = s % 2
        for j in range(per_speaker):
            speaker.append(f"s{s:02d}")
            y.append(label)
            plat.append(f"p{s % 4}")
            cell.append(f"p{s % 4}|{'en' if s % 2 else 'zh'}")
    speaker = np.asarray(speaker)
    y = np.asarray(y)
    cell = np.asarray(cell)
    plat = np.asarray(plat)
    Z = rng.normal(size=(len(y), dim)).astype(np.float32)
    Z[:, 0] += 2.0 * y
    Z += np.repeat(np.arange(n_speakers), per_speaker)[:, None] * 0.15
    return Z, y, cell, plat, speaker


def test_clips_mask_holds_out_clips_not_speakers():
    Z, y, cell, plat, speaker = _toy()
    pool, ev = u4.pool_and_eval(0, "clips", Z, y, cell, speaker)
    assert set(pool.tolist()) & set(ev.tolist()) == set()
    assert len(pool) + len(ev) == len(y)
    assert set(speaker[pool].tolist()) == set(speaker[ev].tolist())   # every speaker both sides
    assert abs(len(ev) / len(y) - 0.25) < 0.02


def test_strict_draw_mask_splits_speakers():
    Z, y, cell, plat, speaker = _toy()
    pool, ev = u4.pool_and_eval(0, "strict_draw", Z, y, cell, speaker)
    assert set(speaker[pool].tolist()) & set(speaker[ev].tolist()) == set()
    assert len(pool) + len(ev) == len(y)
    assert abs(len(set(speaker[pool])) - len(set(speaker[ev]))) <= 1


def test_run_selector_returns_k_distinct_pool_clips():
    Z, y, cell, plat, speaker = _toy()
    graph = knn_graph(Z, k=15)
    pool, _ev = u4.pool_and_eval(0, "clips", Z, y, cell, speaker)
    cfg = (10, 0.9, 1.0, 0.5)
    for selector in ("U", "E", "K", "TypiClust", "DWU", "SC-DWU", "CC-DWU", "full"):
        picked = u4.run_selector(selector, Z, y, cell, speaker, pool, 60, graph, cfg,
                                 "clips", seed=0)
        assert picked.size <= 60, selector
        assert len(set(picked.tolist())) == picked.size, selector
        assert set(picked.tolist()) <= set(pool.tolist()), selector
    typi = u4.run_selector("TypiClust", Z, y, cell, speaker, pool, 60, graph, cfg,
                           "clips", seed=0)
    assert typi.size == 60


def test_speaker_coverage_variant_touches_more_speakers_than_cell_coverage():
    Z, y, cell, plat, speaker = _toy()
    graph = knn_graph(Z, k=15)
    pool, _ev = u4.pool_and_eval(0, "clips", Z, y, cell, speaker)
    cfg = (10, 0.9, 1.0, 0.5)
    sc = u4.run_selector("SC-DWU", Z, y, cell, speaker, pool, 60, graph, cfg, "clips", 0)
    cc = u4.run_selector("CC-DWU", Z, y, cell, speaker, pool, 60, graph, cfg, "clips", 0)
    assert len(set(speaker[sc].tolist())) >= len(set(speaker[cc].tolist()))


def test_learners_return_one_finite_score_per_clip():
    Z, y, cell, plat, speaker = _toy()
    graph = knn_graph(Z, k=15)
    labelled = np.arange(0, len(y), 7)
    cfg = (10, 0.9, 1.0, 0.5)
    for learner in u4.LEARNERS:
        sc = u4.learn(learner, Z, y, graph, labelled, cfg)
        assert sc.shape == (len(y),) and np.all(np.isfinite(sc)), learner
        assert np.mean(sc[y == 1]) > np.mean(sc[y == 0]), learner


def _df(rows):
    return pd.DataFrame(rows)


def test_marginals_compare_each_factor_against_its_reference():
    rows = []
    for mask in ("clips", "strict_draw"):
        for k in (100, 290):
            for sel, base in (("U", 15.0), ("full", 12.0), ("E", 14.0)):
                for learner, off in (("LR", 0.0), ("LP", -1.0), ("LR+LP", -2.0)):
                    rows.append({"mask": mask, "k": k, "selector": sel,
                                 "learner": learner, "pooled_eer": base + off,
                                 "draw": 0, "method": f"{sel}|{learner}"})
    mg = u4.marginals(_df(rows))
    got = mg[(mg["factor"] == "selector") & (mg["level"] == "full")
             & (mg["learner"] == "LR") & (mg["k"] == 100)]
    assert float(got.iloc[0]["effect"]) == -3.0
    got = mg[(mg["factor"] == "learner") & (mg["level"] == "LR+LP")
             & (mg["selector"] == "U")]
    assert float(got.iloc[0]["effect"]) == -2.0


def _gate_frame(malp=11.0, uni=15.0, other=11.4, sc=10.0, cc=12.0, k100=11.0,
                k290=10.8):
    rows = []
    for mask in ("clips", "strict_draw"):
        for k in (100, 290):
            for sel in u4.SELECTORS:
                for learner in u4.LEARNERS:
                    v = 20.0
                    if sel == "full" and learner == "LR+LP":
                        v = malp if k == 100 else k100
                    elif sel == "U" and learner == "LR+LP":
                        v = uni if k == 100 else k290
                    elif sel == "E" and learner == "LR+LP":
                        v = other
                    elif sel == "SC-DWU" and learner == "LR+LP":
                        v = sc
                    elif sel == "CC-DWU" and learner == "LR+LP":
                        v = cc
                    rows.append({"mask": mask, "k": k, "selector": sel,
                                 "learner": learner, "pooled_eer": v, "draw": 0,
                                 "method": f"{sel}|{learner}"})
    iv = pd.DataFrame([
        {"mask": "strict_draw", "k": 100, "method_a": "full|LR+LP",
         "method_b": "U|LR+LP", "delta": -4.0, "b_lo": -5.0, "b_hi": -3.0,
         "b_contains_zero": False},
        {"mask": "strict_draw", "k": 100, "method_a": "full|LR+LP",
         "method_b": "E|LR+LP", "delta": -0.4, "b_lo": -0.6, "b_hi": -0.2,
         "b_contains_zero": False},
        {"mask": "strict_draw", "k": 100, "method_a": "SC-DWU|LR+LP",
         "method_b": "CC-DWU|LR+LP", "delta": -2.0, "b_lo": -3.0, "b_hi": -1.0,
         "b_contains_zero": False},
    ])
    return _df(rows), iv


def test_gates_report_every_verdict_and_stay_serialisable():
    df, iv = _gate_frame()
    gate = u4.gates(df, iv)
    assert gate["G12"]["status"] == "MALP_INFO"        # informational after 追加 02
    assert gate["G13"]["status"] in ("SEL_FAIL", "SEL_ONLY_VS_UNIFORM", "SEL_PASS")
    assert gate["G14"]["status"] in ("SPEAKER_COVERAGE_PASS", "SPEAKER_COVERAGE_FAIL")
    assert gate["G15"]["status"] in ("LABEL_EFFICIENCY_PASS", "LABEL_EFFICIENCY_FAIL")
    assert gate["G16"]["status"] == "SOTA_F2_DEV_NO"
    assert json.dumps(gate, default=str)


def test_a_selector_that_only_beats_uniform_is_labelled_as_such():
    df, iv = _gate_frame()
    iv.loc[iv["method_b"] == "E|LR+LP", "delta"] = -0.1     # cannot separate from E
    gate = u4.gates(df, iv)
    assert gate["G13"]["status"] == "SEL_ONLY_VS_UNIFORM"


def test_g14_fails_when_speaker_coverage_is_not_one_point_better():
    df, iv = _gate_frame()
    iv.loc[(iv["method_a"] == "SC-DWU|LR+LP"), "delta"] = -0.2
    gate = u4.gates(df, iv)
    assert gate["G14"]["status"] == "SPEAKER_COVERAGE_FAIL"

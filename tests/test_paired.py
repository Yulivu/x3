"""Pins for the V-1 estimator in ``src/paired.py``.

The whole point of V-1 is that the *old* rule (2.5/97.5 quantiles of the per-draw
differences) measures single-label-draw fluctuation, so the two intervals must be
structurally different: (a) collapses when the draws repeat, (b) does not.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics3 import eer  # noqa: E402
from src.paired import (  # noqa: E402
    load_draw_scores,
    pooled_eer,
    save_draw_scores,
    two_intervals,
)


def _draw(y, speaker, sa, sb):
    return {
        "y": np.asarray(y),
        "speaker": np.asarray(speaker),
        "sa": np.asarray(sa, float),
        "sb": np.asarray(sb, float),
    }


def _fixed_eval_set(n_per_speaker=60, n_speakers=24, seed=7):
    """One evaluation set with per-speaker difficulty offsets."""
    rng = np.random.default_rng(seed)
    y, speaker, base = [], [], []
    for s in range(n_speakers):
        off = rng.normal(0.0, 2.0)          # speaker-specific difficulty
        n = n_per_speaker
        yy = np.tile([0, 1], n // 2)
        y.append(yy)
        speaker.append(np.full(n, s))
        base.append(off + 1.5 * yy + rng.normal(0.0, 1.0, n))
    return (np.concatenate(y), np.concatenate(speaker), np.concatenate(base))


def test_pooled_eer_is_the_plain_eer():
    y, speaker, base = _fixed_eval_set()
    assert pooled_eer(y, base, speaker) == pytest.approx(eer(y, base))


def test_empty_input_has_no_intervals():
    out = two_intervals([])
    assert out["n_draws"] == 0


def test_repeated_draw_collapses_the_sampling_interval_but_not_the_speaker_one():
    # ``base - y`` shrinks the class separation, which is the only kind of change
    # EER can see (a constant shift of both classes leaves EER untouched).
    y, speaker, base = _fixed_eval_set()
    one = _draw(y, speaker, base, base - y)
    out = two_intervals([one] * 40, B=200, B_s=200)
    # (a) every draw is the same number -> bootstrapping the mean changes nothing
    assert out["a_lo"] == pytest.approx(out["a_hi"])
    assert out["a_lo"] == pytest.approx(out["mean"])
    assert out["mean"] < 0.0                       # "b" is the better method
    # (b) speakers differ in difficulty -> resampling them still moves the mean
    assert out["b_hi"] - out["b_lo"] > 0.0


def test_a_interval_matches_the_normal_approximation_over_draws():
    y, speaker, base = _fixed_eval_set()
    rng = np.random.default_rng(11)
    draws = []
    for _ in range(100):
        shift = 0.9 + rng.normal(0.0, 0.3)          # L varies -> the effect varies
        draws.append(_draw(y, speaker, base, base - shift * y))
    out = two_intervals(draws, B=4000, B_s=50)
    deltas = np.array([eer(d["y"], d["sa"]) - eer(d["y"], d["sb"]) for d in draws])
    se = deltas.std(ddof=1) / np.sqrt(len(deltas))
    assert out["a_lo"] == pytest.approx(deltas.mean() - 1.96 * se, abs=0.25)
    assert out["a_hi"] == pytest.approx(deltas.mean() + 1.96 * se, abs=0.25)
    assert out["mean"] == pytest.approx(deltas.mean())


def test_speaker_interval_is_wider_than_the_sampling_interval_when_speakers_differ():
    """V-1's core motivation: a small, real effect lives inside (b) or outside it."""
    y, speaker, base = _fixed_eval_set(n_speakers=40)
    rng = np.random.default_rng(3)
    # half of the speakers carry the whole effect, the other half none
    eff = np.where(np.arange(40) % 2 == 0, 1.6, 0.0)
    per_speaker = eff[speaker] * y
    draws = []
    for _ in range(60):
        draws.append(_draw(y, speaker, base, base - per_speaker
                           + rng.normal(0.0, 0.05, size=y.size)))
    out = two_intervals(draws, B=1000, B_s=400)
    assert (out["b_hi"] - out["b_lo"]) > (out["a_hi"] - out["a_lo"])
    assert out["b_hi"] - out["b_lo"] > 3.0


def test_identical_scores_contain_zero_and_do_not_exclude_it():
    y, speaker, base = _fixed_eval_set()
    out = two_intervals([_draw(y, speaker, base, base)] * 20, B=100, B_s=100)
    assert out["mean"] == pytest.approx(0.0)
    assert out["b_contains_zero"] is True
    assert out["b_excludes_zero"] is False
    assert (out["a_lo"], out["a_hi"]) == pytest.approx((0.0, 0.0))


def test_consistent_effect_excludes_zero_on_both_intervals():
    y, speaker, base = _fixed_eval_set(n_speakers=30)
    draws = [_draw(y, speaker, base, base - 2.0 * y)] * 25
    out = two_intervals(draws, B=500, B_s=300)
    # EER is an error rate: "b" removes the separation, so delta = EER_a - EER_b < 0
    assert out["mean"] < 0.0
    assert out["b_hi"] < 0.0
    assert out["b_excludes_zero"] is True


def test_seed_makes_the_intervals_deterministic():
    y, speaker, base = _fixed_eval_set()
    draws = [_draw(y, speaker, base, base + 0.7 + 0.01 * i) for i in range(30)]
    a = two_intervals(draws, seed=99)
    b = two_intervals(draws, seed=99)
    assert a == b


def test_save_and_load_draw_scores_roundtrip(tmp_path):
    y, speaker, base = _fixed_eval_set()
    scores = {"B1": base, "B9": base - 0.4}
    for d in range(3):
        save_draw_scores(tmp_path, d, y, scores, speaker, extra={"k": 290})
    back = load_draw_scores(tmp_path, "B9", "B1")
    assert len(back) == 3
    assert np.allclose(back[0]["y"], y)
    assert np.allclose(back[0]["speaker"], speaker)
    assert np.allclose(back[0]["sa"], scores["B9"])
    assert np.allclose(back[0]["sb"], scores["B1"])
    out = two_intervals(back, B=200, B_s=50)
    assert out["n_draws"] == 3
    # a method that was never persisted is skipped rather than crashing
    assert load_draw_scores(tmp_path, "C1", "B1") == []


def test_save_and_load_draw_scores_with_a_pandas_object_speaker_column(tmp_path):
    """Speaker ids come out of a manifest as ``object``; they must stay loadable."""
    import pandas as pd

    y = np.tile([0, 1], 4)
    speaker = pd.Series(["n000001", "n000001", "n000001", "n000001",
                         "n000002", "n000002", "n000002", "n000002"]).to_numpy()
    assert speaker.dtype == object
    save_draw_scores(tmp_path, 0, y, {"A": y * 1.0, "B": y * 0.5}, speaker)
    back = load_draw_scores(tmp_path, "A", "B")
    assert len(back) == 1
    assert back[0]["speaker"].tolist() == speaker.tolist()


def test_cluster_bootstrap_resamples_whole_speakers_not_clips():
    """Every replicate must hand the metric complete speaker blocks."""
    y = np.tile([0, 1], 4)
    speaker = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    sa = y.astype(float)
    sb = 1.0 - y
    draw = _draw(y, speaker, sa, sb)
    # eer(y, y) == 0 and eer(y, 1 - y) == 100 for any union of complete blocks
    out = two_intervals([draw] * 5, B=50, B_s=50)
    assert out["mean"] == pytest.approx(-100.0)
    assert (out["b_lo"], out["b_hi"]) == pytest.approx((-100.0, -100.0))
    # a metric that needs *both* classes would return nan on a clip-wise resample
    assert np.isfinite(out["b_lo"])

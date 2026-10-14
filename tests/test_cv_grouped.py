"""Pins for V-3: speaker-grouped folds and the shared CV selector."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cv import select_config, speaker_grouped_folds, stratified_folds  # noqa: E402


def _toy(n_speakers=20, per_speaker=6, seed=0):
    rng = np.random.default_rng(seed)
    speaker = np.repeat(np.arange(n_speakers), per_speaker)
    y = np.tile([0, 1, 0, 1, 1, 0], n_speakers)
    # a speaker-level offset: the point of grouping is that it must not leak
    return speaker, y, rng


def test_grouped_folds_never_split_a_speaker():
    speaker, y, _ = _toy()
    folds = speaker_grouped_folds(speaker, y, n_splits=5)
    for s in np.unique(speaker):
        assert len(set(folds[speaker == s])) == 1
    assert set(folds) == {0, 1, 2, 3, 4}


def test_grouped_folds_keep_both_classes_in_every_fold():
    speaker, y, _ = _toy()
    folds = speaker_grouped_folds(speaker, y, n_splits=5)
    for f in np.unique(folds):
        assert set(np.unique(y[folds == f])) == {0, 1}


def test_grouped_folds_differ_from_stratified_window_folds():
    speaker, y, _ = _toy()
    grouped = speaker_grouped_folds(speaker, y, n_splits=5)
    window = stratified_folds(y, n_splits=5)
    # the window version scatters clips of one speaker across folds
    leaked = 0
    for s in np.unique(speaker):
        if len(set(window[speaker == s])) > 1:
            leaked += 1
    assert leaked > 0
    assert len(set(grouped)) == len(set(window)) == 5


def test_grouped_folds_are_deterministic_and_handle_small_inputs():
    speaker, y, _ = _toy()
    assert np.array_equal(speaker_grouped_folds(speaker, y),
                          speaker_grouped_folds(speaker, y))
    tiny = speaker_grouped_folds([0, 0, 1], [0, 1, 1], n_splits=5)
    assert set(tiny) <= {0, 1}


def test_select_config_picks_the_better_grid_point():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(200, 4))
    y = (Z[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(int)
    folds = np.arange(200) % 5

    def evaluate(cfg, tr, te):
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(C=cfg, max_iter=2000).fit(Z[tr], y[tr])
        return m.decision_function(Z[te])

    best, info = select_config([1e-3, 1.0, 100.0], evaluate, y, folds)
    assert best in {1e-3, 1.0, 100.0}
    assert len(info["grid"]) == 3
    assert info["cv_loglik"] == max(g["cv_loglik"] for g in info["grid"])
    # the ranking is stable, not a random pick
    again, _ = select_config([1e-3, 1.0, 100.0], evaluate, y, folds)
    assert again == best


def test_select_config_ignores_folds_with_a_single_class():
    # fold 0 holds every bona fide clip, so training without it has one class only
    y = np.array([0, 0, 1, 1, 1, 1])
    folds = np.array([0, 0, 1, 1, 2, 2])

    def evaluate(cfg, tr, te):
        return np.zeros(int(te.sum()))

    cfg, info = select_config(["only"], evaluate, y, folds)
    assert cfg == "only"
    assert np.isfinite(info["cv_loglik"])

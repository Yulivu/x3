"""The U8 synthetic check demanded by NEW_TASKS 02: on unseen speakers, with an
independent per-speaker mean offset plus a class effect, ESN must beat B1 by at
least 2 pooled EER points.

Why the offset needs a component along the class direction
---------------------------------------------------------
A speaker offset that is *orthogonal* to the class direction is a nuisance the
probe can already ignore, so centring it buys almost nothing (measured: <2 points
even at a 6-sigma offset).  The expensive case - and the one X3 actually sees - is
when the offset partly *predicts the label* inside ``L``: the probe then scores a
new speaker by that speaker's own offset, and every one of its clips shifts up or
down.  ``theta`` is that component, and the second test pins the no-offset control
so the gain cannot be manufactured by the extra feature block alone.

Hyper-parameters are chosen with speaker-grouped folds (V-3), i.e. the CV itself
also simulates an unseen speaker.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import fit_logreg_cv  # noqa: E402
from src.cv import speaker_grouped_folds  # noqa: E402
from src.esn import esn_features, speaker_means  # noqa: E402
from src.metrics3 import eer  # noqa: E402


def synth(n_speakers=30, per_class=20, dim=16, offset_sd=4.0, class_gap=2.0,
          noise=1.0, theta=0.5, seed=0):
    """Per-speaker mean offset (``theta`` of it along the class direction) + class effect."""
    rng = np.random.default_rng(seed)
    class_dir = rng.normal(size=dim)
    class_dir /= np.linalg.norm(class_dir)
    Z, y, speaker = [], [], []
    for s in range(n_speakers):
        d = rng.normal(size=dim)
        d /= np.linalg.norm(d)
        d = theta * class_dir + (1.0 - theta) * d
        d /= np.linalg.norm(d)
        offset = rng.normal() * offset_sd * d
        for _ in range(per_class):
            for label in (0, 1):
                Z.append(offset + label * class_gap * class_dir
                         + rng.normal(size=dim) * noise)
                y.append(label)
                speaker.append(f"spk{s:03d}")
    return (np.asarray(Z, np.float32), np.asarray(y), np.asarray(speaker))


def split_unseen(speaker, y, n_label_speakers=15, n_label=8, m_enroll=5, seed=1):
    """The first half of the speakers supplies ``L``; the rest is the unseen set."""
    rng = np.random.default_rng(seed)
    all_spk = np.unique(speaker)
    s_l = set(all_spk[:n_label_speakers].tolist())
    enroll = np.zeros(len(y), bool)
    for s in all_spk:                           # every speaker enrols m genuine clips
        pos = np.where((speaker == s) & (y == 0))[0]
        enroll[rng.choice(pos, size=max(1, min(m_enroll, pos.size // 2)),
                          replace=False)] = True
    lab = np.zeros(len(y), bool)
    for s in s_l:
        pos = np.where((speaker == s) & ~enroll)[0]
        lab[rng.choice(pos, size=min(n_label, pos.size), replace=False)] = True
    ev = (~np.isin(speaker, list(s_l))) & ~enroll
    return lab, enroll, ev


def _eers(Z, y, speaker, lab, enroll, ev, m=5):
    folds = speaker_grouped_folds(speaker[lab], y[lab])
    b1, _ = fit_logreg_cv(Z[lab], y[lab], folds=folds)
    means = speaker_means(Z, speaker, enroll, None, m=m)
    Zt = esn_features(Z, speaker, means, None, variant="plain")
    esn, _ = fit_logreg_cv(Zt[lab], y[lab], folds=folds)
    return (eer(y[ev], b1.decision_function(Z)[ev]),
            eer(y[ev], esn.decision_function(Zt)[ev]))


def test_esn_beats_b1_on_unseen_speakers_by_at_least_two_points():
    Z, y, speaker = synth()
    lab, enroll, ev = split_unseen(speaker, y)
    eer_b1, eer_esn = _eers(Z, y, speaker, lab, enroll, ev)
    assert eer_esn <= eer_b1 - 2.0, (eer_b1, eer_esn)


def test_esn_beats_b1_for_several_seeds():
    """Not a lucky split: the mechanism has to hold across draws."""
    gains = []
    for seed in range(3):
        Z, y, speaker = synth(seed=seed)
        lab, enroll, ev = split_unseen(speaker, y, seed=seed + 10)
        eer_b1, eer_esn = _eers(Z, y, speaker, lab, enroll, ev)
        gains.append(eer_b1 - eer_esn)
    assert min(gains) >= 2.0, gains
    assert np.mean(gains) >= 5.0, gains


def test_the_gap_is_small_when_the_offset_is_orthogonal_to_the_class_direction():
    """Control: with no label-confounding offset there is nothing to remove.

    Averaged over three seeds on purpose - a single seed can land within ~2.5 points
    of the null by chance (the extra feature block costs a little), and the claim is
    that the gain is *not* manufactured on average.
    """
    gaps = []
    for seed in range(3):
        Z, y, speaker = synth(offset_sd=0.0, seed=seed)
        lab, enroll, ev = split_unseen(speaker, y, seed=seed + 10)
        eer_b1, eer_esn = _eers(Z, y, speaker, lab, enroll, ev)
        gaps.append(eer_b1 - eer_esn)
    assert abs(float(np.mean(gaps))) < 2.0, gaps
    assert max(abs(g) for g in gaps) < 4.0, gaps


def test_the_gain_grows_with_the_confounding_and_shrinks_without_an_offset():
    rows = {}
    for theta in (0.0, 0.5):
        Z, y, speaker = synth(theta=theta)
        lab, enroll, ev = split_unseen(speaker, y)
        eer_b1, eer_esn = _eers(Z, y, speaker, lab, enroll, ev)
        rows[theta] = eer_b1 - eer_esn
    assert rows[0.5] > rows[0.0]
    assert rows[0.5] >= 2.0


def test_enrollment_helps_monotonically_in_the_synthetic_setting():
    Z, y, speaker = synth()
    errs = {}
    for m in (0, 3, 5, 10):
        lab, enroll, ev = split_unseen(speaker, y, m_enroll=max(m, 1))
        if m == 0:
            enroll = np.zeros(len(y), bool)
        folds = speaker_grouped_folds(speaker[lab], y[lab])
        means = speaker_means(Z, speaker, enroll, None, m=m)
        Zt = esn_features(Z, speaker, means, None, variant="plain")
        model, _ = fit_logreg_cv(Zt[lab], y[lab], folds=folds)
        errs[m] = eer(y[ev], model.decision_function(Zt)[ev])
    assert errs[0] == max(errs.values())              # m=0 is exactly B1
    assert errs[5] <= errs[0] - 2.0
    assert errs[10] <= errs[0] - 2.0

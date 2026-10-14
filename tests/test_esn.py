"""Pins for the ESN building blocks (U8 / N5)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import fit_logreg_cv  # noqa: E402
from src.esn import (  # noqa: E402
    choose_enrollment,
    esn_features,
    fuse,
    one_class_scores,
    speaker_means,
    standardise,
)


def _toy(n_speakers=6, n_real=6, n_fake=6, dim=5, seed=0):
    rng = np.random.default_rng(seed)
    speaker, y, plat = [], [], []
    for s in range(n_speakers):
        for k in range(n_real):
            speaker.append(f"s{s:02d}")
            y.append(0)
            plat.append("p0" if k < 4 else "p1")
        for k in range(n_fake):
            speaker.append(f"s{s:02d}")
            y.append(1)
            plat.append("p0")
    speaker = np.array(speaker)
    y = np.array(y)
    plat = np.array(plat)
    Z = rng.normal(size=(len(y), dim))
    Z[y == 1, 0] += 2.0
    return Z, y, speaker, plat


def test_choose_enrollment_honours_the_m_and_half_budget_rules():
    Z, y, speaker, plat = _toy(n_real=6)
    assert not choose_enrollment(speaker, y, 0).any()
    e3 = choose_enrollment(speaker, y, 3)
    assert e3.sum() == 6 * 3
    for s in np.unique(speaker):
        assert int(e3[(speaker == s)].sum()) == 3
    # never more than half of a speaker's genuine supply
    e10 = choose_enrollment(speaker, y, 10)
    assert e10.sum() == 6 * 3
    # spoof clips are never enrolled
    assert not (y[e10] == 1).any()
    # a speaker with two genuine clips can only give one
    speaker2 = np.array(["a"] * 4)
    y2 = np.array([0, 0, 1, 1])
    assert choose_enrollment(speaker2, y2, 3).sum() == 1


def test_choose_enrollment_is_seeded_and_spreads_across_the_available_clips():
    Z, y, speaker, plat = _toy(n_real=8)
    a = choose_enrollment(speaker, y, 3, seed=1)
    b = choose_enrollment(speaker, y, 3, seed=1)
    assert np.array_equal(a, b)
    seen = set()
    for seed in range(8):
        seen |= set(np.where(choose_enrollment(speaker, y, 3, seed=seed))[0].tolist())
    assert len(seen) > 6 * 3                    # not always the same first three


def test_speaker_means_are_the_enrollment_means_including_the_platform_split():
    Z, y, speaker, plat = _toy()
    enroll = choose_enrollment(speaker, y, 3)
    means = speaker_means(Z, speaker, enroll, plat, m=3)
    s = "s00"
    idx = np.where(enroll & (speaker == s))[0]
    assert np.allclose(means["speaker"][s], Z[idx].mean(0))
    key = (s, "p0")
    j = idx[plat[idx] == "p0"]
    assert np.allclose(means["platform"][key], Z[j].mean(0))
    assert means["platform_counts"][key] == len(j)
    assert means["m"] == 3
    assert np.allclose(means["global"], Z[enroll].mean(0))


def test_esn_features_centre_each_speaker_and_keep_both_blocks():
    Z, y, speaker, plat = _toy()
    enroll = choose_enrollment(speaker, y, 3)
    means = speaker_means(Z, speaker, enroll, plat, m=3)
    got = esn_features(Z, speaker, means, plat, variant="plain")
    assert got.shape == (len(Z), 2 * Z.shape[1])
    assert np.allclose(got[:, :Z.shape[1]], Z)
    for s in np.unique(speaker):
        rows = np.where(speaker == s)[0]
        assert np.allclose(got[rows, Z.shape[1]:],
                           Z[rows] - means["speaker"][s], atol=1e-5)
    # the enrolment rows of one speaker average to ~0 in the centred block
    for s in np.unique(speaker):
        idx = np.where(enroll & (speaker == s))[0]
        assert np.allclose(got[idx, Z.shape[1]:].mean(0), 0.0, atol=1e-5)
    # a speaker with a single enrolment clip centres that clip to exactly ~0
    once = choose_enrollment(speaker, np.where(y == 0, 0, 1), 1)
    got1 = esn_features(Z, speaker, speaker_means(Z, speaker, once, plat, m=1),
                        plat, variant="plain")
    assert np.allclose(got1[once, Z.shape[1]:], 0.0, atol=1e-5)


def test_esn_platform_variant_uses_the_platform_mean_only_when_it_can():
    Z, y, speaker, plat = _toy()
    enroll = choose_enrollment(speaker, y, 5)          # 3 clips, all from p0
    means = speaker_means(Z, speaker, enroll, plat, m=5)
    P = esn_features(Z, speaker, means, plat, variant="platform")
    rows_p1 = np.where(plat == "p1")[0]
    # p1 clips have no p1 enrolment -> they must fall back to the speaker mean
    s = speaker[rows_p1[0]]
    assert np.allclose(P[rows_p1[0], Z.shape[1]:], Z[rows_p1[0]] - means["speaker"][s],
                       atol=1e-5)
    # a clip whose platform has >= 2 enrolment clips uses its platform mean
    rows_p0 = np.where(plat == "p0")[0]
    i = rows_p0[0]
    assert np.allclose(P[i, Z.shape[1]:],
                       Z[i] - means["platform"][(speaker[i], "p0")], atol=1e-5)


def test_esn_features_fall_back_to_the_population_mean_when_a_speaker_is_not_enrolled():
    Z, y, speaker, plat = _toy()
    enroll = choose_enrollment(speaker, y, 3)
    enroll[speaker == "s05"] = False                   # this speaker cannot enrol
    means = speaker_means(Z, speaker, enroll, plat, m=3)
    got = esn_features(Z, speaker, means, plat, variant="plain")
    rows = np.where(speaker == "s05")[0]
    assert np.allclose(got[rows, Z.shape[1]:], Z[rows] - means["global"], atol=1e-5)


def test_m_zero_returns_the_plain_design_matrix():
    Z, y, speaker, plat = _toy()
    enroll = choose_enrollment(speaker, y, 0)
    means = speaker_means(Z, speaker, enroll, plat, m=0)
    got = esn_features(Z, speaker, means, plat, variant="plain")
    assert got.shape == Z.shape
    assert np.array_equal(got, Z)


def test_esn_at_m_zero_is_bit_identical_to_b1():
    Z, y, speaker, plat = _toy(n_speakers=10, n_real=8, n_fake=8, seed=3)
    enroll = choose_enrollment(speaker, y, 0)
    means = speaker_means(Z, speaker, enroll, plat, m=0)
    Zt = esn_features(Z, speaker, means, plat, variant="plain")
    lab = np.zeros(len(y), bool)
    lab[::2] = True
    m_b1, i_b1 = fit_logreg_cv(Z[lab], y[lab])
    m_esn, i_esn = fit_logreg_cv(Zt[lab], y[lab])
    assert np.array_equal(m_b1.decision_function(Z), m_esn.decision_function(Zt))
    assert i_b1["C"] == i_esn["C"]


def test_one_class_scores_rank_enrolled_real_clips_lowest():
    Z, y, speaker, plat = _toy()
    enroll = choose_enrollment(speaker, y, 3)
    s, _ = one_class_scores(Z, speaker, enroll)
    real = (y == 0) & ~enroll
    fake = y == 1
    assert np.nanmean(s[fake]) > np.nanmean(s[real])
    # an enrolment clip is at distance ~0 from itself
    assert np.allclose(s[enroll], -1.0, atol=1e-6)
    assert np.all(np.isfinite(s))


def test_one_class_scores_report_clips_without_their_own_enrolment():
    Z, y, speaker, plat = _toy()
    enroll = choose_enrollment(speaker, y, 3)
    enroll[speaker == "s02"] = False
    s, info = one_class_scores(Z, speaker, enroll)
    assert info["n_clips_without_own_enrollment"] == int((speaker == "s02").sum())
    assert np.all(np.isfinite(s))


def test_standardise_and_fuse():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([4.0, 4.0, 6.0, 8.0])
    assert abs(float(np.mean(standardise(a)))) < 1e-9
    assert abs(float(np.std(standardise(a))) - 1.0) < 1e-6
    assert np.allclose(fuse(a, b, 0.0), standardise(a))
    assert np.allclose(fuse(a, b, 1.0), standardise(a) + standardise(b))

"""Pins for the W3 speaker split and the frozen-count check."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import w3_split  # noqa: E402
from src.data import EXPECTED_PLATFORM_COUNTS, add_cell, counts_match  # noqa: E402


def _skewed_manifest(seed=1):
    """Mimics ML-ITW: one platform mostly fake, one mostly real, one mixed."""
    rng = np.random.default_rng(seed)
    rows = []
    spec = {"youtube": (0.08, 320), "douyin": (0.86, 320), "tiktok": (0.45, 240)}
    for plat, (rate, n) in spec.items():
        per = n // 40
        for s in range(40):
            for _ in range(per):
                rows.append({
                    "speaker": f"{plat}_s{s:02d}",
                    "language": "english",
                    "platform": plat,
                    "label": "spoof" if rng.random() < rate else "bona-fide",
                })
    df = pd.DataFrame(rows)
    df["y"] = (df["label"] == "spoof").astype(int)
    df["clip_id"] = np.arange(len(df))
    return add_cell(df)


def test_speakers_never_cross_the_split():
    meta = _skewed_manifest()
    split = w3_split.make_split(meta, dev_min=10, test_min=30)
    dev, test = set(split["dev_speakers"]), set(split["test_speakers"])
    assert dev and test and dev.isdisjoint(test)
    assert len(dev) + len(test) == meta["speaker"].nunique()
    for names in (dev, test):
        assert set(meta.loc[meta["speaker"].isin(names), "speaker"]) == names


def test_floors_are_met_for_every_feasible_platform_class():
    meta = _skewed_manifest()
    split = w3_split.make_split(meta, dev_min=10, test_min=30)
    for pair, dev_n in split["dev_counts"].items():
        if pair in split["infeasible_pairs"]:
            continue
        assert dev_n >= 10, f"{pair}: only {dev_n} dev clips"
        assert split["test_counts"][pair] >= 30, f"{pair}: only {split['test_counts'][pair]} test clips"


def test_split_is_reproducible_from_its_recorded_seed():
    meta = _skewed_manifest()
    split = w3_split.make_split(meta, dev_min=10, test_min=30)
    again = w3_split.try_split(meta, split["seed"], dev_min=10, test_min=30)
    assert again is not None and again["dev_speakers"] == split["dev_speakers"]


def test_dev_fraction_stays_in_a_sane_band():
    meta = _skewed_manifest()
    split = w3_split.make_split(meta, dev_frac=0.30, dev_min=10, test_min=30)
    assert 0.10 <= split["dev_frac"] <= 0.50, split["dev_frac"]


def test_infeasible_pairs_are_named_not_silently_dropped():
    meta = _skewed_manifest()
    split = w3_split.make_split(meta, dev_min=200, test_min=200)
    assert split["infeasible_pairs"], "a pair that cannot satisfy both floors must be listed"


def test_video_id_rule_coverage():
    assert w3_split.video_id_rule([])["rule"] is None
    paths = [f"abc12345678_{i:04d}.wav" for i in range(100)]
    assert w3_split.video_id_rule(paths)["coverage"] == pytest.approx(1.0)
    assert w3_split.video_id_rule(["x.wav", "y.wav"])["coverage"] < 0.95


def test_counts_match_against_the_frozen_table():
    rows = []
    for plat, (real, fake) in EXPECTED_PLATFORM_COUNTS.items():
        rows += [{"platform": plat, "y": 0}] * real
        rows += [{"platform": plat, "y": 1}] * fake
    df = pd.DataFrame(rows)
    ok, table, total = counts_match(df)
    assert ok and total == (18168, 6361, 24529)
    assert len(table) == len(EXPECTED_PLATFORM_COUNTS)
    ok2, _, _ = counts_match(df.iloc[:-1])
    assert ok2 is False

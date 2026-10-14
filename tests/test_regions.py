"""Pins for the v3 region/spec loader (``src/regions.py``)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src import regions as R  # noqa: E402
import w5_backbones as w5  # noqa: E402


def _blocks(n=6, n_mean=49, n_std=7, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=(n, n_mean, dim)).astype(np.float32)
    std = rng.normal(size=(n, n_std, dim)).astype(np.float32)
    zero = rng.normal(size=n).astype(np.float32)
    return mean, std, zero


def test_spec_slices_reproduce_the_w5_spec_menu():
    mean, std, zero = _blocks()
    want = w5.specs_for(mean, std, zero, 48)
    assert set(want) == set(R.SPECS)
    for spec in R.SPECS:
        parts = []
        for src, idx in R.spec_slices(spec, 48, mean.shape[1], std.shape[1]):
            if src == "zero":
                parts.append(zero[:, None])
            elif src == "mean":
                parts.append(mean[:, idx, :])
            else:
                parts.append(std[:, idx, :])
        got = np.concatenate(parts, axis=1)
        assert np.allclose(got, want[spec]), spec


def test_spec_slices_rejects_an_unknown_spec():
    try:
        R.spec_slices("S9", 48, 49, 7)
    except KeyError as exc:
        assert "S9" in str(exc)
    else:                                            # pragma: no cover
        raise AssertionError("an unknown spec must raise")


def test_linea_fractions_are_shared_with_w5():
    # spec_slices must use W5's own fractions, not a private copy
    assert R.spec_slices("S1", 48, 49, 7) == [("mean", i) for i in
                                              [max(1, min(48, int(round(f * 48))))
                                               for f in w5.LINEA_FRACTIONS]]


def _write_feature_dir(tmp_path, mean, std, zero):
    fdir = tmp_path / "x3" / "feats" / "toy"
    fdir.mkdir(parents=True)
    np.save(fdir / "mean.npy", mean.astype(np.float16))
    np.save(fdir / "std.npy", std.astype(np.float16))
    pd.DataFrame({"logit_fake_minus_real": zero,
                  "clip_id": np.arange(len(zero))}).to_csv(fdir / "scores.csv", index=False)
    (fdir / "layers.json").write_text(json.dumps(
        {"backbone": "toy", "n_layers": 48, "n_hidden_states": mean.shape[1],
         "std_layers": list(range(std.shape[1]))}), encoding="utf-8")
    return fdir


def test_load_spec_block_matches_w5_for_any_row_subset(tmp_path):
    mean, std, zero = _blocks(n=9, seed=5)
    _write_feature_dir(tmp_path, mean, std, zero)
    rows = np.array([0, 3, 4, 8])
    want = w5.specs_for(mean, std, zero, 48)
    for spec in R.SPECS:
        got = R.load_spec_block(tmp_path, "toy", spec, rows)
        assert got.shape == (len(rows), want[spec].shape[1])
        assert np.allclose(got, want[spec][rows], atol=1e-2), spec   # fp16 storage


def test_load_spec_block_default_is_every_row(tmp_path):
    mean, std, zero = _blocks(n=5, seed=6)
    _write_feature_dir(tmp_path, mean, std, zero)
    got = R.load_spec_block(tmp_path, "toy", "S3")
    assert got.shape[0] == 5


def test_split_lookup_and_region_keep(tmp_path):
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / R.SPLIT_V3).write_text(json.dumps(
        {"dev_speakers": ["a", "b"], "test_speakers": ["c"]}), encoding="utf-8")
    split = R.load_split(tmp_path)
    assert R.speaker_sets(split) == ({"a", "b"}, {"c"})
    meta = pd.DataFrame({"speaker": ["a", "c", "b", "c"]})
    assert R.region_keep(meta, split, "dev").tolist() == [True, False, True, False]
    assert R.region_keep(meta, split, "test").tolist() == [False, True, False, True]
    assert R.region_rows(meta, split, "test").tolist() == [1, 3]


def test_missing_split_file_is_reported_by_name(tmp_path):
    try:
        R.load_split(tmp_path, "speaker_split_v3.json")
    except FileNotFoundError as exc:
        assert "speaker_split_v3.json" in str(exc)
    else:                                            # pragma: no cover
        raise AssertionError("a missing split must raise")

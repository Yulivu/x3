"""Pins for the W10 view parameters and the transcode memmap contract."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w10_views  # noqa: E402
from src.extract_layers import N_SAMPLES, SAMPLE_RATE  # noqa: E402
from src.views import FAM_A  # noqa: E402

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def test_fam_a1_and_fam_a2_pin_their_codec_branch():
    for cid in range(200):
        a1 = w10_views.params_for("famA1", cid)
        a2 = w10_views.params_for("famA2", cid)
        assert a1["codec"] == "libopus" and a1["view"] == "A1"
        assert a2["codec"] == "aac" and a2["view"] == "A2"
        assert a1["bitrate_kbps"] in FAM_A["A1"]["bitrates"]
        assert a2["bitrate_kbps"] in FAM_A["A2"]["bitrates"]
        # and the pair is reproducible
        assert a1 == w10_views.params_for("famA1", cid)
        assert a2 == w10_views.params_for("famA2", cid)


def test_fam_b1_is_the_held_out_chain_and_differs_from_every_training_view():
    chain = tuple(tuple(x) for x in w10_views.params_for("famB1", 0)["chain"])
    assert len(chain) == 2
    for cid in range(100):
        b = w10_views.params_for("famB1", cid)
        a1 = w10_views.params_for("famA1", cid)
        a2 = w10_views.params_for("famA2", cid)
        assert tuple(tuple(x) for x in b["chain"]) == chain
        assert (a1["codec"], a1["bitrate_kbps"]) != ("libmp3lame", 64)
        assert (a2["codec"], a2["bitrate_kbps"]) != ("libmp3lame", 64)
        # the Fam-B band limit is a state the training views never take
        assert b["band_limit_hz"] in (None, 8000)


def test_unknown_view_is_rejected():
    with pytest.raises(ValueError):
        w10_views.params_for("nope", 0)


def test_wav_roundtrip(tmp_path):
    x = np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32) * 0.5
    p = tmp_path / "t.wav"
    w10_views.write_wav(p, x)
    y = w10_views.read_wav(p)
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1e-3


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed locally")
def test_transcode_view_writes_a_rectangular_memmap(tmp_path):
    rng = np.random.default_rng(0)
    waves = [rng.normal(0, 0.1, SAMPLE_RATE).astype(np.float32) for _ in range(3)]
    ids = [0, 1, 2]
    info = w10_views.transcode_view("famA1", waves, ids, tmp_path / "famA1", jobs=1)
    assert info["n_errors"] == 0, info
    arr = np.memmap(tmp_path / "famA1" / "view_audio.int16", dtype="<i2", mode="r",
                    shape=(3, N_SAMPLES))
    assert arr.shape == (3, N_SAMPLES)
    assert np.abs(np.asarray(arr)).max() > 0
    params = pd.read_csv(tmp_path / "famA1" / "params.csv")
    assert len(params) == 3 and (params["view"] == "A1").all()
    md5 = pd.read_csv(tmp_path / "famA1" / "audio_md5.csv")
    assert md5["n_errors"].iloc[0] == 0 and len(md5["audio_md5"].iloc[0]) == 32

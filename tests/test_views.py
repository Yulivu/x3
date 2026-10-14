"""Pins for src/views.py (TASKBOOK W10)."""

import shutil
import struct
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from src.views import (
    FAM_A,
    FAM_A_EXTRA,
    FAM_B,
    MAX_DURATION_DRIFT_S,
    SAMPLE_RATE,
    codec_hops,
    duration_s,
    encode_cmd,
    fam_a_params,
    fam_b_params,
    ffmpeg_cmd,
    params_table,
    run_view,
    view_seed,
    waveform_ops,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def test_seed_rule_is_stable_for_ints_and_strings():
    assert view_seed(0) == 20261015
    assert view_seed(7) == 20261022
    assert view_seed("clip-42") == view_seed("clip-42")
    assert view_seed("clip-42") != view_seed("clip-43")


def test_fam_a_params_respect_the_declared_grid():
    for cid in range(300):
        p = fam_a_params(cid)
        assert p["codec"] in {cfg["codec"] for cfg in FAM_A.values()}
        assert p["bitrate_kbps"] in FAM_A[p["view"]]["bitrates"]
        assert -FAM_A_EXTRA["gain_db"] <= p["gain_db"] <= FAM_A_EXTRA["gain_db"]
        if p["pink_snr_db"] is not None:
            lo, hi = FAM_A_EXTRA["pink_snr_db"]
            assert lo <= p["pink_snr_db"] <= hi
        assert p["downsample_hz"] in (None, FAM_A_EXTRA["downsample_hz"])
        assert p["family"] == "fam_a"


def test_fam_a_and_fam_b_parameter_sets_are_disjoint():
    """Fam-B is a two-hop transcode; no Fam-A view (single hop) can reproduce it.

    The first Opus hop is shared with A1 on purpose (TASKBOOK 7/W10); what makes
    Fam-B held out is the second hop plus the band limit, not the codec identity.
    """
    a_pairs = {(cfg["codec"], int(b)) for cfg in FAM_A.values() for b in cfg["bitrates"]}
    chain = tuple((str(c), int(b)) for c, b in FAM_B["B1"]["chain"])
    assert len(chain) == 2, "Fam-B must stay a double transcode"
    assert chain not in {(p,) for p in a_pairs}, "a training view reproduces the evaluation chain"
    assert ("libmp3lame", 64) not in a_pairs
    for cid in range(200):
        a, b = fam_a_params(cid), fam_b_params(cid)
        assert a["family"] != b["family"]
        assert "chain" not in a and len(b["chain"]) == 2


def test_fam_b_is_deterministic_and_single_chain():
    seen = {fam_b_params(c)["band_limit_hz"] for c in range(200)}
    assert seen <= {None, 8000}
    assert len(seen) == 2, "both band-limited and clean Fam-B variants should occur"
    assert fam_b_params(3) == fam_b_params(3)


def test_ffmpeg_cmd_carries_codec_bitrate_and_band_limit():
    a = ffmpeg_cmd("in.wav", "out.wav", fam_a_params(1))
    assert "-c:a" in a and any(t.endswith("k") for t in a)
    # walk the whole chain, not just the first hop
    hops = codec_hops(fam_b_params(1))
    assert [h[0] for h in hops] == ["libopus", "libmp3lame"]
    first = encode_cmd("in.wav", "hop0.ogg", *hops[0])
    assert "libopus" in first and "-b:a" in first and first[-1].endswith(".ogg")
    second = encode_cmd("hop0.wav", "hop1.mp3", *hops[1])
    assert "libmp3lame" in second and second[-1].endswith(".mp3")
    banded = ffmpeg_cmd("in.wav", "out.wav", {"band_limit_hz": 8000})
    assert any("lowpass" in t for t in banded)


def test_waveform_ops_apply_gain_noise_and_downsample_deterministically():
    x = np.sin(2 * np.pi * 220 * np.arange(16000) / 16000).astype(np.float32) * 0.3
    p = {"gain_db": 6.0, "pink_snr_db": 20.0, "downsample_hz": 8000}
    a = waveform_ops(x, p, clip_id=7)
    b = waveform_ops(x, p, clip_id=7)
    assert a.shape == x.shape and np.allclose(a, b)
    quiet = waveform_ops(x, {"gain_db": 0.0}, clip_id=7)
    loud = waveform_ops(x, {"gain_db": 6.0}, clip_id=7)
    # 6 dB is a factor of 10**(6/20) = 1.99526, not 2.0
    assert np.isclose(np.sqrt(np.mean(loud ** 2)) / np.sqrt(np.mean(quiet ** 2)),
                      10 ** 0.3, rtol=1e-4)


def test_params_table_shape():
    rows = params_table([0, 1, 2], "fam_a")
    assert len(rows) == 3 and all(r["family"] == "fam_a" for r in rows)
    rows_b = params_table([0, 1, 2], "fam_b")
    assert all(r["family"] == "fam_b" for r in rows_b)


def _write_tone(path: Path, seconds: float = 1.0, freq: float = 440.0, sr: int = SAMPLE_RATE):
    t = np.arange(int(sr * seconds)) / sr
    x = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed locally")
def test_transcode_keeps_duration_within_50ms(tmp_path):
    src = tmp_path / "tone.wav"
    _write_tone(src, seconds=1.0)
    dst = tmp_path / "view.wav"
    info = run_view(str(src), str(dst), fam_a_params(5))
    assert info["ok"], info
    assert abs(duration_s(str(dst)) - 1.0) <= MAX_DURATION_DRIFT_S


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed locally")
def test_fam_b_chain_runs(tmp_path):
    src = tmp_path / "tone.wav"
    _write_tone(src, seconds=0.8)
    dst = tmp_path / "fam_b.wav"
    info = run_view(str(src), str(dst), fam_b_params(11))
    assert info["ok"], info

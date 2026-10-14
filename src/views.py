"""W10 augmentation views: the transcoding chains PAST and W19 are built on.

Fam-A is the *training* view family (Opus or AAC at a handful of bit rates, plus
optional gain / pink noise / 8 kHz round trip).  Fam-B is the *held-out*
evaluation chain (Opus 24k -> MP3 64k, then a band limit at 8 kHz with p=0.5).
The two parameter sets are disjoint by construction, which is what makes W19 a
statement about an unseen chain rather than a memorised one.

Every view's random parameters are a pure function of ``20261015 + clip_id``, so
the same clip always gets the same chain on any machine and the parameters can be
re-derived from ``views/<name>/params.csv`` alone.
"""

from __future__ import annotations

import subprocess
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

SEED_BASE = 20261015
SAMPLE_RATE = 16000
MAX_DURATION_DRIFT_S = 0.05

FAM_A = {
    "A1": {"codec": "libopus", "bitrates": (12, 16, 24, 32, 48)},
    "A2": {"codec": "aac", "bitrates": (24, 32, 48, 64, 96)},
}
FAM_A_EXTRA = {
    "gain_db": 6.0,
    "gain_p": 0.5,
    "pink_snr_db": (10.0, 30.0),
    "pink_p": 0.3,
    "downsample_hz": 8000,
    "downsample_p": 0.3,
}
# Held-out chains.  B1 is the taskbook's family (Opus 24 -> MP3 64 with an optional
# 8 kHz band limit); B2 and B3 are the W22 backlog's "more families": a longer
# low-bitrate chain and a narrow-band chain, both structurally different from B1.
FAM_B = {
    "B1": {"chain": (("libopus", 24), ("libmp3lame", 64)), "band_limit_hz": 8000,
           "band_p": 0.5},
    "B2": {"chain": (("aac", 32), ("libmp3lame", 32), ("libopus", 16)),
           "band_limit_hz": 6000, "band_p": 0.5},
    "B3": {"chain": (("libmp3lame", 48), ("libopus", 16)), "band_limit_hz": 4000,
           "band_p": 0.5},
}

__all__ = [
    "SEED_BASE", "SAMPLE_RATE", "MAX_DURATION_DRIFT_S", "FAM_A", "FAM_A_EXTRA", "FAM_B",
    "view_seed", "fam_a_params", "fam_b_params", "ffmpeg_cmd", "run_view",
    "duration_s", "params_table", "apply_view", "CODEC_CONTAINER",
]

# Opus/AAC/MP3 cannot live in a WAV container; the chain hop writes the codec's
# native container and the next step decodes it back to WAV.
CODEC_CONTAINER = {"libopus": ".ogg", "opus": ".ogg", "aac": ".m4a", "libmp3lame": ".mp3",
                   "libvorbis": ".ogg", "libfdk_aac": ".m4a"}


def view_seed(clip_id: Any) -> int:
    """``20261015 + clip_id``; string ids are folded into a stable integer first."""
    if isinstance(clip_id, (int, np.integer)):
        return SEED_BASE + int(clip_id)
    return SEED_BASE + (zlib.crc32(str(clip_id).encode("utf-8")) & 0x7FFFFFFF)


def fam_a_params(clip_id: Any, view_key: Optional[str] = None) -> Dict[str, Any]:
    """Fam-A parameters for one clip.

    ``view_key`` pins the codec branch (``"A1"`` = Opus, ``"A2"`` = AAC) so the two
    training views can be generated from the same clip without ever drawing a
    different branch for the same seed.  ``None`` draws the branch itself.
    """
    rng = np.random.default_rng(view_seed(clip_id))
    if view_key is not None:
        if view_key not in FAM_A:
            raise ValueError(f"unknown Fam-A view {view_key!r}")
        view = view_key
    else:
        view = "A1" if rng.random() < 0.5 else "A2"
    cfg = FAM_A[view]
    bitrate = int(rng.choice(cfg["bitrates"]))
    gain = float(rng.uniform(-FAM_A_EXTRA["gain_db"], FAM_A_EXTRA["gain_db"])) \
        if rng.random() < FAM_A_EXTRA["gain_p"] else 0.0
    lo, hi = FAM_A_EXTRA["pink_snr_db"]
    snr = float(rng.uniform(lo, hi)) if rng.random() < FAM_A_EXTRA["pink_p"] else None
    ds = bool(rng.random() < FAM_A_EXTRA["downsample_p"])
    return {
        "family": "fam_a",
        "view": view,
        "codec": cfg["codec"],
        "bitrate_kbps": bitrate,
        "gain_db": round(gain, 3),
        "pink_snr_db": None if snr is None else round(snr, 3),
        "downsample_hz": FAM_A_EXTRA["downsample_hz"] if ds else None,
    }


def fam_b_params(clip_id: Any, view_key: str = "B1") -> Dict[str, Any]:
    """Fam-B parameters for one clip; ``view_key`` picks the held-out chain."""
    if view_key not in FAM_B:
        raise ValueError(f"unknown Fam-B view {view_key!r}; expected one of "
                         f"{sorted(FAM_B)}")
    rng = np.random.default_rng(view_seed(clip_id))
    cfg = FAM_B[view_key]
    band = bool(rng.random() < cfg["band_p"])
    return {
        "family": "fam_b",
        "view": view_key,
        "chain": [[c, b] for c, b in cfg["chain"]],
        "codec": cfg["chain"][0][0],
        "bitrate_kbps": cfg["chain"][0][1],
        "band_limit_hz": cfg["band_limit_hz"] if band else None,
    }


def _resample(x: np.ndarray, sr_out: int, sr_in: int = SAMPLE_RATE) -> np.ndarray:
    if sr_out == sr_in:
        return x
    n = max(1, int(round(len(x) * sr_out / sr_in)))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def waveform_ops(x: np.ndarray, params: Mapping[str, Any], clip_id: Any) -> np.ndarray:
    """The waveform-domain part of a Fam-A view: gain, pink noise, 8 kHz round trip.

    Done in numpy rather than with ffmpeg filters: ``anoisesrc`` is a SOURCE and
    mixing it needs a complex filtergraph, and the SNR is easier to honour exactly
    when the noise amplitude is derived from the clip's own RMS.
    """
    x = np.asarray(x, np.float32)
    if params.get("gain_db"):
        x = x * float(10.0 ** (float(params["gain_db"]) / 20.0))
    snr = params.get("pink_snr_db")
    if snr is not None:
        rng = np.random.default_rng(view_seed(f"{clip_id}#pink"))
        # 1/f noise via a random walk, then scaled to the requested SNR
        w = rng.normal(size=x.size).cumsum()
        w = w - w.mean()
        w = w / (np.std(w) + 1e-9)
        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
        amp = rms / (10.0 ** (float(snr) / 20.0))
        x = x + (w * amp).astype(np.float32)
    if params.get("downsample_hz"):
        low = _resample(x, int(params["downsample_hz"]))
        x = _resample(low, SAMPLE_RATE, int(params["downsample_hz"]))
    return x.astype(np.float32)


def codec_hops(params: Mapping[str, Any]) -> List[tuple]:
    """The (codec, bitrate) hops to apply, Fam-A single hop or Fam-B double."""
    chain = params.get("chain")
    if chain:
        return [(str(c), int(b)) for c, b in chain]
    return [(str(params.get("codec", "libopus")), int(params.get("bitrate_kbps", 32)))]


def encode_cmd(src: str, dst: str, codec: str, bitrate: int) -> List[str]:
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", codec, "-b:a", f"{int(bitrate)}k",
            str(dst)]


def decode_cmd(src: str, dst: str) -> List[str]:
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)]


def lowpass_cmd(src: str, dst: str, hz: int) -> List[str]:
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
            "-af", f"lowpass=f={int(hz)}", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le", str(dst)]


def ffmpeg_cmd(src: str, dst: str, params: Mapping[str, Any]) -> List[str]:
    """Backwards-compatible single command for a one-hop view (used by tests)."""
    codec, bitrate = codec_hops(params)[0]
    if params.get("band_limit_hz"):
        return lowpass_cmd(src, dst, int(params["band_limit_hz"]))
    if codec in ("pcm_s16le", "wav"):
        return decode_cmd(src, dst)
    return encode_cmd(src, dst, codec, bitrate)


def apply_view(x: np.ndarray, params: Mapping[str, Any], clip_id: Any,
               tmpdir: Optional[str] = None) -> np.ndarray:
    """Full view pipeline in memory: waveform ops -> codec hops -> band limit."""
    import tempfile

    own_tmp = tmpdir is None
    tmp = tempfile.mkdtemp() if own_tmp else tmpdir
    try:
        y = waveform_ops(x, params, clip_id)
        cur = Path(tmp) / "cur.wav"
        write_wav_mono(cur, y)
        for hop, (codec, bitrate) in enumerate(codec_hops(params)):
            if codec in ("none", "pcm_s16le"):
                continue
            enc = Path(tmp) / f"hop{hop}{CODEC_CONTAINER.get(codec, '.mka')}"
            res = subprocess.run(encode_cmd(str(cur), str(enc), codec, bitrate),
                                 capture_output=True, timeout=300)
            if res.returncode != 0:
                raise RuntimeError(f"{codec} encode failed: {res.stderr[-200:]!r}")
            nxt = Path(tmp) / f"dec{hop}.wav"
            res = subprocess.run(decode_cmd(str(enc), str(nxt)), capture_output=True, timeout=300)
            if res.returncode != 0:
                raise RuntimeError(f"{codec} decode failed: {res.stderr[-200:]!r}")
            cur = nxt
        if params.get("band_limit_hz"):
            out = Path(tmp) / "band.wav"
            res = subprocess.run(lowpass_cmd(str(cur), str(out), int(params["band_limit_hz"])),
                                 capture_output=True, timeout=300)
            if res.returncode != 0:
                raise RuntimeError(f"lowpass failed: {res.stderr[-200:]!r}")
            cur = out
        import soundfile as sf

        data, sr = sf.read(str(cur), dtype="float32", always_2d=True)
        return _resample(data.mean(axis=1), SAMPLE_RATE, sr) if sr != SAMPLE_RATE \
            else data.mean(axis=1).astype(np.float32)
    finally:
        if own_tmp:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


def write_wav_mono(path: Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    import wave

    xi = np.clip(np.asarray(x, np.float32), -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((xi * 32767.0).astype("<i2").tobytes())


def duration_s(path: str) -> float:
    """Duration in seconds via ffprobe; NaN when ffprobe is unavailable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=120,
        )
        return float(out.stdout.strip())
    except Exception:
        return float("nan")


def run_view(src: str, dst: str, params: Mapping[str, Any], check_duration: bool = True,
             clip_id: Any = 0) -> Dict[str, Any]:
    """Transcode one clip and (optionally) enforce the <=50 ms duration budget."""
    import soundfile as sf

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    try:
        x, sr = sf.read(str(src), dtype="float32", always_2d=True)
        y = apply_view(x.mean(axis=1), params, clip_id)
        write_wav_mono(Path(dst), y)
    except Exception as exc:
        return {"ok": False, "stderr": f"{type(exc).__name__}: {exc}"}
    info: Dict[str, Any] = {"ok": True}
    if check_duration:
        d_src, d_dst = duration_s(src), duration_s(dst)
        drift = abs(d_dst - d_src)
        info["duration_drift_s"] = drift
        if not np.isnan(drift) and drift > MAX_DURATION_DRIFT_S:
            info["ok"] = False
            info["why"] = f"duration drift {drift:.3f}s > {MAX_DURATION_DRIFT_S}s"
    return info


def params_table(clip_ids: Sequence[Any], family: str = "fam_a") -> List[Dict[str, Any]]:
    fn = fam_a_params if family == "fam_a" else fam_b_params
    rows = []
    for cid in clip_ids:
        row = {"clip_id": cid}
        row.update(fn(cid))
        rows.append(row)
    return rows

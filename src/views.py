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
FAM_B = {
    "B1": {"chain": (("libopus", 24), ("libmp3lame", 64)), "band_limit_hz": 8000, "band_p": 0.5},
}

__all__ = [
    "SEED_BASE", "SAMPLE_RATE", "MAX_DURATION_DRIFT_S", "FAM_A", "FAM_A_EXTRA", "FAM_B",
    "view_seed", "fam_a_params", "fam_b_params", "ffmpeg_cmd", "run_view",
    "duration_s", "params_table",
]


def view_seed(clip_id: Any) -> int:
    """``20261015 + clip_id``; string ids are folded into a stable integer first."""
    if isinstance(clip_id, (int, np.integer)):
        return SEED_BASE + int(clip_id)
    return SEED_BASE + (zlib.crc32(str(clip_id).encode("utf-8")) & 0x7FFFFFFF)


def fam_a_params(clip_id: Any) -> Dict[str, Any]:
    rng = np.random.default_rng(view_seed(clip_id))
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


def fam_b_params(clip_id: Any) -> Dict[str, Any]:
    rng = np.random.default_rng(view_seed(clip_id))
    cfg = FAM_B["B1"]
    band = bool(rng.random() < cfg["band_p"])
    return {
        "family": "fam_b",
        "view": "B1",
        "chain": [[c, b] for c, b in cfg["chain"]],
        "codec": cfg["chain"][0][0],
        "bitrate_kbps": cfg["chain"][0][1],
        "band_limit_hz": cfg["band_limit_hz"] if band else None,
    }


def ffmpeg_cmd(src: str, dst: str, params: Mapping[str, Any]) -> List[str]:
    """Build the ffmpeg command for one view (16 kHz mono WAV in and out)."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    filters: List[str] = []
    if params.get("gain_db"):
        filters.append(f"volume={float(params['gain_db'])}dB")
    if params.get("pink_snr_db") is not None:
        # pink noise mixed at the SNR implied by the parameter
        filters.append(f"anoisesrc=c=pink:r={SAMPLE_RATE}:a=0.01[noise]")
        filters.append(f"amix=inputs=2:duration=first:normalize=0")
    if params.get("downsample_hz"):
        filters.append(f"aresample={int(params['downsample_hz'])},aresample={SAMPLE_RATE}")
    if params.get("band_limit_hz"):
        filters.append(f"lowpass=f={int(params['band_limit_hz'])}")
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-ac", "1", "-ar", str(SAMPLE_RATE)]
    chain = params.get("chain")
    if chain:
        for codec, bitrate in chain:
            cmd += ["-c:a", str(codec), "-b:a", f"{int(bitrate)}k"]
        cmd += ["-f", "wav"]
    else:
        cmd += ["-c:a", str(params.get("codec", "libopus")),
                "-b:a", f"{int(params.get('bitrate_kbps', 32))}k", "-f", "wav"]
    cmd += [str(dst)]
    return cmd


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


def run_view(src: str, dst: str, params: Mapping[str, Any], check_duration: bool = True) -> Dict[str, Any]:
    """Transcode one clip and (optionally) enforce the <=50 ms duration budget."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = ffmpeg_cmd(src, dst, params)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    info: Dict[str, Any] = {"ok": res.returncode == 0, "cmd": cmd, "stderr": res.stderr[-500:]}
    if res.returncode != 0:
        return info
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

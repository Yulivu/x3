#!/usr/bin/env python3
"""G-A1 / G-A2: codec-normalised features for the dev split.

Section 3: every dev clip goes through 16 kHz mono -> AAC 64 kbps -> decode and is
then re-embedded with the same backbone, so S3-F can ask how much of a method's gain
survives when the channel is normalised.  Two things the taskbook is explicit about:

* ``ffmpeg`` concurrency stays at <= 3 and lane B is paused while transcoding
  (a ``control/PAUSE_B`` marker, honoured by the scheduler);
* only the dev split is touched - the memmap refuses test rows itself.

Outputs: a normal feature block ``<data>/x3/feats/<backbone>_<view>/`` (so every
existing loader works unchanged) plus a sprint-local index
``sprint1/feats/recode_<backbone>_<view>.json`` holding the global dev rows, the
zero-shot column and the fp16 S3 block S3-F reads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sprint1 import config as cfg  # noqa: E402
from scripts.sprint1 import state as st  # noqa: E402

PAUSE_B = "PAUSE_B"


def read_memmap(index_path: Path):
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    shape = tuple(int(v) for v in index["shape"])
    mm = np.memmap(index["path"], dtype=index["dtype"], mode="r", shape=shape)
    return mm, index


def write_wav(path: Path, x: np.ndarray, sr: int = 16000) -> None:
    y = np.clip(x, -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    data, _sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.asarray(data.mean(axis=1), np.float32)


def transcode_one(wave_i16: np.ndarray, bitrate: str, tmpdir: Path) -> np.ndarray:
    src = tmpdir / "in.wav"
    enc = tmpdir / "in.m4a"
    dec = tmpdir / "out.wav"
    write_wav(src, np.asarray(wave_i16, np.float32) / 32767.0)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-c:a", "aac", "-b:a", bitrate, str(enc)],
                   check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(enc),
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dec)],
                   check=True, capture_output=True)
    return read_wav(dec)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X3 sprint 1 G-A codec-normalised features")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--view", default="aac64", help="aac64 -> AAC 64 kbps")
    ap.add_argument("--side", default="dev", choices=("dev",))
    ap.add_argument("--bitrate", default="64k")
    ap.add_argument("--jobs", type=int, default=cfg.FFMPEG_JOBS_MAX)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--data", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else cfg.REPO_ROOT
    data_root = cfg.data_root(args.data)
    out_json = root / f"sprint1/feats/recode_{args.backbone}_{args.view}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    index_path = root / f"sprint1/audio/{args.side}_index.json"
    if not index_path.exists():
        print(f"missing {index_path}: run P-1 first")
        return 2
    mm, index = read_memmap(index_path)
    if index.get("side") != "dev":
        print("REFUSED: only the dev memmap may be recoded before FREEZE_OK")
        return 3
    n = len(index["rows"]) if not args.limit else min(args.limit, len(index["rows"]))

    model_dir = args.model_dir or os.environ.get(
        f"X3_MODEL_DIR_{args.backbone.upper().replace('-', '_')}",
        str(Path("/root/autodl-tmp/models") / args.backbone))
    if not Path(model_dir).exists():
        print(f"model dir {model_dir} not found; pass --model-dir or set "
              f"X3_MODEL_DIR_{args.backbone.upper().replace('-', '_')}")
        return 2

    cfg.CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    pause = cfg.CONTROL_DIR / PAUSE_B
    t0 = time.time()
    waves: List[np.ndarray] = []
    try:
        pause.write_text(f"transcoding {args.view} started {st.now_iso()}\n", encoding="utf-8")
        for i in range(n):
            with tempfile.TemporaryDirectory(prefix="x3recode_") as td:
                waves.append(transcode_one(mm[i], args.bitrate, Path(td)))
            if (i + 1) % 200 == 0:
                print(f"  transcoded {i + 1}/{n}", flush=True)
    finally:
        if pause.exists():
            pause.unlink()
    transcode_s = time.time() - t0

    from scripts.w2_features import run_backbone  # reused writer, same feature layout
    from src.data import load_mlitw
    from src.regions import load_spec_block

    meta_all = load_mlitw(data_root)
    rows = np.asarray(index["rows"], int)[:n]
    meta = meta_all.iloc[rows].reset_index(drop=True)
    feat_dir = data_root / "x3" / "feats" / f"{args.backbone}_{args.view}"
    info = run_backbone(Path(model_dir), feat_dir, waves, meta,
                        device=("cuda" if args.device == "auto" else args.device),
                        dtype=args.dtype)
    block = load_spec_block(data_root, f"{args.backbone}_{args.view}", "S3")
    block_path = root / f"sprint1/feats/recode_{args.backbone}_{args.view}_S3_fp16.npy"
    np.save(block_path, block.astype(np.float16))
    zero = np.loadtxt(feat_dir / "scores.csv", delimiter=",", skiprows=1,
                      usecols=1) if (feat_dir / "scores.csv").exists() else np.array([])
    payload: Dict[str, Any] = {
        "backbone": args.backbone, "view": args.view, "bitrate": args.bitrate,
        "side": "dev", "rows": [int(r) for r in rows], "n_clips": int(n),
        "feat_dir": str(feat_dir), "spec": "S3",
        "shape": list(block.shape), "dtype": "float16",
        "block_path": str(block_path.relative_to(root)),
        "zero_shot_path": str((feat_dir / "scores.csv")),
        "extract_info": info, "transcode_seconds": round(transcode_s, 1),
        "created": st.now_iso(),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"json": str(out_json), "block": str(block_path),
                      "n_clips": int(n), "block_shape": list(block.shape),
                      "transcode_s": round(transcode_s, 1),
                      "z0_len": int(zero.size)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

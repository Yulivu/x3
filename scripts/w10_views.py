#!/usr/bin/env python3
"""W10 - augmented views for the dev region (TASKBOOK W10).

Three chains, all applied to the SAME clip and all deterministic in
``20261015 + clip_id`` (``src/views.py``):

* ``famA1`` / ``famA2`` - training-time views (Opus / AAC), used by PAST's
  consistency term, C4's TTA and B8's true strong view;
* ``famB1`` - the held-out evaluation chain (Opus 24k -> MP3 64k, optional 8 kHz
  band limit), used by W19 and never during training.

Two phases, because 32 workers cannot each hold a 2 B-parameter encoder:

1. transcode in parallel into a fixed-shape int16 memmap (every clip is already
   exactly 64,600 samples, so the array is rectangular); the temporary audio is
   deleted afterwards and only the md5 + parameters are kept;
2. extract the chosen spec's vector once per view on the GPU, writing
   ``views/<view>/<backbone>/spec_feats.npy`` and ``scores.csv``.

Writes ``$DATA/x3/views/<view>/{params.csv,audio_md5.csv}` and
``$DATA/x3/views/<view>/<backbone>/{spec_feats.npy,scores.csv,DONE}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import w2_features as w2  # noqa: E402
import w5_backbones as w5  # noqa: E402
from src.data import data_root, load_mlitw  # noqa: E402
from src.extract_layers import N_SAMPLES, SAMPLE_RATE, prepare_waveform  # noqa: E402
from src.views import fam_a_params, fam_b_params, ffmpeg_cmd  # noqa: E402

VIEWS = ("famA1", "famA2", "famB1")

__all__ = ["params_for", "write_wav", "read_wav", "transcode_view", "extract_spec", "main"]


def params_for(view: str, clip_id: Any) -> Dict[str, Any]:
    """The deterministic parameter set of one (view, clip) pair.

    ``famA1`` pins the Opus branch and ``famA2`` the AAC branch of the Fam-A menu;
    ``famB1`` is the held-out chain.  Every draw is a pure function of
    ``20261015 + clip_id`` plus the view name, so re-running reproduces it exactly.
    """
    if view == "famA1":
        return fam_a_params(clip_id, view_key="A1")
    if view == "famA2":
        return fam_a_params(clip_id, view_key="A2")
    if view == "famB1":
        return fam_b_params(clip_id)
    raise ValueError(f"unknown view {view!r}")


def write_wav(path: Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    xi = np.clip(np.asarray(x, np.float32), -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((xi * 32767.0).astype("<i2").tobytes())


def read_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(round(len(x) * SAMPLE_RATE / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x.astype(np.float32)


def _one(args) -> Tuple[int, Optional[str], Optional[bytes]]:
    idx, x, view, params, tmp, clip_id = args
    try:
        from src.views import apply_view

        # EVERY worker needs its own scratch directory: sharing one makes all 32
        # ffmpeg processes write the same cur.wav/hop0.ogg and silently fail
        own = Path(tmp) / f"c{idx}"
        own.mkdir(parents=True, exist_ok=True)
        y = apply_view(x, params, clip_id, tmpdir=str(own))
        out = prepare_waveform(y)
        return idx, None, (np.clip(out, -1, 1) * 32767.0).astype("<i2").tobytes()
    except Exception as exc:
        return idx, f"{type(exc).__name__}: {exc}", None
    finally:
        try:
            shutil.rmtree(Path(tmp) / f"c{idx}", ignore_errors=True)
        except Exception:
            pass


def transcode_view(view: str, waves: Sequence[np.ndarray], clip_ids: Sequence[Any],
                   out_dir: Path, jobs: int = 16) -> Dict[str, Any]:
    """Transcode every clip into an int16 memmap; return the md5 of that array."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(waves)
    path = out_dir / "view_audio.int16"
    mm = np.memmap(path, dtype="<i2", mode="w+", shape=(n, N_SAMPLES))
    params_rows, errors = [], []
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tasks = [(i, waves[i], view, params_for(view, clip_ids[i]), tmp, clip_ids[i])
                 for i in range(n)]
        if jobs > 1:
            import multiprocessing as mp

            ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
            with ctx.Pool(jobs) as pool:
                for idx, err, blob in pool.imap_unordered(_one, tasks, chunksize=16):
                    if err:
                        errors.append(f"{idx}: {err}")
                        continue
                    mm[idx] = np.frombuffer(blob, dtype="<i2")
        else:
            for t in tasks:
                idx, err, blob = _one(t)
                if err:
                    errors.append(f"{idx}: {err}")
                    continue
                mm[idx] = np.frombuffer(blob, dtype="<i2")
    mm.flush()
    for i, cid in enumerate(clip_ids):
        row = {"clip_id": cid}
        row.update(params_for(view, cid))
        params_rows.append(row)
    pd.DataFrame(params_rows).to_csv(out_dir / "params.csv", index=False)
    h = hashlib.md5()
    for chunk in np.array_split(np.asarray(mm[:], dtype="<i2"), 64):
        h.update(chunk.tobytes())
    pd.DataFrame([{"view": view, "n_clips": n, "n_errors": len(errors),
                   "audio_md5": h.hexdigest(), "seconds": round(time.time() - t0, 1)}]
                 ).to_csv(out_dir / "audio_md5.csv", index=False)
    del mm
    if errors:
        # a failed clip leaves an all-zero row, i.e. the features of silence. Never
        # let that reach a downstream method: fail the view instead.
        raise RuntimeError(
            f"{view}: {len(errors)}/{n} clips failed to transcode; first: {errors[:3]}"
        )
    return {"view": view, "n": n, "errors": errors[:5], "n_errors": len(errors),
            "md5": h.hexdigest()}


def extract_spec(backbone: str, view: str, audio_path: Path, meta: pd.DataFrame,
                 out_dir: Path, model_dir: Path, spec: str, data_dir: Path,
                 device: str = "cuda") -> Dict[str, Any]:
    """One GPU pass over a transcoded view: spec vector + zero-shot score."""
    from src.extract_layers import AntiDeepfakeExtractor

    extractor = AntiDeepfakeExtractor(model_dir, device=device, dtype="float16")
    mm = np.memmap(audio_path, dtype="<i2", mode="r", shape=(len(meta), N_SAMPLES))
    info = json.loads((data_dir / "x3" / "feats" / backbone / "layers.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    means, stds, heads = [], [], []
    t0 = time.time()
    step = 64
    for start in range(0, len(meta), step):
        stop = min(start + step, len(meta))
        waves = [np.asarray(mm[i], np.float32) / 32768.0 for i in range(start, stop)]
        res = extractor.extract(waves)
        means.append(res["mean"])
        stds.append(res["std"])
        if "head_logits" in res:
            heads.append(res["head_logits"])
    mean = np.concatenate(means, axis=0)
    std = np.concatenate(stds, axis=0)
    specs = w5.specs_for(mean.astype(np.float32), std.astype(np.float32),
                         np.zeros(len(meta), np.float32), info["n_layers"])
    np.save(out_dir / "spec_feats.npy", specs[spec].astype(np.float16))
    off = info.get("fake_index")
    if heads and off is not None:
        head = np.concatenate(heads, axis=0)
        score = head[:, off] - head[:, 1 - off]
        pd.DataFrame({"clip_id": meta["clip_id"].to_numpy(), "logit_fake_minus_real": score}
                     ).to_csv(out_dir / "scores.csv", index=False)
    (out_dir / "DONE").write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n",
                                  encoding="utf-8")
    return {"view": view, "backbone": backbone, "spec": spec,
            "shape": list(np.asarray(specs[spec]).shape), "seconds": round(time.time() - t0, 1)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W10 augmented views")
    parser.add_argument("--data", default=None)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--views", default=",".join(VIEWS))
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args(argv)
    data_dir = data_root(args.data)
    meta = load_mlitw(data_dir)
    root = REPO
    split = json.loads((root / "splits" / "speaker_split_v2.json").read_text(encoding="utf-8"))
    keep = meta["speaker"].isin(set(split["dev_speakers"])).to_numpy()
    dev = meta.loc[keep].reset_index(drop=True)
    views_root = data_dir / "x3" / "views"
    print(f"dev clips={len(dev)} views={args.views}")
    for view in [v.strip() for v in args.views.split(",") if v.strip()]:
        vdir = views_root / view
        audio = vdir / "view_audio.int16"
        if not args.extract_only or not audio.exists():
            # decode the WHOLE corpus so the parquet/manifest alignment assertion
            # still runs against the full manifest, then keep only the dev rows
            all_waves, align = w2.load_audio(data_dir, meta)
            print(f"decoded {len(all_waves)} clips; alignment={align}")
            waves = [all_waves[i] for i in np.where(keep)[0]]
            info = transcode_view(view, waves, dev["clip_id"].to_numpy(), vdir, jobs=args.jobs)
            print(f"{view}: {info}")
        out = extract_spec(args.backbone, view, audio, dev,
                           vdir / args.backbone, Path(args.model_dir), args.spec, data_dir)
        print(f"{view}: {out}")
        try:
            audio.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

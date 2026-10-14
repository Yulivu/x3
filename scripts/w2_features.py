#!/usr/bin/env python3
"""W2 - AntiDeepfake hidden states and zero-shot scores on ML-ITW (GPU).

For one backbone:

1. decode every clip the way Line A did (mono 16 kHz, first 64,600 samples,
   ``np.tile`` padding - never zero padding) and layer-normalise the waveform the
   way the official model card does;
2. run length-sorted dynamic batches (total <= 160 s of audio per batch);
3. store the frame MEAN of every hidden state and the frame STD of the standard
   deviation layers, float16, plus the zero-shot ``proj_fc`` logits;
4. write ``layers.json`` (including which index is fake and how that was decided)
   and ``DONE`` so W5 can pick the directory up.

The parquet order is asserted against ``mlitw_meta.csv`` row by row: a silent
misalignment here would poison every downstream number.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data import data_root, load_mlitw  # noqa: E402
from src.extract_layers import (  # noqa: E402
    N_SAMPLES,
    SAMPLE_RATE,
    AntiDeepfakeExtractor,
    prepare_waveform,
)
from src.metrics3 import auc  # noqa: E402

MAX_BATCH_SECONDS = 160.0

__all__ = ["load_audio", "decode_wav", "run_backbone", "main"]


def decode_wav(blob: bytes, fast: bool = True) -> np.ndarray:
    """16-bit/mono-or-multi PCM WAV -> float32 mono, no resampling (all 16 kHz here)."""
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        # linear resample; the corpora are 16 kHz but never assume it
        n = int(round(len(x) * SAMPLE_RATE / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x.astype(np.float32)


def load_audio(data_dir: Path, meta: pd.DataFrame,
               limit: int = 0) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Decode every clip in manifest order and check the alignment while doing it."""
    import pyarrow.parquet as pq

    shards = sorted((data_dir / "mlitw" / "data").glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {data_dir / 'mlitw' / 'data'}")
    cols = ["file", "speaker", "language", "platform", "label"]
    waves: List[np.ndarray] = []
    seen: List[Dict[str, Any]] = []
    for shard in shards:
        table = pq.read_table(shard, columns=cols)
        d = table.to_pydict()
        for i in range(len(d["file"])):
            if limit and len(waves) >= limit:
                break
            blob = d["file"][i]["bytes"]
            waves.append(prepare_waveform(decode_wav(blob)))
            seen.append({
                "speaker": d["speaker"][i], "language": d["language"][i],
                "platform": d["platform"][i], "label": str(d["label"][i]).lower(),
            })
        if limit and len(waves) >= limit:
            break
    got = pd.DataFrame(seen)
    info: Dict[str, Any] = {"n_clips": len(got), "n_shards": len(shards)}
    expected = meta.iloc[:limit] if limit else meta
    if len(got) != len(expected):
        raise ValueError(f"audio has {len(got)} clips but the manifest has {len(expected)}")
    same = (
        (got["speaker"].astype(str).values == expected["speaker"].astype(str).values)
        & (got["platform"].astype(str).values == expected["platform"].astype(str).values)
        & (got["language"].astype(str).values == expected["language"].astype(str).values)
        & (got["label"].values == expected["label"].values)
    )
    info["alignment_ok"] = bool(same.all())
    info["misaligned_rows"] = int((~same).sum())
    if not info["alignment_ok"]:
        raise ValueError(
            f"parquet order does not match mlitw_meta.csv ({info['misaligned_rows']} rows differ); "
            "refusing to write features that would be silently misaligned"
        )
    return waves, info


def run_backbone(model_dir: Path, out_dir: Path, waves: Sequence[np.ndarray],
                 meta: pd.DataFrame, device: str = "cuda", dtype: str = "float16",
                 batch_seconds: float = MAX_BATCH_SECONDS) -> Dict[str, Any]:
    extractor = AntiDeepfakeExtractor(model_dir, device=device, dtype=dtype)
    n = len(waves)
    order = np.argsort([len(w) for w in waves], kind="stable")   # all equal, but harmless
    means, stds, heads = [], [], []
    t0 = time.time()
    done = 0
    # dynamic batches: accumulate whole clips until the second budget is hit
    i = 0
    while i < n:
        j, secs = i, 0.0
        while j < n and (secs + len(waves[int(order[j])]) / SAMPLE_RATE) <= batch_seconds:
            secs += len(waves[int(order[j])]) / SAMPLE_RATE
            j += 1
        j = max(j, i + 1)
        idx = [int(k) for k in order[i:j]]
        res = extractor.extract([waves[k] for k in idx])
        means.append(res["mean"])
        stds.append(res["std"])
        if "head_logits" in res:
            heads.append(res["head_logits"])
        i = j
        done += len(idx)
        if done % 2000 < len(idx):
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{n} clips  ({rate:.0f}/s)", flush=True)
    mean = np.concatenate(means, axis=0)
    std = np.concatenate(stds, axis=0)
    # undo the length sort so rows are back in manifest order
    inv = np.empty(n, int)
    inv[order] = np.arange(n)
    mean = mean[inv]
    std = std[inv]
    head = np.concatenate(heads, axis=0)[inv] if heads else None

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mean.npy", mean.astype(np.float16))
    np.save(out_dir / "std.npy", std.astype(np.float16))
    y = meta["y"].to_numpy()
    info: Dict[str, Any] = {
        "backbone": model_dir.name,
        "model_dir": str(model_dir),
        "n_clips": int(n),
        "n_hidden_states": int(mean.shape[1]),
        "hidden_size": int(mean.shape[2]),
        "std_layers": extractor.std_layers,
        "n_layers": extractor.n_layers,
        "load_info": extractor.info,
        "batch_seconds": batch_seconds,
        "elapsed_s": round(time.time() - t0, 1),
        "sample_rate": SAMPLE_RATE,
        "n_samples": N_SAMPLES,
    }
    if head is not None:
        # the fake/real index is decided by measurement, never assumed
        auc0 = auc(y, head[:, 0])
        fake_index = 1 if auc0 < 0.5 else 0
        real_index = 1 - fake_index
        score = head[:, fake_index] - head[:, real_index]
        info.update({
            "head_classes": int(head.shape[1]),
            "fake_index": int(fake_index),
            "real_index": int(real_index),
            "auc_index0": float(auc0),
            "auc_chosen": float(auc(y, score)),
        })
        pd.DataFrame({
            "clip_id": meta["clip_id"].to_numpy(),
            "logit_fake_minus_real": score,
            "p_real": 1.0 / (1.0 + np.exp(-(head[:, real_index] - head[:, fake_index]))),
        }).to_csv(out_dir / "scores.csv", index=False)
    meta[["clip_id", "speaker", "language", "platform", "label"]].to_csv(
        out_dir / "clip_order.csv", index=False
    )
    (out_dir / "layers.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    (out_dir / "DONE").write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n",
                                  encoding="utf-8")
    return info


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W2 feature extraction")
    parser.add_argument("--data", default=None)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--name", default=None, help="output directory name (default: model dir)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--limit", type=int, default=0, help="debug: only the first N clips")
    parser.add_argument("--out-root", default=None)
    args = parser.parse_args(argv)
    data_dir = data_root(args.data)
    out_root = Path(args.out_root) if args.out_root else (data_dir / "x3" / "feats")
    name = args.name or Path(args.model_dir).name
    meta = load_mlitw(data_dir)
    waves, align = load_audio(data_dir, meta, limit=args.limit)
    print(f"decoded {len(waves)} clips; alignment={align}")
    if args.limit:
        meta = meta.iloc[:args.limit].reset_index(drop=True)
    info = run_backbone(Path(args.model_dir), out_root / name, waves, meta,
                        device=args.device, dtype=args.dtype)
    print(json.dumps(info, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""P-1: decode the dev split to an int16 memmap once (section 1).

Every GPU package shares this memmap, so the corpus is decoded exactly once instead
of once per package.  Alignment against ``mlitw_meta.csv`` is verified while
decoding and a mismatch is a hard error - misaligned audio would silently poison
every later number.

Test clips are refused unless ``FREEZE_OK`` exists (section 0.6: no sprint package
may read test clips before the freeze).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sprint1 import config as cfg  # noqa: E402
from scripts.sprint1 import state as st  # noqa: E402

SAMPLE_RATE = 16000
N_SAMPLES = 64600  # 4.0375 s, the ML-ITW / SSL_Anti-spoofing protocol clip length


def decode_wav(blob: bytes) -> np.ndarray:
    """16-bit PCM (any channel count) -> float32 mono at 16 kHz."""
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(round(len(x) * SAMPLE_RATE / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return np.asarray(x, np.float32)


def prepare_waveform(x: np.ndarray, target: int = N_SAMPLES) -> np.ndarray:
    """Truncate or tile-pad to ``target`` samples (never zero padding)."""
    x = np.asarray(x, np.float32).ravel()
    if x.size == target:
        return x
    if x.size > target:
        return x[:target]
    reps = int(np.ceil(target / max(1, x.size)))
    return np.tile(x, reps)[:target]


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 32767.0), -32768, 32767).astype(np.int16)


def keep_positions(rows: Sequence[int], n_total: int) -> List[int]:
    """Global clip indices to decode, in manifest order.

    The parquet shards hold the whole corpus in manifest order, so a *subset* split
    has to filter by global index while walking them - reading only the first N
    clips would silently fill the memmap with the wrong rows.
    """
    keep = set(int(r) for r in rows)
    return [g for g in range(int(n_total)) if g in keep]


def dev_rows(meta, root: Path, side: str):
    from src.regions import load_split, region_rows

    split = load_split(root, cfg.SPLIT_FILE)
    return region_rows(meta, split, side)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X3 sprint 1 P-1 audio memmap")
    ap.add_argument("--side", default="dev", choices=("dev", "test"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--index", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N clips")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else cfg.REPO_ROOT
    data_root = cfg.data_root(args.data)
    out = Path(args.out) if args.out else root / f"sprint1/audio/{args.side}_i16.mm"
    index_path = Path(args.index) if args.index else root / f"sprint1/audio/{args.side}_index.json"
    out = out if out.is_absolute() else root / out
    index_path = index_path if index_path.is_absolute() else root / index_path
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.side == "test" and not cfg.has_control("FREEZE_OK"):
        print("REFUSED: test clips may not be read before FREEZE_OK (NEW_TASKS 10 section 0.6)")
        return 3

    import pandas as pd
    import pyarrow.parquet as pq

    from src.data import load_mlitw

    meta = load_mlitw(data_root)
    rows = dev_rows(meta, root, args.side)
    if args.limit:
        rows = rows[: args.limit]
    want = meta.iloc[rows].reset_index(drop=True)

    shards = sorted((data_root / "mlitw" / "data").glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {data_root / 'mlitw' / 'data'}")

    # A raw int16 memmap (no .npy header) so any reader can open it with
    # np.memmap(path, dtype="int16", mode="r", shape=index["shape"]).
    positions = keep_positions(rows, len(meta))
    if len(positions) != len(rows):
        raise ValueError(f"only {len(positions)} of {len(rows)} split rows exist in the manifest")
    mm = np.memmap(out, dtype=np.int16, mode="w+", shape=(len(rows), N_SAMPLES))
    seen_speaker: List[str] = []
    seen_platform: List[str] = []
    seen_language: List[str] = []
    seen_label: List[str] = []
    files: List[str] = []
    i = 0
    g = 0
    cols = ["file", "speaker", "language", "platform", "label"]
    for shard in shards:
        if i >= len(rows):
            break
        table = pq.read_table(shard, columns=cols).to_pydict()
        for j in range(len(table["file"])):
            if i >= len(rows):
                break
            global_index = g
            g += 1
            if global_index != positions[i]:
                continue          # not part of this split: skip without decoding
            wave = prepare_waveform(decode_wav(table["file"][j]["bytes"]))
            mm[i] = to_int16(wave)
            seen_speaker.append(str(table["speaker"][j]))
            seen_platform.append(str(table["platform"][j]))
            seen_language.append(str(table["language"][j]))
            seen_label.append(str(table["label"][j]).strip().lower())
            files.append(shard.name)
            i += 1
            if i % 500 == 0:
                print(f"  decoded {i}/{len(rows)}", flush=True)
    mm.flush()
    del mm

    got = pd.DataFrame({"speaker": seen_speaker, "platform": seen_platform,
                        "language": seen_language, "label": seen_label})
    same = (
        (got["speaker"].to_numpy() == want["speaker"].astype(str).to_numpy())
        & (got["platform"].to_numpy() == want["platform"].astype(str).to_numpy())
        & (got["language"].to_numpy() == want["language"].astype(str).to_numpy())
        & (got["label"].to_numpy() == want["label"].to_numpy())
    )
    if not bool(same.all()):
        raise ValueError(f"parquet order does not match the manifest "
                         f"({int((~same).sum())} rows differ); refusing to write a memmap")

    payload: Dict[str, Any] = {
        "side": args.side,
        "path": str(out),
        "shape": [int(len(rows)), int(N_SAMPLES)],
        "dtype": "int16",
        "layout": "raw memmap, row-major, int16, shape [n_clips, 64600]",
        "n_clips": int(len(rows)),
        "limit": int(args.limit),
        "rows": [int(r) for r in rows],
        "clip_id": [int(meta.iloc[r]["clip_id"]) for r in rows],
        "meta": {
            "speaker": seen_speaker,
            "platform": seen_platform,
            "language": seen_language,
            "label": seen_label,
            "y": [int(v) for v in want["y"].to_numpy()],
            "file": files,
        },
        "created": st.now_iso(),
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"clips": len(rows), "memmap": str(out),
                      "index": str(index_path),
                      "bytes": out.stat().st_size if out.exists() else 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

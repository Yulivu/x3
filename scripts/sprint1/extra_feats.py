#!/usr/bin/env python3
"""G-B: handcrafted descriptors plus per-layer pooled encoder features (section 3).

Two blocks, written even when one of them cannot be produced:

* **handcrafted** - log-spectral and prosodic descriptors from the waveform itself
  (RMS, ZCR, duration, spectral centroid/bandwidth/rolloff/flatness, an SNR proxy and
  a silence ratio).  No model, no data disk cost;
* **per-layer pooled** - the encoder's hidden states mean-pooled over frames at a
  fixed set of layer fractions, for each requested backbone.  Section 1 caps this at
  ~2.4 GB, so only the pooled vectors are kept (not frames), and the run stops at
  ``--max-minutes`` recording ``partial`` instead of dying.

The manifest records which weights were used.  The box ships *fine-tuned* backbone
directories, so ``weights_note`` says so explicitly: S3-F has to know whether a
retention number came from pre-head fine-tuned features or from raw SSL weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sprint1 import config as cfg  # noqa: E402
from scripts.sprint1 import state as st  # noqa: E402

SAMPLE_RATE = 16000
FRAME = 400          # 25 ms
HOP = 160            # 10 ms
LAYER_FRACTIONS = (0.25, 0.5, 0.625, 0.75, 0.875)
DESC_NAMES = ("rms_mean", "rms_std", "zcr", "duration_s", "silence_ratio", "snr_db",
              "centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std",
              "rolloff85_mean", "rolloff85_std", "flatness_mean", "flatness_std")


def frames(x: np.ndarray) -> np.ndarray:
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    if n <= 0:
        return np.zeros((1, FRAME), np.float32)
    idx = (np.arange(n)[:, None] * HOP) + np.arange(FRAME)[None, :]
    return x[idx]


def handcrafted(x: np.ndarray) -> np.ndarray:
    F = frames(np.asarray(x, np.float32))
    win = np.hanning(FRAME).astype(np.float32)
    mag = np.abs(np.fft.rfft(F * win, axis=1)) + 1e-10
    freqs = np.fft.rfftfreq(FRAME, 1.0 / SAMPLE_RATE)
    power = mag ** 2
    total = power.sum(axis=1) + 1e-12
    centroid = (power * freqs).sum(axis=1) / total
    band = np.sqrt(((freqs[None, :] - centroid[:, None]) ** 2 * power).sum(axis=1) / total)
    csum = np.cumsum(power, axis=1) / total[:, None]
    rolloff = freqs[np.argmax(csum >= 0.85, axis=1)]
    flatness = np.exp(np.log(mag).mean(axis=1)) / (mag.mean(axis=1) + 1e-12)
    rms = np.sqrt((F ** 2).mean(axis=1) + 1e-12)
    zcr = float((np.diff(np.sign(F), axis=1) != 0).mean())
    energy_db = 10 * np.log10(rms ** 2 + 1e-12)
    snr = float(np.percentile(energy_db, 95) - np.percentile(energy_db, 10))
    silence = float((rms < 0.01 * rms.max()).mean()) if rms.max() > 0 else 1.0
    return np.array([
        float(rms.mean()), float(rms.std()), zcr, len(x) / SAMPLE_RATE, silence, snr,
        float(centroid.mean()), float(centroid.std()),
        float(band.mean()), float(band.std()),
        float(rolloff.mean()), float(rolloff.std()),
        float(flatness.mean()), float(flatness.std()),
    ], np.float32)


def layer_indices(n_layers: int) -> List[int]:
    return sorted({min(n_layers, max(1, int(round(f * n_layers)))) for f in LAYER_FRACTIONS})


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X3 sprint 1 G-B extra features")
    ap.add_argument("--out", default="sprint1/feats/extra_layers_dev.npz")
    ap.add_argument("--index-out", default="sprint1/feats/extra_index.json")
    ap.add_argument("--backbones", default=",".join(cfg.BACKBONES))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-root", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else cfg.REPO_ROOT
    data_root = cfg.data_root(args.data)
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.index_out)
    index_path = index_path if index_path.is_absolute() else root / index_path

    index_file = root / "sprint1/audio/dev_index.json"
    if not index_file.exists():
        print(f"missing {index_file}: run P-1 first")
        return 2
    audio_index = json.loads(index_file.read_text(encoding="utf-8"))
    if audio_index.get("side") != "dev":
        print("REFUSED: only the dev memmap may be used before FREEZE_OK")
        return 3
    mm = np.memmap(audio_index["path"], dtype=audio_index["dtype"], mode="r",
                   shape=tuple(audio_index["shape"]))
    n = len(audio_index["rows"]) if not args.limit else min(args.limit, len(audio_index["rows"]))

    t0 = time.time()
    hand = np.zeros((n, len(DESC_NAMES)), np.float32)
    for i in range(n):
        hand[i] = handcrafted(np.asarray(mm[i], np.float32) / 32767.0)
        if (i + 1) % 2000 == 0:
            print(f"  handcrafted {i + 1}/{n}", flush=True)
    print(f"handcrafted block done in {time.time() - t0:.0f}s", flush=True)

    payload: Dict[str, Any] = {"rows": np.asarray(audio_index["rows"], int)[:n],
                               "handcrafted": hand,
                               "desc_names": np.array(DESC_NAMES)}
    status: Dict[str, Any] = {"handcrafted": {"ok": True, "n": int(n),
                                              "dim": len(DESC_NAMES)}}
    backbones = [b.strip() for b in str(args.backbones).split(",") if b.strip()]
    model_root = Path(args.model_root) if args.model_root else data_root / "x3" / "models"
    weights_note = ("fine-tuned encoder directories under <data>/x3/models (the box ships "
                    "no separate raw SSL weights); '-' means the block was not produced")

    for bb in backbones:
        if time.time() - t0 > args.max_minutes * 60:
            status[bb] = {"ok": False, "error": "time cap reached before this backbone"}
            continue
        model_dir = model_root / bb
        if not model_dir.exists():
            status[bb] = {"ok": False, "error": f"model dir {model_dir} not found"}
            continue
        try:
            import torch

            from src.extract_layers import AntiDeepfakeExtractor

            dev = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
                args.device if args.device != "auto" else "cpu")
            ex = AntiDeepfakeExtractor(str(model_dir), device=dev, dtype=args.dtype)
            sel = layer_indices(ex.n_layers)
            dim = None
            pooled, stdnorm, zs = [], [], []
            for start in range(0, n, args.batch):
                stop = min(n, start + args.batch)
                waves = [np.asarray(mm[i], np.float32) / 32767.0 for i in range(start, stop)]
                res = ex.extract(waves)
                mean = res["mean"].astype(np.float32)          # (b, L+1, D)
                if dim is None:
                    dim = mean.shape[2]
                pooled.append(mean[:, sel, :].reshape(mean.shape[0], -1).astype(np.float16))
                stdnorm.append(res["std"].astype(np.float32).mean(axis=2).astype(np.float16))
                if "head_logits" in res:
                    z = res["head_logits"]
                    zs.append((z[:, 1] - z[:, 0]).astype(np.float32))
                if (start // args.batch) % 20 == 0:
                    print(f"  {bb}: {stop}/{n} clips", flush=True)
                if time.time() - t0 > args.max_minutes * 60 and stop < n:
                    status[bb] = {"ok": True, "partial": True, "n": int(stop),
                                  "note": "time cap reached; only the first clips are stored"}
                    break
            payload[f"npt_{bb}"] = np.concatenate(pooled, axis=0)
            payload[f"npt_{bb}_layers"] = np.array(sel, int)
            payload[f"stdnorm_{bb}"] = np.concatenate(stdnorm, axis=0)
            if zs:
                payload[f"zs_{bb}"] = np.concatenate(zs, axis=0)
            status.setdefault(bb, {"ok": True, "n": int(len(payload[f"npt_{bb}"])),
                                   "layers": sel, "dim_per_layer": int(dim or 0),
                                   "device": dev})
        except Exception as exc:                                # noqa: BLE001 - reported
            status[bb] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"{bb}: FAILED - {status[bb]['error']}", flush=True)

    np.savez(out, **payload)
    index_path.write_text(json.dumps({
        "created": st.now_iso(), "npz": str(out.relative_to(root)), "n": int(n),
        "desc_names": list(DESC_NAMES), "weights_note": weights_note,
        "status": status, "seconds": round(time.time() - t0, 1),
    }, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"npz": str(out), "status": status,
                      "seconds": round(time.time() - t0, 1)}, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

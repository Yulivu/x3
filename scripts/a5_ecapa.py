#!/usr/bin/env python3
"""A5: speaker-verification OC versus SSL OC (NEW_TASKS 07 section 1).

The one-class arm on SSL features is strong.  If the same construction on a
dedicated speaker-verification embedding (SpeechBrain ECAPA-TDNN) is also
strong, the signal is identity fidelity; if SV-OC is weak while SSL-OC is
strong, the signal is channel / artefact / session, not "who is speaking".

Audio lives inside the ML-ITW parquet shards (``file.bytes``) and the shard
order is the meta order, so row *i* of the concatenated shards is row *i* of
``load_mlitw``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# The box cannot reach huggingface.co; the mirror is the only route, and
# speechbrain's fetch helpers go through HF_ENDPOINT.  It must be set before the
# speechbrain import, and the model files are already on disk.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]

from src import regions as R  # noqa: E402
from src.data import data_root  # noqa: E402
from src.metrics3 import summarize  # noqa: E402
from src.newtasks05 import (  # noqa: E402
    enrol_mask, l2_normalise, oc_scores, write_csv_atomic, zero_scores,
)


def load_dev_audio(data_dir: Path, keep: np.ndarray) -> List[bytes]:
    """Audio bytes for the kept rows, in meta order."""
    import pyarrow.parquet as pq

    shards = sorted((data_dir / "mlitw" / "data").glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {data_dir / 'mlitw' / 'data'}")
    out: List[bytes] = []
    i = 0
    for shard in shards:
        tbl = pq.read_table(shard, columns=["file"])
        for blob in tbl.column("file").to_pylist():
            if i >= keep.size:
                break
            if keep[i]:
                out.append(blob["bytes"])
            i += 1
    return out


def embed_audio(blobs: List[bytes], device: str, batch: int = 16) -> np.ndarray:
    import io

    import soundfile as sf
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    clf = EncoderClassifier.from_hparams(
        source="/root/autodl-tmp/data/ecapa", savedir="/root/autodl-tmp/data/ecapa",
        run_opts={"device": device})
    embs: List[np.ndarray] = []
    from torch.nn.utils.rnn import pad_sequence

    for i in range(0, len(blobs), batch):
        chunk = []
        for b in blobs[i:i + batch]:
            wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                import librosa
                wav = librosa.resample(np.asarray(wav, np.float32),
                                       orig_sr=sr, target_sr=16000)
            chunk.append(torch.from_numpy(np.asarray(wav, np.float32)))
        lengths = torch.tensor([c.numel() for c in chunk], dtype=torch.float32)
        lengths = lengths / lengths.max()
        padded = pad_sequence(chunk, batch_first=True)
        with torch.no_grad():
            e = clf.encode_batch(padded, wav_lens=lengths).squeeze(1).cpu().numpy()
        embs.append(e)
        if (i // batch) % 50 == 0:
            print(f"  embedded {min(i + batch, len(blobs))}/{len(blobs)}", flush=True)
    return np.vstack(embs)


def main(argv=None):
    ap = argparse.ArgumentParser(description="X3 A5 ECAPA vs SSL one-class")
    ap.add_argument("--root", default=".")
    ap.add_argument("--data")
    ap.add_argument("--backbone", default="xls-r-2b")
    ap.add_argument("--spec", default="S3")
    ap.add_argument("--split", default=R.SPLIT_V3)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results/a5_ecapa.csv")
    a = ap.parse_args(argv)

    root = Path(a.root)
    data = data_root(a.data)
    sub, X = R.load_region(root, data, a.backbone, a.spec, a.split)
    emb_path = root / "results" / "a5_ecapa_emb.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
        print(f"reusing embeddings {emb.shape}", flush=True)
    else:
        keep = np.ones(len(sub), bool)
        blobs = load_dev_audio(data, keep)
        print(f"collected {len(blobs)} dev audio blobs", flush=True)
        device = a.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"embedding on {device}", flush=True)
        emb = embed_audio(blobs, device, a.batch)
        np.save(emb_path, emb)
    En = l2_normalise(emb)
    Xn = l2_normalise(X)
    y = sub["y"].to_numpy(int)
    sp = sub["speaker"].to_numpy()
    pl = sub["platform"].to_numpy()
    s0 = zero_scores(data, a.backbone, sub["clip_id"].to_numpy())
    rows: List[Dict[str, Any]] = []
    for d in range(int(a.draws)):
        en = enrol_mask(y, sp, int(a.m), int(d))
        ev = ~en
        for name, sc in (("SV-OC", oc_scores(En, sp, en)),
                         ("SSL-OC", oc_scores(Xn, sp, en)),
                         ("Z0", s0)):
            rows.append(summarize(y[ev], sc[ev], pl[ev], extra={
                "protocol": "a5_ecapa", "region": "dev", "method": name,
                "m": int(a.m), "draw": int(d), "mask": "E0_all",
                "n_labeled": 0, "n_eval": int(ev.sum()),
                "n_enroll": int(en.sum()), "backbone": a.backbone, "spec": a.spec}))
    df = pd.DataFrame(rows)
    write_csv_atomic(df, root / a.out)
    piv = df.groupby("method")[["pooled_eer", "within_macro_eer", "auc"]].mean().round(3)
    print(piv.to_string())
    (root / "results" / "a5_ecapa.json").write_text(
        json.dumps({k: float(v) for k, v in piv["pooled_eer"].items()},
                   indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

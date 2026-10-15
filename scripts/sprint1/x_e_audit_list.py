#!/usr/bin/env python3
"""X-E: the human listening list (section 5.5).

150 dev clips stratified by platform x label, written as WAV files plus a CSV with
the metadata and the *existing* model scores.  It contains no judgement of ours -
that is the point: hongyu listens and decides.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sprint1 import config as cfg  # noqa: E402
from scripts.sprint1 import state as st  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X3 sprint 1 X-E listening list")
    ap.add_argument("--out", default="paper_tables/audit_list.csv")
    ap.add_argument("--audio-dir", default="paper_tables/audit_audio")
    ap.add_argument("--per-cell", type=int, default=0,
                    help="clips per platform x label (default: 150 spread evenly)")
    ap.add_argument("--total", type=int, default=150)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else cfg.REPO_ROOT
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(args.audio_dir)
    audio_dir = audio_dir if audio_dir.is_absolute() else root / audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)
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
    meta = audio_index["meta"]
    rows = np.asarray(audio_index["rows"], int)
    y = np.asarray(meta["y"], int)
    plat = np.asarray(meta["platform"]).astype(str)

    from src.views import write_wav_mono

    cells = [(p, lab) for p in sorted(set(plat)) for lab in (0, 1)]
    per = args.per_cell or max(1, args.total // len(cells))
    gen = np.random.default_rng(cfg.DRAW_BASE + 99)
    picked: List[int] = []
    for p, lab in cells:
        idx = np.where((plat == p) & (y == lab))[0]
        if idx.size == 0:
            continue
        picked.extend(int(v) for v in gen.choice(idx, size=min(per, idx.size), replace=False))
    picked = sorted(set(picked))

    z0 = None
    for bb in cfg.BACKBONES:
        p = cfg.data_root() / "x3" / "feats" / bb / "scores.csv"
        if p.exists():
            z0 = pd.read_csv(p)["logit_fake_minus_real"].to_numpy()
            z0_bb = bb
            break
    records: List[Dict[str, Any]] = []
    for i in picked:
        clip_id = int(audio_index["clip_id"][i])
        wav = audio_dir / f"{clip_id}_{plat[i]}_{'spoof' if y[i] else 'bonafide'}.wav"
        write_wav_mono(wav, np.asarray(mm[i], np.float32) / 32767.0)
        rec = {
            "clip_id": clip_id, "row": int(rows[i]), "speaker": meta["speaker"][i],
            "platform": plat[i], "language": meta["language"][i],
            "label": "spoof" if y[i] else "bonafide",
            "model_score_z0": float(z0[rows[i]]) if z0 is not None and rows[i] < len(z0) else "",
            "model_score_backbone": z0_bb if z0 is not None else "",
            "wav": str(wav.relative_to(root)),
            "source_shard": meta["file"][i],
        }
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(out, index=False)
    print(json.dumps({"csv": str(out), "wav_dir": str(audio_dir), "n": len(df),
                      "score_backbone": z0_bb if z0 is not None else None},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

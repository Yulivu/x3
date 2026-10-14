#!/usr/bin/env python3
"""W0 - restate the Line A protocol, inventory the machine, check the counts.

Produces ``results/w0_recon.csv`` (platform x label counts vs the frozen table),
``W0_RESTATEMENT.md`` (the five open items of TASKBOOK 1.2, each with the Line A
file it was read from) and ``REPORT.md`` (delegated to make_report.py).

It never edits a frozen artifact: everything it writes is new and lives under
``x3/``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data import EXPECTED_TOTAL, counts_match, data_root, load_mlitw  # noqa: E402

# The five items TASKBOOK 1.2 lists as "restate these"; each is answered from the
# Line A source, with the file it came from.
RESTATEMENT: Sequence[Dict[str, str]] = (
    {
        "item": "cell filter rule",
        "answer": (
            "A cell is platform|language (29 of them). A cell is only eligible to donate "
            "labels when it holds more than k+20 clips; smaller cells contribute no labels "
            "and fall back to their platform rate. Label counts per cell are therefore "
            "k=10 per cell at k=290, not 290/29 rounded."
        ),
        "read_from": "wx2/scripts/fsc_var.py (len(idx) > k + 20), wx2/scripts/p0_fewshot.py",
    },
    {
        "item": "is the label draw speaker-restricted?",
        "answer": (
            "No. Clips are drawn with rng.choice(idx, k, replace=False) per cell with no "
            "speaker constraint, so the same speaker can appear in L and in E. X3 adds the "
            "strict mask precisely to measure how much that matters."
        ),
        "read_from": "wx2/scripts/fsc_var.py, wx2/scripts/p0_fewshot.py",
    },
    {
        "item": "exact definition of the clips / strict masks",
        "answer": (
            "clips = every unlabelled clip in the region. strict = additionally drop "
            "evaluation clips whose SPEAKER also appears in L. Line A reports both as "
            "13.64 pooled at k=290, and W4 checks whether they are bit-identical; X3 "
            "implements the strict mask as an explicit speaker-disjoint filter "
            "(src/draws.py::strict_eval_mask) instead of relying on that coincidence."
        ),
        "read_from": "wx2/docs/COMPARISON.md E.5; X3 src/draws.py",
    },
    {
        "item": "probe structure and hyper-parameters",
        "answer": (
            "An MLP probe over layers [8,12,16,18,20] of the 25 mean-pooled layer vectors: "
            "LayerNorm of the flattened input, Linear(flat_dim, 512), GELU, then the output "
            "unit; trained with the script's fixed recipe (not a CV grid). The per-clip "
            "variant additionally z-scores each layer per clip (clip_z_layer)."
        ),
        "read_from": "wx2/scripts/train_corpus.py (LAYERS, class Net, clip_z_layer)",
    },
    {
        "item": "which layer and pooling the features use",
        "answer": (
            "All 25 hidden states of facebook/wav2vec2-xls-r-300m are kept, each mean-pooled "
            "over time, after decoding to mono 16 kHz and taking the FIRST 64,600 samples "
            "(4.0375 s) with np.tile padding for short clips (never zero padding). Stored as "
            "(24529, 25, 1024) float16 in mlitw_feats.npy."
        ),
        "read_from": "wx2/scripts/extract_ssl.py (SR, N, Wav2Vec2Model hidden states, mean over time)",
    },
)


def _shell(cmd: Sequence[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return (out.stdout or out.stderr).strip()
    except Exception as exc:
        return f"unavailable ({exc})"


def inventory() -> Dict[str, str]:
    return {
        "hostname": platform.node(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nproc": _shell(["nproc"]),
        "memory": _shell(["free", "-g"]),
        "disk": _shell(["df", "-h", str(data_root()), "/"]),
        "gpu": _shell(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                       "--format=csv"]),
        "ffmpeg": shutil.which("ffmpeg") or "absent",
        "tmux": shutil.which("tmux") or "absent",
        "python": sys.version.split()[0],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W0 reconnaissance")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(parents=True, exist_ok=True)

    t0 = root / "T0.txt"
    if not t0.exists():
        t0.write_text(f"T0_UTC={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n", encoding="utf-8")

    inv = inventory()
    for k, v in inv.items():
        print(f"{k}: {v}")

    drops = data_root(args.data)
    print(f"data_root: {drops}  exists={drops.exists()}")

    ok, rows, total = False, None, None
    if (drops / "mlitw_meta.csv").exists():
        meta = load_mlitw(drops)
        ok, rows, total = counts_match(meta)
        rows.to_csv(root / "results" / "w0_recon.csv", index=False)
        print(f"platform x label counts: match={ok} total={total} expected={EXPECTED_TOTAL}")
        print(rows.to_string(index=False))
    else:
        print(f"mlitw_meta.csv not found under {drops}; count check skipped")

    lines = ["# W0 restatement (TASKBOOK 1.2)", ""]
    lines.append(f"- utc: {inv['utc']}   host: {inv['hostname']}")
    lines.append(f"- platform x label counts match the frozen X2 table: **{ok}** (total {total})")
    lines.append("")
    for i, item in enumerate(RESTATEMENT, 1):
        lines.append(f"## {i}. {item['item']}")
        lines.append("")
        lines.append(item["answer"])
        lines.append("")
        lines.append(f"read from: `{item['read_from']}`")
        lines.append("")
    lines.append("## Inventory")
    lines.append("")
    for k, v in inv.items():
        lines.append(f"- **{k}**: `{v}`")
    (root / "W0_RESTATEMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {root / 'W0_RESTATEMENT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

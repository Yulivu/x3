"""Line A data contracts: the ML-ITW manifest and the frozen platform counts.

Line A's frozen artifacts live at ``$DATA`` (default ``/root/autodl-tmp/data``):

* ``mlitw_meta.csv``  - ``speaker, language, platform, label`` (24,529 rows, no header index);
  the ROW INDEX is the clip id every Line A result file is aligned to.
* ``mlitw_feats.npy`` - ``(24529, 25, 1024)`` float16, XLS-R-300m hidden states.

The counts below are transcribed from ``wx2/docs/X2_mlitw_probe.md`` section 1.1,
which is part of the frozen protocol.  They are a CHECK, never an input: W0 fails
its reconciliation if the manifest disagrees with them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_DATA = "/root/autodl-tmp/data"
SPOOF, BONAFIDE = 1, 0

# platform -> (bona fide, spoof), X2_mlitw_probe.md section 1.1
EXPECTED_PLATFORM_COUNTS: Dict[str, Tuple[int, int]] = {
    "youtube": (5948, 531),
    "bilibili": (4263, 793),
    "facebook": (3027, 1129),
    "twitter": (3046, 1061),
    "rednote": (1333, 1393),
    "douyin": (182, 1149),
    "tiktok": (369, 305),
}
EXPECTED_TOTAL = (18168, 6361, 24529)

__all__ = [
    "DEFAULT_DATA", "SPOOF", "BONAFIDE", "EXPECTED_PLATFORM_COUNTS", "EXPECTED_TOTAL",
    "data_root", "load_mlitw", "add_cell", "platform_counts", "counts_match",
]


def data_root(explicit: Optional[str] = None) -> Path:
    return Path(explicit or os.environ.get("DATA", DEFAULT_DATA))


def load_mlitw(root: Optional[Path] = None) -> pd.DataFrame:
    """The Line A manifest with ``clip_id``, ``cell`` and a numeric ``y``."""
    root = Path(root) if root is not None else data_root()
    meta = pd.read_csv(root / "mlitw_meta.csv")
    for col in ("speaker", "language", "platform", "label"):
        if col not in meta.columns:
            raise ValueError(f"mlitw_meta.csv is missing the {col!r} column")
    meta = meta.reset_index(drop=True)
    meta["clip_id"] = np.arange(len(meta))
    meta["label"] = meta["label"].astype(str).str.strip().str.lower()
    meta["y"] = (meta["label"] == "spoof").astype(int)
    return add_cell(meta)


def add_cell(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cell"] = df["platform"].astype(str) + "|" + df["language"].astype(str)
    return df


def platform_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-platform bona fide / spoof / total counts."""
    counts = (
        df.assign(real=(df["y"] == 0).astype(int), fake=(df["y"] == 1).astype(int))
        .groupby(df["platform"].astype(str).str.lower())[["real", "fake"]]
        .sum()
    )
    counts["total"] = counts["real"] + counts["fake"]
    return counts.sort_index()


def counts_match(df: pd.DataFrame, expected: Dict[str, Tuple[int, int]] = EXPECTED_PLATFORM_COUNTS):
    """Compare the manifest against the frozen table; returns ``(ok, rows)``."""
    got = platform_counts(df)
    rows = []
    ok = True
    for platform, (real, fake) in expected.items():
        g = got.loc[platform] if platform in got.index else None
        gr = int(g["real"]) if g is not None else 0
        gf = int(g["fake"]) if g is not None else 0
        match = (gr == real) and (gf == fake)
        ok = ok and match
        rows.append({"platform": platform, "expected_real": real, "real": gr,
                     "expected_fake": fake, "fake": gf, "match": bool(match)})
    extra = [p for p in got.index if p not in expected]
    if extra:
        ok = False
        rows.append({"platform": f"unexpected: {extra}", "match": False})
    total = (int((df["y"] == 0).sum()), int((df["y"] == 1).sum()), int(len(df)))
    ok = ok and total == EXPECTED_TOTAL
    return ok, pd.DataFrame(rows), total

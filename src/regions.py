"""Region and feature-block loading for the frozen v3 protocol (NEW_TASKS 02).

Everything that reads the dev region used to hard-code ``speaker_split_v2.json``
inside W3/W5/W6.  The protocol correction in NEW_TASKS 02 section 1 adds a second,
frozen split, so the lookup lives here: one function for the split, one for the
"which rows are dev" mask, and a memory-lean builder for the W5 feature specs
(the full ``mean.npy`` block is 4.6 GB fp16; the v3 dev region needs only 2-3
layers of it).

Nothing in this module decides anything: it only reads frozen files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import data_root, load_mlitw

SPLIT_V2 = "speaker_split_v2.json"
SPLIT_V3 = "speaker_split_v3.json"
SPECS = ("S1", "S2", "S3", "S4", "S5")

__all__ = [
    "SPLIT_V2", "SPLIT_V3", "SPECS",
    "load_split", "speaker_sets", "region_keep", "region_rows",
    "spec_slices", "spec_names", "load_spec_block", "load_region",
]


def load_split(root: Path, name: str = SPLIT_V3) -> Dict[str, Any]:
    """Read a frozen split file by name (``speaker_split_v2.json`` stays readable)."""
    path = Path(root) / "splits" / name
    if not path.exists():
        raise FileNotFoundError(f"split file {path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def speaker_sets(split: Dict[str, Any]) -> Tuple[set, set]:
    return set(map(str, split["dev_speakers"])), set(map(str, split["test_speakers"]))


def region_keep(meta: pd.DataFrame, split: Dict[str, Any], side: str = "dev") -> np.ndarray:
    """Boolean row mask for one side of a frozen split."""
    dev, test = speaker_sets(split)
    want = {"dev": dev, "test": test}[side]
    return meta["speaker"].astype(str).isin(want).to_numpy()


def region_rows(meta: pd.DataFrame, split: Dict[str, Any], side: str = "dev") -> np.ndarray:
    return np.where(region_keep(meta, split, side))[0]


def spec_names() -> Tuple[str, ...]:
    return SPECS


def spec_slices(spec: str, n_layers: int, n_mean_cols: int,
                n_std_cols: int) -> List[Tuple[str, int]]:
    """The ``(source, index)`` pairs W5's ``specs_for`` concatenates for ``spec``.

    Kept as a literal transcription of ``w5_backbones.specs_for`` (mid-layer block
    ``q3 = round(0.75 L)``, std block at ``round(0.75 L) - 1`` clamped to the
    std stack) and pinned against it by ``tests/test_regions.py``.
    """
    from w5_backbones import LINEA_FRACTIONS  # noqa: PLC0415  (scripts/ is on sys.path)

    spec = str(spec).upper()
    if spec not in SPECS:
        raise KeyError(f"unknown spec {spec!r}; expected one of {SPECS}")
    hs = int(n_mean_cols) - 1                 # column 0 is the projection, not a state
    L = int(n_layers)
    q3 = max(1, min(hs, int(round(0.75 * L))))
    q2 = max(1, min(hs, int(round(0.5 * L))))
    last = hs
    sidx = min(int(n_std_cols) - 1, max(0, int(round(0.75 * L)) - 1))
    frac = [max(1, min(hs, int(round(f * hs)))) for f in LINEA_FRACTIONS]
    if spec == "S1":
        return [("mean", i) for i in frac]
    if spec == "S2":
        return [("mean", last), ("std", int(n_std_cols) - 1)]
    if spec == "S3":
        return [("mean", q3), ("std", sidx)]
    if spec == "S4":
        return [("mean", q2), ("mean", q3), ("mean", last)]
    return [("mean", q3), ("std", sidx), ("zero", 0)]


def _feature_dir(data_dir: Path, backbone: str) -> Path:
    return Path(data_dir) / "x3" / "feats" / backbone


def load_spec_block(data_dir: Path, backbone: str, spec: str,
                    rows: Optional[Sequence[int]] = None,
                    want_zero: bool = True) -> np.ndarray:
    """Build ``spec`` for ``rows`` (default: every clip), without loading all layers."""
    fdir = _feature_dir(data_dir, backbone)
    info = json.loads((fdir / "layers.json").read_text(encoding="utf-8"))
    mean_mm = np.load(fdir / "mean.npy", mmap_mode="r")
    std_mm = np.load(fdir / "std.npy", mmap_mode="r")
    if mean_mm.ndim != 3 or std_mm.ndim != 3:
        raise ValueError(f"{fdir}: mean/std must be (N, layers, dim)")
    if mean_mm.shape[0] != std_mm.shape[0]:
        raise ValueError(f"{fdir}: mean has {mean_mm.shape[0]} rows, std {std_mm.shape[0]}")
    slices = spec_slices(spec, info.get("n_layers", mean_mm.shape[1] - 1),
                         mean_mm.shape[1], std_mm.shape[1])
    rows = np.arange(mean_mm.shape[0]) if rows is None else np.asarray(rows, int)
    need_mean = sorted({i for src, i in slices if src == "mean"})
    need_std = sorted({i for src, i in slices if src == "std"})
    mean = (np.asarray(mean_mm[rows][:, need_mean, :], np.float32)
            if need_mean else None)
    std = (np.asarray(std_mm[rows][:, need_std, :], np.float32)
           if need_std else None)
    zero = None
    if any(src == "zero" for src, _ in slices):
        if not want_zero:
            raise ValueError(f"spec {spec} needs the zero-shot column")
        col = pd.read_csv(fdir / "scores.csv")
        if "logit_fake_minus_real" not in col.columns:
            raise KeyError(f"{fdir/'scores.csv'} has no logit_fake_minus_real column")
        zero = col["logit_fake_minus_real"].to_numpy(np.float32)[rows]
    mpos = {v: k for k, v in enumerate(need_mean)}
    spos = {v: k for k, v in enumerate(need_std)}
    parts = []
    for src, idx in slices:
        if src == "mean":
            parts.append(mean[:, mpos[idx], :])
        elif src == "std":
            parts.append(std[:, spos[idx], :])
        else:
            parts.append(zero[:, None])
    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)


def load_region(root: Path, data_dir: Path, backbone: str, spec: str,
                split_name: str = SPLIT_V3, side: str = "dev"):
    """``(sub, F)`` for one side of a frozen split, in manifest row order."""
    split = load_split(root, split_name)
    meta = load_mlitw(data_dir)
    keep = region_keep(meta, split, side)
    sub = meta.loc[keep].reset_index(drop=True)
    F = load_spec_block(data_dir, backbone, spec, np.where(keep)[0])
    if len(sub) != F.shape[0]:
        raise ValueError(f"{len(sub)} rows but {F.shape[0]} feature rows")
    return sub, F


def default_data(explicit: Optional[str] = None) -> Path:      # thin re-export
    return data_root(explicit)


def _scripts_on_path() -> None:                               # pragma: no cover
    p = Path(__file__).resolve().parents[1] / "scripts"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


_scripts_on_path()

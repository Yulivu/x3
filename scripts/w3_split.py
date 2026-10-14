#!/usr/bin/env python3
"""W3 - freeze the speaker-disjoint dev/test split (TASKBOOK W3).

Groups are SPEAKERS, so a clip never changes side because of a lucky draw.  For
every (platform, class) the dev side must keep at least ``--dev-min`` clips and
the test side at least ``--test-min``; pairs whose total cannot satisfy both are
reported as infeasible instead of silently lowering the bar.  The split is
retried with a new seed until it satisfies every feasible constraint (<= 200
tries) and then written once and never touched again.

Writes ``splits/speaker_split_v2.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data import data_root, load_mlitw  # noqa: E402

DEV_FRAC = 0.30
DEV_MIN = 30
TEST_MIN = 100
MAX_TRIES = 200

VIDEO_ID_PATTERNS = (
    re.compile(r"(?P<vid>[A-Za-z0-9_-]{11})_[0-9]{3,6}\.wav$"),
    re.compile(r"(?P<vid>[A-Za-z0-9_-]{11})[/_]"),
)


def _key(p, y) -> Tuple[str, int]:
    return (str(p), int(y))


def speaker_counts(meta) -> Dict[str, Dict[Tuple[str, int], int]]:
    out: Dict[str, Dict[Tuple[str, int], int]] = {}
    for speaker, plat, y in zip(meta["speaker"], meta["platform"], meta["y"]):
        out.setdefault(str(speaker), {})
        out[str(speaker)][_key(plat, y)] = out[str(speaker)].get(_key(plat, y), 0) + 1
    return out


def try_split(meta, seed: int, dev_frac: float = DEV_FRAC, dev_min: int = DEV_MIN,
              test_min: int = TEST_MIN) -> Optional[Dict[str, Any]]:
    """One attempt: every speaker starts on test, then speakers move to dev."""
    rng = np.random.default_rng(seed)
    counts = speaker_counts(meta)
    speakers = sorted(counts)
    total: Dict[Tuple[str, int], int] = {}
    for c in counts.values():
        for k, v in c.items():
            total[k] = total.get(k, 0) + v
    keys = sorted(total)
    feasible = {k: total[k] >= dev_min + test_min for k in keys}

    side = {s: "test" for s in speakers}
    test_counts = dict(total)
    dev_counts = {k: 0 for k in keys}
    n = int(sum(total.values()))

    def move_to_dev(s: str) -> None:
        side[s] = "dev"
        for k, v in counts[s].items():
            test_counts[k] = test_counts.get(k, 0) - v
            dev_counts[k] = dev_counts.get(k, 0) + v

    def removable(s: str) -> bool:
        for k, v in counts[s].items():
            if test_counts.get(k, 0) - v < test_min:
                return False
        return True

    for k in sorted((k for k in keys if feasible[k]), key=lambda k: total[k]):
        guard = 0
        while dev_counts[k] < dev_min and guard < 10000:
            guard += 1
            pool = [s for s in speakers if side[s] == "test" and counts[s].get(k, 0) > 0 and removable(s)]
            if not pool:
                break
            pool.sort(key=lambda s: (sum(counts[s].values()), rng.random()))
            move_to_dev(pool[0])
        if dev_counts[k] < dev_min:
            return None

    rest = [s for s in speakers if side[s] == "test"]
    rng.shuffle(rest)
    dev_now = sum(dev_counts.values())
    for s in rest:
        if dev_now >= dev_frac * n:
            break
        if removable(s):
            move_to_dev(s)
            dev_now += sum(counts[s].values())

    for k in keys:
        if feasible[k] and (dev_counts[k] < dev_min or test_counts[k] < test_min):
            return None
    return {
        "seed": int(seed),
        "dev_speakers": sorted(s for s in speakers if side[s] == "dev"),
        "test_speakers": sorted(s for s in speakers if side[s] == "test"),
        "dev_counts": {f"{k[0]}|{k[1]}": int(v) for k, v in sorted(dev_counts.items())},
        "test_counts": {f"{k[0]}|{k[1]}": int(v) for k, v in sorted(test_counts.items())},
        "infeasible_pairs": sorted(f"{k[0]}|{k[1]}" for k in keys if not feasible[k]),
        "n_dev": int(sum(dev_counts.values())),
        "n_total": n,
        "dev_frac": float(sum(dev_counts.values()) / max(n, 1)),
    }


def make_split(meta, dev_frac: float = DEV_FRAC, dev_min: int = DEV_MIN,
               test_min: int = TEST_MIN, max_tries: int = MAX_TRIES,
               seed0: int = 20261015) -> Dict[str, Any]:
    for i in range(int(max_tries)):
        split = try_split(meta, seed0 + i, dev_frac, dev_min, test_min)
        if split is not None:
            split["tries"] = i + 1
            return split
    raise RuntimeError(f"no speaker split satisfied the constraints in {max_tries} tries")


def video_id_rule(paths: Sequence[str]) -> Dict[str, Any]:
    """Find a regex that extracts a source-video id from >= 95 % of the paths."""
    paths = [p for p in paths if isinstance(p, str)]
    if not paths:
        return {"rule": None, "coverage": 0.0, "why": "no audio path column available"}
    best: Dict[str, Any] = {"rule": None, "coverage": 0.0}
    for pat in VIDEO_ID_PATTERNS:
        hits = sum(1 for p in paths if pat.search(p))
        cov = hits / len(paths)
        if cov > best["coverage"]:
            best = {"rule": pat.pattern, "coverage": cov}
    if best["coverage"] >= 0.95:
        return best
    return {**best, "why": "best rule below the 95 % coverage bar"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W3 speaker split")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--dev-frac", type=float, default=DEV_FRAC)
    parser.add_argument("--dev-min", type=int, default=DEV_MIN)
    parser.add_argument("--test-min", type=int, default=TEST_MIN)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    drops = data_root(args.data)
    meta = load_mlitw(drops)
    split = make_split(meta, args.dev_frac, args.dev_min, args.test_min)
    split["manifest_md5"] = hashlib.md5((drops / "mlitw_meta.csv").read_bytes()).hexdigest()
    split["video_id"] = video_id_rule([])
    out = root / "splits" / "speaker_split_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"{out}: seed={split['seed']} tries={split['tries']} "
        f"dev_frac={split['dev_frac']:.3f} infeasible={split['infeasible_pairs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""W3b - the frozen protocol-v3 split (NEW_TASKS 02 section 1).

v2 left the dev region with 68 speakers, so ``strict_draw`` gave each half ~34
speakers and the unseen-speaker estimate was noisy.  v3 grows dev by moving
speakers in from the v2 TEST side - which had never been evaluated - and keeps
the rest as the new test region:

* moves are drawn label-stratified (a speaker's stratum is its majority label);
* after every move the new TEST region must still hold ``--test-min`` clips for
  each (platform, class) cell, the W3 rule;
* dev only grows, so the ``--dev-min`` rule cannot be broken by a move, but it is
  re-checked on the frozen result anyway.

The moved count is ``--from-test`` (default 54, the number NEW_TASKS 02 names).
The task text also predicts "the remaining 72 speakers", which follows from a dev
size of 54; the real v2 dev has 68 speakers, so the file records both the
requested 54 moves and the realised new-test size instead of silently aiming at
72.

``results/split_v3_audit.csv`` records the read-audit: which code paths can even
select test rows, and whether any result file carries a non-dev ``region``.

Writes ``splits/speaker_split_v3.json`` (frozen once written) and
``results/split_v3_audit.csv``.
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
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.data import data_root, load_mlitw  # noqa: E402
from src.regions import SPLIT_V2, SPLIT_V3, load_split, speaker_sets  # noqa: E402

FROM_TEST = 54
DEV_MIN = 30
TEST_MIN = 100
MAX_TRIES = 500
SEED0 = 20261015

# code that is allowed to name the test side at all: the split builders and the
# split reader.  Nothing here scores a clip.
WHITELIST = {"w3_split.py", "w3b_split_v3.py", "regions.py"}
TEST_PATTERNS = (
    re.compile(r"test_speakers"),
    re.compile(r"\bside\s*=\s*[\"']test[\"']"),
    re.compile(r"region\s*==\s*[\"']test[\"']"),
    re.compile(r"\"region\"\s*:\s*\"test\""),
)

__all__ = [
    "speaker_cell_counts", "speaker_majority", "largest_remainder",
    "stratified_order", "try_move", "final_counts", "make_split_v3",
    "audit_test_usage", "main",
]


def _key(platform: Any, y: Any) -> Tuple[str, int]:
    return (str(platform), int(y))


def speaker_cell_counts(meta: pd.DataFrame) -> Dict[str, Dict[Tuple[str, int], int]]:
    """Per speaker: how many clips it contributes to each (platform, class) cell."""
    out: Dict[str, Dict[Tuple[str, int], int]] = {}
    for spk, plat, y in zip(meta["speaker"].astype(str), meta["platform"], meta["y"]):
        d = out.setdefault(spk, {})
        k = _key(plat, y)
        d[k] = d.get(k, 0) + 1
    return out


def speaker_majority(meta: pd.DataFrame) -> Dict[str, int]:
    """Each speaker's majority label (0 bona fide / 1 spoof); ties go to 0."""
    out: Dict[str, int] = {}
    for spk, y in zip(meta["speaker"].astype(str), meta["y"]):
        real, fake = out.get(spk, (0, 0))
        out[spk] = (real + int(y == 0), fake + int(y == 1))
    return {s: int(fake > real) for s, (real, fake) in out.items()}


def largest_remainder(sizes: Dict[int, int], total: int) -> Dict[int, int]:
    """Largest-remainder allocation of ``total`` slots across strata."""
    n = sum(sizes.values())
    if total > n:
        raise ValueError("cannot allocate more slots than there are speakers")
    exact = {g: total * s / n for g, s in sizes.items()}
    quota = {g: int(np.floor(v)) for g, v in exact.items()}
    left = int(total - sum(quota.values()))
    order = sorted(sizes, key=lambda g: (-(exact[g] - quota[g]), -sizes[g], g))
    for g in order[:left]:
        quota[g] += 1
    return quota


def stratified_order(speakers: Sequence[str], majority: Dict[str, int], n: int,
                     rng: np.random.Generator) -> List[str]:
    """A full permutation whose first ``n`` entries are label-stratified.

    The head is built from a largest-remainder allocation over the label strata, so
    a greedy "take the first ``n`` that are still allowed" keeps the label mix
    proportional.  The tail keeps the remaining speakers so the caller can fall
    back when a stratified candidate is blocked by the (platform, class) floor.
    """
    groups: Dict[int, List[str]] = {}
    for s in speakers:
        groups.setdefault(int(majority[s]), []).append(s)
    for g in groups:
        groups[g] = list(groups[g])
        rng.shuffle(groups[g])
    n = min(int(n), len(speakers))
    quota = largest_remainder({g: len(v) for g, v in groups.items()}, n)
    tags: List[int] = []
    for g, q in quota.items():
        tags.extend([g] * q)
    rng.shuffle(tags)
    cursor = {g: 0 for g in groups}
    order: List[str] = []
    for g in tags:
        order.append(groups[g][cursor[g]])
        cursor[g] += 1
    used = set(order)
    leftover = [s for s in speakers if s not in used]
    rng.shuffle(leftover)
    return order + leftover


def final_counts(base_dev: Dict[Tuple[str, int], int],
                 base_test: Dict[Tuple[str, int], int],
                 counts: Dict[str, Dict[Tuple[str, int], int]],
                 moved: Sequence[str]) -> Tuple[Dict[Tuple[str, int], int],
                                                Dict[Tuple[str, int], int]]:
    dev = dict(base_dev)
    test = dict(base_test)
    for s in moved:
        for k, v in counts[s].items():
            dev[k] = dev.get(k, 0) + v
            test[k] = test.get(k, 0) - v
    return dev, test


def try_move(meta: pd.DataFrame, old_dev: Sequence[str], old_test: Sequence[str],
             n_move: int, seed: int, dev_min: int = DEV_MIN,
             test_min: int = TEST_MIN) -> Optional[Dict[str, Any]]:
    """One attempt: move ``n_move`` speakers from v2 test into dev, constraints held."""
    counts = speaker_cell_counts(meta)
    majority = speaker_majority(meta)
    # only cells the side actually owns: a cell that dev does not cover at all is
    # W3's "infeasible pair" and must not fail the split on its own
    base_dev: Dict[Tuple[str, int], int] = {}
    base_test: Dict[Tuple[str, int], int] = {}
    for s in old_dev:
        for k, v in counts.get(s, {}).items():
            base_dev[k] = base_dev.get(k, 0) + v
    for s in old_test:
        for k, v in counts.get(s, {}).items():
            base_test[k] = base_test.get(k, 0) + v
    if min(base_dev.values()) < dev_min or min(base_test.values()) < test_min:
        # the frozen v2 split must satisfy the W3 rule before anything moves
        return None

    rng = np.random.default_rng(int(seed))
    order = stratified_order(list(old_test), majority, min(int(n_move), len(old_test)), rng)
    test = dict(base_test)
    moved: List[str] = []
    for s in order:
        if len(moved) >= int(n_move):
            break
        if all(test.get(k, 0) - v >= test_min for k, v in counts[s].items()):
            for k, v in counts[s].items():
                test[k] = test.get(k, 0) - v
            moved.append(s)
    if len(moved) != int(n_move):
        return None

    dev, test_final = final_counts(base_dev, base_test, counts, moved)
    if min(dev.values()) < dev_min or min(test_final.values()) < test_min:
        return None
    dev_speakers = sorted(set(map(str, old_dev)) | set(moved))
    test_speakers = sorted(set(map(str, old_test)) - set(moved))
    return {
        "seed": int(seed),
        "moved": sorted(moved),
        "dev_speakers": dev_speakers,
        "test_speakers": test_speakers,
        "dev_counts": {f"{k[0]}|{k[1]}": int(v) for k, v in sorted(dev.items())},
        "test_counts": {f"{k[0]}|{k[1]}": int(v) for k, v in sorted(test_final.items())},
        "n_dev": int(sum(dev.values())),
        "n_total": int(sum(dev.values()) + sum(test_final.values())),
        "n_dev_speakers": len(dev_speakers),
        "n_test_speakers": len(test_speakers),
        "moved_strata": {
            "real_majority": int(sum(1 for s in moved if speaker_majority(meta)[s] == 0)),
            "fake_majority": int(sum(1 for s in moved if speaker_majority(meta)[s] == 1)),
        },
    }


def make_split_v3(meta: pd.DataFrame, v2: Dict[str, Any], n_move: int = FROM_TEST,
                  dev_min: int = DEV_MIN, test_min: int = TEST_MIN,
                  max_tries: int = MAX_TRIES, seed0: int = SEED0) -> Dict[str, Any]:
    old_dev, old_test = speaker_sets(v2)
    last: Optional[Dict[str, Any]] = None
    for i in range(int(max_tries)):
        got = try_move(meta, sorted(old_dev), sorted(old_test), n_move, seed0 + i,
                       dev_min, test_min)
        if got is not None:
            got["tries"] = i + 1
            got["dev_frac"] = got["n_dev"] / max(got["n_total"], 1)
            return got
        last = got
    raise RuntimeError(
        f"could not move {n_move} speakers out of v2 test in {max_tries} tries "
        f"without breaking dev>={dev_min} / test>={test_min} (last={last})")


def audit_test_usage(root: Path, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Evidence that no design decision ever read the v2 test region."""
    root = Path(root)
    rows: List[Dict[str, Any]] = []

    # 1. no result file may be a TEST-region result.  "full" is allowed but
    #    reported: W2/W4 reproduced frozen Line A numbers over every clip, before
    #    this split existed, and no design decision read those numbers.
    offenders: List[str] = []
    full_users: List[str] = []
    n_checked = 0
    seen_regions: set = set()
    rdir = root / "results"
    if rdir.exists():
        for csv in sorted(rdir.glob("*.csv")):
            try:
                head = pd.read_csv(csv, nrows=2000)
            except Exception:                       # a non-tabular csv, skip
                continue
            if "region" not in head.columns:
                continue
            n_checked += 1
            vals = set(head["region"].astype(str).unique())
            seen_regions |= vals
            if "test" in vals:
                offenders.append(f"{csv.name}:{sorted(vals)}")
            if "full" in vals:
                full_users.append(csv.name)
    rows.append({"check": "result_files_region_column", "detail":
                 f"{n_checked} result csv with a region column; values={sorted(seen_regions)}; "
                 f"test_region={offenders}; full_region (pre-split reproduction of frozen "
                 f"Line A numbers, no design decision)={full_users}", "n_hits": len(offenders),
                 "ok": bool(not offenders)})

    # 2. which source files can name the test side at all
    hits: List[str] = []
    for d in ("scripts", "src", "tools"):
        for py in sorted((root / d).glob("*.py")):
            if py.name in WHITELIST:
                continue
            try:
                text = py.read_text(encoding="utf-8", errors="ignore")
            except OSError:                         # pragma: no cover
                continue
            for pat in TEST_PATTERNS:
                for m in pat.finditer(text):
                    line = text[:m.start()].count("\n") + 1
                    hits.append(f"{d}/{py.name}:{line}:{m.group(0)}")
    rows.append({"check": "source_paths_touching_test_side",
                 "detail": f"whitelisted={sorted(WHITELIST)}; hits={hits[:10]}",
                 "n_hits": len(hits), "ok": bool(not hits)})

    # 3. the run log
    log = root / "progress.log"
    log_hits: List[str] = []
    if log.exists():
        for i, line in enumerate(log.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if "test" in line.lower() and "latest" not in line.lower():
                log_hits.append(f"progress.log:{i}:{line.strip()[:120]}")
    rows.append({"check": "progress_log_mentions", "detail": f"hits={log_hits[:5]}",
                 "n_hits": len(log_hits), "ok": bool(not log_hits)})

    # 4. explicit caveat: the frozen Line A reproduction scored every clip
    zs = root / "results" / "zeroshot_mlitw.csv"
    detail = "results/zeroshot_mlitw.csv missing"
    if zs.exists():
        try:
            zdf = pd.read_csv(zs)
            detail = (f"{len(zdf)} rows scored by the frozen Line A probe (W2 reproduction); "
                      "a published model applied to all clips, no test label and no design decision")
        except Exception as exc:                    # pragma: no cover
            detail = f"unreadable: {exc}"
    rows.append({"check": "frozen_model_scored_every_clip", "detail": detail,
                 "n_hits": 0, "ok": True})

    # 5. the v2 split's own speaker counts, for the record
    if data_dir is not None:
        meta = load_mlitw(data_dir)
        v2 = load_split(root, SPLIT_V2)
        dev, test = speaker_sets(v2)
        rows.append({"check": "v2_split_shape",
                     "detail": f"dev={len(dev)} speakers/{int(meta['speaker'].astype(str).isin(dev).sum())} clips; "
                               f"test={len(test)} speakers/{int(meta['speaker'].astype(str).isin(test).sum())} clips",
                     "n_hits": 0, "ok": True})
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 protocol v3 split")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--from-test", type=int, default=FROM_TEST)
    parser.add_argument("--dev-min", type=int, default=DEV_MIN)
    parser.add_argument("--test-min", type=int, default=TEST_MIN)
    parser.add_argument("--seed0", type=int, default=SEED0)
    parser.add_argument("--max-tries", type=int, default=MAX_TRIES)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    data_dir = data_root(args.data)
    (root / "results").mkdir(parents=True, exist_ok=True)

    audit = audit_test_usage(root, data_dir)
    audit.to_csv(root / "results" / "split_v3_audit.csv", index=False)
    print(audit.to_string(index=False))
    if args.audit_only:
        return 0

    meta = load_mlitw(data_dir)
    v2 = load_split(root, SPLIT_V2)
    split = make_split_v3(meta, v2, args.from_test, args.dev_min, args.test_min,
                          args.max_tries, args.seed0)
    split.update({
        "protocol": "speaker_split_v3",
        "supersedes": SPLIT_V2,
        "from_test_requested": int(args.from_test),
        "dev_min": int(args.dev_min),
        "test_min": int(args.test_min),
        "manifest_md5": hashlib.md5((data_dir / "mlitw_meta.csv").read_bytes()).hexdigest(),
        "test_region_audit": audit.to_dict(orient="records"),
        "notes": (
            "NEW_TASKS 02 section 1: new dev = v2 dev + speakers moved from v2 test, "
            "label-stratified, (platform, class) test >= test_min. The task predicted a "
            "new test of 72 speakers from a dev of 54; the real v2 dev has 68 speakers, so "
            "the requested 54 moves leave the realised new-test size recorded here."
        ),
    })
    out = root / "splits" / SPLIT_V3
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"\n{out}: moved={len(split['moved'])} tries={split['tries']} "
        f"dev={split['n_dev_speakers']} speakers/{split['n_dev']} clips "
        f"test={split['n_test_speakers']} speakers/{split['n_total'] - split['n_dev']} clips "
        f"dev_frac={split['dev_frac']:.3f}\n"
        f"  strata of the moves: {split['moved_strata']}\n"
        f"  dev cell floor={min(split['dev_counts'].values())} "
        f"test cell floor={min(split['test_counts'].values())}"
    )
    print(json.dumps(split["dev_counts"], indent=2))
    print(json.dumps(split["test_counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

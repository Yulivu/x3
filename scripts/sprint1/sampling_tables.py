#!/usr/bin/env python3
"""P-2: the shared sampling tables every later package reuses (section 3).

Everything here is deterministic from ``config.DRAW_BASE`` so a rerun reproduces
the same libraries:

* **P1** (the headline protocol, section 0.1): target platform ``p``, source = every
  labelled dev clip whose platform is not ``p`` *and* whose speaker never appears on
  ``p``.  The library is other speakers' genuine clips of ``p``.  A platform whose
  target has fewer than 30 genuine or 30 spoof clips after the library is drawn is
  marked ``macro_eligible=false`` and reported separately.
* **P2** (the fallback of F2, U8's speaker-library setting): a speaker-disjoint
  50/50 dev split with a 100-clip labelled subset of ``S_L`` and per-speaker genuine
  libraries from ``S_E``.
* **bootstrap**: the speaker-clustered resample recipes (B=200) the (b) intervals
  use, so every package resamples the *same* speakers in the *same* order.

Only v3 **dev** clips are ever indexed; test rows are refused (section 0.6).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sprint1 import config as cfg  # noqa: E402
from scripts.sprint1 import state as st  # noqa: E402

P2_DRAWS = 20
P2_LABELLED = 100
MIN_PER_CLASS = 30


def rng_for(*parts: Any) -> np.random.Generator:
    seed = cfg.DRAW_BASE
    for part in parts:
        seed = (seed * 1000003 + int(part)) % (2 ** 61 - 1)
    return np.random.default_rng(seed)


def sample(rows: np.ndarray, n: int, gen: np.random.Generator) -> List[int]:
    rows = np.asarray(rows)
    if n <= 0 or rows.size == 0:
        return []
    if n >= rows.size:
        return [int(r) for r in rows]
    return [int(r) for r in gen.choice(rows, size=n, replace=False)]


def build_p1(meta, dev_rows: np.ndarray, draws: int, sizes: Sequence[int],
             boot: int) -> Dict[str, Any]:
    y = meta["y"].to_numpy()
    sp = meta["speaker"].astype(str).to_numpy()
    plat = meta["platform"].astype(str).to_numpy()
    lang = meta["language"].astype(str).to_numpy()
    dev = set(int(r) for r in dev_rows)
    targets: Dict[str, Any] = {}

    for p in sorted(set(plat[dev_rows])):
        p_rows = np.array(sorted(r for r in dev if plat[r] == p), dtype=int)
        p_speakers = set(sp[p_rows])
        src = np.array(sorted(
            r for r in dev
            if plat[r] != p and sp[r] not in p_speakers
        ), dtype=int)
        src_speakers = set(sp[src])
        # Library candidates: genuine clips recorded on p, from *any* p speaker.
        # Section 0.2 defines the library as "other speakers' genuine clips of the
        # target platform", i.e. the query speaker's own members are masked at
        # scoring time (test b of section 0.6), not removed from the pool here.
        lib_pool = np.array(sorted(r for r in p_rows if y[r] == 0), dtype=int)
        lib_speakers = sorted(set(sp[lib_pool]))

        lib_by_draw: Dict[str, Dict[str, List[int]]] = {}
        counts: List[Dict[str, int]] = []
        for d in range(int(draws)):
            gen = rng_for(1, abs(hash(p)) % 10 ** 6 if False else sum(map(ord, p)), d)
            entry: Dict[str, List[int]] = {}
            for n in sizes:
                lib = sample(lib_pool, int(n), gen)
                entry[str(int(n))] = lib
            lib_by_draw[str(d)] = entry
            biggest = set(entry[str(int(max(sizes)))])
            ev = [r for r in p_rows if r not in biggest]
            counts.append({"n_real": int(sum(1 for r in ev if y[r] == 0)),
                           "n_fake": int(sum(1 for r in ev if y[r] == 1))})

        target_speakers = sorted(set(sp[p_rows]))
        recipes: List[List[int]] = []
        gen = rng_for(2, sum(map(ord, p)))
        for _ in range(int(boot)):
            recipes.append([int(v) for v in gen.integers(0, len(target_speakers),
                                                         size=len(target_speakers))])
        min_real = min(c["n_real"] for c in counts) if counts else 0
        min_fake = min(c["n_fake"] for c in counts) if counts else 0
        targets[p] = {
            "source_rows": [int(r) for r in src],
            "target_rows": [int(r) for r in p_rows],
            "library_pool_rows": [int(r) for r in lib_pool],
            "library_pool_speakers": lib_speakers,
            "library_pool_clip_speaker": [str(sp[r]) for r in lib_pool],
            "library_by_draw": lib_by_draw,
            "target_speakers": target_speakers,
            "target_labels": [int(y[r]) for r in p_rows],
            "target_language": [str(lang[r]) for r in p_rows],
            "bootstrap_speaker_index": recipes,
            "counts_per_draw": counts,
            "macro_eligible": bool(min_real >= MIN_PER_CLASS and min_fake >= MIN_PER_CLASS),
            "min_real": int(min_real),
            "min_fake": int(min_fake),
            "n_source_speakers": len(src_speakers),
            "n_target_speakers": len(target_speakers),
        }

    lopo = {}
    for p, block in targets.items():
        src_platforms = sorted({plat[r] for r in block["source_rows"]})
        lopo[p] = {q: [int(r) for r in block["source_rows"] if plat[r] != q]
                   for q in src_platforms}
    return {"protocol": "P1 speaker-disjoint source, dev only (NEW_TASKS 10 section 0.1)",
            "draws": int(draws), "sizes": [int(s) for s in sizes], "bootstrap_b": int(boot),
            "min_per_class": MIN_PER_CLASS, "targets": targets, "lopo_folds": lopo}


def build_p2(meta, dev_rows: np.ndarray, draws: int, sizes: Sequence[int],
             boot: int) -> Dict[str, Any]:
    from src.newtasks05 import enrol_mask, speaker_split

    y = meta["y"].to_numpy()
    sp = meta["speaker"].astype(str).to_numpy()
    dev = np.array(sorted(int(r) for r in dev_rows), dtype=int)
    out: Dict[str, Any] = {}
    for d in range(int(draws)):
        left, _right = speaker_split(sp, d, 0.5)
        s_l = dev[left[dev]]
        s_e = dev[~left[dev]]
        gen = rng_for(3, d)
        lab = sample(s_l, P2_LABELLED, gen)
        lib_by_size: Dict[str, Dict[str, List[int]]] = {}
        for n in sizes:
            lib_by_size[str(int(n))] = {"per_speaker": [], "budget": sample(s_e[y[s_e] == 0], int(n), gen)}
        # per-speaker genuine libraries (m in {5,10}) reuse the U8 enrolment sampler
        per_m: Dict[str, List[int]] = {}
        for m in (5, 10):
            mask = enrol_mask(y, sp, m, d)
            per_m[str(m)] = [int(r) for r in np.where(mask)[0] if r in set(int(v) for v in dev)]
        eval_rows = [int(r) for r in s_e if r not in set(per_m["5"])]
        eval_speakers = sorted({sp[r] for r in eval_rows})
        recipes = []
        g2 = rng_for(4, d)
        for _ in range(int(boot)):
            recipes.append([int(v) for v in g2.integers(0, len(eval_speakers),
                                                        size=len(eval_speakers))])
        out[str(d)] = {
            "labelled_rows": lab,
            "eval_rows": eval_rows,
            "library_by_size": lib_by_size,
            "library_per_speaker": per_m,
            "eval_speakers": eval_speakers,
            "bootstrap_speaker_index": recipes,
        }
    return {"protocol": "P2 speaker-library fallback (F2 in NEW_TASKS 10 section 4)",
            "draws": int(draws), "sizes": [int(s) for s in sizes], "bootstrap_b": int(boot),
            "labelled": P2_LABELLED, "draws_by_index": out}


def assert_no_test_rows(obj: Any, test_rows: set, key: str = "table") -> None:
    """Section 0.6: a sprint table that references a test clip is a hard error."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_test_rows(v, test_rows, k)
    elif isinstance(obj, list) and obj and isinstance(obj[0], int):
        if key.endswith(("rows", "_rows", "index", "pool")):
            bad = test_rows.intersection(obj)
            if bad:
                raise ValueError(f"{key} references test clips: {sorted(bad)[:5]}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X3 sprint 1 P-2 sampling tables")
    ap.add_argument("--out-dir", default="sprint1/tables")
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--p2-draws", type=int, default=P2_DRAWS)
    ap.add_argument("--sizes", default=",".join(str(s) for s in cfg.LIBRARY_SIZES))
    ap.add_argument("--boot", type=int, default=cfg.BOOTSTRAP_B)
    ap.add_argument("--data", default=None)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else cfg.REPO_ROOT
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = cfg.data_root(args.data)
    sizes = [int(s) for s in str(args.sizes).split(",") if s.strip()]

    import pandas as pd  # noqa: F401  (kept for the manifest loader's contract)

    from src.data import load_mlitw
    from src.regions import load_split, region_rows

    meta = load_mlitw(data_root)
    split = load_split(root, cfg.SPLIT_FILE)
    dev_rows = region_rows(meta, split, "dev")
    test_rows = set(int(r) for r in region_rows(meta, split, "test"))

    p1 = build_p1(meta, dev_rows, args.draws, sizes, args.boot)
    p2 = build_p2(meta, dev_rows, args.p2_draws, sizes, args.boot)

    assert_no_test_rows(p1, test_rows, "p1")
    assert_no_test_rows(p2, test_rows, "p2")

    (out_dir / "p1_folds.json").write_text(json.dumps(p1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "p2_folds.json").write_text(json.dumps(p2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "bootstrap.json").write_text(json.dumps({
        "protocol": "speaker-clustered (b) recipes, B per target",
        "p1": {p: {"speakers": b["target_speakers"],
                   "index": b["bootstrap_speaker_index"]} for p, b in p1["targets"].items()},
        "p2": {d: {"speakers": b["eval_speakers"],
                   "index": b["bootstrap_speaker_index"]} for d, b in p2["draws_by_index"].items()},
    }, ensure_ascii=False), encoding="utf-8")

    summary = {p: {"target_rows": len(b["target_rows"]), "source_rows": len(b["source_rows"]),
                   "library_pool": len(b["library_pool_rows"]),
                   "macro_eligible": b["macro_eligible"],
                   "min_real": b["min_real"], "min_fake": b["min_fake"]}
               for p, b in p1["targets"].items()}
    print(json.dumps({"created": st.now_iso(), "p1": summary,
                      "p2_draws": len(p2["draws_by_index"]),
                      "out_dir": str(out_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

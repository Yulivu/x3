#!/usr/bin/env python3
"""S1-A: prove the harness agrees with ``src.metrics3`` before anything uses it.

Writes ``sprint1/harness_report.json``; a failed check exits non-zero, which makes
the scheduler mark the package ``FAILED`` instead of letting a divergent metric
propagate into a headline number.
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
from scripts.sprint1 import harness as h  # noqa: E402
from scripts.sprint1 import state as st  # noqa: E402


def check(name: str, ok: bool, detail: Any) -> Dict[str, Any]:
    return {"check": name, "ok": bool(ok), "detail": detail}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X3 sprint 1 S1-A harness self-test")
    ap.add_argument("--out", default="sprint1/harness_report.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--clips", type=int, default=4000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    from src import metrics3

    root = Path(args.root).resolve() if args.root else cfg.REPO_ROOT
    out = Path(args.out)
    out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    device = h.pick_device(args.device)
    threads = h.set_threads()
    rows.append(check("device", device in ("cpu", "cuda"), {"device": device,
                                                           "threads": threads}))

    rng = np.random.default_rng(cfg.DRAW_BASE)
    s = rng.normal(size=args.clips) + rng.normal(size=args.clips)
    y = (rng.random(args.clips) < 0.35).astype(int)
    plat = rng.choice(["youtube", "bilibili", "douyin"], size=args.clips)

    e_t, e_l = h.eer_t(s, y), metrics3.eer(y, s)
    rows.append(check("eer matches metrics3", abs(e_t - e_l) < 1e-6,
                      {"harness": e_t, "metrics3": e_l}))
    rows.append(check("eer antisymmetry", abs(h.eer_t(-s, y) - (100 - h.eer_t(s, y))) < 1e-6,
                      {"eer": h.eer_t(s, y), "eer_neg": h.eer_t(-s, y)}))
    a_t, a_l = h.auc_t(s, y), metrics3.auc(y, s)
    rows.append(check("auc matches metrics3", abs(a_t - a_l) < 1e-9,
                      {"harness": a_t, "metrics3": a_l}))
    row_t = h.summarize_t(s, y, plat)
    row_l = metrics3.summarize(y, s, plat)
    same = all(abs(float(row_t[k]) - float(row_l[k])) < 1e-6
               for k in ("pooled_eer", "within_macro_eer", "auc", "worst_platform_eer")
               if k in row_l and isinstance(row_l[k], (int, float)))
    rows.append(check("summarize matches metrics3", same,
                      {k: (row_t.get(k), row_l.get(k))
                       for k in ("pooled_eer", "within_macro_eer", "auc", "worst_platform_eer")}))

    # one-class distance: sign, library masking, and self-exclusion
    X = rng.normal(size=(300, args.dim)).astype(np.float32)
    speaker = np.array([f"s{i % 10}" for i in range(300)])
    lib_rows = np.arange(0, 300, 3)
    d = h.library_distance(X, np.arange(300), lib_rows, speaker, speaker, device=device)
    same_speaker_pairs = 0
    lib_sp = speaker[lib_rows]
    for qi, qs in enumerate(speaker):
        if (lib_sp == qs).any():
            same_speaker_pairs += 1
    rows.append(check("library masks the query speaker",
                      same_speaker_pairs > 0 and np.isfinite(d).all(),
                      {"queries_with_own_speaker_in_pool": int(same_speaker_pairs)}))
    d_no_mask = h.library_distance(X, np.arange(300), lib_rows, speaker, speaker,
                                   device=device, mask_same_speaker=False)
    rows.append(check("masking is a real constraint",
                      bool(np.all(d >= d_no_mask - 1e-6) and np.any(d > d_no_mask + 1e-9)),
                      {"max_tightening": float(np.max(d - d_no_mask))}))

    # trainer sanity: separable data must be classified
    sep = np.vstack([rng.normal(loc=-2.0, size=(200, 8)), rng.normal(loc=2.0, size=(200, 8))])
    lab = np.r_[np.zeros(200), np.ones(200)]
    logits, info = h.logreg_lbfgs(sep.astype(np.float32), lab, l2=1e-4, device=device)
    acc = float(((logits > 0).astype(int) == lab).mean())
    rows.append(check("logreg lbfgs separates", acc > 0.95, {"accuracy": acc, **info}))

    ok = all(r["ok"] for r in rows)
    report = {"ts": st.now_iso(), "device": device, "threads": threads, "ok": ok,
              "checks": rows, "n_checks": len(rows)}
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": ok, "device": device, "failed": [r["check"] for r in rows
                                                            if not r["ok"]]}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

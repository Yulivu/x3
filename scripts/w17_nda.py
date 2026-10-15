#!/usr/bin/env python3
"""W17 - the NDA variants and the extra backbones (TASKBOOK W17).

W17 asks for the ``-nda`` variant of the two best zero-shot backbones plus
``mms-300m`` and ``wav2vec-large`` (the ensemble candidates), extracted with the same
pipeline as W2, and for the result to be merged into the backbone ranking.  The G3
rule is explicit: a candidate replaces the incumbent only if its **dev M2
within-platform macro** is lower by >= 0.5 points with the interval clear of 0 - and
then W6's ``final`` is rerun, with the old rows kept as ``superseded``.

Three stages, each safe to rerun:

* ``download`` - stream ``config.json`` + ``model.safetensors`` into
  ``$DATA/x3/models/<repo>/``; a file whose size already matches is skipped, and the
  rename is atomic, so an interrupted download never leaves a half model behind.
* ``extract`` - ``scripts/w2_features.py`` per model (same self-checks as W2), which
  appends the zero-shot row to ``results/zeroshot_mlitw.csv``.
* ``decide`` - the G3 rule on the dev region, written to ``results/w17_g3.json`` and
  ``results/w17_nda.csv``.  It **never** rewrites ``backbone_rank.csv`` or
  ``chosen_backbone.json``; it reports the verdict so the rerun is a separate,
  visible step.

The weights are CC BY-NC-SA 4.0 and stay under ``$DATA`` (never in git).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HF = "https://huggingface.co/nii-yamagishilab"
TARGETS: Tuple[Tuple[str, str], ...] = (
    ("xls-r-2b-anti-deepfake-nda", "nda of the chosen backbone (G3 incumbent)"),
    ("xls-r-1b-anti-deepfake-nda", "nda of the runner-up"),
    ("mms-1b-anti-deepfake-nda", "nda of the third backbone"),
    ("mms-300m-anti-deepfake", "extra backbone (ensemble)"),
    ("wav2vec-large-anti-deepfake", "extra backbone (ensemble)"),
)
FILES = ("config.json", "model.safetensors")
G3_MARGIN = 0.5

__all__ = ["remote_size", "download_model", "extract_model", "g3_rule",
           "dev_m2_table", "main"]


def remote_size(url: str, timeout: int = 30) -> Optional[int]:
    """Content-Length of ``url`` (``None`` when the server will not say)."""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "x3-w17"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            n = resp.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def download_model(repo: str, dest: Path, base: str = HF, chunk: int = 1 << 20,
                   max_seconds: int = 3600) -> Dict[str, Any]:
    """Fetch one HF repo into ``dest`` with curl (resume + retry).

    ``urllib`` was the obvious tool and the wrong one here: an 8 GB transfer through
    the box's proxy stalls for minutes at a time without raising, and a Python read
    loop then hangs forever.  ``curl -C -`` resumes from the partial file and
    ``--retry`` rides out the drops, and the file is only renamed once its size
    matches the server's ``Content-Length``.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    got: Dict[str, Any] = {"repo": repo, "dir": str(dest), "files": {}}
    for name in FILES:
        url = f"{base}/{repo}/resolve/main/{name}"
        want = remote_size(url)
        target = dest / name
        if target.exists() and (want is None or target.stat().st_size == want):
            got["files"][name] = {"skipped": True, "bytes": target.stat().st_size}
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        t0 = time.time()
        cmd = ["curl", "-L", "--fail", "--retry", "8", "--retry-delay", "5",
               "--connect-timeout", "20", "--max-time", str(int(max_seconds)),
               "-C", "-", "-o", str(tmp), url]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=int(max_seconds) + 120)
        except Exception as exc:                        # pragma: no cover
            got["files"][name] = {"ok": False,
                                  "reason": f"{type(exc).__name__}: {exc}",
                                  "bytes": tmp.stat().st_size if tmp.exists() else 0}
            got["ok"] = False
            return got
        got["files"][name] = {"ok": out.returncode == 0,
                              "returncode": out.returncode,
                              "stderr": (out.stderr or "").strip()[-200:],
                              "bytes": tmp.stat().st_size if tmp.exists() else 0}
        if out.returncode != 0:
            got["ok"] = False
            return got
        if want is not None and tmp.stat().st_size != want:
            got["files"][name]["ok"] = False
            got["files"][name]["reason"] = (f"size {tmp.stat().st_size} != "
                                            f"{want} after a clean curl exit")
            got["ok"] = False
            return got
        tmp.replace(target)
        secs = max(time.time() - t0, 1e-6)
        got["files"][name].update({"seconds": round(secs, 1),
                                   "mb_per_s": round(target.stat().st_size / secs / 2 ** 20, 2)})
    got["ok"] = all(v.get("ok") or v.get("skipped")
                    for v in got["files"].values())
    return got


def extract_model(root: Path, repo: str, model_dir: Path, name: Optional[str] = None,
                  jobs: int = 16, timeout: int = 6 * 3600) -> Dict[str, Any]:
    """Run W2's extractor on one model (features + zero-shot row)."""
    name = name or repo
    cmd = [sys.executable, str(root / "scripts" / "w2_features.py"),
           "--model-dir", str(model_dir), "--name", name]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:                            # pragma: no cover
        return {"repo": repo, "ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {"repo": repo, "ok": out.returncode == 0, "returncode": out.returncode,
            "tail": (out.stdout or "")[-300:], "stderr": (out.stderr or "")[-300:]}


def g3_rule(rows: Sequence[Dict[str, Any]], incumbent: str = "xls-r-2b",
            margin: float = G3_MARGIN) -> Dict[str, Any]:
    """The frozen G3 rule: replace the incumbent only on a clear >= 0.5 win.

    ``rows`` carry ``backbone``, ``m2_within_macro`` and the paired interval
    ``lo``/``hi`` against the incumbent (negative = better than the incumbent).
    """
    by_name = {r["backbone"]: r for r in rows}
    base = by_name.get(incumbent)
    out: Dict[str, Any] = {"incumbent": incumbent, "margin": margin,
                           "candidates": [], "verdict": "G3_KEEP"}
    if base is None:
        out["reason"] = f"no dev M2 row for the incumbent {incumbent}"
        return out
    best = None
    for r in rows:
        if r["backbone"] == incumbent:
            continue
        gain = float(base["m2_within_macro"]) - float(r["m2_within_macro"])
        clears = bool(r.get("hi", 1.0) < 0.0)           # interval clear of zero
        cand = {"backbone": r["backbone"], "gain_points": gain, "interval": [r.get("lo"),
                                                                             r.get("hi")],
                "clears_interval": clears, "meets_margin": bool(gain >= margin),
                "replaces": bool(gain >= margin and clears)}
        out["candidates"].append(cand)
        if cand["replaces"] and (best is None or gain > best["gain_points"]):
            best = cand
    if best is not None:
        out["verdict"] = "G3_UPDATE"
        out["winner"] = best
        out["next_step"] = ("rerun W6 final for the winner and keep the old rows as "
                            "superseded (per TASKBOOK W17)")
    return out


def dev_m2_table(root: Path, data_dir: Path, names: Sequence[str],
                 incumbent: str = "xls-r-2b", draws: int = 30, jobs: int = 8
                 ) -> pd.DataFrame:
    """Dev M2 (cell-prior corrected) within-macro per backbone, draw-paired.

    Runs the canonical W6 runner with ``--methods B2`` (B2 *is* the M2 correction) for
    every candidate including the incumbent, with its own suffix so nothing stored is
    touched, then pairs the draws against the incumbent.
    """
    import w17_extra_backbone as w17b

    def run(name: str) -> Optional[Path]:
        suffix = f"_w17_{name}"
        draws_file = root / "results" / f"w6_draws_{name}_S3{suffix}.csv"
        if not draws_file.exists():
            cmd = [sys.executable, str(root / "scripts" / "w6_baselines.py"),
                   "--root", str(root), "--backbone", name, "--spec", "S3",
                   "--split", "speaker_split_v3.json", "--suffix", suffix,
                   "--methods", "B2", "--draws", str(int(draws)),
                   "--jobs", str(int(jobs))]
            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=6 * 3600)
            except Exception as exc:                    # pragma: no cover
                print(f"  {name}: {type(exc).__name__}: {exc}", flush=True)
                return None
        return draws_file if draws_file.exists() else None

    frames: Dict[str, pd.DataFrame] = {}
    for name in names:
        path = run(name)
        if path is not None:
            frames[name] = pd.read_csv(path)
    rows: List[Dict[str, Any]] = []
    base = frames.get(incumbent)
    for name, df in frames.items():
        mean = float(df["within_macro_eer"].mean())
        entry: Dict[str, Any] = {"backbone": name, "spec": "S3",
                                 "m2_within_macro": mean, "n_draws": int(len(df))}
        if base is not None and name != incumbent:
            paired = w17b.pair_runs(df, base, name, incumbent,
                                    metrics=("within_macro_eer",))
            if not paired.empty:
                r = paired.iloc[0]
                entry.update({"lo": float(r["a_lo"]), "hi": float(r["a_hi"]),
                              "delta": float(r["delta"])})
        rows.append(entry)
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W17 NDA variants / extra backbones")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--stage", default="all",
                        choices=["download", "extract", "decide", "all"])
    parser.add_argument("--repos", default=",".join(r for r, _ in TARGETS))
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--draws", type=int, default=30)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root / "scripts"))
    from src.data import data_root

    data_dir = data_root(args.data)
    repos = [r.strip() for r in str(args.repos).split(",") if r.strip()]
    known = dict(TARGETS)
    results: List[Dict[str, Any]] = []
    (root / "results").mkdir(parents=True, exist_ok=True)
    for repo in repos:
        if args.stage in ("download", "all"):
            got = download_model(repo, data_dir / "x3" / "models" / repo)
            print(f"download {repo}: ok={got.get('ok')} "
                  f"{ {k: v.get('mb_per_s') for k, v in got['files'].items()} }",
                  flush=True)
            results.append({"repo": repo, "role": known.get(repo, ""), **got})
        if args.stage in ("extract", "all"):
            got = extract_model(root, repo, data_dir / "x3" / "models" / repo,
                                jobs=args.jobs)
            print(f"extract {repo}: {got}", flush=True)
            results.append({"repo": repo, "stage": "extract", **got})
    if args.stage in ("decide", "all"):
        table = dev_m2_table(root, data_dir, repos, draws=args.draws, jobs=args.jobs)
        table.to_csv(root / "results" / "w17_nda.csv", index=False)
        verdict = g3_rule(table.to_dict(orient="records"))
        (root / "results" / "w17_g3.json").write_text(
            json.dumps(verdict, indent=2, sort_keys=True, default=str),
            encoding="utf-8")
        print(json.dumps(verdict, indent=2, sort_keys=True, default=str))
    (root / "results" / "w17_run.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

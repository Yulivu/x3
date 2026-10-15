#!/usr/bin/env python3
"""W1 - the environment record (TASKBOOK W1, first P0 work package).

TASKBOOK W1 asks for the compute environment to be pinned before anything runs:
interpreter, torch/CUDA, GPU, the AntiDeepfake stack, ffmpeg, tmux, cores, RAM and
free disk.  It writes two things:

* ``ENV_READY`` - the marker the queue waits for, written atomically and only when
  every *required* check passed;
* ``results/w1_env.csv`` - one row per check with its measured value, so the paper's
  reproducibility section can quote the box instead of a guess.

``--allow-degraded`` records a missing optional tool (e.g. ffmpeg) without the
marker; the default is strict, because a missing ffmpeg silently turns every
transcode family into a copy.

Writes ``ENV_READY`` and ``results/w1_env.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REQUIRED = ("python", "torch", "transformers", "sklearn", "numpy", "ffmpeg", "tmux")
OPTIONAL = ("fairseq", "soundfile", "scipy", "pandas", "yaml")

__all__ = ["module_version", "tool_version", "gpu_info", "checks", "main"]


def module_version(name: str) -> Optional[str]:
    """Import ``name`` and return its ``__version__`` (``None`` when unavailable)."""
    try:
        mod = __import__(name)
    except Exception:
        return None
    return str(getattr(mod, "__version__", "unknown"))


def tool_version(exe: str, args: Sequence[str] = ("--version",),
                 timeout: int = 20) -> Optional[str]:
    if shutil.which(exe) is None:
        return None
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True,
                             timeout=timeout)
    except Exception:                                   # pragma: no cover
        return None
    text = (out.stdout or out.stderr or "").strip().splitlines()
    return text[0][:120] if text else "present"


def gpu_info(timeout: int = 20) -> Dict[str, Any]:
    """``{n_gpu, name, memory_mib}`` from ``nvidia-smi`` (empty when absent)."""
    if shutil.which("nvidia-smi") is None:
        return {"n_gpu": 0, "reason": "nvidia-smi not on PATH"}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=timeout)
    except Exception as exc:                            # pragma: no cover
        return {"n_gpu": 0, "reason": f"{type(exc).__name__}: {exc}"}
    rows = [r for r in (out.stdout or "").strip().splitlines() if r.strip()]
    if not rows:
        return {"n_gpu": 0, "reason": (out.stderr or "no output").strip()[:120]}
    name, mem = (rows[0].split(",") + ["", ""])[:2]
    return {"n_gpu": len(rows), "name": name.strip(),
            "memory_mib": int(float(mem)) if mem.strip().replace(".", "").isdigit()
            else None}


def checks(data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """One row per environment fact, with ``required`` marking the blockers."""
    rows: List[Dict[str, Any]] = []

    rows.append({"check": "python", "required": True,
                 "value": sys.version.split()[0], "present": True})
    for name in ("torch", "transformers", "sklearn", "numpy"):
        v = module_version(name)
        rows.append({"check": name, "required": True, "value": v, "present": v is not None})
    for name in OPTIONAL:
        v = module_version(name)
        rows.append({"check": name, "required": False, "value": v,
                     "present": v is not None})
    gpu = gpu_info()
    rows.append({"check": "gpu", "required": False,
                 "value": json.dumps(gpu), "present": bool(gpu.get("n_gpu"))})
    for exe, args in (("ffmpeg", ("-version",)), ("tmux", ("-V",)),
                      ("git", ("--version",))):
        v = tool_version(exe, args)
        rows.append({"check": exe, "required": exe in REQUIRED, "value": v,
                     "present": v is not None})
    rows.append({"check": "cpus", "required": False,
                 "value": os.cpu_count(), "present": True})
    try:
        import shutil as _sh

        total, used, free = _sh.disk_usage("/")
        rows.append({"check": "disk_root_gb", "required": False,
                     "value": round(free / 2 ** 30, 1), "present": True})
    except Exception:                                   # pragma: no cover
        rows.append({"check": "disk_root_gb", "required": False, "value": None,
                     "present": False})
    if data_dir is not None:
        rows.append({"check": "data_dir", "required": False,
                     "value": str(data_dir), "present": Path(data_dir).exists()})
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W1 environment check")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--allow-degraded", action="store_true",
                        help="record missing optional tools without the marker")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root / "scripts"))
    from src.data import data_root

    data_dir = data_root(args.data)
    rows = checks(data_dir)
    df = pd.DataFrame(rows)
    (root / "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "results" / "w1_env.csv", index=False)
    missing_required = [r["check"] for r in rows
                        if r["required"] and not r["present"]]
    missing_optional = [r["check"] for r in rows
                        if not r["required"] and not r["present"]]
    print(df[["check", "present", "value"]].to_string(index=False))
    if missing_required:
        print(f"MISSING REQUIRED: {missing_required}", file=sys.stderr)
    if missing_optional:
        print(f"missing optional: {missing_optional}")
    if missing_required or (missing_optional and not args.allow_degraded):
        print("ENV_READY not written", file=sys.stderr)
        return 1
    payload = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "python": sys.version.split()[0], "cpus": os.cpu_count(),
               "missing_optional": missing_optional}
    tmp = root / "ENV_READY.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(root / "ENV_READY")
    print(f"ENV_READY written ({len(rows)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

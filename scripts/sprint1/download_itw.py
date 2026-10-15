#!/usr/bin/env python3
"""N-1: fetch the In-The-Wild set in the background (lane N, network only).

The machine may be offline apart from mirrors, so the package tries, in order:

1. ``huggingface_hub.snapshot_download`` against ``HF_ENDPOINT`` (default
   ``https://hf-mirror.com``, the mirror the user asked for);
2. ``aria2c`` on the single archive URL with 16 connections and ``--continue``;
3. ModelScope, when the client is importable.

``source /etc/network_turbo`` is applied when that file exists.  The package always
finishes by writing ``sprint1/itw_status.json`` with ``OK`` or ``BLOCKED`` plus a
reason (``network`` / ``unzip`` / ``license`` / ``disk``) - section F7 wants the
status recorded, never a chain that stalls on a download.
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
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sprint1 import config as cfg  # noqa: E402
from scripts.sprint1 import state as st  # noqa: E402

DEFAULT_DATASET = "mueller91/In-The-Wild"
DEFAULT_ZIP = ("https://hf-mirror.com/datasets/mueller91/In-The-Wild/resolve/main/"
               "release_in_the_wild.zip?download=true")


def turbo_prefix() -> str:
    return "source /etc/network_turbo 2>/dev/null || true; " if Path("/etc/network_turbo").exists() else ""


def run(cmd: str, timeout: int) -> Dict[str, Any]:
    t0 = time.time()
    try:
        out = subprocess.run(f"{turbo_prefix()}{cmd}", shell=True, capture_output=True,
                             text=True, timeout=timeout)
        return {"rc": int(out.returncode), "stdout": out.stdout[-2000:],
                "stderr": out.stderr[-2000:], "seconds": round(time.time() - t0, 1)}
    except subprocess.TimeoutExpired:
        return {"rc": 124, "stdout": "", "stderr": f"timeout after {timeout}s",
                "seconds": round(time.time() - t0, 1)}
    except Exception as exc:
        return {"rc": 1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.time() - t0, 1)}


def try_snapshot(dataset: str, out: Path, endpoint: str, timeout: int) -> Dict[str, Any]:
    env_endpoint = endpoint
    script = (
        "import os\n"
        f"os.environ.setdefault('HF_ENDPOINT', {env_endpoint!r})\n"
        "from huggingface_hub import snapshot_download\n"
        f"p = snapshot_download({dataset!r}, repo_type='dataset', local_dir={str(out)!r})\n"
        "print('SNAPSHOT_OK', p)\n"
    )
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "itw_snapshot_dl.py"
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(script, encoding="utf-8")
    except OSError:
        tmp = Path("itw_snapshot_dl.py")
        tmp.write_text(script, encoding="utf-8")
    return run(f"python {tmp} ", timeout)


def try_aria2(url: str, out_zip: Path, timeout: int, conns: int = 16) -> Dict[str, Any]:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    cmd = (f"aria2c --continue=true --max-connection-per-server={conns} --split={conns} "
           f"--min-split-size=8M --file-allocation=none --max-tries=8 --retry-wait=10 "
           f"--summary-interval=60 --console-log-level=warn "
           f"--dir={out_zip.parent} --out={out_zip.name} {url!r}")
    return run(cmd, timeout)


def verify(out: Path, zip_path: Path) -> Dict[str, Any]:
    """Integrity check: ``unzip -t`` for the archive, a file count for a snapshot."""
    if zip_path.exists():
        res = run(f"unzip -tq {zip_path}", timeout=1800)
        return {"kind": "zip", "bytes": zip_path.stat().st_size, "test_rc": res["rc"],
                "detail": res["stderr"][-300:] or res["stdout"][-300:],
                "ok": res["rc"] == 0}
    files = [p for p in out.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    return {"kind": "snapshot", "files": len(files), "bytes": total,
            "ok": len(files) > 0 and total > 0}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X3 sprint 1 N-1 In-The-Wild download")
    ap.add_argument("--out", default="/root/autodl-tmp/data/itw")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--zip-url", default=DEFAULT_ZIP)
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    ap.add_argument("--timeout", type=int, default=5400, help="per-source budget (s)")
    ap.add_argument("--status", default="sprint1/itw_status.json")
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else cfg.REPO_ROOT
    out = Path(args.out)
    status_path = Path(args.status)
    status_path = status_path if status_path.is_absolute() else root / status_path
    status_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    attempts: List[Dict[str, Any]] = []
    status: Dict[str, Any] = {"status": "BLOCKED", "reason": "network",
                              "dataset": args.dataset, "out": str(out)}

    free = cfg.free_gb(out if out.exists() else Path("/root/autodl-tmp"))
    status["free_gb_before"] = None if free != free else round(float(free), 1)
    if free == free and free < cfg.SYSTEM_FREE_MIN_GB:
        status.update(status="BLOCKED", reason="disk",
                      note=f"free {free:.1f} GB < {cfg.SYSTEM_FREE_MIN_GB:g} GB guard")
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False))
        return 0

    zip_path = out / "release_in_the_wild.zip"
    # An earlier run may already have staged the set (the previous phase extracted it):
    # verify first so a present corpus never triggers an 8 GB re-download.
    check = verify(out, zip_path)
    if check["ok"]:
        attempts.append({"source": "already-present", "rc": 0,
                         "stderr": f"verified in place: {check}"})
    else:
        attempts.append({"source": "snapshot", **(try_snapshot(args.dataset, out, args.endpoint,
                                                              min(args.timeout, 3600)) or {})})
        check = verify(out, zip_path)
    if not check["ok"]:
        attempts.append({"source": "aria2", **try_aria2(args.zip_url, zip_path,
                                                       min(args.timeout, 5400))})
        check = verify(out, zip_path)
    if not check["ok"]:
        try:
            import modelscope  # noqa: F401
            attempts.append({"source": "modelscope", **run(
                f"python -c \"from modelscope import snapshot_download; "
                f"snapshot_download('iic/speech_ecapa-tdnn_zh-cn_16k-common')\"",
                min(args.timeout, 3600))})
        except ImportError:
            attempts.append({"source": "modelscope", "rc": None,
                             "stderr": "modelscope client not installed"})
        check = verify(out, zip_path)

    status.update({
        "attempts": attempts,
        "verification": check,
        "elapsed_s": round(time.time() - t0, 1),
        "free_gb_after": (None if cfg.free_gb(out) != cfg.free_gb(out)
                          else round(float(cfg.free_gb(out)), 1)),
    })
    if check["ok"]:
        status.update(status="OK", reason="")
        if attempts and attempts[0].get("source") == "already-present":
            status["note"] = "the staged corpus was already on the data disk; nothing downloaded"
        elif zip_path.exists():
            status["note"] = "archive verified; extract on first use and delete the zip"
    else:
        reasons = " ".join(str(a.get("stderr", ""))[-200:] for a in attempts)
        if "licen" in reasons.lower() or "gated" in reasons.lower():
            status["reason"] = "license"
        elif "unzip" in reasons.lower():
            status["reason"] = "unzip"
        status["note"] = ("ITW_UNAVAILABLE - section F7: record the reason and continue; "
                          "the FSW subset is only used when a download costs <= 2 h")
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: status[k] for k in ("status", "reason", "elapsed_s")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

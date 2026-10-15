#!/usr/bin/env python3
"""P-0 preflight (NEW_TASKS 10 section 3).

Writes ``preflight.md`` (human) and ``logs/sprint1_env.json`` (machine): disk
inventory, the cache/manifest inventory the rest of the sprint depends on, the
lane budgets actually in force, and the blocking list.  It also publishes the CPU
budget where ``src.newtasks05.worker_slots`` reads it, so every driver inherits it.

Exit code is 0 even when data is missing: a missing input is reported, not a crash.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sprint1 import config as cfg  # noqa: E402
from scripts.sprint1 import state as st  # noqa: E402


def sh(cmd: Sequence[str], timeout: int = 30) -> str:
    try:
        out = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)
        return (out.stdout or out.stderr).strip()
    except Exception as exc:  # pragma: no cover - depends on the machine
        return f"<{type(exc).__name__}: {exc}>"


def disk_rows(roots: Sequence[Path]) -> List[Dict[str, Any]]:
    rows = []
    for p in roots:
        probe = p
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            u = shutil.disk_usage(str(probe))
            rows.append({"path": str(p), "probed": str(probe),
                         "total_gb": round(u.total / 1024 ** 3, 1),
                         "free_gb": round(u.free / 1024 ** 3, 1)})
        except OSError as exc:
            rows.append({"path": str(p), "probed": str(probe), "error": str(exc)})
    return rows


def manifest_inventory(data_root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"data_root": str(data_root), "exists": data_root.exists()}
    meta = data_root / "mlitw_meta.csv"
    info["manifest"] = {"path": str(meta), "exists": meta.exists(),
                        "bytes": meta.stat().st_size if meta.exists() else 0}
    shard_dir = data_root / "mlitw" / "data"
    shards = sorted(shard_dir.glob("*.parquet")) if shard_dir.exists() else []
    info["parquet_shards"] = {"dir": str(shard_dir), "count": len(shards),
                              "bytes": sum(p.stat().st_size for p in shards)}
    try:
        import pandas as pd
        if meta.exists():
            m = pd.read_csv(meta)
            info["manifest"]["rows"] = int(len(m))
            if "platform" in m.columns:
                info["manifest"]["platform_rows"] = {
                    str(k): int(v) for k, v in m["platform"].astype(str).value_counts().items()}
    except Exception as exc:
        info["manifest"]["read_error"] = f"{type(exc).__name__}: {exc}"
    feats = data_root / "x3" / "feats"
    blocks: Dict[str, Any] = {}
    if feats.exists():
        for bb in sorted(p for p in feats.iterdir() if p.is_dir()):
            files = sorted(bb.iterdir())
            blocks[bb.name] = {
                "bytes": sum(f.stat().st_size for f in files if f.is_file()),
                "files": [f.name for f in files[:20]],
            }
    info["feature_blocks"] = blocks
    return info


def sprint_inputs(root: Path, data_root: Path) -> Dict[str, Any]:
    split = root / "splits" / cfg.SPLIT_FILE
    out: Dict[str, Any] = {}
    out["split_file"] = {"path": str(split), "exists": split.exists()}
    if split.exists():
        try:
            obj = json.loads(split.read_text(encoding="utf-8"))
            out["split_file"]["dev_speakers"] = len(obj.get("dev_speakers") or [])
            out["split_file"]["test_speakers"] = len(obj.get("test_speakers") or [])
        except Exception as exc:
            out["split_file"]["read_error"] = f"{type(exc).__name__}: {exc}"
    audio_mm = root / "sprint1" / "audio"
    out["audio_memmap"] = {"dir": str(audio_mm), "exists": audio_mm.exists(),
                           "bytes": sum(f.stat().st_size for f in audio_mm.glob("*")
                                        if f.is_file()) if audio_mm.exists() else 0}
    itw_status = root / "sprint1" / "itw_status.json"
    out["itw_status"] = {"path": str(itw_status), "exists": itw_status.exists()}
    if itw_status.exists():
        try:
            out["itw_status"]["value"] = json.loads(itw_status.read_text(encoding="utf-8"))
        except Exception:
            pass
    for name in ("FREEZE_OK", "PAUSE", "SHUTDOWN_OK"):
        out[f"control_{name}"] = cfg.has_control(name)
    return out


def environment_blocking(env: Dict[str, Any]) -> List[str]:
    """The inputs the sprint cannot start without."""
    blocking: List[str] = []
    if not env["manifest"]["exists"]:
        blocking.append("mlitw_meta.csv is missing under the data root")
    if env["parquet_shards"]["count"] == 0:
        blocking.append("no parquet shards under <data>/mlitw/data")
    if not env["split_file"]["exists"]:
        blocking.append(f"split file {cfg.SPLIT_FILE} is missing")
    if not env["feature_blocks"]:
        blocking.append("no feature blocks under <data>/x3/feats (needed by S1-B and later)")
    free = [r.get("free_gb") for r in env["disk"] if isinstance(r.get("free_gb"), (int, float))]
    if free and min(free) < cfg.SYSTEM_FREE_MIN_GB:
        blocking.append(f"free disk {min(free):.1f} GB < the {cfg.SYSTEM_FREE_MIN_GB:g} GB guard")
    return blocking


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X3 sprint 1 preflight (P-0)")
    ap.add_argument("--out", default="preflight.md")
    ap.add_argument("--json", default="logs/sprint1_env.json")
    ap.add_argument("--data", default=None)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else cfg.REPO_ROOT
    data_root = cfg.data_root(args.data)
    cfg.ensure_dirs(root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "x3_cpu_budget").write_text(str(cfg.CPU_BUDGET), encoding="utf-8")

    env: Dict[str, Any] = {
        "ts": st.now_iso(),
        "host": platform.node(),
        "uname": sh(["uname", "-a"]),
        "python": sys.version.split()[0],
        "nproc": os.cpu_count(),
        "loadavg": sh(["cat", "/proc/loadavg"]),
        "free": sh(["free", "-g"]),
        "nvidia_smi": sh(["nvidia-smi"]),
        "df": sh(["df", "-h"]),
        "data_root": str(data_root),
    }
    env.update(manifest_inventory(data_root))
    env["sprint_inputs"] = sprint_inputs(root, data_root)
    env["disk"] = disk_rows([root, data_root, Path("/tmp"), Path("/root/autodl-tmp")
                             if Path("/root/autodl-tmp").exists() else root])
    env["budget"] = {
        "vcpu": cfg.VCPU,
        "cpu_budget": cfg.CPU_BUDGET,
        "ram_limit_gb": cfg.RAM_LIMIT_GB,
        "vram_total_gb": cfg.VRAM_TOTAL_GB,
        "vram_budget_gb": cfg.VRAM_BUDGET_GB,
        "lane_threads": dict(cfg.LANE_THREADS),
        "lane_capacity": dict(cfg.LANE_CAPACITY),
        "lane_nice": dict(cfg.LANE_NICE),
        "ffmpeg_jobs_max": cfg.FFMPEG_JOBS_MAX,
    }
    env["scratch"] = {k: {"path": str(v), "exists": v.exists()} for k, v in cfg.SCRATCH_DIRS.items()}
    env["blocking"] = environment_blocking(env)

    json_path = root / args.json
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(env, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    md = [
        "# X3 sprint 1 preflight (P-0)",
        "",
        f"- generated: {env['ts']}  host: {env['host']}",
        f"- python: {env['python']}  nproc: {env['nproc']}  data root: {data_root}",
        f"- budgets: CPU {cfg.CPU_BUDGET}/{cfg.VCPU} vCPU, RAM <= {cfg.RAM_LIMIT_GB:g} GiB, "
        f"VRAM <= {cfg.VRAM_BUDGET_GB:g} GiB of {cfg.VRAM_TOTAL_GB:g}",
        f"- lane threads: A {cfg.LANE_THREADS['A']} / B {cfg.LANE_THREADS['B']} (nice "
        f"{cfg.LANE_NICE['B']}) / G {cfg.LANE_THREADS['G']} / N {cfg.LANE_THREADS['N']}; "
        f"capacity A{cfg.LANE_CAPACITY['A']} B{cfg.LANE_CAPACITY['B']} "
        f"G{cfg.LANE_CAPACITY['G']} N{cfg.LANE_CAPACITY['N']}",
        "",
        "## Disk",
        "",
        "| path | probed | total GB | free GB |",
        "|---|---|---|---|",
    ]
    for r in env["disk"]:
        md.append(f"| {r.get('path')} | {r.get('probed', '')} | {r.get('total_gb', '')} "
                  f"| {r.get('free_gb', r.get('error', ''))} |")
    md += [
        "",
        "## Inventory",
        "",
        f"- manifest: {env['manifest'].get('path')} exists={env['manifest']['exists']} "
        f"rows={env['manifest'].get('rows', '?')}",
        f"- parquet shards: {env['parquet_shards']['count']} "
        f"({env['parquet_shards']['bytes'] / 1024 ** 3:.2f} GiB)",
        f"- feature blocks: {', '.join(sorted(env['feature_blocks'])) or 'none'}",
        f"- split file: {env['sprint_inputs']['split_file']}",
        f"- audio memmap: {env['sprint_inputs']['audio_memmap']}",
        f"- ITW: {env['sprint_inputs']['itw_status']}",
        f"- controls: FREEZE_OK={env['sprint_inputs']['control_FREEZE_OK']} "
        f"PAUSE={env['sprint_inputs']['control_PAUSE']} "
        f"SHUTDOWN_OK={env['sprint_inputs']['control_SHUTDOWN_OK']}",
        "",
        "## Blocking",
        "",
    ]
    md += [f"- {b}" for b in env["blocking"]] or ["- none"]
    md += ["", "## nvidia-smi", "", "```", env["nvidia_smi"][:4000], "```", "",
           "## free -g", "", "```", env["free"], "```", ""]
    (root / args.out).write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({"blocking": env["blocking"],
                      "cpu_budget": cfg.CPU_BUDGET,
                      "free_gb": [r.get("free_gb") for r in env["disk"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

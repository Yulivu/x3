"""Machine profile and lane budgets for NEW_TASKS 10 sprint 1.

Section 1 of the taskbook fixes the thread budgets, the nice level of lane B, the
cache locations and the disk guard.  Every driver and the scheduler read this one
module so the numbers cannot drift apart.

The machine is 16 vCPU (Xeon Platinum 8470Q) / 80 GiB RAM / vGPU-32GB, matching the
taskbook.  The data disk has ~50 GB free (the taskbook planned for 70), so the disk
guard matters and every stage reports free space before it starts.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- lanes -------------------------------------------------------------------
LANES = ("A", "B", "G", "N")
#: One package at a time per CPU lane; the GPU lane may host two, one of them light.
LANE_CAPACITY: Dict[str, int] = {"A": 1, "B": 1, "G": 2, "N": 1}
#: OMP/MKL/OPENBLAS threads handed to each lane's children (section 1).
LANE_THREADS: Dict[str, int] = {
    "A": int(os.environ.get("X3_THREADS_A", "5")),
    "B": int(os.environ.get("X3_THREADS_B", "3")),
    "G": int(os.environ.get("X3_THREADS_G", "4")),
    "N": 1,
}
LANE_NICE: Dict[str, int] = {"A": 0, "B": 10, "G": 0, "N": 0}

# --- machine ----------------------------------------------------------------
VCPU = int(os.environ.get("X3_VCPU", "16"))
RESERVED_VCPU = 2  # "always leave >= 2 vCPU for the system and the agent" (section 1)
CPU_BUDGET = max(1, VCPU - RESERVED_VCPU)
RAM_LIMIT_GB = float(os.environ.get("X3_RAM_GB", "80"))
VGPU_COUNT = int(os.environ.get("X3_VGPU", "1"))
VRAM_TOTAL_GB = float(os.environ.get("X3_VRAM_GB", "32"))
#: Concurrent GPU memory the sprint is allowed to occupy (section 1: <= 24 GB).
VRAM_BUDGET_GB = float(os.environ.get("X3_VRAM_BUDGET_GB", "24"))
GPU_HEAVY_VRAM_GB = 6.0  # a package tagged heavy takes the whole lane to itself
FFMPEG_JOBS_MAX = 3

# --- storage ----------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("X3_DATA_ROOT", "/root/autodl-tmp/data"))
SYSTEM_FREE_MIN_GB = float(os.environ.get("X3_MIN_FREE_GB", "15"))
SCRATCH_DIRS: Dict[str, Path] = {
    "HF_HOME": Path(os.environ.get("HF_HOME", "/root/autodl-tmp/cache/hf")),
    "TORCH_HOME": Path(os.environ.get("TORCH_HOME", "/root/autodl-tmp/cache/torch")),
    "PIP_CACHE_DIR": Path(os.environ.get("PIP_CACHE_DIR", "/root/autodl-tmp/cache/pip")),
    "TMPDIR": Path(os.environ.get("TMPDIR", "/root/autodl-tmp/tmp")),
}
SPRINT_DIR = REPO_ROOT / "sprint1"
LOG_DIR = REPO_ROOT / "logs" / "sprint1"
CONTROL_DIR = REPO_ROOT / "control"

# --- contracts --------------------------------------------------------------
SPLIT_FILE = "speaker_split_v3.json"
BACKBONES = ("xls-r-2b", "mms-1b")
LIBRARY_SIZES = (10, 40, 160)
DRAW_BASE = 20261015
BOOTSTRAP_B = 200
PROGRESS_EVERY_S = 30 * 60
WATCHDOG_EVERY_S = 5 * 60
IDLE_GRACE_S = 120  # "a lane idle for more than 2 minutes must take work"
TIMEOUT_FACTOR = 1.5
GPU_IDLE_UTIL = 40.0  # percent, 10-minute average

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def lane_env(lane: str, base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment for a package on ``lane``: thread caps plus the cache roots."""
    env = dict(os.environ if base is None else base)
    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r}")
    threads = str(LANE_THREADS[lane])
    for key in THREAD_ENV_KEYS:
        env[key] = threads
    env["TORCH_NUM_THREADS"] = threads
    env["X3_CPU_BUDGET"] = str(CPU_BUDGET)
    env["X3_LANE"] = lane
    env["PYTHONUNBUFFERED"] = "1"
    for name, path in SCRATCH_DIRS.items():
        env.setdefault(name, str(path))
    env.setdefault("X3_DATA_ROOT", str(DATA_ROOT))
    env.setdefault("DATA", str(DATA_ROOT))
    return env


def lane_prefix(lane: str) -> List[str]:
    """``nice -n 10`` for lane B, nothing otherwise."""
    nice = LANE_NICE.get(lane, 0)
    return ["nice", "-n", str(nice)] if nice else []


def ensure_dirs(root: Path = REPO_ROOT) -> List[Path]:
    dirs = [SPRINT_DIR, LOG_DIR, CONTROL_DIR, root / "results", root / "paper_tables",
            root / "paper_figs", root / "export"]
    dirs.extend(SCRATCH_DIRS.values())
    made = []
    for d in dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            made.append(d)
        except OSError:
            pass  # a Windows dry run cannot create /root/...; preflight reports it
    return made


def free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / 1024 ** 3
    except OSError:
        return float("nan")


def control_path(name: str) -> Path:
    """``FREEZE_OK`` / ``PAUSE`` / ``SHUTDOWN_OK`` may sit in either accepted place."""
    for cand in (CONTROL_DIR / name, REPO_ROOT / name):
        if cand.exists():
            return cand
    return CONTROL_DIR / name


def has_control(name: str) -> bool:
    return control_path(name).exists()


def data_root(explicit: Optional[str] = None) -> Path:
    return Path(explicit) if explicit else DATA_ROOT

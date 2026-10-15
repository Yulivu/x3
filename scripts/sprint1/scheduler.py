#!/usr/bin/env python3
"""Work-conserving four-lane scheduler for NEW_TASKS 10 sprint 1 (section 2).

What the taskbook asks for and where it lives here:

* lanes A (5 threads), B (3 threads, ``nice -n 10``), G (GPU, up to two packages),
  N (network) - ``config.LANE_*``;
* work-conserving: a lane idle for more than ``config.IDLE_GRACE_S`` takes the next
  ready package, and if none is ready it takes the first backlog entry of its lane,
  so every lane always has work queued;
* time boxes: ``eta_min * 1.5`` then ``TIMEOUT`` with the partial results kept;
* a watchdog line every 5 minutes into ``util.csv`` (GPU util / memory, load, the
  package on each lane) plus the four rules of section 2;
* a progress line every 30 minutes into ``sprint1_progress.md``;
* every decision is made by this script - the only things that wait for a human are
  ``FREEZE_OK`` (unlocks ``needs_freeze`` packages) and ``SHUTDOWN_OK``.

Usage:
    python -m scripts.sprint1.scheduler --packages packages.yaml --state state.json
    python -m scripts.sprint1.scheduler ... --dry-run     # print the plan, touch nothing
    python -m scripts.sprint1.scheduler ... --once        # one pass, for tests
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

if __package__ in (None, ""):  # direct execution from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sprint1 import config as cfg  # noqa: E402
from scripts.sprint1 import state as st  # noqa: E402


def load_packages(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read the package table; backlog entries get ``backlog=True`` and a low priority."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    packages = [dict(p) for p in (raw.get("packages") or [])]
    backlog = [dict(p, backlog=True) for p in (raw.get("backlog") or [])]
    for i, pkg in enumerate(packages):
        pkg.setdefault("priority", "P3")
        pkg["order"] = i
    for i, pkg in enumerate(backlog):
        pkg.setdefault("priority", "P9")
        pkg["order"] = 1000 + i
    return packages, backlog


def cmd_missing(cmd: str, root: Path) -> Optional[str]:
    """Return the missing script/module name when the command cannot run yet."""
    for token in shlex.split(cmd):
        if token.endswith(".py"):
            if not (root / token).exists():
                return token
        if token.startswith(("scripts.", "src.")):
            rel = Path(*token.split("."))
            if not ((root / rel).with_suffix(".py").exists()
                    or (root / rel / "__init__.py").exists()):
                return token
    return None


def gpu_sample() -> Tuple[Optional[float], Optional[float]]:
    """``(utilisation %, used MB)`` of the first GPU, or ``(None, None)`` off-GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception:
        return None, None
    parts = [p.strip() for p in out.split(",")]
    try:
        return float(parts[0]), float(parts[1])
    except (IndexError, ValueError):
        return None, None


def load_average() -> Optional[float]:
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return None


class Sprint:
    def __init__(self, packages_path: Path, state_path: Path, *, poll: int = 10,
                 dry_run: bool = False, once: bool = False, root: Optional[Path] = None,
                 verbose: bool = True) -> None:
        self.root = Path(root or cfg.REPO_ROOT)
        self.packages_path = Path(packages_path)
        self.packages, self.backlog = load_packages(packages_path)
        self.by_id = {p["id"]: p for p in self.packages + self.backlog}
        self.state_path = Path(state_path)
        self.state = st.load_state(self.state_path)
        self.poll = int(poll)
        self.dry_run = dry_run
        self.once = once
        self.verbose = verbose
        self.started_at = time.time()
        self.running: Dict[str, Dict[str, Any]] = {}
        self.retired: Dict[str, str] = {}
        self.just_finished: List[str] = []
        self.last_util = 0.0
        self.last_progress = 0.0
        self.util_path = self.root / "util.csv"
        self.progress_path = self.root / "sprint1_progress.md"
        self.reasons: Dict[str, str] = {}

    # --- plumbing ---------------------------------------------------------
    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[H{st.elapsed_h(self.started_at):5.2f}] {msg}", flush=True)

    def hours(self) -> float:
        return st.elapsed_h(self.started_at)

    def busy(self) -> Dict[str, int]:
        counts = {lane: 0 for lane in cfg.LANES}
        for info in self.running.values():
            counts[info["lane"]] += 1
        return counts

    def lane_free(self, lane: str) -> bool:
        return self.busy()[lane] < cfg.LANE_CAPACITY[lane]

    def gpu_free_for(self, pkg: Dict[str, Any]) -> bool:
        """A heavy GPU package needs the whole lane; two light ones may share it."""
        g_running = [i for i in self.running.values() if i["lane"] == "G"]
        if len(g_running) >= cfg.LANE_CAPACITY["G"]:
            return False
        if pkg.get("heavy"):
            return not g_running
        if any(i["pkg"].get("heavy") for i in g_running):
            return False
        vram = sum(float(i["pkg"].get("vram_gb") or 0.0) for i in g_running)
        return vram + float(pkg.get("vram_gb") or 0.0) <= cfg.VRAM_BUDGET_GB

    # --- state ------------------------------------------------------------
    def row(self, pid: str) -> Dict[str, Any]:
        return st.entry(self.state, pid)

    def persist(self) -> None:
        st.save_state(self.state_path, self.state)

    def mark_blocked(self) -> None:
        for pkg in self.packages + self.backlog:
            pid = pkg["id"]
            if pid in self.running or self.row(pid)["status"] not in st.STARTABLE:
                continue
            bad = st.blocked_by_failed(pkg, self.state)
            if bad:
                st.mark(self.state, pid, "BLOCKED", note=f"dependency {bad} did not finish")
                self.retired[pid] = "BLOCKED"

    def missing_scripts(self) -> List[str]:
        out = []
        for pkg in self.packages + self.backlog:
            if self.row(pkg["id"])["status"] in st.STARTABLE:
                miss = cmd_missing(pkg.get("cmd", ""), self.root)
                if miss:
                    out.append(f"{pkg['id']}:{miss}")
        return out

    # --- selection --------------------------------------------------------
    def ready(self, lane: str) -> Optional[Dict[str, Any]]:
        if cfg.has_control("PAUSE"):
            self.reasons["PAUSE"] = "control file PAUSE is present"
            return None
        cands = sorted((p for p in self.packages if p["lane"] == lane),
                       key=lambda p: (p["priority"], p["order"]))
        for pkg in cands:
            pid = pkg["id"]
            if pid in self.running or pid in self.retired:
                continue
            if self.row(pid)["status"] not in st.STARTABLE:
                continue
            if pkg.get("needs_freeze") and not cfg.has_control("FREEZE_OK"):
                self.reasons[pid] = "waiting for FREEZE_OK"
                continue
            ok, dep = st.deps_satisfied(pkg, self.state)
            if not ok:
                self.reasons[pid] = f"waiting for {dep}"
                continue
            if lane == "G" and not self.gpu_free_for(pkg):
                self.reasons[pid] = "GPU lane busy or over the VRAM budget"
                continue
            return pkg
        return None

    def from_backlog(self, lane: str) -> Optional[Dict[str, Any]]:
        """Take the first backlog entry of ``lane`` whose dependencies are satisfied."""
        if cfg.has_control("PAUSE"):
            return None
        cands = sorted((p for p in self.backlog if p["lane"] == lane),
                       key=lambda p: (p["priority"], p["order"]))
        for pkg in cands:
            pid = pkg["id"]
            if pid in self.running or pid in self.retired:
                continue
            if self.row(pid)["status"] not in st.STARTABLE:
                continue
            ok, _ = st.deps_satisfied(pkg, self.state)
            if not ok:
                continue
            if lane == "G" and not self.gpu_free_for(pkg):
                continue
            if pkg not in self.packages:
                self.packages.append(pkg)   # it is queued for real from now on
            return pkg
        return None

    # --- process control --------------------------------------------------
    def start(self, pkg: Dict[str, Any], lane: str) -> None:
        pid = pkg["id"]
        row = self.row(pid)
        miss = cmd_missing(pkg.get("cmd", ""), self.root)
        if miss:
            st.mark(self.state, pid, "MISSING", note=f"{miss} not written yet")
            self.retired[pid] = "MISSING"
            self.persist()
            self.log(f"start {pid}: MISSING ({miss})")
            return
        if self.dry_run:
            self.log(f"dry-run would start {pid} on lane {lane}: {pkg.get('cmd')}")
            self.retired[pid] = "DRY_RUN"
            return
        prefix = " ".join(cfg.lane_prefix(lane))
        shell_cmd = f"{prefix} {pkg['cmd']}" if prefix else pkg["cmd"]
        kwargs: Dict[str, Any] = dict(shell=True, cwd=str(self.root), env=cfg.lane_env(lane))
        if os.name == "posix":
            kwargs["preexec_fn"] = os.setsid
        cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
        handle = open(cfg.LOG_DIR / f"{pid}.log", "ab", buffering=0)
        kwargs["stdout"] = handle
        kwargs["stderr"] = subprocess.STDOUT
        proc = subprocess.Popen(shell_cmd, **kwargs)  # noqa: S602 - commands come from packages.yaml
        row["attempts"] = int(row.get("attempts") or 0) + 1
        row["started"].append(st.now_iso())
        st.mark(self.state, pid, "RUNNING", note=f"lane {lane} pid {proc.pid}",
                eta_min=pkg.get("eta_min"))
        self.persist()
        self.running[pid] = {"proc": proc, "handle": handle, "lane": lane,
                             "pkg": pkg, "t0": time.time()}
        self.log(f"start {pid} on lane {lane} (pid {proc.pid}, eta {pkg.get('eta_min')}m)")

    def kill(self, pid: str, why: str) -> None:
        info = self.running.get(pid)
        if not info:
            return
        proc = info["proc"]
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:  # pragma: no cover - Windows only runs dry runs
                proc.terminate()
        except Exception:
            pass
        deadline = time.time() + 30
        while proc.poll() is None and time.time() < deadline:
            time.sleep(1)
        if proc.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:  # pragma: no cover
                    proc.kill()
            except Exception:
                pass
        self.log(f"{pid}: killed ({why})")

    def reap(self) -> None:
        for pid, info in list(self.running.items()):
            proc = info["proc"]
            rc = proc.poll()
            if rc is None:
                continue
            pkg = info["pkg"]
            self.running.pop(pid)
            try:
                info["handle"].close()
            except Exception:
                pass
            row = self.row(pid)
            row["ended"].append(st.now_iso())
            row["rc"] = int(rc)
            ok, missing = st.artifacts_ok(pkg.get("artifacts") or [], self.root)
            st.record_artifacts(self.state, pid, pkg.get("artifacts") or [], self.root)
            if rc == 0 and ok:
                status = pkg.get("done_status") or "DONE"
                st.mark(self.state, pid, status, note=pkg.get("note", ""))
                self.retired[pid] = status
                self.just_finished.append(pid)
                self.log(f"finish {pid}: {status}")
            elif rc == 0:
                st.mark(self.state, pid, "FAILED",
                        note="exit 0 but artifacts missing: " + ", ".join(missing))
                self.retired[pid] = "FAILED"
                self.log(f"finish {pid}: FAILED (artifacts missing: {', '.join(missing)})")
            else:
                attempts = int(row.get("attempts") or 1)
                retries = int(pkg.get("retries") or 0)
                if attempts <= retries:
                    st.mark(self.state, pid, "TODO", note=f"rc={rc}; retry {attempts}/{retries}")
                    self.log(f"reap {pid}: rc={rc}, retrying")
                else:
                    st.mark(self.state, pid, "FAILED", note=f"rc={rc}")
                    self.retired[pid] = "FAILED"
                    self.log(f"finish {pid}: FAILED (rc={rc})")
            self.persist()

    def timebox(self) -> None:
        for pid, info in list(self.running.items()):
            eta = float(info["pkg"].get("eta_min") or 0)
            if not eta:
                continue
            if time.time() - info["t0"] > eta * cfg.TIMEOUT_FACTOR * 60:
                self.kill(pid, f"time box {eta:g}m x {cfg.TIMEOUT_FACTOR}")
                self.reap()
                st.record_artifacts(self.state, pid, info["pkg"].get("artifacts") or [], self.root)
                st.mark(self.state, pid, "TIMEOUT",
                        note=f"exceeded {cfg.TIMEOUT_FACTOR:g} x eta {eta:g}m; partial results kept")
                self.retired[pid] = "TIMEOUT"
                self.persist()

    def drain(self, timeout: float = 300.0) -> None:
        """Wait for the packages started so far; used by ``--once`` and by tests."""
        deadline = time.time() + float(timeout)
        while self.running and time.time() < deadline:
            time.sleep(0.5)
            self.reap()

    # --- reporting --------------------------------------------------------
    def write_util(self, force: bool = False, note: str = "") -> None:
        if not force and time.time() - self.last_util < cfg.WATCHDOG_EVERY_S:
            return
        self.last_util = time.time()
        gpu_util, gpu_mem = gpu_sample()
        load1 = load_average()
        if not self.util_path.exists():
            self.util_path.write_text(
                "ts,elapsed_h,gpu_util,gpu_mem_mb,load1,lane_A,lane_B,lane_G,lane_N,note\n",
                encoding="utf-8")
        current = {lane: next((p for p, i in self.running.items() if i["lane"] == lane), "")
                   for lane in cfg.LANES}
        row = ",".join([
            st.now_iso(),
            f"{self.hours():.2f}",
            "" if gpu_util is None else f"{gpu_util:.0f}",
            "" if gpu_mem is None else f"{gpu_mem:.0f}",
            "" if load1 is None else f"{load1:.2f}",
            current["A"], current["B"], current["G"], current["N"],
            (note or "; ".join(f"{k}={v}" for k, v in sorted(self.reasons.items())))
            .replace(",", " ")[:240],
        ])
        with self.util_path.open("a", encoding="utf-8") as fh:
            fh.write(row + "\n")
        self.reasons.clear()

    def write_progress(self, force: bool = False, note: str = "") -> None:
        if not force and time.time() - self.last_progress < cfg.PROGRESS_EVERY_S:
            return
        self.last_progress = time.time()
        busy = self.busy()
        done = sorted(p for p, r in self.state.items() if r.get("status") in st.SATISFIED)
        terminal = sorted(p for p, r in self.state.items() if r.get("status") in st.TERMINAL)
        lines = [
            f"## [H+{self.hours():.2f}] {st.now_iso()}",
            "",
            f"- 已用时 {self.hours():.2f} h",
            "- 各车道当前包: " + ", ".join(
                f"{lane}={next((p for p, i in self.running.items() if i['lane'] == lane), '-')}"
                for lane in cfg.LANES),
            "- 刚完成: " + (", ".join(self.just_finished) or "-"),
            f"- 已终态: {', '.join(terminal) or '-'} (DONE {len(done)})",
            "- 下一步: " + ", ".join(
                f"{lane}->{(self.ready(lane) or {}).get('id', 'backlog')}"
                for lane in cfg.LANES if busy[lane] < cfg.LANE_CAPACITY[lane]),
            "- 阻塞: " + (", ".join(f"{k}:{v}" for k, v in sorted(self.reasons.items())[:6]) or "-"),
            f"- 遥测: {note or '见 util.csv'}",
            "",
        ]
        self.just_finished = []
        with self.progress_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    # --- watchdog (section 2 rules i-iv) ----------------------------------
    def watchdog(self) -> str:
        gpu_util, _ = gpu_sample()
        load1 = load_average()
        busy = self.busy()
        notes: List[str] = []
        g_running = busy["G"] > 0
        if gpu_util is not None and gpu_util < cfg.GPU_IDLE_UTIL:
            if not g_running:
                nxt = self.ready("G") or self.from_backlog("G")
                if nxt:
                    self.start(nxt, "G")
                    notes.append(f"rule i: GPU idle, started {nxt['id']}")
                else:
                    notes.append("rule i: GPU idle but no ready/backlog G package")
            else:
                note = ("rule ii: GPU busy but <40% - check data loading / batch size / "
                        "CPU contention")
                notes.append(note)
                with self.progress_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"- [{st.now_iso()}] {note}\n")
        if load1 is not None and load1 < 0.5 * cfg.VCPU:
            started = []
            for lane in ("A", "B"):
                if self.lane_free(lane):
                    nxt = self.ready(lane) or self.from_backlog(lane)
                    if nxt:
                        self.start(nxt, lane)
                        started.append(f"{lane}:{nxt['id']}")
            notes.append("rule iii: A+B under half the CPU - " +
                         (", ".join(started) if started else "nothing ready"))
        if not notes:
            notes.append("rule iv: watchdog found nothing to change")
        return "; ".join(notes)

    # --- main loop --------------------------------------------------------
    def plan(self) -> List[str]:
        rows = []
        for pkg in self.packages:
            miss = cmd_missing(pkg.get("cmd", ""), self.root)
            dep_ok, dep = st.deps_satisfied(pkg, self.state)
            flag = ("MISSING:" + str(miss)) if miss else ("ready" if dep_ok else f"wait:{dep}")
            rows.append(f"{pkg['id']:<6} lane {pkg['lane']} eta {pkg.get('eta_min')}m "
                        f"prio {pkg['priority']} {flag}")
        return rows

    def loop(self) -> int:
        cfg.ensure_dirs(self.root)
        if not self.progress_path.exists():
            self.progress_path.write_text(
                "# X3 sprint 1 progress (NEW_TASKS 10)\n\n"
                "每 30 分钟由 `scripts/sprint1/scheduler.py` 追加一行；"
                "利用率明细见 `util.csv`。\n\n", encoding="utf-8")
        self.log(f"scheduler up: {len(self.packages)} packages, {len(self.backlog)} backlog, "
                 f"lanes {cfg.LANE_CAPACITY}")
        while True:
            self.reap()
            self.mark_blocked()
            self.timebox()
            for lane in cfg.LANES:
                if not self.lane_free(lane):
                    continue
                pkg = self.ready(lane) or self.from_backlog(lane)
                if pkg:
                    self.start(pkg, lane)
            note = "" if self.dry_run else self.watchdog()
            self.write_util(note=note)
            self.write_progress(note=note)
            self.persist()
            if self.once:
                break
            if st.all_terminal(self.state, [p["id"] for p in self.packages]):
                if not (self.root / "ALL_DONE").exists():
                    (self.root / "ALL_DONE").write_text(
                        f"all sprint packages terminal at {st.now_iso()}\n", encoding="utf-8")
                    self.log("ALL_DONE written; continuing with the backlog queue")
                    continue
                backlog_open = any(self.row(p["id"])["status"] in st.STARTABLE
                                   for p in self.backlog)
                if not backlog_open and not self.running:
                    (self.root / "SPRINT_WAIT").write_text(
                        "backlog exhausted; waiting for FREEZE_OK / SHUTDOWN_OK / PAUSE / "
                        "NEW_TASKS.md\n", encoding="utf-8")
                    self.log("backlog exhausted - entering wait mode (no shutdown)")
                    if cfg.has_control("SHUTDOWN_OK"):
                        self.log("SHUTDOWN_OK honoured; nothing left to run")
                        return 0
                    time.sleep(max(30, self.poll))
                    continue
                if cfg.has_control("SHUTDOWN_OK") and not self.running:
                    self.log("SHUTDOWN_OK honoured; nothing left to run")
                    return 0
            time.sleep(max(2, self.poll))
        return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X3 sprint 1 scheduler")
    ap.add_argument("--packages", default="packages.yaml")
    ap.add_argument("--state", default="state.json")
    ap.add_argument("--poll", type=int, default=10)
    ap.add_argument("--root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--plan", action="store_true",
                    help="print the package table with readiness and exit")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve() if args.root else cfg.REPO_ROOT
    sprint = Sprint(Path(args.packages), Path(args.state), poll=args.poll,
                    dry_run=args.dry_run, once=args.once, root=root)
    if args.plan:
        for line in sprint.plan():
            print(line)
        return 0
    return sprint.loop()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

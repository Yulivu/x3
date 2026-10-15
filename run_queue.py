#!/usr/bin/env python3
"""X3 queue orchestrator (TASKBOOK section 3).

Reads ``queue.yaml``, schedules the work packages on two lanes (``G`` = GPU /
network, ``C`` = CPU) while respecting dependencies, and keeps the run alive
without supervision:

* a finished work package (all ``outputs`` present + its ``DONE`` marker) is skipped;
* a failed work package is retried once, then marked ``FAILED`` and the run continues;
* work packages that depend on a ``FAILED`` one are marked ``BLOCKED``;
* a work package whose command points at a script that does not exist yet is marked
  ``MISSING``; ``MISSING`` never counts as progress and blocks ``ALL_DONE``;
* ``PAUSE`` stops new starts, ``FREEZE_OK`` unlocks the ``WF`` work package;
* a heartbeat line is appended to ``progress.log`` every 30 minutes and
  ``CHECKPOINT_REPORT.md`` is refreshed every 3 hours;
* once every work package is terminal, ``ALL_DONE`` is written and the process enters
  wait mode: it polls the control files and shuts the machine down only when
  ``SHUTDOWN_OK`` exists and no other ``probe-*`` tmux session is alive.

The command executed for a work package is a plain shell command run from the repo
root, so ``scripts/wN_*.py`` can be written in any language.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import yaml

TIER_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
LANES = ("G", "C")
# NEW_TASKS 05 permits two CPU work packages while a GPU task is active.  A
# GPU/network job remains exclusive so feature extraction is never contended.
LANE_CAPACITY = {"G": 1, "C": 2}
CONTROL_DEPS = {"FREEZE_OK", "PAUSE", "SHUTDOWN_OK"}
HEARTBEAT_SEC = 30 * 60
CHECKPOINT_SEC = 3 * 3600
POLL_SEC = 10
MAX_ATTEMPTS = 2  # one attempt + one retry (TASKBOOK hard rule 10)


def _now() -> float:
    return time.time()


def _stamp(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_now() if ts is None else ts))


def parse_lanes(lane: str) -> Set[str]:
    """``"C"`` -> ``{"C"}``; ``"G+C"`` -> ``{"G", "C"}``."""
    out = {part.strip() for part in str(lane).replace(",", "+").split("+") if part.strip()}
    unknown = out - set(LANES)
    if unknown:
        raise ValueError(f"unknown lane(s) {sorted(unknown)} in {lane!r}")
    return out


def load_queue(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    wps = data.get("wps") or data.get("work_packages") or []
    if not wps:
        raise ValueError(f"{path} contains no work packages")
    seen: Set[str] = set()
    for wp in wps:
        for key in ("id", "cmd"):
            if key not in wp:
                raise ValueError(f"work package missing {key!r}: {wp}")
        if wp["id"] in seen:
            raise ValueError(f"duplicate work package id {wp['id']!r}")
        seen.add(wp["id"])
        wp.setdefault("tier", "P3")
        wp.setdefault("lane", "C")
        wp.setdefault("deps", [])
        wp.setdefault("outputs", [])
        wp.setdefault("est_h", 0.0)
        wp.setdefault("gating", True)  # gating=False -> backlog; never blocks ALL_DONE
    return wps


def cmd_missing(cmd: str, root: Path) -> Optional[str]:
    """Return the offending token when ``cmd`` calls a local script that does not exist."""
    for token in str(cmd).split():
        bare = token.strip("\"'")
        if bare.endswith((".py", ".sh", ".bash", ".r")) and not os.path.isabs(bare):
            if not (root / bare).exists():
                return bare
    return None


class Orchestrator:
    def __init__(
        self,
        root: Path,
        wps: Sequence[Dict[str, Any]],
        *,
        poll_sec: float = POLL_SEC,
        heartbeat_sec: float = HEARTBEAT_SEC,
        checkpoint_sec: float = CHECKPOINT_SEC,
        git_sync: bool = True,
        allow_shutdown: bool = True,
        once_mode: bool = False,
    ) -> None:
        self.root = Path(root)
        self.wps = {wp["id"]: wp for wp in wps}
        self.poll_sec = poll_sec
        self.heartbeat_sec = heartbeat_sec
        self.checkpoint_sec = checkpoint_sec
        self.git_sync = git_sync
        self.allow_shutdown = allow_shutdown
        self.once_mode = once_mode
        self.order: Dict[str, int] = {wp["id"]: i for i, wp in enumerate(wps)}

        self.logs = self.root / "logs"
        (self.logs / "done").mkdir(parents=True, exist_ok=True)
        (self.logs / "wps").mkdir(parents=True, exist_ok=True)
        self.state_path = self.logs / "queue_state.json"
        self.progress_path = self.root / "progress.log"
        self.checkpoint_path = self.root / "CHECKPOINT_REPORT.md"
        self.all_done_path = self.root / "ALL_DONE"

        self.state: Dict[str, Dict[str, Any]] = self._load_state()
        self.running: Dict[str, Dict[str, Any]] = {}
        self.last_heartbeat = 0.0
        self.last_checkpoint = 0.0

    # ---------------------------------------------------------------- state
    def _load_state(self) -> Dict[str, Dict[str, Any]]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_state(self) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def _entry(self, wp_id: str) -> Dict[str, Any]:
        return self.state.setdefault(
            wp_id, {"status": "PENDING", "attempts": 0, "started": [], "ended": [], "rc": None}
        )

    # ------------------------------------------------------------ controls
    def paused(self) -> bool:
        return (self.root / "PAUSE").exists()

    def freeze_ok(self) -> bool:
        return (self.root / "FREEZE_OK").exists()

    def shutdown_ok(self) -> bool:
        return (self.root / "SHUTDOWN_OK").exists()

    # ------------------------------------------------------------- helpers
    def outputs_present(self, wp: Dict[str, Any]) -> bool:
        outs = wp.get("outputs") or []
        return all((self.root / str(o)).exists() for o in outs) if outs else False

    def done_marker(self, wp_id: str) -> Path:
        return self.logs / "done" / f"{wp_id}.DONE"

    def is_finished(self, wp: Dict[str, Any]) -> bool:
        return self.done_marker(wp["id"]).exists() and self.outputs_present(wp)

    def deps_met(self, wp: Dict[str, Any]) -> bool:
        for dep in wp.get("deps", []):
            if dep in CONTROL_DEPS:
                if dep == "FREEZE_OK" and not self.freeze_ok():
                    return False
                continue
            if "/" in dep or dep.endswith((".json", ".csv", ".md")):
                if not (self.root / dep).exists():
                    return False
                continue
            entry = self.state.get(dep, {})
            if entry.get("status") not in ("DONE",):
                return False
        return True

    def blocked_by_failed_dep(self, wp: Dict[str, Any]) -> Optional[str]:
        for dep in wp.get("deps", []):
            if dep in CONTROL_DEPS or "/" in dep or dep.endswith((".json", ".csv", ".md")):
                continue
            if self.state.get(dep, {}).get("status") in ("FAILED", "BLOCKED", "MISSING"):
                return dep
        return None

    def lane_busy(self, lanes: Set[str]) -> bool:
        return any(lanes & info["lanes"] for info in self.running.values())

    def pick_runnable(self) -> List[Dict[str, Any]]:
        """Deterministic scheduling: highest priority tier first, then queue order."""
        ordered = sorted(
            self.wps.values(),
            key=lambda wp: (TIER_ORDER.get(str(wp["tier"]), 99), self.order[wp["id"]]),
        )
        chosen: List[Dict[str, Any]] = []
        busy = {lane: 0 for lane in LANES}
        for lanes in (info["lanes"] for info in self.running.values()):
            for lane in lanes:
                busy[lane] += 1
        for wp in ordered:
            wid = wp["id"]
            entry = self._entry(wid)
            if wid in self.running or entry["status"] in ("DONE", "FAILED", "BLOCKED"):
                continue
            if self.is_finished(wp):
                continue
            if not self.deps_met(wp):
                continue
            if self.blocked_by_failed_dep(wp):
                continue
            lanes = parse_lanes(wp["lane"])
            if any(busy[lane] >= LANE_CAPACITY[lane] for lane in lanes):
                continue
            if cmd_missing(wp["cmd"], self.root):
                continue
            chosen.append(wp)
            for lane in lanes:
                busy[lane] += 1
        return chosen

    # -------------------------------------------------------------- running
    def start(self, wp: Dict[str, Any]) -> None:
        wid = wp["id"]
        entry = self._entry(wid)
        log_path = self.logs / "wps" / f"{wid}.log"
        handle = open(log_path, "a", encoding="utf-8")
        handle.write(f"\n===== {_stamp()} attempt {entry['attempts'] + 1} :: {wp['cmd']}\n")
        handle.flush()
        proc = subprocess.Popen(
            wp["cmd"],
            shell=True,
            cwd=str(self.root),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        entry["status"] = "RUNNING"
        entry["attempts"] += 1
        entry["started"].append(_now())
        self.running[wid] = {"proc": proc, "handle": handle, "lanes": parse_lanes(wp["lane"])}
        self._save_state()
        self.log(f"start {wid} (attempt {entry['attempts']})")

    def reap(self) -> None:
        for wid, info in list(self.running.items()):
            proc = info["proc"]
            if proc.poll() is None:
                continue
            info["handle"].close()
            rc = int(proc.returncode)
            entry = self._entry(wid)
            entry["rc"] = rc
            entry["ended"].append(_now())
            if rc == 0 and self.outputs_present(self.wps[wid]):
                self.done_marker(wid).parent.mkdir(parents=True, exist_ok=True)
                self.done_marker(wid).write_text(f"{_stamp()} rc=0\n", encoding="utf-8")
                entry["status"] = "DONE"
                self.log(f"done {wid}")
                self.git_sync_commit(wid)
            elif entry["attempts"] < MAX_ATTEMPTS:
                entry["status"] = "PENDING"
                self.log(f"retry {wid} (rc={rc})")
            else:
                entry["status"] = "FAILED"
                self.log(f"FAILED {wid} (rc={rc}, {entry['attempts']} attempts)")
            del self.running[wid]
            self._save_state()

    def mark_missing(self) -> List[str]:
        missing = []
        for wp in self.wps.values():
            wid = wp["id"]
            entry = self._entry(wid)
            if entry["status"] in ("DONE", "FAILED", "BLOCKED", "MISSING", "RUNNING"):
                continue
            token = cmd_missing(wp["cmd"], self.root)
            if token:
                entry["status"] = "MISSING"
                entry["missing"] = token
                missing.append(f"{wid}:{token}")
        if missing:
            self._save_state()
        return missing

    def mark_blocked(self) -> None:
        changed = True
        while changed:
            changed = False
            for wp in self.wps.values():
                wid = wp["id"]
                entry = self._entry(wid)
                if entry["status"] in ("DONE", "FAILED", "BLOCKED", "MISSING", "RUNNING"):
                    continue
                dep = self.blocked_by_failed_dep(wp)
                if dep:
                    entry["status"] = "BLOCKED"
                    entry["blocked_by"] = dep
                    changed = True
                    self.log(f"BLOCKED {wid} (dep {dep})")
        self._save_state()

    # ------------------------------------------------------------- logging
    def log(self, message: str) -> None:
        line = f"[{_stamp()}] {message}"
        print(line, flush=True)

    def _gpu(self) -> str:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0]
        except Exception:  # pragma: no cover - depends on the host
            pass
        return "n/a"

    def _disk(self) -> str:
        try:
            out = subprocess.run(["df", "-h", str(self.root)], capture_output=True, text=True, timeout=20)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[-1].split()[-2]
        except Exception:  # pragma: no cover - depends on the host
            pass
        return "n/a"

    def heartbeat(self) -> None:
        running = ",".join(sorted(self.running)) or "idle"
        counts: Dict[str, int] = {}
        for wp in self.wps:
            status = self._entry(wp)["status"]
            counts[status] = counts.get(status, 0) + 1
        summary = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        line = f"{_stamp()} | {running} | GPU {self._gpu()} | disk {self._disk()} | {summary}"
        with open(self.progress_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self.last_heartbeat = _now()

    def checkpoint(self) -> None:
        script = self.root / "scripts" / "make_report.py"
        if script.exists():
            try:
                subprocess.run(
                    [sys.executable, str(script)],
                    cwd=str(self.root),
                    capture_output=True,
                    timeout=600,
                )
                self.last_checkpoint = _now()
                return
            except Exception as exc:  # pragma: no cover
                self.log(f"make_report.py failed: {exc}")
        lines = [f"# X3 checkpoint ({_stamp()})", ""]
        lines.append("| WP | tier | lane | status | attempts |")
        lines.append("|---|---|---|---|---|")
        for wp in self.wps.values():
            entry = self._entry(wp["id"])
            lines.append(
                f"| {wp['id']} | {wp['tier']} | {wp['lane']} | {entry['status']} | {entry['attempts']} |"
            )
        self.checkpoint_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.last_checkpoint = _now()

    def git_sync_commit(self, wp_id: str) -> None:
        if not self.git_sync or not (self.root / ".git").exists():
            return
        try:
            subprocess.run(["git", "add", "-A"], cwd=str(self.root), capture_output=True, timeout=300)
            commit = subprocess.run(
                ["git", "commit", "-m", f"{wp_id}: work package complete"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if commit.returncode != 0 and "nothing to commit" not in (commit.stdout + commit.stderr):
                self.log(f"git commit for {wp_id} failed: {commit.stderr.strip()[:200]}")
                return
            push = subprocess.run(
                ["git", "push", "origin", "x3/main"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=900,
            )
            if push.returncode != 0:
                self.log(f"git push for {wp_id} failed: {push.stderr.strip()[:200]}")
        except Exception as exc:  # pragma: no cover
            self.log(f"git sync for {wp_id} raised: {exc}")

    # ------------------------------------------------------------- terminal
    def terminal_statuses(self) -> Dict[str, str]:
        return {wp_id: self._entry(wp_id)["status"] for wp_id in self.wps}

    def gating_ids(self) -> List[str]:
        return [wp["id"] for wp in self.wps.values() if wp.get("gating", True)]

    def all_terminal(self) -> bool:
        statuses = self.terminal_statuses()
        return all(statuses[wid] in ("DONE", "FAILED", "BLOCKED") for wid in self.gating_ids())

    def any_missing(self) -> bool:
        statuses = self.terminal_statuses()
        return any(statuses[wid] == "MISSING" for wid in self.gating_ids())

    def other_probe_sessions(self) -> List[str]:
        try:
            out = subprocess.run(
                ["tmux", "ls", "-F", "#{session_name}"], capture_output=True, text=True, timeout=20
            )
        except Exception:
            return []
        names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return [n for n in names if n.startswith("probe-")]

    def write_all_done(self) -> None:
        if not self.all_done_path.exists():
            self.all_done_path.write_text(f"{_stamp()}\n", encoding="utf-8")
            self.log("ALL_DONE written; entering wait mode")
            self.git_sync_commit("ALL_DONE")

    def wait_mode(self) -> None:
        """Everything is terminal: poll the control files until shutdown is allowed."""
        self.write_all_done()
        while True:
            self.reap()
            self.mark_blocked()
            self.mark_missing()
            if not self.paused():
                for wp in self.pick_runnable():
                    self.start(wp)
            self.heartbeat()
            self.checkpoint()
            if self.shutdown_ok() and self.allow_shutdown:
                probes = self.other_probe_sessions()
                if probes:
                    self.log(f"SHUTDOWN_OK but probe sessions alive: {probes}; not shutting down")
                else:
                    self.log("SHUTDOWN_OK and no probe sessions -> shutting down")
                    subprocess.run(["shutdown", "-h", "now"], capture_output=True)
                    return
            time.sleep(max(self.poll_sec, 60))

    # ----------------------------------------------------------- main loop
    def heartbeat_if_due(self) -> None:
        if _now() - self.last_heartbeat >= self.heartbeat_sec:
            self.heartbeat()

    def checkpoint_if_due(self) -> None:
        if _now() - self.last_checkpoint >= self.checkpoint_sec:
            self.checkpoint()

    def run_once(self) -> None:
        self.reap()
        self.mark_blocked()
        missing = self.mark_missing()
        if missing:
            self.log(f"missing commands (write the scripts, then re-run): {missing}")
        if self.all_terminal():
            if self.any_missing():
                self.log("all terminal but some commands are MISSING; not writing ALL_DONE")
            else:
                self.write_all_done()
                if not self.once_mode:
                    self.wait_mode()
                    return
        elif not self.paused():
            for wp in self.pick_runnable():
                self.start(wp)
        self.heartbeat_if_due()
        self.checkpoint_if_due()

    def run(self) -> None:
        self.log(f"queue root={self.root} wps={len(self.wps)}")
        self.last_heartbeat = 0.0
        self.last_checkpoint = 0.0
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:  # pragma: no cover
                self.log("interrupted; leaving markers in place for resume")
                return
            time.sleep(self.poll_sec)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 queue orchestrator")
    parser.add_argument("--queue", default="queue.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--poll", type=float, default=POLL_SEC)
    parser.add_argument("--heartbeat", type=float, default=HEARTBEAT_SEC)
    parser.add_argument("--checkpoint", type=float, default=CHECKPOINT_SEC)
    parser.add_argument("--no-git", action="store_true", help="skip commit/push after each work package")
    parser.add_argument("--no-shutdown", action="store_true", help="never call shutdown in wait mode")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--once", action="store_true", help="single scheduling pass (used by tests)")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    queue_path = Path(args.queue)
    if not queue_path.is_absolute():
        queue_path = root / queue_path
    wps = load_queue(queue_path)

    if args.dry_run:
        for wp in wps:
            missing = cmd_missing(wp["cmd"], root)
            print(
                f"{wp['id']:<5} {wp['tier']:<3} {wp['lane']:<3} deps={','.join(wp['deps']) or '-':<12} "
                f"est={wp['est_h']}h missing={missing or '-'}"
            )
        return 0

    orch = Orchestrator(
        root,
        wps,
        poll_sec=args.poll,
        heartbeat_sec=args.heartbeat,
        checkpoint_sec=args.checkpoint,
        git_sync=not args.no_git,
        allow_shutdown=not args.no_shutdown,
        once_mode=args.once,
    )
    if args.once:
        orch.run_once()
    else:
        orch.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

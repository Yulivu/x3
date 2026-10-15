"""``state.json``: the only contract the sprint resume logic is allowed to read.

Section 2 is explicit: a package counts as finished when its **actual artifacts**
are on disk, not when its declared file name matches something.  So every entry
records the artifacts it produced (size + mtime) next to the status.

Statuses: ``TODO`` (never started), ``RUNNING``, ``DONE``, ``DONE_ALT`` (finished
with a documented fallback), ``FAILED``, ``BLOCKED`` (a dependency did not
finish), ``TIMEOUT``, ``MISSING`` (the command's script does not exist yet) and
``PAUSED`` (parked by the load rule, not a timeout).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TERMINAL = ("DONE", "DONE_ALT", "FAILED", "BLOCKED", "TIMEOUT", "MISSING")
SATISFIED = ("DONE", "DONE_ALT")
STARTABLE = ("TODO", "PAUSED")


def load_state(path: Path) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def save_state(path: Path, state: Dict[str, Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def entry(state: Dict[str, Dict[str, Any]], pid: str) -> Dict[str, Any]:
    return state.setdefault(
        pid,
        {"status": "TODO", "attempts": 0, "started": [], "ended": [],
         "rc": None, "artifacts": {}, "note": ""},
    )


def mark(state: Dict[str, Dict[str, Any]], pid: str, status: str, **extra: Any) -> Dict[str, Any]:
    row = entry(state, pid)
    row["status"] = status
    row.update(extra)
    return row


def record_artifacts(state: Dict[str, Dict[str, Any]], pid: str, paths: Iterable[str],
                     root: Path) -> Dict[str, Dict[str, Any]]:
    """Stat every declared artifact; the result goes into ``state.json``."""
    root = Path(root)
    found: Dict[str, Dict[str, Any]] = {}
    for rel in paths:
        ap = Path(rel)
        full = ap if ap.is_absolute() else root / ap
        if full.exists():
            st = full.stat()
            found[str(rel)] = {"bytes": int(st.st_size), "mtime": int(st.st_mtime)}
        else:
            found[str(rel)] = {"bytes": 0, "mtime": 0}
    entry(state, pid)["artifacts"] = found
    return found


def artifacts_ok(paths: Iterable[str], root: Path, min_bytes: int = 1) -> Tuple[bool, List[str]]:
    """True when every declared artifact exists and is non-empty."""
    root = Path(root)
    missing: List[str] = []
    for rel in paths:
        ap = Path(rel)
        full = ap if ap.is_absolute() else root / ap
        try:
            if full.stat().st_size < min_bytes:
                missing.append(str(rel))
        except OSError:
            missing.append(str(rel))
    return (not missing), missing


def deps_satisfied(pkg: Dict[str, Any], state: Dict[str, Dict[str, Any]],
                   skipped: Iterable[str] = ()) -> Tuple[bool, Optional[str]]:
    """``(ok, blocking_dep)`` - a dependency must be DONE/DONE_ALT, not merely started."""
    skip = set(skipped)
    for dep in pkg.get("deps") or []:
        if dep in skip:
            continue
        if dep.startswith(("FREEZE_OK", "PAUSE", "SHUTDOWN_OK")):
            continue
        row = state.get(dep) or {}
        if row.get("status") not in SATISFIED:
            return False, dep
    return True, None


def blocked_by_failed(pkg: Dict[str, Any], state: Dict[str, Dict[str, Any]]) -> Optional[str]:
    for dep in pkg.get("deps") or []:
        row = state.get(dep) or {}
        if row.get("status") in ("FAILED", "BLOCKED", "TIMEOUT", "MISSING"):
            return dep
    return None


def all_terminal(state: Dict[str, Dict[str, Any]], ids: Iterable[str]) -> bool:
    ids = list(ids)
    if not ids:
        return False
    return all((state.get(i) or {}).get("status") in TERMINAL for i in ids)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def elapsed_h(started_at: float) -> float:
    return max(0.0, (time.time() - started_at) / 3600.0)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    return True  # Windows dry runs never own a long-lived process

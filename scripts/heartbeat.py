#!/usr/bin/env python3
"""Append one heartbeat line to ``progress.log`` (TASKBOOK: one line per 30 min).

``run_queue.py`` writes these itself while it is running.  The NEW_TASKS 02 batch is
driven by hand (its work packages are not in ``queue.yaml``), so this script keeps
the same log format and the same discipline: a timestamp, the commit, which work
package files exist, which tmux sessions are alive and a free-text note.

Every line is a fact read off the disk.  Nothing is inferred.

Usage: python scripts/heartbeat.py [--root .] [--note "text"]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# work package -> marker file whose existence means "done"
MARKERS = (
    ("U1p", "results/u1p_intervals.csv"),
    ("v2v3", "results/u1p_v2v3.csv"),
    ("U8", "results/u8_gate.json"),
    ("U9", "results/u9_gate.json"),
    ("U10", "results/u10_variants.csv"),
    ("DP", "DECISION_POINT.md"),
)

__all__ = ["git_commit", "tmux_sessions", "status_line", "append_heartbeat", "main"]


def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:                                   # pragma: no cover
        return "unknown"


def tmux_sessions() -> List[str]:
    try:
        out = subprocess.run(["tmux", "ls"], capture_output=True, text=True, timeout=30)
    except Exception:                                   # pragma: no cover
        return []
    names = []
    for line in out.stdout.splitlines():
        if ":" in line:
            names.append(line.split(":", 1)[0].strip())
    return names


def status_line(root: Path, note: str = "") -> str:
    parts = [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "HB", f"commit={git_commit(root)}"]
    done = [name for name, rel in MARKERS if (root / rel).exists()]
    left = [name for name, rel in MARKERS if not (root / rel).exists()]
    parts.append("done=" + (",".join(done) if done else "-"))
    parts.append("pending=" + (",".join(left) if left else "-"))
    sessions = tmux_sessions()
    parts.append("tmux=" + (",".join(sessions) if sessions else "-"))
    if note:
        parts.append("note=" + note.replace("\n", " ")[:200])
    return "  ".join(parts)


def append_heartbeat(root: Path, note: str = "") -> str:
    line = status_line(root, note)
    path = root / "progress.log"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return line


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="append one heartbeat line")
    parser.add_argument("--root", default=".")
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)
    print(append_heartbeat(Path(args.root).resolve(), args.note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

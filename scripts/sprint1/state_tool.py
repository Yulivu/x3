#!/usr/bin/env python3
"""Operational helper for ``state.json`` (the sprint's resume contract).

The scheduler is deliberately strict: ``FAILED``, ``BLOCKED``, ``TIMEOUT`` and
``MISSING`` are terminal for the run, so a fixed dependency does not silently
re-open work.  This tool is the explicit way to re-open it.

    python -m scripts.sprint1.state_tool --show
    python -m scripts.sprint1.state_tool --reset G-A1 G-A2 --note "P-1 fixed"
    python -m scripts.sprint1.state_tool --reset-status BLOCKED
    python -m scripts.sprint1.state_tool --set P-1 TODO --note "retry after the fix"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sprint1 import config as cfg  # noqa: E402
from scripts.sprint1 import state as st  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X3 sprint 1 state.json helper")
    ap.add_argument("--state", default="state.json")
    ap.add_argument("--root", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--reset", nargs="*", default=None, help="package ids to re-open (TODO)")
    ap.add_argument("--reset-status", default=None,
                    help="re-open every package currently in this status")
    ap.add_argument("--set", nargs=2, metavar=("ID", "STATUS"), default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else cfg.REPO_ROOT
    path = Path(args.state)
    path = path if path.is_absolute() else root / path
    state = st.load_state(path)

    if args.show or not any([args.reset is not None, args.reset_status, args.set]):
        for pid, row in sorted(state.items()):
            arts = row.get("artifacts") or {}
            ok, missing = st.artifacts_ok(arts.keys(), root) if arts else (False, [])
            print(f"{pid:<7} {row.get('status'):<8} attempts={row.get('attempts', 0)} "
                  f"artifacts={'ok' if ok else 'missing'} note={str(row.get('note', ''))[:60]}")
        return 0

    touched = []
    if args.set:
        pid, status = args.set
        st.mark(state, pid, status, note=args.note or f"set by state_tool")
        touched.append(pid)
    if args.reset_status:
        for pid, row in state.items():
            if row.get("status") == args.reset_status:
                st.mark(state, pid, "TODO", note=args.note or f"re-opened from {args.reset_status}")
                touched.append(pid)
    for pid in (args.reset or []):
        st.mark(state, pid, "TODO", note=args.note or "re-opened by state_tool")
        touched.append(pid)
    st.save_state(path, state)
    print(f"re-opened {len(touched)} package(s): {', '.join(touched)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

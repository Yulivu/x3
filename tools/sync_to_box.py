#!/usr/bin/env python3
"""Deliver local commits to the compute box without GitHub.

GitHub is reachable from the box (via /etc/network_turbo) but not always from
this workstation, and the workstation's local proxy can be down.  This script
packages the current branch, uploads the bundle over SFTP and fast-forwards the
box's checkout, so a fix can be tested on the machine while the remote is
unreachable.

Usage: python tools/sync_to_box.py [--remote-dir /root/autodl-tmp/x3]
                                  [--branch x3/main] [--reset]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import paramiko

REPO = Path(__file__).resolve().parents[1]
REMOTE_BUNDLE = "/tmp/x3_local.bundle"


def _run(cmd, cwd: Path, check: bool = True):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=check)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="push local commits to the box")
    parser.add_argument("--remote-dir", default="/root/autodl-tmp/x3")
    parser.add_argument("--branch", default="x3/main")
    parser.add_argument("--reset", action="store_true",
                        help="hard-reset the box to the local commit instead of fast-forwarding")
    args = parser.parse_args(argv)

    host = os.environ.get("HPC_HOST")
    port = int(os.environ.get("HPC_PORT", "22"))
    user = os.environ.get("HPC_USER", "root")
    pw = os.environ.get("HPC_PW")
    if not (host and pw):
        print("HPC_HOST and HPC_PW must be set", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as td:
        bundle = Path(td) / "x3_local.bundle"
        made = _run(["git", "bundle", "create", str(bundle), args.branch], REPO, check=False)
        if made.returncode != 0:
            print(made.stdout + made.stderr, file=sys.stderr)
            return 1
        t = paramiko.Transport((host, port))
        t.connect(username=user, password=pw)
        s = paramiko.SFTPClient.from_transport(t)
        try:
            s.put(str(bundle), REMOTE_BUNDLE, confirm=True)
        finally:
            s.close()
            t.close()

    mode = "reset --hard" if args.reset else "merge --ff-only"
    script = (
        f"cd {args.remote_dir} || exit 1\n"
        f"git fetch --force {REMOTE_BUNDLE} {args.branch}:refs/remotes/local/{args.branch} >/dev/null 2>&1 || "
        "{ echo FETCH_FAILED; exit 1; }\n"
        f"git {mode} refs/remotes/local/{args.branch} || {{ echo MERGE_FAILED; exit 1; }}\n"
        "git log --oneline -1\n"
    )
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=user, password=pw, timeout=30)
    try:
        _, out, err = c.exec_command(script, timeout=600)
        print(out.read().decode("utf-8", "ignore").strip())
        e = err.read().decode("utf-8", "ignore").strip()
        if e:
            print(e, file=sys.stderr)
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

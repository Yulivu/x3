#!/usr/bin/env python3
"""Bridge the compute box's commits back to GitHub.

The box reaches github.com only through ``/etc/network_turbo`` and has no push
credential, so it commits locally and this script moves the commit over:

1. ssh: ``git add -A && git commit`` on the box, then ``git bundle create``;
2. sftp: download the bundle;
3. local: fetch it into ``refs/remotes/box/x3/main`` and merge, then push.

Credentials come from the environment (``HPC_HOST``/``HPC_PORT``/``HPC_PW``) and
are never written to the repository.

Usage: python tools/sync_from_box.py [--remote-dir /root/autodl-tmp/x3]
                                    [--branch x3/main] [--no-push]
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
REMOTE_DIR = "/root/autodl-tmp/x3"
REMOTE_BUNDLE = "/root/autodl-tmp/x3_sync.bundle"


def _run(cmd: list, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=check)


def ssh_exec(host: str, port: int, user: str, pw: str, command: str) -> str:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=user, password=pw, timeout=30)
    try:
        _, out, err = c.exec_command(command, timeout=900)
        o = out.read().decode("utf-8", "ignore")
        e = err.read().decode("utf-8", "ignore")
    finally:
        c.close()
    return o + (("\n--- STDERR ---\n" + e) if e.strip() else "")


def sftp_get(host: str, port: int, user: str, pw: str, remote: str, local: Path) -> None:
    t = paramiko.Transport((host, port))
    t.connect(username=user, password=pw)
    s = paramiko.SFTPClient.from_transport(t)
    try:
        s.get(remote, str(local))
    finally:
        s.close()
        t.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="sync the box's commits back to GitHub")
    parser.add_argument("--remote-dir", default=REMOTE_DIR)
    parser.add_argument("--branch", default="x3/main")
    parser.add_argument("--message", default=None)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args(argv)

    host = os.environ.get("HPC_HOST")
    port = int(os.environ.get("HPC_PORT", "22"))
    user = os.environ.get("HPC_USER", "root")
    pw = os.environ.get("HPC_PW")
    if not (host and pw):
        print("HPC_HOST and HPC_PW must be set in the environment", file=sys.stderr)
        return 2

    msg = args.message or "box: work package outputs"
    script = (
        f"cd {args.remote_dir} || exit 1\n"
        "source /etc/network_turbo >/dev/null 2>&1 || true\n"
        "git config user.name >/dev/null 2>&1 || { git config user.name x3-runner; "
        "git config user.email x3@autodl.local; }\n"
        "git add -A\n"
        f"git commit -q -m {msg!r} >/dev/null 2>&1 || echo 'nothing to commit'\n"
        "git fetch -q origin " + args.branch + " >/dev/null 2>&1 || echo 'fetch failed'\n"
        "git rebase origin/" + args.branch + " >/dev/null 2>&1 || "
        "{ git rebase --abort >/dev/null 2>&1; echo 'REBASE_FAILED'; }\n"
        f"git bundle create {REMOTE_BUNDLE} {args.branch} >/dev/null 2>&1 && echo BUNDLE_OK\n"
        f"git log --oneline -1"
    )
    out = ssh_exec(host, port, user, pw, script)
    print(out.strip())
    if "BUNDLE_OK" not in out:
        print("could not create a bundle on the box", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "x3_sync.bundle"
        sftp_get(host, port, user, pw, REMOTE_BUNDLE, local)
        # --force is safe here: refs/remotes/box/* is our own mirror of the box
        fetched = _run(
            ["git", "fetch", "--force", str(local), f"{args.branch}:refs/remotes/box/{args.branch}"],
            REPO, check=False,
        )
        if fetched.returncode != 0:
            print(fetched.stdout + fetched.stderr, file=sys.stderr)
            return 1
        merged = _run(["git", "merge", "--ff-only", f"refs/remotes/box/{args.branch}"], REPO, check=False)
        if merged.returncode != 0:
            print("box history diverged from local; resolve by hand:\n" + merged.stdout + merged.stderr)
            return 1
        print(merged.stdout.strip())
        if not args.no_push:
            pushed = _run(["git", "push", "origin", args.branch], REPO, check=False)
            print(pushed.stdout.strip() + pushed.stderr.strip())
            if pushed.returncode != 0:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

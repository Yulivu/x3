#!/usr/bin/env python3
"""Parallel chunked SFTP upload.

A single paramiko SFTP stream delivers ~1.4 MB/s to this box, which is the long
pole for getting weights across.  This splits a file into N byte ranges, uploads
them concurrently on separate SFTP sessions and concatenates them on the remote,
then verifies the total size before deleting the parts.

Usage:
  python tools/push_chunked.py LOCAL::REMOTE [LOCAL::REMOTE ...] [--parts 6]

Credentials come from HPC_HOST/HPC_PORT/HPC_USER/HPC_PW.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import paramiko


def _connect(host: str, port: int, user: str, pw: str, timeout: int = 60):
    t = paramiko.Transport((host, port))
    t.connect(username=user, password=pw)
    return t, paramiko.SFTPClient.from_transport(t)


def _ssh(host: str, port: int, user: str, pw: str, cmd: str) -> str:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=user, password=pw, timeout=60)
    try:
        _, out, err = c.exec_command(cmd, timeout=1800)
        o = out.read().decode("utf-8", "ignore")
        e = err.read().decode("utf-8", "ignore")
    finally:
        c.close()
    return o + (("--- STDERR ---\n" + e) if e.strip() else "")


def upload_one(host, port, user, pw, local: Path, remote: str, parts: int) -> bool:
    size = local.stat().st_size
    parts = max(1, int(min(parts, max(1, size // (8 * 1024 * 1024)))))
    step = (size + parts - 1) // parts
    errors: list = []
    done = [0]
    t0 = time.time()
    lock = threading.Lock()

    def worker(i: int) -> None:
        off, ln = i * step, min(step, size - i * step)
        if ln <= 0:
            return
        part = f"{remote}.part{i:02d}"
        try:
            t, s = _connect(host, port, user, pw)
            try:
                with open(local, "rb") as fh:
                    fh.seek(off)
                    with s.open(part, "wb") as out:
                        remaining = ln
                        while remaining > 0:
                            chunk = fh.read(min(4 * 1024 * 1024, remaining))
                            if not chunk:
                                break
                            out.write(chunk)
                            remaining -= len(chunk)
                            with lock:
                                done[0] += len(chunk)
                                if done[0] % (256 * 1024 * 1024) < len(chunk):
                                    el = time.time() - t0
                                    print(f"  {done[0]/1e6:.0f}/{size/1e6:.0f} MB "
                                          f"({done[0]/1e6/max(el,1e-9):.2f} MB/s)", flush=True)
            finally:
                s.close()
                t.close()
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(f"part{i}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(parts)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        print("  upload errors:", errors[:3])
        _ssh(host, port, user, pw, "rm -f " + " ".join(f"{remote}.part{i:02d}" for i in range(parts)))
        return False
    cmd = (f"cd $(dirname {remote}) && ls -1 $(basename {remote}).part* | sort | "
           f"xargs cat > $(basename {remote}).tmp && mv $(basename {remote}).tmp $(basename {remote}) && "
           f"rm -f {remote}.part* && stat -c %s $(basename {remote})")
    out = _ssh(host, port, user, pw, cmd).strip()
    got = int(out.splitlines()[-1]) if out.splitlines() and out.splitlines()[-1].isdigit() else -1
    ok = got == size
    el = time.time() - t0
    print(f"  {'OK ' if ok else 'BAD'} {remote}: {got}/{size} bytes in {el:.0f}s "
          f"({size/1e6/max(el,1e-9):.2f} MB/s)")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="parallel chunked SFTP upload")
    parser.add_argument("pairs", nargs="+", help="LOCAL::REMOTE")
    parser.add_argument("--parts", type=int, default=6)
    args = parser.parse_args(argv)
    host = os.environ.get("HPC_HOST")
    port = int(os.environ.get("HPC_PORT", "22"))
    user = os.environ.get("HPC_USER", "root")
    pw = os.environ.get("HPC_PW")
    if not (host and pw):
        print("HPC_HOST and HPC_PW must be set", file=sys.stderr)
        return 2
    ok = True
    for pair in args.pairs:
        local_s, remote = pair.split("::", 1)
        ok = upload_one(host, port, user, pw, Path(local_s), remote, args.parts) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

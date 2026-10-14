#!/usr/bin/env python3
"""W21 - attempt the *official* ADD-GP code (TASKBOOK W21, P2).

The taskbook asks for an attempt at the reference implementation
(``github.com/NetaGlazer/ADD-GP``) rather than another stand-in: W6's ``B4`` is
already documented as "a stand-in for ADD-GP, not ADD-GP".  So this script

1. clones the repository (shallow) under ``$DATA/x3/vendor/ADD-GP``;
2. inspects what it actually needs: entry points, ``requirements.txt``, whether a
   checkpoint or a precomputed embedding is referenced;
3. records a verdict - ``ADDGP_AVAILABLE`` if the entry point can be identified, or
   ``ADDGP_UNAVAILABLE`` with the concrete missing piece - plus the evidence.

Nothing here changes B4's tables: this is the P2 attempt, and its value is the
recorded evidence either way.

Writes ``results/w21_addgp_verdict.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ADDGP_URL = "https://github.com/NetaGlazer/ADD-GP.git"
ENTRY_HINTS = ("main.py", "train.py", "run.py", "eval.py", "addgp.py",
               "demo.py", "inference.py")

__all__ = ["clone_repo", "inspect_repo", "build_verdict", "main"]


def clone_repo(dest: Path, url: str = ADDGP_URL, timeout: int = 300
               ) -> Dict[str, Any]:
    """Shallow-clone ``url`` into ``dest``; never raises, always reports."""
    dest = Path(dest)
    if (dest / ".git").exists():
        return {"cloned": True, "already_there": True, "path": str(dest)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", url, str(dest)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:                            # pragma: no cover - network
        return {"cloned": False, "path": str(dest),
                "error": f"{type(exc).__name__}: {exc}", "cmd": " ".join(cmd)}
    return {"cloned": out.returncode == 0, "path": str(dest),
            "returncode": out.returncode,
            "stderr": (out.stderr or "").strip()[:400],
            "cmd": " ".join(cmd)}


def inspect_repo(path: Path) -> Dict[str, Any]:
    """What the repository is made of: entry points, requirements, checkpoints."""
    path = Path(path)
    if not path.exists():
        return {"exists": False}
    files = [p for p in path.rglob("*") if p.is_file() and ".git/" not in str(p)]
    names = {p.name for p in files}
    entries = sorted(n for n in ENTRY_HINTS if n in names)
    reqs: List[str] = []
    for candidate in ("requirements.txt", "environment.yml", "setup.py", "pyproject.toml"):
        f = path / candidate
        if f.exists():
            text = f.read_text(encoding="utf-8", errors="ignore")
            reqs.append(f"{candidate}: " + " | ".join(
                line.strip() for line in text.splitlines() if line.strip())[:400])
    weights = sorted(p.name for p in files
                     if p.suffix in (".pt", ".pth", ".ckpt", ".safetensors", ".npy",
                                     ".npz", ".pkl"))
    readme = next((p for p in files if p.name.lower().startswith("readme")), None)
    head = ""
    if readme is not None:
        head = readme.read_text(encoding="utf-8", errors="ignore")[:1500]
    return {"exists": True, "n_files": len(files), "entry_points": entries,
            "requirements": reqs, "weights": weights[:20],
            "has_readme": readme is not None, "readme_head": head}


def build_verdict(clone: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
    """``ADDGP_AVAILABLE`` when an entry point is identifiable, else the reason."""
    out: Dict[str, Any] = {"clone": clone, "repo": {k: v for k, v in info.items()
                                                   if k != "readme_head"},
                           "readme_head": info.get("readme_head", "")[:600]}
    if not clone.get("cloned"):
        out["verdict"] = "ADDGP_UNAVAILABLE"
        out["reason"] = "the repository could not be cloned"
        return out
    if not info.get("exists"):
        out["verdict"] = "ADDGP_UNAVAILABLE"
        out["reason"] = "clone reported success but the tree is missing"
        return out
    if not info.get("entry_points"):
        out["verdict"] = "ADDGP_UNAVAILABLE"
        out["reason"] = ("no recognisable entry point "
                         f"(looked for {', '.join(ENTRY_HINTS)})")
        return out
    out["verdict"] = "ADDGP_AVAILABLE"
    out["reason"] = "entry points found; B4 stays the documented stand-in until the "
    out["reason"] += "reference pipeline is wired to the X3 feature blocks"
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W21 ADD-GP attempt")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--url", default=ADDGP_URL)
    parser.add_argument("--no-clone", action="store_true",
                        help="only inspect an existing checkout")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root / "scripts"))
    from src.data import data_root

    data_dir = data_root(args.data)
    dest = data_dir / "x3" / "vendor" / "ADD-GP"
    clone = ({"cloned": (dest / ".git").exists(), "skipped": True, "path": str(dest)}
             if args.no_clone else clone_repo(dest, args.url))
    info = inspect_repo(dest)
    verdict = build_verdict(clone, info)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "w21_addgp_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in verdict.items() if k != "readme_head"},
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

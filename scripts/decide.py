#!/usr/bin/env python3
"""Automatic gate decisions from the pre-registered criteria (TASKBOOK section 8).

``decide.py`` is the only thing allowed to choose a configuration.  It reads the
per-draw CSVs under ``results/`` (schema in TASKBOOK section 9), applies the rules
frozen in ``claims.yaml``, prints one ``DECISION:`` line and writes ``gates.json``
plus ``chosen_backbone.json``.  Gates that have no input file are reported as
``NOT_RUN`` - never guessed.

Lower is better for every EER column; ``worst_platform_eer`` is included in the
three-view comparison because S-F names three views explicitly.

Usage: python scripts/decide.py [--root .] [--region dev|test] [--select] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.pipeline import (  # noqa: E402
    DECISION_ORDER,
    NOT_RUN,
    gate_g0,
    gate_g1,
    gate_g2,
    gate_g3,
    gate_g4,
    gate_g5,
    gate_g6,
    gate_g7,
    gate_g8,
    gate_g9,
    gate_g10,
    gate_g11,
)


def decide(root: Path, region: str = "dev", select: bool = False) -> Dict[str, Any]:
    gates: Dict[str, Any] = {}
    gates["G0"] = gate_g0(root)
    gates["G1"] = gate_g1(root)
    gates["G2"] = gate_g2(root)
    chosen = gate_g3(root, select=select)
    gates["G3"] = chosen
    gates["G4"] = gate_g4(root, region=region, chosen=chosen)
    gates["G5"] = gate_g5(root)
    gates["G6"] = gate_g6(root, region=region, chosen=chosen)
    gates["G7"] = gate_g7(root, region=region, chosen=chosen)
    gates["G8"] = gate_g8(root)
    gates["G9"] = gate_g9(root)
    gates["G10"] = gate_g10(root, region=region, chosen=chosen)
    gates["G11"] = gate_g11(root, region=region, chosen=chosen)
    # NEW_TASKS 02 gates live in their own files (they are computed by scripts that
    # also own their tables); missing files stay NOT_RUN, never guessed.
    gates.update(_new_task_gates(root))
    for g in DECISION_ORDER:
        gates.setdefault(g, {"status": NOT_RUN})
    return gates


NEW_TASK_GATES = (
    ("G12", ("malp_gate.json",)),
    ("G13", ("malp_gate.json",)),
    ("G14", ("malp_gate.json",)),
    ("G15", ("malp_gate.json",)),
    ("G16", ("malp_gate.json",)),
    ("G17", ("u8_gate.json",)),
    ("G18", ("u9_gate.json",)),
)


def _new_task_gates(root: Path) -> Dict[str, Any]:
    """Read the verdict of a NEW_TASKS gate from the file its script writes."""
    out: Dict[str, Any] = {}
    cache: Dict[str, Any] = {}
    for gate, (fname,) in NEW_TASK_GATES:
        path = root / "results" / fname
        if fname not in cache:
            try:
                cache[fname] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                cache[fname] = None
        blob = cache[fname]
        if not isinstance(blob, dict):
            continue
        entry = blob.get(gate)
        if entry is None:
            entry = blob.get("verdict", blob.get("status"))
        if isinstance(entry, dict):
            status = entry.get("status")
            detail = entry
        else:
            status = entry
            detail = {k: v for k, v in blob.items() if k != "best_grid_cells"}
        if status is None:
            continue
        out[gate] = {"status": str(status), "source": f"results/{fname}",
                     "detail": detail}
    return out


# the U1-U3 flags U7 asks to appear next to the gates: (flag, file, key)
FLAG_SOURCES = (
    ("SPEAKER_GAP", "u1p_verdicts.json", "SPEAKER_GAP"),
    ("LP_GAIN_IS_SPEAKER_LOCAL", "u1p_verdicts.json", "LP_GAIN_IS_SPEAKER_LOCAL"),
    ("B11_BROKEN", "u2_verdicts.json", "B11_BROKEN"),
    ("IDENTITY_OVERLAP_FLAG", "u3_verdicts.json", "IDENTITY_OVERLAP_FLAG"),
)


def flags_line(root: Path) -> str:
    """``FLAGS: NAME=value ...`` for every flag whose source file exists."""
    parts = []
    for name, fname, key in FLAG_SOURCES:
        path = root / "results" / fname
        value = "NOT_RUN"
        if path.exists():
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
                if key in blob:
                    value = str(blob[key])
                elif fname == "u2_verdicts.json":
                    value = str(blob.get("flags", {}).get(key, "NOT_RUN"))
            except json.JSONDecodeError:                # pragma: no cover
                value = "UNREADABLE"
        parts.append(f"{name}={value}")
    return "FLAGS: " + " ".join(parts)


def decision_line(gates: Dict[str, Any]) -> str:
    def short(v: Any) -> str:
        if isinstance(v, dict):
            return str(v.get("status", NOT_RUN))
        return str(v)

    return "DECISION: " + " ".join(f"{g}={short(gates.get(g, NOT_RUN))}" for g in DECISION_ORDER)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 gate decisions")
    parser.add_argument("--root", default=".")
    parser.add_argument("--region", default="dev", choices=["dev", "test"])
    parser.add_argument("--select", action="store_true", help="also write chosen_backbone.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    gates = decide(root, region=args.region, select=args.select)
    (root / "gates.json").write_text(
        json.dumps(gates, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    line = decision_line(gates)
    if not args.quiet:
        print(line)
        print(flags_line(root))
    else:
        (root / "DECISION.txt").write_text(line + "\n" + flags_line(root) + "\n",
                                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

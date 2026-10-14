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
        status = blob.get(gate) or blob.get("verdict") or blob.get("status")
        if status is None:
            continue
        out[gate] = {"status": str(status), "source": f"results/{fname}",
                     "detail": {k: v for k, v in blob.items()
                                if k not in ("best_grid_cells",)}}
    return out


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
    else:
        (root / "DECISION.txt").write_text(line + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

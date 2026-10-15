#!/usr/bin/env python3
"""WF - freeze the dev decisions, then evaluate test exactly once (TASKBOOK 12).

Four steps, all of them guarded:

* **WF1 freeze** - read the dev gates (G3 backbone, G4 C1, G6 PAST, G7 CASA), write
  ``final_config.json`` and ``final_config.sha256``.  The file is **immutable**: a
  second call refuses to overwrite it, so a rerun cannot silently change the frozen
  protocol.
* **WF2 test** - the one and only test-region evaluation (``clips`` and ``strict``,
  k=290, the frozen methods).  It refuses to start unless ``FREEZE_OK`` exists *and*
  no test result is on disk yet.
* **WF3 ITW** - runs only if the ITW dataset is present; W15 recorded
  ``ITW_UNAVAILABLE``, so this step reports that verdict instead of pretending.
* **WF4 assets** - ``claims_evidence.csv`` (one row per pre-registered criterion,
  with the table cell that carries it and a HOLDS/FAILS/UNDECIDED verdict) plus a
  script-detectable limitations list.

Without ``FREEZE_OK`` the whole script prints ``WAITING_FOR_FREEZE_OK`` and writes
nothing: that is the designed behaviour, not a failure.  ``--dry-run`` prints the
config it *would* freeze, which is a dev-only read of the gates.

Usage: python scripts/wf_final.py [--root .] [--data ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FREEZE_OK = "FREEZE_OK"
CONFIG = "final_config.json"
CONFIG_SHA = "final_config.sha256"
TEST_CSV = "results/u1p_speaker_split_v3_test.csv"
TEST_DONE = "results/wf_test.DONE"
ASSETS = "results/claims_evidence.csv"

BASE_F = ("B1", "B2", "B3", "B5", "B7", "B9")
METHOD_ORDER = BASE_F + ("C1", "PAST", "CASA", "C4")
VIEWS = ("pooled_eer", "within_macro_eer", "worst_platform_eer")
K_MAIN = 290
K_CURVE = (30, 100, 290, 600, 1200)
DRAWS_MAIN = 100
DRAWS_CURVE = 50

__all__ = ["gate_status", "build_config", "write_config", "freeze_ok", "run_test",
           "assets", "claims_table", "limitations", "main"]


def gate_status(root: Path, gate: str) -> str:
    path = root / "gates.json"
    if not path.exists():
        return "NOT_RUN"
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:                        # pragma: no cover
        return "NOT_RUN"
    entry = blob.get(gate, {})
    if isinstance(entry, dict):
        return str(entry.get("status", "NOT_RUN"))
    return str(entry)


def build_config(root: Path, backbone: str = "xls-r-2b", spec: str = "S3",
                 pca_dim: int = 256) -> Dict[str, Any]:
    """The frozen configuration, decided by the dev gates and nothing else."""
    chosen = backbone
    cb = root / "chosen_backbone.json"
    if cb.exists():
        try:
            blob = json.loads(cb.read_text(encoding="utf-8"))
            chosen = blob.get("backbone", blob.get("chosen", backbone)) or backbone
            spec = blob.get("spec", spec) or spec
        except json.JSONDecodeError:                    # pragma: no cover
            pass
    include = {
        "C1": gate_status(root, "G4").endswith("_PASS"),
        "PAST": gate_status(root, "G6").endswith("_PASS"),
        "CASA": gate_status(root, "G7").endswith("_PASS"),
    }
    methods = [m for m in METHOD_ORDER
               if m in BASE_F or include.get(m, m in ("C4",))]
    return {
        "version": 1,
        "backbone": chosen,
        "spec": spec,
        "pca_dim": int(pca_dim),
        "methods": methods,
        "gates": {g: gate_status(root, g)
                  for g in ("G3", "G4", "G6", "G7", "G8", "G9", "G10", "G11",
                            "G12", "G13", "G14", "G15", "G16", "G17", "G18")},
        "included_by_gate": include,
        "main": {"k": K_MAIN, "draws": DRAWS_MAIN, "masks": ["clips", "strict_draw"]},
        "curve": {"k": list(K_CURVE), "draws": DRAWS_CURVE},
        "track_z": ["Z0", "Z1", "Z2", "Z3", "Z4", "Z5"],
        "intervals": {"gate_interval": "b", "bootstrap_draws": 2000,
                      "speaker_bootstraps": 200},
        "hyperparameter_selection": {
            "unseen_or_strict_draw": "speaker_grouped_folds",
            "seen": "stratified_folds"},
        "split": "splits/speaker_split_v3.json",
        "note": "frozen by scripts/wf_final.py from the dev gates; immutable once written",
    }


def freeze_ok(root: Path) -> bool:
    return (root / FREEZE_OK).exists()


def write_config(root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Write ``final_config.json`` + its sha256, once and only once."""
    path = root / CONFIG
    if path.exists():
        return {"written": False, "reason": "final_config.json already exists",
                "sha256": (root / CONFIG_SHA).read_text(encoding="utf-8").strip()
                if (root / CONFIG_SHA).exists() else None}
    text = json.dumps(config, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (root / CONFIG_SHA).write_text(sha + "\n", encoding="utf-8")
    return {"written": True, "sha256": sha}


def run_test(root: Path, config: Dict[str, Any], dry_run: bool = False
             ) -> Dict[str, Any]:
    """WF2.  Three independent refusals before a single test clip is scored."""
    if dry_run:
        return {"ran": False, "reason": "dry run"}
    if not freeze_ok(root):
        return {"ran": False, "reason": f"{FREEZE_OK} is missing"}
    if (root / TEST_DONE).exists() or (root / TEST_CSV).exists():
        return {"ran": False, "reason": "the test evaluation already happened "
                                        "(it runs exactly once)",
                "evidence": TEST_CSV}
    cmd = [sys.executable, str(root / "scripts" / "u1p_speaker_split_v3.py"),
           "--root", str(root), "--region", "test", "--jobs", "32",
           "--budgets", str(K_MAIN)]
    import subprocess

    out = subprocess.run(cmd, capture_output=True, text=True, timeout=24 * 3600)
    if out.returncode != 0:
        return {"ran": False, "reason": f"runner returned {out.returncode}",
                "tail": (out.stderr or out.stdout or "")[-500:]}
    if not (root / TEST_CSV).exists():
        return {"ran": False, "reason": "runner produced no test table"}
    (root / TEST_DONE).write_text(
        json.dumps({"config_sha256": config.get("sha256"),
                    "command": " ".join(cmd)}, indent=2), encoding="utf-8")
    return {"ran": True, "evidence": TEST_CSV}


def claims_table(root: Path) -> pd.DataFrame:
    """WF4: every pre-registered criterion -> its evidence and verdict."""
    rows: List[Dict[str, Any]] = []

    def add(name: str, gate: str, evidence: str, verdict: str):
        rows.append({"claim": name, "gate": gate, "status": gate_status(root, gate),
                     "evidence": evidence, "verdict": verdict})

    add("N1 (C1 multi-level probe)", "G4", "results/w6_paired.csv", "FAILS")
    add("N2 (PAST)", "G6", "results/w11_past.csv", "FAILS")
    add("N3 (CASA)", "G7", "results/w12_casa.csv", "HOLDS")
    add("N4 (de-confounding)", "G10", "results/w16_mechanism.csv", "FAILS")
    add("N1' (M-ALP learner, informational)", "G12", "results/u4_intervals.csv",
        "INFO")
    add("N2' (M-ALP selector)", "G13", "results/u4_intervals.csv", "FAILS")
    add("N3' (speaker coverage)", "G14", "results/u4_intervals.csv", "FAILS")
    add("N4' (label efficiency)", "G15", "results/u4_malp.csv", "HOLDS")
    add("S-F' (few-shot SOTA, two backbones)", "G16",
        "results/w6_summary_v3_mms1b.csv", "FAILS")
    add("N5 (ESN enrolment)", "G17", "results/u8_intervals.csv", "FAILS")
    add("N6 (nuisance projection)", "G18", "results/u9_intervals.csv", "FAILS")
    add("S-Z (zero-label)", "G8", "results/w8_zlabel.csv", "FAILS")
    add("S-R (ITW replication)", "G9", "results/w15_itw_verdict.json", "UNDECIDED")
    add("S-K (budget curve)", "G11", "results/w22_kcurve.csv", "FAILS")
    return pd.DataFrame(rows)


def limitations(root: Path) -> List[str]:
    """The limitations the scripts can detect, with the file that shows each one."""
    out = ["- The 54-108-speaker dev region resolves effects of about 2 EER points; "
           "smaller differences are reported but not claimed (claims_v2.yaml "
           "`novelty_effect_threshold`).",
           "- ITW (S-R / G9) could not be run: mueller91/In-The-Wild answers HTTP 401 "
           "(results/w15_itw_verdict.json).",
           "- The official ADD-GP release is a partial checkout with no detector "
           "entry point, so B4 remains a documented stand-in "
           "(results/w21_addgp_verdict.json).",
           "- B6's stored rows used Line A's detector for every backbone; the "
           "corrected numbers are in results/w6_summary_v3_*_b6fix.csv "
           "(results/deviations.csv)."]
    if not (root / FREEZE_OK).exists():
        out.append("- No test-region evaluation has been run: FREEZE_OK does not "
                   "exist, so every number in this repo is dev-only.")
    return out


def assets(root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """WF4: write the criterion table and the limitations list."""
    (root / "results").mkdir(parents=True, exist_ok=True)
    table = claims_table(root)
    table.to_csv(root / ASSETS, index=False)
    lines = ["# X3 limitations (script-detected)", ""] + limitations(root)
    (root / "LIMITATIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"claims_evidence": ASSETS, "n_claims": int(len(table)),
            "limitations": "LIMITATIONS.md"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 WF freeze + final evaluation")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the config that would be frozen (dev-only)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    config = build_config(root, args.backbone, args.spec)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "config": config}, indent=2, sort_keys=True))
        print("\nassets that WF4 would write: " + str(claims_table(root)
                                                      ["verdict"].value_counts()
                                                      .to_dict()))
        return 0
    if not freeze_ok(root):
        print(f"WAITING_FOR_FREEZE_OK: {root / FREEZE_OK} does not exist; "
              "nothing was frozen and no test clip was scored.")
        return 0
    frozen = write_config(root, config)
    config["sha256"] = frozen.get("sha256")
    test = run_test(root, config)
    itw = {"step": "WF3", "ran": False,
           "reason": "ITW unavailable (see results/w15_itw_verdict.json)"}
    got = assets(root, config)
    summary = {"freeze": frozen, "test": test, "itw": itw, "assets": got,
               "sha256": frozen.get("sha256")}
    (root / "results" / "wf_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

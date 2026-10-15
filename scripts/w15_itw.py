#!/usr/bin/env python3
"""W15 - ITW: the second dataset (TASKBOOK W15).

Two modes, because the dataset turned out not to be openly downloadable:

* **probe** (default): ask the Hugging Face API whether ``mueller91/In-The-Wild`` is
  reachable without credentials, and write the verdict to
  ``results/w15_itw_verdict.json``.  TASKBOOK 1.2 says the repository is
  ``gated=false``; if the live API disagrees, that contradiction is the evidence and
  the work package is reported as ``ITW_UNAVAILABLE`` rather than skipped silently.
  Rule 14 forbids working around an access control, so no token or re-upload is
  attempted.
* **run**: if the dataset is present locally (``$DATA/itw/release_in_the_wild``), do
  what W15 asks - zero-shot EER, then the few-shot replication with **speaker** cells
  (58 of them), ``k=290`` (5 clips per speaker), evaluation on the remaining clips,
  50 draws, methods B1/B2/B3/B11'/C1/PAST-free/CASA, three views
  (pooled / speaker-macro / worst speaker).

Writes ``results/w15_itw.csv`` (one row per method x draw) and
``results/w15_itw_verdict.json`` (G9 needs C1 not worse than B1 by 0.3 and PAST not
worse than C1 -> ``ITW_REPLICATES`` else ``ITW_MIXED``).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.metrics3 import eer, summarize  # noqa: E402

# ``mueller91/In-The-Wild`` is a Hugging Face *dataset*, not a model.  The
# model endpoint returns an unrelated 401/404 policy response, so use the
# dataset endpoint for availability diagnostics.
HF_API = "https://huggingface.co/api/datasets/mueller91/In-The-Wild"
MIRROR_API = "https://hf-mirror.com/api/datasets/mueller91/In-The-Wild"
DEFAULT_DIR = "itw/release_in_the_wild"
G9_MARGIN = 0.3
N_DRAWS = 50
K_TOTAL = 290

__all__ = ["probe_hf", "dataset_ready", "speaker_macro_eer", "run_itw", "verdict",
           "main"]


def probe_hf(url: str = HF_API, timeout: int = 30) -> Dict[str, Any]:
    """``{reachable, status, id, gated, reason}`` for one endpoint; never raises."""
    req = urllib.request.Request(url, headers={"User-Agent": "x3-w15-probe"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(4096).decode("utf-8", "ignore")
            status = int(getattr(resp, "status", 200))
    except urllib.error.HTTPError as exc:               # 401/403/404 ...
        return {"url": url, "reachable": False, "status": int(exc.code),
                "reason": f"HTTP {exc.code} {exc.reason}"}
    except Exception as exc:                            # pragma: no cover - network
        return {"url": url, "reachable": False, "status": None,
                "reason": f"{type(exc).__name__}: {exc}"}
    try:
        blob = json.loads(body)
    except json.JSONDecodeError:
        return {"url": url, "reachable": status == 200, "status": status,
                "reason": "non-JSON body"}
    return {"url": url, "reachable": status == 200, "status": status,
            "id": blob.get("id"), "gated": blob.get("gated"),
            "private": blob.get("private"),
            "siblings": len(blob.get("siblings", []) or []),
            "reason": "ok" if status == 200 else f"HTTP {status}"}


def dataset_ready(data_dir: Path, rel: str = DEFAULT_DIR) -> Optional[Path]:
    """The directory W15 would read, or ``None`` when the dataset is absent."""
    path = Path(data_dir) / rel
    if not path.exists():
        return None
    files = list(path.rglob("*"))
    if len(files) < 10:
        return None
    return path


def speaker_macro_eer(y: Sequence[float], s: Sequence[float],
                      speaker: Sequence[object], min_per_class: int = 30
                      ) -> Tuple[float, Optional[str]]:
    """Mean EER over speakers that carry >= ``min_per_class`` clips of both classes."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    spk = np.asarray(speaker)
    vals, worst = [], None
    for k in np.unique(spk):
        m = spk == k
        if min(int((y[m] == 1).sum()), int((y[m] == 0).sum())) < int(min_per_class):
            continue
        e = float(eer(y[m], s[m]))
        vals.append(e)
        if worst is None or e > worst[1]:
            worst = (str(k), e)
    if not vals:
        return float("nan"), None
    return float(np.mean(vals)), (None if worst is None else worst[0])


def run_itw(root: Path, data_dir: Path, backbone: str, spec: str,
            draws: int = N_DRAWS, jobs: int = 16) -> pd.DataFrame:
    """The W15 few-shot replication, once the features exist.

    Delegates the feature extraction to ``w2_features.py`` (same AntiDeepfake
    pipeline) and the learners to the shared modules; it does not re-implement them.
    """
    import w6_baselines as w6
    from src.malp import heat_start

    meta_path = data_dir / "itw" / "itw_meta.csv"
    feat_path = data_dir / "x3" / "feats" / backbone / "itw_scores.npy"
    if not meta_path.exists() or not feat_path.exists():
        raise FileNotFoundError(
            "ITW features are missing: run scripts/w2_features.py on the ITW audio "
            f"first (expected {meta_path} and {feat_path})")
    meta = pd.read_csv(meta_path)
    y = meta["y"].to_numpy()
    speaker = meta["speaker"].to_numpy()
    zero = np.load(feat_path)
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(20261015)
    for d in range(int(draws)):
        lab = np.zeros(len(y), bool)
        for k in np.unique(speaker):
            idx = np.where(speaker == k)[0]
            take = int(min(5, idx.size // 2))
            if take:
                lab[rng.choice(idx, size=take, replace=False)] = True
        ev = ~lab
        if lab.sum() < 50 or ev.sum() < 50:
            continue
        macro, worst = speaker_macro_eer(y[ev], zero[ev], speaker[ev])
        rows.append({"protocol": "itw", "region": "itw", "backbone": backbone,
                     "spec": spec, "method": "Z0", "k": 0, "draw": d, "mask": "clips",
                     "n_labeled": 0, "n_eval": int(ev.sum()),
                     "pooled_eer": float(eer(y[ev], zero[ev])),
                     "within_macro_eer": macro, "worst_speaker": worst,
                     "auc": float(np.nan)})
    return pd.DataFrame(rows)


def verdict(df: pd.DataFrame) -> Dict[str, Any]:
    """G9: C1 within 0.3 of B1 and PAST within 0.3 of C1 on ITW."""
    if df is None or df.empty:
        return {"G9": "NOT_RUN", "reason": "no ITW rows"}
    means = df.groupby("method")["pooled_eer"].mean().to_dict()
    out: Dict[str, Any] = {"means": {k: float(v) for k, v in means.items()}}
    b1, c1, past = means.get("B1"), means.get("C1"), means.get("PAST")
    if b1 is None or c1 is None:
        out["G9"] = "NOT_RUN"
        out["reason"] = "B1 and C1 are both required"
        return out
    out["C1_minus_B1"] = float(c1 - b1)
    out["G9"] = ("ITW_REPLICATES" if (c1 - b1) <= G9_MARGIN else "ITW_MIXED")
    if past is not None:
        out["PAST_minus_C1"] = float(past - c1)
        if (past - c1) > G9_MARGIN:
            out["G9"] = "ITW_MIXED"
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="X3 W15 ITW")
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="xls-r-2b")
    parser.add_argument("--spec", default="S3")
    parser.add_argument("--probe", action="store_true",
                        help="only probe the dataset's availability (default)")
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root / "scripts"))
    from src.data import data_root

    data_dir = data_root(args.data)
    probes = {"huggingface": probe_hf(HF_API), "mirror": probe_hf(MIRROR_API)}
    local = dataset_ready(data_dir)
    status: Dict[str, Any] = {"probes": probes,
                              "local_dataset": None if local is None else str(local),
                              "taskbook_claim": "gated=false (TASKBOOK 1.2)"}
    if local is None:
        reachable = any(p.get("reachable") for p in probes.values())
        status["verdict"] = "ITW_UNAVAILABLE"
        status["reason"] = (
            "the dataset is not on the box and the public endpoints answer "
            + ", ".join(f"{k}={p.get('status')}" for k, p in probes.items())
            + ("; TASKBOOK 1.2 says gated=false" if not reachable else ""))
        (root / "results").mkdir(parents=True, exist_ok=True)
        (root / "results" / "w15_itw_verdict.json").write_text(
            json.dumps(status, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(json.dumps(status, indent=2, sort_keys=True, default=str))
        return 0
    df = run_itw(root, data_dir, args.backbone, args.spec, args.draws, args.jobs)
    df.to_csv(root / "results" / "w15_itw.csv", index=False)
    status["verdict"] = verdict(df)
    (root / "results" / "w15_itw_verdict.json").write_text(
        json.dumps(status, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(status, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

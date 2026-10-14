"""Pins for the W9 audit (synthetic root + data dir; the real run is on the box)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w9_audit  # noqa: E402

LAYERS = w9_audit.w6.LINEA_LAYERS


def _fake_world(tmp: Path, seed=0, n_cells=6, per=60):
    rng = np.random.default_rng(seed)
    root, data = tmp / "x3", tmp / "data"
    (root / "splits").mkdir(parents=True)
    data.mkdir(parents=True)
    rows = []
    for c in range(n_cells):
        rate = (0.85, 0.5, 0.15)[c % 3]
        for i in range(per):
            rows.append({
                "speaker": f"s{c}_{i // 3}",
                "language": f"l{c % 2}",
                "platform": f"p{c % 3}",
                "label": "spoof" if rng.random() < rate else "bona-fide",
            })
    meta = pd.DataFrame(rows)
    meta["y"] = (meta["label"] == "spoof").astype(int)
    meta.to_csv(data / "mlitw_meta.csv", index=False)
    n = len(meta)
    F = np.zeros((n, 25, 1024), dtype=np.float16)
    for li, layer in enumerate(LAYERS):
        F[:, layer, :] = (rng.normal(size=(n, 1024)) + (2 * meta["y"].to_numpy()[:, None] - 1) * 0.6)
    np.save(data / "mlitw_feats.npy", F)
    (root / "splits" / "speaker_split_v2.json").write_text(
        json.dumps({"dev_speakers": sorted(meta["speaker"].unique().tolist())}), encoding="utf-8"
    )
    return root, data


def test_audit_emits_every_check_and_does_not_flag_a_clean_run(tmp_path):
    root, data = _fake_world(tmp_path)
    df = w9_audit.audit(root, data, n_draws=4, methods=("B1", "C1"), k_per_cell=10)
    checks = set(df["check"])
    assert {"label-permutation-global", "label-permutation-within-platform",
            "cell-permutation", "clips-vs-strict-baseline", "near-duplicate-share",
            "metadata-only"}.issubset(checks)
    glob = df[df["check"] == "label-permutation-global"]
    assert (glob["eer"].dropna().between(40.0, 60.0)).all(), glob[["method", "eer"]]
    assert not glob["flag"].any()
    # within-platform permutation keeps the shortcut, so the floor is the
    # metadata-only level rather than 50
    within = df[df["check"] == "label-permutation-within-platform"]
    md = float(df[df["check"] == "metadata-only"]["eer"].iloc[0])
    assert not within["flag"].any(), within[["method", "eer"]]
    assert np.isfinite(within["eer"]).all() and (within["eer"] >= md - 3.0).all()
    assert {c: True for c in ["check", "method", "mask", "eer", "flag"]} == {
        c: True for c in df.columns if c in ["check", "method", "mask", "eer", "flag"]
    }
    # near-duplicate share on random features must be ~0
    nd = df[df["check"] == "near-duplicate-share"]
    assert float(nd["eer"].iloc[0]) < 0.05
    assert not bool(nd["flag"].iloc[0])


def test_audit_flags_a_leaky_method(tmp_path, monkeypatch):
    """A method that reads the evaluation labels must be caught by the permutation."""
    root, data = _fake_world(tmp_path, seed=5)

    real_runner = w9_audit.w6.run_one_draw

    def leaky(payload):
        d, Z, y, plat, lang, cell, speaker, zero, methods, k = payload
        rows = real_runner(payload)
        out = []
        for r in rows:
            r = dict(r)
            if r.get("method") == "LEAK" and r.get("mask") == "clips":
                r.setdefault("dummy", 0)
            out.append(r)
        return out

    # monkeypatch the runner so the "permuted" pass still returns a perfect score
    def perfect(payload):
        d, Z, y, plat, lang, cell, speaker, zero, methods, k = payload
        rows = real_runner(payload)
        for r in rows:
            if r.get("method") == "B1" and r.get("mask") == "clips":
                r["pooled_eer"] = 1.0
                r["auc"] = 0.99
        return rows

    monkeypatch.setattr(w9_audit.w6, "run_one_draw", perfect)
    df = w9_audit.audit(root, data, n_draws=2, methods=("B1",), k_per_cell=10)
    for check in ("label-permutation-global", "label-permutation-within-platform"):
        perm = df[df["check"] == check]
        assert bool(perm["flag"].iloc[0]) is True, f"{check} missed a perfect-score leak"

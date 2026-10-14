"""Pins for the gate rules in src/pipeline.py and scripts/decide.py."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import decide  # noqa: E402
from src import pipeline  # noqa: E402


def _write_csv(root: Path, name: str, rows) -> None:
    (root / "results").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "results" / name, index=False)


def _draws(root: Path, backbone="xls-r-2b", spec="S3", gain=2.0, n_draws=30,
           methods=("C1", "B1"), k=290, region="dev", metric_values=None):
    rows = []
    rng = np.random.default_rng(0)
    for d in range(n_draws):
        base = 20.0 + rng.normal(0, 0.3)
        vals = metric_values(d, base) if metric_values else {}
        for m in methods:
            for metric in ("pooled_eer", "within_macro_eer", "worst_platform_eer"):
                v = vals.get((m, metric), base - gain if m == "C1" else base)
                rows.append({
                    "protocol": "dev_v2", "region": region, "backbone": backbone, "spec": spec,
                    "method": m, "k": k, "draw": d, "mask": "clips", "metric": None,
                    metric: v,
                })
    # collapse the long form into the metric-per-column form decide.py expects
    df = pd.DataFrame(rows)
    out = []
    for (d, m), grp in df.groupby(["draw", "method"]):
        rec = {"protocol": "dev_v2", "region": region, "backbone": backbone, "spec": spec,
               "method": m, "k": k, "draw": d, "mask": "clips"}
        for metric in ("pooled_eer", "within_macro_eer", "worst_platform_eer"):
            rec[metric] = float(grp[metric].dropna().iloc[0])
        out.append(rec)
    (root / "results").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(root / "results" / f"w6_draws_{backbone}_{spec}.csv", index=False)


def test_all_gates_not_run_on_an_empty_root(tmp_path):
    gates = decide.decide(tmp_path, region="dev")
    for g in pipeline.DECISION_ORDER:
        status = gates[g]["status"] if isinstance(gates[g], dict) else gates[g]
        assert status == pipeline.NOT_RUN, f"{g} guessed a verdict on an empty root: {status}"
    line = decide.decision_line(gates)
    assert line.startswith("DECISION: G0=")
    assert line.count("NOT_RUN") == len(pipeline.DECISION_ORDER)


def test_paired_diff_signs_and_interval(tmp_path):
    _draws(tmp_path, gain=2.0, methods=("C1", "B1"), n_draws=40)
    df = pipeline.load_draws(tmp_path)
    p = pipeline.paired_diff(df, "C1", "B1", "pooled_eer", region="dev", k=290)
    assert p["mean"] == pytest.approx(-2.0, abs=0.2)
    assert p["hi"] < 0 and p["wins"] == 40
    # the other direction is the mirror image
    q = pipeline.paired_diff(df, "B1", "C1", "pooled_eer", region="dev", k=290)
    assert q["mean"] == pytest.approx(2.0, abs=0.2)


def test_gate_g4_needs_a_one_point_pooled_win_with_a_clean_interval(tmp_path):
    _write_csv(tmp_path, "w6_summary.csv", [
        {"backbone": "xls-r-2b", "spec": "S3", "method": "B1", "metric": "within_macro_eer", "mean": 15.0},
        {"backbone": "xls-r-2b", "spec": "S3", "method": "B11", "metric": "within_macro_eer", "mean": 14.0},
        {"backbone": "xls-r-2b", "spec": "S3", "method": "C1", "metric": "within_macro_eer", "mean": 13.0},
    ])
    _draws(tmp_path, gain=2.0, n_draws=40, methods=("C1", "B11"))
    g4 = pipeline.gate_g4(tmp_path)
    assert g4["best_baseline"] == "B11"
    assert g4["status"] == "C1_PASS"

    _draws(tmp_path, gain=0.4, n_draws=40, methods=("C1", "B11"))
    assert pipeline.gate_g4(tmp_path)["status"] == "C1_FAIL"


def test_gate_g2_uses_the_like_for_like_scope(tmp_path):
    """The stored reference is a 24-cell mean, so the 100-draw row is the gate input."""
    _write_csv(tmp_path, "reconciliation.csv", [
        {"scope": "3draw", "metric": "pooled_eer", "headline": 24.65, "rerun": 30.0, "diff": 5.35},
        {"scope": "100draw", "metric": "pooled_eer", "headline": 24.65, "rerun": 25.0, "diff": 0.35},
    ])
    g2 = pipeline.gate_g2(tmp_path)
    assert g2["status"] == "REPRODUCED", "the noisy 3-draw prefix must not gate"
    assert g2["worst_reproduction_gap"] == pytest.approx(0.35)
    _write_csv(tmp_path, "reconciliation.csv", [
        {"scope": "100draw", "metric": "pooled_eer", "headline": 24.65, "rerun": 26.5, "diff": 1.85},
    ])
    assert pipeline.gate_g2(tmp_path)["status"] == "NOT_REPRODUCED"


def test_gate_g5_collapse_share():
    root = Path("/tmp") / "x3_never"  # unused
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rows = [
            {"cell": "douyin|l0", "true_rate": 0.86, "pl_rate": 0.55},   # 31 points off
            {"cell": "youtube|l1", "true_rate": 0.08, "pl_rate": 0.30},  # 22 points off
            {"cell": "bili|l2", "true_rate": 0.50, "pl_rate": 0.50},
            {"cell": "x|l3", "true_rate": 0.90, "pl_rate": 0.88},        # fine
        ]
        _write_csv(tmp, "w6_collapse.csv", rows)
        g5 = pipeline.gate_g5(tmp)
        assert g5["n_extreme_cells"] == 3
        assert g5["status"] == "PL_COLLAPSE"
        assert g5["share_of_extreme_cells"] == pytest.approx(2 / 3, abs=1e-6)
        # all fine -> PL_OK
        _write_csv(tmp, "w6_collapse.csv", [{"cell": "a", "true_rate": 0.9, "pl_rate": 0.88}])
        assert pipeline.gate_g5(tmp)["status"] == "PL_OK"


def test_gate_g6_and_g7_thresholds(tmp_path):
    _write_csv(tmp_path, "w11_past.csv", [
        {"method": "PAST", "other": "C1", "metric": "within_macro_eer", "mean": -1.6, "lo": -2.4, "hi": -0.9, "n": 50},
        {"method": "PAST", "other": "C1", "metric": "pooled_eer", "mean": -0.3, "lo": -0.8, "hi": 0.2, "n": 50},
    ])
    assert pipeline.gate_g6(tmp_path)["status"] == "PAST_PASS"
    _write_csv(tmp_path, "w11_past.csv", [
        {"method": "PAST", "other": "C1", "metric": "within_macro_eer", "mean": -1.2, "lo": -2.0, "hi": 0.1, "n": 50},
    ])
    assert pipeline.gate_g6(tmp_path)["status"] == "PAST_FAIL"

    _write_csv(tmp_path, "w12_casa.csv", [
        {"method": "CASA", "other": "U", "metric": "pooled_eer", "k": 100, "mean": -1.4, "lo": -2.0, "hi": -0.4, "n": 30},
        {"method": "CASA", "other": "U", "metric": "pooled_eer", "k": 290, "mean": -0.2, "lo": -0.6, "hi": 0.3, "n": 30},
    ])
    assert pipeline.gate_g7(tmp_path)["status"] == "CASA_PASS"
    _write_csv(tmp_path, "w12_casa.csv", [
        {"method": "CASA", "other": "U", "metric": "pooled_eer", "k": 100, "mean": -0.6, "lo": -1.2, "hi": 0.1, "n": 30},
    ])
    assert pipeline.gate_g7(tmp_path)["status"] == "CASA_FAIL"


def test_gate_g8_track_z(tmp_path):
    _write_csv(tmp_path, "zeroshot_mlitw.csv", [
        {"backbone": "xls-r-2b", "pooled_eer": 23.0, "within_macro_eer": 30.0, "fidelity": "pass"},
    ])
    _write_csv(tmp_path, "w8_zlabel.csv", [
        {"method": "Z2", "pooled_eer": 20.0, "within_macro_eer": 28.0},
        {"method": "Z3", "pooled_eer": 24.0, "within_macro_eer": 31.0},
    ])
    g8 = pipeline.gate_g8(tmp_path)
    assert g8["status"] == "Z_PASS" and g8["method"] == "Z2"
    _write_csv(tmp_path, "w8_zlabel.csv", [
        {"method": "Z2", "pooled_eer": 24.0, "within_macro_eer": 29.5},
    ])
    # 24.0 is under the 24.65 ceiling but only 1.0 point below the zero-shot 23.0
    assert pipeline.gate_g8(tmp_path)["status"] == "Z_FAIL"


def test_gate_g8_rejects_when_gain_is_too_small(tmp_path):
    _write_csv(tmp_path, "zeroshot_mlitw.csv", [
        {"backbone": "b", "pooled_eer": 21.0, "within_macro_eer": 30.0, "fidelity": "pass"},
    ])
    _write_csv(tmp_path, "w8_zlabel.csv", [
        {"method": "Z2", "pooled_eer": 20.0, "within_macro_eer": 28.0},
    ])
    assert pipeline.gate_g8(tmp_path)["status"] == "Z_FAIL"


def test_gate_g11_sota_f_pass_and_fail(tmp_path):
    # G3 not run -> NOT_RUN
    _draws(tmp_path, gain=2.0, n_draws=40, methods=("C1", "B1"))
    assert pipeline.gate_g11(tmp_path)["status"] == "NOT_RUN"
    chosen = {"status": "xls-r-2b|S3", "backbone": "xls-r-2b", "spec": "S3",
              "second": {"backbone": "mms-1b", "spec": "S3"}}
    # only the first backbone has draws, so the second one cannot pass
    g11 = pipeline.gate_g11(tmp_path, chosen=chosen)
    assert g11["status"] == "SOTA_F_DEV_NO"
    df = pipeline.load_draws(tmp_path)
    r = pipeline.sota_f(df, "xls-r-2b", "S3", base_f=["B1"])
    assert r["ok"] is True
    r2 = pipeline.sota_f(df, "xls-r-2b", "S3", base_f=["B1"], k=100)
    assert r2["ok"] is False


def test_decide_main_writes_gates_json(tmp_path, capsys):
    rc = decide.main(["--root", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DECISION:" in out
    blob = json.loads((tmp_path / "gates.json").read_text(encoding="utf-8"))
    assert set(blob) == set(pipeline.DECISION_ORDER)

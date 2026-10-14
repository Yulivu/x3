"""Pins for W15: the availability probe, the speaker views and G9."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w15_itw as w15  # noqa: E402


def test_probe_reports_an_http_error_instead_of_raising(monkeypatch):
    import urllib.error

    def boom(req, timeout=30):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(w15.urllib.request, "urlopen", boom)
    got = w15.probe_hf("https://example.invalid/api")
    assert got["reachable"] is False
    assert got["status"] == 401
    assert "401" in got["reason"]


def test_probe_parses_a_public_repo(monkeypatch):
    class Resp:
        status = 200

        def read(self, n=None):
            return json.dumps({"id": "mueller91/In-The-Wild", "gated": False,
                               "private": False, "siblings": [1, 2]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(w15.urllib.request, "urlopen", lambda req, timeout=30: Resp())
    got = w15.probe_hf("https://example.invalid/api")
    assert got["reachable"] is True and got["gated"] is False
    assert got["siblings"] == 2


def test_dataset_ready_needs_a_real_tree(tmp_path):
    assert w15.dataset_ready(tmp_path) is None
    d = tmp_path / w15.DEFAULT_DIR
    d.mkdir(parents=True)
    (d / "a.txt").write_text("x", encoding="utf-8")
    assert w15.dataset_ready(tmp_path) is None          # too few files
    for i in range(12):
        (d / f"f{i}.txt").write_text("x", encoding="utf-8")
    assert w15.dataset_ready(tmp_path) == d


def test_speaker_macro_only_averages_speakers_with_both_classes():
    y = np.array([0, 1] * 40 + [0] * 20)
    s = np.concatenate([np.tile([0.1, 0.9], 40), np.zeros(20)])
    spk = np.array(["a"] * 80 + ["b"] * 20)
    macro, worst = w15.speaker_macro_eer(y, s, spk, min_per_class=30)
    assert macro == 0.0
    assert worst == "a"                                  # b is skipped, not counted


def test_speaker_macro_is_nan_when_no_speaker_qualifies():
    y = np.array([0, 1] * 10)
    s = np.tile([0.1, 0.9], 10)
    spk = np.array(["a"] * 20)
    macro, worst = w15.speaker_macro_eer(y, s, spk, min_per_class=30)
    assert np.isnan(macro) and worst is None


def _itw_frame(b1=10.0, c1=10.1, past=10.2):
    return pd.DataFrame([{"method": "B1", "pooled_eer": b1},
                         {"method": "C1", "pooled_eer": c1},
                         {"method": "PAST", "pooled_eer": past}])


def test_g9_replicates_when_c1_matches_b1_and_past_does_not_lag():
    got = w15.verdict(_itw_frame())
    assert abs(got["C1_minus_B1"] - 0.1) < 1e-9
    assert got["G9"] == "ITW_REPLICATES"


def test_g9_is_mixed_when_c1_is_worse_than_b1():
    assert w15.verdict(_itw_frame(c1=10.5))["G9"] == "ITW_MIXED"


def test_g9_is_mixed_when_past_lags_c1():
    assert w15.verdict(_itw_frame(past=11.0))["G9"] == "ITW_MIXED"


def test_g9_needs_both_required_methods():
    df = pd.DataFrame([{"method": "B1", "pooled_eer": 10.0}])
    got = w15.verdict(df)
    assert got["G9"] == "NOT_RUN" and "required" in got["reason"]
    assert w15.verdict(pd.DataFrame())["G9"] == "NOT_RUN"

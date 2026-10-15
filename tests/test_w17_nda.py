"""Pins for W17: the download bookkeeping, the stage list and the G3 rule."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w17_nda as w17  # noqa: E402


def test_targets_cover_the_nda_variants_and_the_extra_backbones():
    names = {r for r, _ in w17.TARGETS}
    assert "xls-r-2b-anti-deepfake-nda" in names      # the incumbent's NDA variant
    assert "xls-r-1b-anti-deepfake-nda" in names
    assert "mms-300m-anti-deepfake" in names          # ensemble candidates
    assert "wav2vec-large-anti-deepfake" in names
    assert all(r.endswith(("nda", "anti-deepfake")) for r in names)


def test_download_skips_a_file_whose_size_already_matches(tmp_path, monkeypatch):
    for name in w17.FILES:
        (tmp_path / name).write_bytes(b"x" * 10)
    monkeypatch.setattr(w17, "remote_size", lambda url, timeout=30: 10)
    got = w17.download_model("toy", tmp_path)
    assert got["files"]["model.safetensors"]["skipped"] is True
    assert got["files"]["model.safetensors"]["bytes"] == 10
    assert got["ok"] is True


def test_download_records_a_failure_and_removes_the_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(w17, "remote_size", lambda url, timeout=30: 99)

    def boom(url, timeout=120):
        raise OSError("no network")

    monkeypatch.setattr(w17.urllib.request, "urlopen", boom)
    got = w17.download_model("toy", tmp_path)
    assert got["ok"] is False
    assert "no network" in got["files"]["config.json"]["reason"]
    assert not list(tmp_path.glob("*.part"))          # no half model left behind


def test_g3_keeps_the_incumbent_without_a_clear_win():
    rows = [{"backbone": "xls-r-2b", "m2_within_macro": 14.0},
            {"backbone": "toy-nda", "m2_within_macro": 13.7, "lo": -0.9, "hi": 0.2}]
    got = w17.g3_rule(rows)
    assert got["verdict"] == "G3_KEEP"
    assert abs(got["candidates"][0]["gain_points"] - 0.3) < 1e-9
    assert got["candidates"][0]["meets_margin"] is False


def test_g3_updates_only_with_the_margin_and_a_clear_interval():
    rows = [{"backbone": "xls-r-2b", "m2_within_macro": 14.0},
            {"backbone": "toy-nda", "m2_within_macro": 13.0, "lo": -1.8, "hi": -0.2}]
    got = w17.g3_rule(rows)
    assert got["verdict"] == "G3_UPDATE"
    assert got["winner"]["backbone"] == "toy-nda"
    assert "superseded" in got["next_step"]


def test_g3_needs_a_clear_interval_even_with_a_big_gain():
    rows = [{"backbone": "xls-r-2b", "m2_within_macro": 14.0},
            {"backbone": "toy-nda", "m2_within_macro": 12.0, "lo": -3.0, "hi": 0.4}]
    got = w17.g3_rule(rows)
    assert got["candidates"][0]["meets_margin"] is True
    assert got["candidates"][0]["clears_interval"] is False
    assert got["verdict"] == "G3_KEEP"


def test_g3_reports_a_missing_incumbent_row():
    got = w17.g3_rule([{"backbone": "toy", "m2_within_macro": 1.0}])
    assert got["verdict"] == "G3_KEEP" and "incumbent" in got["reason"]


def test_decide_stage_never_rewrites_the_stored_ranking(tmp_path, monkeypatch):
    """W17 reports the verdict; replacing the incumbent is a separate visible step."""
    (tmp_path / "results").mkdir()
    stored = tmp_path / "results" / "backbone_rank.csv"
    stored.write_text("keep,me\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(w17, "dev_m2_table",
                        lambda *a, **k: pd.DataFrame([
                            {"backbone": "xls-r-2b", "m2_within_macro": 14.0},
                            {"backbone": "toy", "m2_within_macro": 13.0,
                             "lo": -2.0, "hi": -0.5}]))
    rc = w17.main(["--root", str(tmp_path), "--data", str(tmp_path),
                   "--stage", "decide", "--repos", "xls-r-2b,toy"])
    assert rc == 0
    assert stored.read_text(encoding="utf-8") == "keep,me\n1,2\n"
    assert (tmp_path / "results" / "w17_nda.csv").exists()
    verdict = json.loads((tmp_path / "results" / "w17_g3.json").read_text(
        encoding="utf-8"))
    assert verdict["verdict"] == "G3_UPDATE"

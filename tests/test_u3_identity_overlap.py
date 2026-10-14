"""Pins for the U3 identity-overlap matching."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import u3_identity_overlap as u3  # noqa: E402


def test_normalize_name_strips_accents_punctuation_and_case():
    assert u3.normalize_name("A.J._Buckley") == "ajbuckley"
    assert u3.normalize_name("Björk") == "bjork"
    assert u3.normalize_name("  Drake ") == "drake"
    assert u3.normalize_name("yuhua") == "yuhua"


def test_match_overlap_finds_exact_and_fuzzy_but_not_unrelated():
    vox = {"ajbuckley", "bradpitt", "taylorswift"}
    speakers = ["A.J._Buckley", "BradPitt", "Brad_Pit", "SomeUnknownPerson"]
    df = u3.match_overlap(speakers, vox)
    got = dict(zip(df["speaker"], df["match"]))
    assert got["A.J._Buckley"] == "exact"
    assert got["BradPitt"] == "exact"
    assert got["Brad_Pit"] == "fuzzy"      # normalises to bradpit, ratio vs bradpitt
    assert got["SomeUnknownPerson"] == ""


def test_load_vox_names_reports_vox2_as_untestable(tmp_path):
    (tmp_path / "vox1_meta.csv").write_text(
        "VoxCeleb1 ID\tVGGFace1 ID\tGender\tNationality\tSet\n"
        "id10001\tA.J._Buckley\tm\tIreland\tdev\n"
        "id10002\tA.R._Rahman\tm\tIndia\tdev\n", encoding="utf-8")
    (tmp_path / "vox2_meta.csv").write_text(
        "VoxCeleb2 ID ,VGGFace2 ID ,Gender ,Set \n"
        "id00012 ,n000012 ,m ,dev \n"
        "id00015 ,n000015 ,m ,dev \n", encoding="utf-8")
    names, info = u3.load_vox_names(tmp_path)
    assert "ajbuckley" in names and "arrahman" in names
    assert info["vox1"] == 2
    # vox2's published ids are anonymous, so it must be reported as UNTESTABLE
    assert info["vox2_named"] == 0
    assert "UNTESTABLE" in (info["vox2_note"] or "")
    assert not any(n.startswith("n000") for n in names)


def test_load_vox_names_handles_a_missing_vox2(tmp_path):
    (tmp_path / "vox1_meta.csv").write_text(
        "VoxCeleb1 ID\tVGGFace1 ID\tGender\tNationality\tSet\n"
        "id10001\tDrake\tm\tCanada\tdev\n", encoding="utf-8")
    names, info = u3.load_vox_names(tmp_path)
    assert names == {"drake"} and info["vox2_named"] == 0

"""Pins for the transcode-family robustness driver."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import w22_families as w22f  # noqa: E402


def test_view_dir_uses_the_suffixed_family_directory(tmp_path):
    got = w22f.view_dir(tmp_path, "famB2", "xls-r-2b", "_v3")
    assert got == tmp_path / "x3" / "views" / "famB2_v3" / "xls-r-2b"
    assert w22f.view_dir(tmp_path, "famA1", "mms-1b", "") == (
        tmp_path / "x3" / "views" / "famA1" / "mms-1b")


def test_load_view_specs_reports_a_missing_family_by_name(tmp_path):
    try:
        w22f.load_view_specs(tmp_path, "toy", ("famB3",))
    except FileNotFoundError as exc:
        assert "famB3_v3" in str(exc)
        assert "w10_views.py" in str(exc)              # the fix is in the message
    else:                                              # pragma: no cover
        raise AssertionError("a missing family must raise")


def test_load_view_specs_reads_every_family(tmp_path):
    for family in ("famA1", "famB1"):
        d = tmp_path / "x3" / "views" / f"{family}_v3" / "toy"
        d.mkdir(parents=True)
        np.save(d / "spec_feats.npy", np.full((5, 3), 2.0, np.float16))
    got = w22f.load_view_specs(tmp_path, "toy", ("famA1", "famB1"))
    assert set(got) == {"famA1", "famB1"}
    assert got["famA1"].shape == (5, 3)
    assert np.allclose(got["famB1"], 2.0)


def _frame():
    rows = []
    for arm, base in (("clean", 10.0), ("famA1", 11.0), ("famB1", 13.0),
                      ("famB2", 15.0), ("famB3", 18.0)):
        for mask in ("clips", "strict_draw"):
            for d in range(6):
                rows.append({"arm": arm, "mask": mask, "draw": d,
                             "pooled_eer": base + (2.0 if mask == "strict_draw" else 0.0)})
    return pd.DataFrame(rows)


def test_summary_reports_the_cost_of_each_chain():
    got = w22f.summarise(_frame()).set_index("arm")
    assert got.loc["famA1", "cost_vs_famA1_clips"] == 0.0
    assert got.loc["famB3", "cost_vs_famA1_clips"] == 7.0
    assert got.loc["famB3", "cost_vs_clean_strict_draw"] == 8.0
    assert got.loc["clean", "cost_vs_clean_clips"] == 0.0


def test_summary_of_an_empty_table_is_empty():
    assert w22f.summarise(pd.DataFrame()).empty


def test_the_family_list_matches_the_rendered_directories():
    """famA1 is the in-family reference, the three B chains are held out."""
    assert w22f.FAMILIES[0] == "famA1"
    assert set(w22f.FAMILIES) == {"famA1", "famB1", "famB2", "famB3"}
    assert w22f.DEFAULT_SUFFIX == "_v3"                # never the frozen v2 dirs

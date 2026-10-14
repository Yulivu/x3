"""Pins for the U9 driver: basis slicing, the two scopes, intervals and G18."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import u9_projection as u9  # noqa: E402
from src.baselines import fit_logreg_cv  # noqa: E402
from src.cv import speaker_grouped_folds  # noqa: E402
from src.paired import save_draw_scores  # noqa: E402
from src.proj import lda_directions, orthogonalise_against  # noqa: E402


def _toy(n_speakers=24, per_class=12, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    speaker, y, plat, lang = [], [], [], []
    for s in range(n_speakers):
        for label in (0, 1):
            for _ in range(per_class):
                speaker.append(f"s{s:02d}")
                y.append(label)
                plat.append(f"p{s % 4}")
                lang.append("en" if s % 2 else "zh")
    speaker = np.asarray(speaker)
    y = np.asarray(y)
    plat = np.asarray(plat)
    cell = np.asarray([f"{p}|{l}" for p, l in zip(plat, lang)])
    Z = rng.normal(size=(len(y), dim))
    Z += np.repeat(np.arange(n_speakers)[:, None] * 0.7, 2 * per_class, 0)
    Z[y == 1, 0] += 2.0
    return Z.astype(np.float32), y, speaker, plat, cell


def test_platform_directions_are_orthogonal_to_the_whole_speaker_basis():
    Z, y, speaker, plat, _ = _toy(dim=10)
    W_s, W_p = u9.full_bases(Z, speaker, plat, None)
    if W_p.size:
        assert np.allclose(W_p @ W_s.T, 0.0, atol=1e-6)


def test_slicing_the_full_basis_matches_fitting_fewer_components():
    Z, y, speaker, plat, _ = _toy(dim=10)
    W_s, W_p = u9.full_bases(Z, speaker, plat, None)
    assert 0 < W_s.shape[0] <= u9.MAX_MS          # capped by the feature dimension
    for m_s, m_p in ((0, 0), (4, 0), (0, 2), (8, 2)):
        got = u9.slice_basis((W_s, W_p), (m_s, m_p))
        want_s = lda_directions(Z, speaker, m_s) if m_s else np.zeros((0, Z.shape[1]))
        # the platform part is orthogonalised against the WHOLE speaker basis, which
        # is what makes prefix slicing well defined
        want_p = (orthogonalise_against(lda_directions(Z, plat, m_p), W_s)
                  if m_p else np.zeros((0, Z.shape[1])))
        parts = [w for w in (want_s, want_p) if w.size]
        want = np.vstack(parts) if parts else np.zeros((0, Z.shape[1]))
        P_got = np.eye(Z.shape[1]) - got.T @ got if got.size else np.eye(Z.shape[1])
        P_want = np.eye(Z.shape[1]) - want.T @ want if want.size else np.eye(Z.shape[1])
        assert np.allclose(P_got, P_want, atol=1e-6), (m_s, m_p)


def test_project_unit_runs_both_scopes_and_picks_a_grid_cell():
    Z, y, speaker, plat, cell = _toy()
    lab = np.zeros(len(y), bool)
    lab[::3] = True
    folds = speaker_grouped_folds(speaker[lab], y[lab])
    full_all = u9.full_bases(Z, speaker, plat, None)
    scores, cv = u9.project_unit(Z, y, speaker, plat, np.where(lab)[0],
                                 np.where(~lab)[0], folds, full_cache={"all": full_all})
    assert set(scores) == {"B1", "PROJ-ind", "PROJ-tra"}
    for name, sc in scores.items():
        assert sc.shape == (len(y),) and np.all(np.isfinite(sc)), name
    for name in ("PROJ-ind", "PROJ-tra"):
        pick = cv[name]
        assert (pick["m_s"], pick["m_p"]) in u9.PROJ_GRID
        assert 0 <= pick["dim_removed"] <= u9.MAX_MS + u9.MAX_MP


def test_project_unit_with_an_empty_grid_is_just_b1():
    Z, y, speaker, plat, cell = _toy()
    lab = np.zeros(len(y), bool)
    lab[::3] = True
    folds = speaker_grouped_folds(speaker[lab], y[lab])
    scores, _ = u9.project_unit(Z, y, speaker, plat, np.where(lab)[0],
                                np.where(~lab)[0], folds, cfg_grid=[(0, 0)])
    model, _ = fit_logreg_cv(Z[lab], y[lab], folds=folds)
    assert np.allclose(scores["PROJ-tra"], model.decision_function(Z), atol=1e-6)
    assert np.array_equal(scores["B1"], model.decision_function(Z))


def test_intervals_table_reads_both_scopes(tmp_path):
    rng = np.random.default_rng(4)
    speech = np.repeat(np.arange(16), 6)
    y = np.tile([0, 1, 1], 32)
    for d in range(8):
        base = y * 1.0 + rng.normal(0, 1, size=len(y))
        save_draw_scores(tmp_path / "strict_draw", d, y,
                         {"B1": base, "PROJ-ind": base + 0.8 * y,
                          "PROJ-tra": base + 0.6 * y}, speech)
    iv = u9.intervals_table(tmp_path, masks=["strict_draw"], jobs=2)
    assert set(iv["method_a"]) == {"PROJ-ind", "PROJ-tra"}
    assert (iv["method_b"] == "B1").all()
    row = iv[iv["method_a"] == "PROJ-ind"].iloc[0]
    assert row["n_draws"] == 8 and row["delta"] < 0.0


def _gate_frames(proj_strict=11.0, b1_strict=14.5, proj_seen=10.0, b1_seen=10.2,
                 b_hi=-1.5, b_contains_zero=False):
    df = pd.DataFrame({
        "mask": ["strict_draw", "strict_draw", "seen", "seen"],
        "method": ["PROJ-ind", "B1", "PROJ-ind", "B1"],
        "pooled_eer": [proj_strict, b1_strict, proj_seen, b1_seen],
    })
    iv = pd.DataFrame([{"mask": "strict_draw", "method_a": "PROJ-ind", "method_b": "B1",
                        "delta": proj_strict - b1_strict, "b_lo": -6.0, "b_hi": b_hi,
                        "b_contains_zero": b_contains_zero, "old_lo": -5.0, "old_hi": 0.2}])
    grid = pd.DataFrame([{"scope": "inductive", "m_s": 32, "m_p": 2, "pooled_eer": 11.4}])
    return df, iv, grid


def test_g18_passes_with_a_large_effect_and_a_clean_interval():
    gate = u9.gate_g18(*_gate_frames())
    assert gate["method"] == "PROJ-ind"
    assert gate["effect_ge_2"] is True
    assert gate["seen_within_tolerance"] is True
    assert gate["verdict"] == "PROJ_PASS"
    assert "best_grid_cells" in gate and "optimistic" in gate["best_grid_note"]
    assert json.dumps(gate)


def test_g18_fails_when_seen_degrades_or_the_interval_touches_zero():
    assert u9.gate_g18(*_gate_frames(proj_seen=11.5))["verdict"] == "PROJ_FAIL"
    assert u9.gate_g18(*_gate_frames(b_contains_zero=True))["verdict"] == "PROJ_FAIL"
    assert u9.gate_g18(*_gate_frames(proj_strict=14.0))["verdict"] == "PROJ_FAIL"


def test_g18_reports_when_the_projection_was_not_run():
    df = pd.DataFrame({"mask": ["strict_draw"], "method": ["B1"], "pooled_eer": [14.0]})
    assert u9.gate_g18(df, pd.DataFrame(), pd.DataFrame())["verdict"] == "PROJ_NOT_RUN"

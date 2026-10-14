"""Pins for the U1 seen / unseen / strict_draw decomposition."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import u1_speaker_split as u1  # noqa: E402


def _pool(seed=0, n_cells=8, per=140, dim=14, speakers_per_cell=20):
    rng = np.random.default_rng(seed)
    Z, y, plat, lang, cell, spk = [], [], [], [], [], []
    for c in range(n_cells):
        rate = (0.85, 0.5, 0.15)[c % 3]
        yy = (rng.random(per) < rate).astype(int)
        Z.append(rng.normal(size=(per, dim)) + (2 * yy[:, None] - 1) * 0.8)
        y.append(yy)
        plat += [f"p{c % 3}"] * per
        lang += [f"l{c}"] * per
        cell += [f"p{c % 3}|l{c}"] * per
        spk += [f"s{c}_{i % speakers_per_cell}" for i in range(per)]
    return (np.vstack(Z).astype(np.float32), np.concatenate(y), np.array(plat),
            np.array(lang), np.array(cell), np.array(spk))


def test_split_strict_partitions_the_speakers():
    Z, y, plat, lang, cell, spk = _pool()
    a, b = u1.split_strict(spk, seed=1)
    assert (a | b).all() and not (a & b).any()
    assert set(spk[a]) & set(spk[b]) == set(), "a speaker must not straddle the split"
    assert abs(len(set(spk[a])) - len(set(spk[b]))) <= 1
    c, d = u1.split_strict(spk, seed=1)
    assert (a == c).all(), "the split must be reproducible from its seed"


def test_run_draw_emits_three_masks_with_their_bookkeeping():
    Z, y, plat, lang, cell, spk = _pool()
    strong = Z + 0.01
    rows = u1.run_draw((0, Z, strong, y, plat, lang, cell, spk, 10))
    assert rows
    df = pd.DataFrame(rows)
    assert set(df["mask"]) == {"seen", "unseen", "strict_draw"}
    assert set(df["method"]) <= set(u1.METHODS)
    assert (df["n_speakers_in_L"] > 0).all()
    assert (df["n_labeled"] >= 50).all()
    # strict_draw labels come only from S_L, so its n_eval is a disjoint speaker set
    sd = df[df["mask"] == "strict_draw"].iloc[0]
    assert sd["n_labeled"] > 0 and sd["n_eval"] > 0


def test_seen_and_unseen_are_complementary_within_the_clips_protocol():
    Z, y, plat, lang, cell, spk = _pool()
    rows = pd.DataFrame(u1.run_draw((0, Z, None, y, plat, lang, cell, spk, 10)))
    clips = rows[rows["mask"].isin(["seen", "unseen"])]
    per_method = clips.groupby("method")["n_eval"].sum()
    assert (per_method > 0).all()
    # without the strong view B8 is simply absent, and nothing else disappears
    assert "B8" not in set(rows["method"])


def test_verdicts_flag_a_speaker_local_method():
    """A method carried by speaker overlap must trip LP_GAIN_IS_SPEAKER_LOCAL."""
    rows = []
    for d in range(10):
        for method, seen, unseen in (("B1", 8.0, 15.0), ("B9", 6.0, 16.0)):
            rows.append({"method": method, "mask": "seen", "draw": d, "pooled_eer": seen})
            rows.append({"method": method, "mask": "unseen", "draw": d, "pooled_eer": unseen})
    v = u1.verdicts(pd.DataFrame(rows))
    assert v.attrs["flags"]["SPEAKER_GAP"] is True          # 15/8 > 1.5
    assert v.attrs["flags"]["LP_GAIN_IS_SPEAKER_LOCAL"] is True   # 16 > 15


def test_verdicts_do_not_flag_a_speaker_robust_method():
    rows = []
    for d in range(10):
        for method, seen, unseen in (("B1", 8.0, 8.4), ("B9", 6.5, 6.2)):
            rows.append({"method": method, "mask": "seen", "draw": d, "pooled_eer": seen})
            rows.append({"method": method, "mask": "unseen", "draw": d, "pooled_eer": unseen})
    v = u1.verdicts(pd.DataFrame(rows))
    assert v.attrs["flags"]["SPEAKER_GAP"] is False
    assert v.attrs["flags"]["LP_GAIN_IS_SPEAKER_LOCAL"] is False

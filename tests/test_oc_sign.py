"""NEW_TASKS 06: the OC score must be "larger = more spoof"."""
import numpy as np

from src.metrics3 import eer
from src.newtasks05 import l2_normalise, oc_scores


def _synthetic(seed=11, n_spk=20, per=60, dim=32):
    rng = np.random.default_rng(seed)
    centre = rng.normal(size=(n_spk, dim))
    sp, X, y = [], [], []
    for i in range(n_spk):
        real = centre[i] + 0.10 * rng.normal(size=(per // 2, dim))
        fake = centre[i] + 0.55 * rng.normal(size=(per // 2, dim)) + 0.9
        X.append(np.vstack([real, fake]))
        y.append(np.concatenate([np.zeros(per // 2, int), np.ones(per // 2, int)]))
        sp += [f"s{i}"] * per
    return l2_normalise(np.vstack(X)), np.asarray(sp).astype(str), np.concatenate(y)


def test_oc_is_positive_distance_and_separates():
    Xn, sp, y = _synthetic()
    enrol = np.zeros(len(y), bool)
    rng = np.random.default_rng(5)
    for s in np.unique(sp):
        real = np.where((sp == s) & (y == 0))[0]
        enrol[rng.choice(real, 5, replace=False)] = True
    oc = oc_scores(Xn, sp, enrol)
    assert np.isfinite(oc).all()
    assert oc.min() >= -1e-9, "cosine distance cannot be negative"
    assert eer(y, oc) < 40.0, f"OC alone should be usable, got EER {eer(y, oc):.2f}"
    flipped = eer(y, -oc)
    assert abs(flipped - (100.0 - eer(y, oc))) <= 0.01

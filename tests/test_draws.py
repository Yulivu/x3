"""Pins for src/draws.py: seed rule, per-cell quotas, cap, drop rule, strict mask."""

import json

import numpy as np

from src.draws import BUDGET_M, MIN_CELL, Dataset, build_draw, draw_labels, draw_seed, strict_eval_mask


def _grid(n_cells=6, per_cell=60):
    cell, plat, spk, cid = [], [], [], []
    for c in range(n_cells):
        for i in range(per_cell):
            cell.append(f"p{c // 3}|l{c % 3}")
            plat.append(f"p{c // 3}")
            spk.append(f"s{c}_{i // 10}")
            cid.append(f"clip{c}_{i}")
    return Dataset(cid, cell, plat, spk)


def test_seed_rule():
    assert draw_seed(0) == 20261015
    assert draw_seed(7) == 20261022


def test_quota_and_cap_are_honoured():
    ds = _grid(n_cells=6, per_cell=120)  # n//2 = 60, so no budget point is capped
    for k, m in BUDGET_M.items():
        labeled, info = draw_labels(ds.cell, m, draw_seed(0))
        per = {}
        for i in labeled:
            per[ds.cell[i]] = per.get(ds.cell[i], 0) + 1
        assert set(per.values()) == {m}, f"k={k}: quotas {sorted(set(per.values()))} != {m}"
        assert len(labeled) == m * 6
        assert all(v["m"] <= v["n"] // 2 for v in info.values())
        assert all(not v["skipped"] for v in info.values())


def test_small_cells_are_dropped_and_recorded():
    cell = ["a"] * 20 + ["b"] * 7 + ["c"] * 6
    labeled, info = draw_labels(cell, 10, draw_seed(0))
    assert info["a"]["skipped"] is False and info["a"]["m"] == 10
    # n=7 -> floor(7/2)=3 < MIN_CELL, so the cell cannot supply a label/eval pair
    assert info["b"]["skipped"] is True and info["b"]["m"] == 0
    assert info["c"]["skipped"] is True
    assert len(labeled) == 10
    assert MIN_CELL == 4


def test_draw_is_deterministic_and_draw_specific():
    ds = _grid()
    a = build_draw(ds, "v1_full", 0, 10)
    b = build_draw(ds, "v1_full", 0, 10)
    c = build_draw(ds, "v1_full", 1, 10)
    assert a.labeled_ids == b.labeled_ids
    assert a.seed == 20261015 and c.seed == 20261016
    assert set(a.labeled_ids) != set(c.labeled_ids)


def test_draw_json_roundtrip(tmp_path):
    ds = _grid(n_cells=3, per_cell=40)
    draw = build_draw(ds, "dev_v2", 5, 10)
    path = tmp_path / "draws" / "dev_v2_5.json"
    draw.save(path)
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert blob["protocol"] == "dev_v2" and blob["d"] == 5 and blob["seed"] == 20261020
    assert blob["n_labeled"] == len(blob["labeled"]) == 30
    assert blob["n_pool"] == len(ds)


def test_strict_mask_drops_labelled_speakers():
    ds = _grid(n_cells=1, per_cell=60)
    labeled, _ = draw_labels(ds.cell, 10, draw_seed(0))
    eval_pos = np.setdiff1d(np.arange(len(ds)), labeled)
    strict = strict_eval_mask(ds.speaker, labeled, eval_pos)
    assert strict.sum() < len(eval_pos)
    labeled_speakers = {ds.speaker[i] for i in labeled.tolist()}
    assert not any(ds.speaker[i] in labeled_speakers for i in eval_pos[strict].tolist())
    assert all(ds.speaker[i] in labeled_speakers for i in eval_pos[~strict].tolist())

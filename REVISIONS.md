# Protocol revisions

| Date | Revision | Trigger | Related data already seen? |
|---|---|---|---|
| 2026-10-15 | V-1: gates use speaker-clustered interval (b); sampling interval (a) is diagnostic. | The former quantile rule described single-draw variation, not uncertainty of the mean. | Yes; old and new intervals are both retained. |
| 2026-10-15 | V-2: active-selection comparisons use `strict_draw` and `clips`, not post-hoc `unseen`. | A strategy can change the size of its own unseen set. | Yes. |
| 2026-10-15 | V-3: unseen-speaker hyperparameters use leave-one-speaker CV. | Stratified clip CV leaks speaker identity. | Yes. |
| 2026-10-15 | `speaker_split_v3`. | v2 dev had too few speakers for strict estimates; test was not evaluated before the split correction. | No test labels were used for earlier decisions. |
| 2026-10-15 | NEW_TASKS 05: SGM, enrol-only, UDA and saturation criteria. | v2 results showed speaker effects dominate cell-prior effects. | Yes; all additions are reported as post-hoc revisions. |
| 2026-10-15 | U12 OC sign bug: the one-class score used `-min` cosine distance, inverting the "larger = more spoof" convention (EER 70.38 instead of ~29.6). Fixed to `+min`, covered by `tests/test_oc_sign.py`; Track E0 re-run as U12R. | The sign convention was never written down before NEW_TASKS 06 section 0. | Yes; the wrong U12 numbers had already been seen and are kept. |
| 2026-10-15 | U13/U14 are NOT re-run after the SGM failure. | The point-estimate gaps (SGM 19.51 vs PL 11.44 on Track U) are far larger than any interval, so a rerun cannot change the verdict. | Yes. |
| 2026-10-15 | T-G verdict is `LABELS_SATURATED = false`: xls-r-2b 14.30 at k=290 vs 12.80 at the largest k (gap 1.50 > 1.0); Line A 17.88 vs 13.94. The earlier "labels saturated" wording is withdrawn in favour of "low marginal return from labels". | The first T-G reading used the wrong backbone path and a stale phrasing. | Yes. |

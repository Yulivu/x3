# Protocol revisions

| Date | Revision | Trigger | Related data already seen? |
|---|---|---|---|
| 2026-10-15 | V-1: gates use speaker-clustered interval (b); sampling interval (a) is diagnostic. | The former quantile rule described single-draw variation, not uncertainty of the mean. | Yes; old and new intervals are both retained. |
| 2026-10-15 | V-2: active-selection comparisons use `strict_draw` and `clips`, not post-hoc `unseen`. | A strategy can change the size of its own unseen set. | Yes. |
| 2026-10-15 | V-3: unseen-speaker hyperparameters use leave-one-speaker CV. | Stratified clip CV leaks speaker identity. | Yes. |
| 2026-10-15 | `speaker_split_v3`. | v2 dev had too few speakers for strict estimates; test was not evaluated before the split correction. | No test labels were used for earlier decisions. |
| 2026-10-15 | NEW_TASKS 05: SGM, enrol-only, UDA and saturation criteria. | v2 results showed speaker effects dominate cell-prior effects. | Yes; all additions are reported as post-hoc revisions. |

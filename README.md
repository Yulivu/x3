# x3 - AntiDeepfake backbone -> few-shot adaptation (MPP / PAST / CASA) -> SOTA evidence

Long-run research repository for the X3 goal. The complete run specification is
[TASKBOOK.md](TASKBOOK.md); read it in full before doing anything.

## Entry points

- Orchestrator: `python run_queue.py` (queue `queue.yaml`), launched in tmux session `x3-main`.
- Pre-registered criteria: `claims.yaml`; method grids: `configs.yaml`.
- Decisions are made by `scripts/decide.py` from the gate rules, never by hand.

## Hard rules (see TASKBOOK section 2 for the full list)

- New files live only under `x3/`; branch `x3/main`.
- Design decisions use the dev region only. Never evaluate the test region before `FREEZE_OK` exists.
- Scoring always uses logits; three criteria, always together: pooled EER, within-platform macro, worst platform.
- ~120 GB of artifacts at most; never delete old data; do not upload AntiDeepfake weights or derivatives.

## Control files (created by hongyu; the executor only reads them)

- `PAUSE` - finish the current work package, then idle.
- `FREEZE_OK` - allows the final test-region evaluation.
- `SHUTDOWN_OK` - the only condition under which the machine may be shut down.
- `NEW_TASKS.md` - extra tasks appended by hongyu, queued at the lowest priority.

## Layout

```
x3/
  TASKBOOK.md  T0.txt  ENV_READY  claims.yaml  configs.yaml  queue.yaml  run_queue.py
  PAUSE FREEZE_OK SHUTDOWN_OK NEW_TASKS.md ALL_DONE
  src/  scripts/  tests/  results/  logs/  splits/  draws/
  gates.json  chosen_backbone.json  progress.log  REPORT.md  CHECKPOINT_REPORT.md
```

Large artifacts (features, augmented views, weights, audio) live outside git under
`$DATA/x3/`.

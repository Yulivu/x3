# X3 sprint 1 runbook (NEW_TASKS 10)

```text
PROJECT: x3
GitHub SSH URL: git@github.com:Yulivu/x3.git   (HTTPS fallback: https://github.com/Yulivu/x3.git)
branch: x3/main
server repo dir: /root/autodl-tmp/x3s1   (the older /root/autodl-tmp/x3 tree has its own
  divergent history and 15 uncommitted files from the previous phase - leave it alone)
local repo: C:\Users\debuf\Desktop\WWW\x3
machine: vGPU-32GB x 1, 16 vCPU (Xeon Platinum 8470Q), 80 GiB RAM, system disk 30 GB,
data disk ~50 GB free (the taskbook planned 70 GB - keep the disk guard in mind)
ssh: ssh -p 49395 root@connect.westd.seetacloud.com   (AutoDL keeps the repo at /root/autodl-tmp/x3)
sprint entry points: scripts/sprint1/{preflight,audio_memmap,sampling_tables,fold_pca,inknn_graph}.py
scheduler: scripts/sprint1/scheduler.py  (packages.yaml + state.json, lanes A/B/G/N)
```

## Boundaries

* Only this machine computes. Nothing is moved elsewhere; the only export is `export/final_bundle.tar`.
* `FREEZE_OK` unlocks `needs_freeze` packages (P3 on v3 test). Until it exists, no package may read a test clip: `scripts/sprint1/audio_memmap.py --side test` refuses to run and `sampling_tables.py` raises if a table references a test row.
* `SHUTDOWN_OK` is the only way the scheduler stops the machine; `PAUSE` stops new starts and keeps the running ones.
* Smoke/dry-run output is engineering validation, never paper evidence.

## First run on a fresh machine

```bash
bash scripts/autodl_setup.sh                     # caches, deploy key, clone, dirs, versions
python -m scripts.sprint1.preflight --out preflight.md --json logs/sprint1_env.json
python -m scripts.sprint1.scheduler --plan       # package table with readiness/MISSING flags
python -m scripts.sprint1.scheduler --dry-run --once   # nothing is started
```

Then the always-on scheduler, in its own tmux window:

```bash
tmux new-session -d -s x3-sprint1 \
  'cd /root/autodl-tmp/x3 && python -m scripts.sprint1.scheduler \
     --packages packages.yaml --state state.json --poll 10 >> logs/sprint1/scheduler.log 2>&1'
tmux ls
```

## What the scheduler guarantees

* Work-conserving: lanes A (5 threads), B (3 threads, `nice -n 10`), G (GPU, two light packages or one heavy), N (network).
* A lane with no ready package takes the first backlog entry of its lane, so every lane keeps work queued.
* Time box: `eta_min x 1.5`, then `TIMEOUT` with the partial results kept and the next package started.
* `util.csv` gets a line every 5 minutes (GPU util/memory, load, per-lane package) and the watchdog applies rules i-iv; `sprint1_progress.md` gets a line every 30 minutes.
* A package counts as `DONE` only when its declared artifacts exist non-empty; the artifacts (bytes, mtime) are recorded in `state.json`.

## Recovery

```bash
python -c "import json;s=json.load(open('state.json'));print({k:v['status'] for k,v in sorted(s.items())})"
tmux capture-pane -pt x3-sprint1:0 -S -20
tail -20 logs/sprint1/*.log
```

A package stuck in `RUNNING` after a crash is restarted by deleting its entry from
`state.json` (the artifacts stay on disk; a package whose artifacts already exist is
only re-run if its status is not `DONE`). `MISSING` means the script for that
package has not been written yet - it blocks `ALL_DONE` on purpose.

## Known gap to close before the run

`NEW_TASKS 09` (the source of `G40/G41/G42`, the `M-A ... M-G` definitions and the
`P1/P2/P3` details) is not in the repo. The packages that need it are exactly the
method packages: `S1-B, S2-A ... S2-H, S3-A ... S3-F, S4-A` and every `X-*` probe.
They are declared in `packages.yaml` already, so the scheduler reports them as
`MISSING` until the code lands - visible, not silently skipped.

#!/usr/bin/env bash
# X3 sprint 1 (NEW_TASKS 10) bootstrap for an AutoDL/HPC box.
#
# Runs once per machine: caches on the data disk, a machine-specific deploy key,
# the repo clone/update, and the directory skeleton the scheduler writes into.
# Nothing here starts a formal run.
#
#   bash scripts/autodl_setup.sh
#
# Environment overrides: REPO_NAME, WORK_ROOT, REPO_DIR, GITHUB_SSH, BRANCH, KEY.
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH   # so plain `python` is the sandbox interpreter

REPO_NAME=${REPO_NAME:-x3}
WORK_ROOT=${WORK_ROOT:-/root/autodl-tmp}
REPO_DIR=${REPO_DIR:-$WORK_ROOT/$REPO_NAME}
GITHUB_SSH=${GITHUB_SSH:-git@github.com:Yulivu/x3.git}
GITHUB_HTTPS=${GITHUB_HTTPS:-https://github.com/Yulivu/x3.git}
BRANCH=${BRANCH:-x3/main}
KEY=${KEY:-/root/.ssh/id_ed25519_x3}

echo "== 1. caches on the data disk (system disk holds code only) =="
mkdir -p "$WORK_ROOT"/cache/{hf,torch,pip} "$WORK_ROOT"/tmp "$WORK_ROOT"/data
export HF_HOME="$WORK_ROOT/cache/hf"
export TORCH_HOME="$WORK_ROOT/cache/torch"
export PIP_CACHE_DIR="$WORK_ROOT/cache/pip"
export TMPDIR="$WORK_ROOT/tmp"
if [ -f /etc/network_turbo ]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
  echo "network_turbo: sourced"
fi
echo "HF_HOME=$HF_HOME TMPDIR=$TMPDIR"

echo "== 2. machine-specific deploy key (never a local private key) =="
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [ ! -f "$KEY" ]; then
  ssh-keygen -t ed25519 -C "autodl_${REPO_NAME}_$(date +%Y%m%d_%H%M)" -f "$KEY" -N ""
fi
cat > /root/.ssh/config <<EOF
Host github.com
  HostName github.com
  User git
  IdentityFile $KEY
  IdentitiesOnly yes
EOF
chmod 600 /root/.ssh/config "$KEY"
chmod 644 "$KEY.pub"
echo "-- public key (add it to the GitHub repo as a deploy key if the clone fails) --"
cat "$KEY.pub"

echo "== 3. clone or fast-forward $BRANCH =="
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$GITHUB_SSH" "$REPO_DIR" \
    || git clone "$GITHUB_HTTPS" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin --prune
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
git pull --ff-only || echo "WARN: pull --ff-only refused (local commits on the box?)"
git rev-parse --short HEAD

echo "== 4. directories the sprint writes into =="
mkdir -p logs/sprint1 control sprint1/audio sprint1/tables sprint1/pca sprint1/graph \
         sprint1/results sprint1/feats sprint1/counterfactual results paper_tables \
         paper_figs export

echo "== 5. runtime versions =="
PY=${PY:-$(command -v python3 || command -v python)}
echo "python at: $PY"
"$PY" - <<'PY'
import platform, sys
print("python", sys.version.split()[0], platform.platform())
for name in ("numpy", "pandas", "sklearn", "torch", "soundfile", "pyarrow", "yaml"):
    try:
        mod = __import__(name)
        print(f"{name:11s}", getattr(mod, "__version__", "?"))
    except Exception as exc:
        print(f"{name:11s} MISSING ({type(exc).__name__})")
try:
    import torch
    print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception:
    pass
PY

echo "== setup done: next is 'python -m scripts.sprint1.preflight --out preflight.md' =="

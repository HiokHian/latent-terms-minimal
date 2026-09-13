#!/usr/bin/env bash
# One-time pod setup.
# Run this once after provisioning a new RunPod pod.
# Safe to re-run: all steps are idempotent.
set -euo pipefail

# 1. Install uv (lives in /root, wiped on restart — reinstall each session if needed)
if ! command -v uv &>/dev/null; then
    curl -Lsf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:$PATH"
fi

# 2. Create env.sh on /workspace (survives restarts) if not already present
ENV_SH=/workspace/env.sh
if [ ! -f "$ENV_SH" ]; then
    cat > "$ENV_SH" <<'EOF'
# Single source of truth for machine-level cache locations on this pod.
# Source from ~/.bashrc:  [ -f /workspace/env.sh ] && source /workspace/env.sh
#
# NOTE: the container image ships HF_HOME=/workspace/.cache/huggingface (a path that
# does not exist). This file overrides it. Anything running in a NON-interactive shell
# that does not source this file will still see the broken container default — so pass
# HF_HOME explicitly in nohup/systemd-style invocations, or source this file first.

export HF_HOME=/workspace/cache/hf
export UV_CACHE_DIR=/workspace/cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/cache/uv/python
export PIP_CACHE_DIR=/workspace/cache/pip
export TMPDIR=/workspace/cache/tmp
export HF_XET_HIGH_PERFORMANCE=1
export PATH="/root/.local/bin:$PATH"
EOF
    echo "Created $ENV_SH"
fi
# shellcheck source=/dev/null
source "$ENV_SH"

# 3. Wire env.sh into .bashrc for interactive shells
if ! grep -q 'workspace/env.sh' ~/.bashrc; then
    echo '[ -f /workspace/env.sh ] && source /workspace/env.sh' >> ~/.bashrc
fi

# 4. Create cache directories
mkdir -p /workspace/cache/{hf,uv,pip,tmp}

# 5. Check disk before syncing (MooseFS volume; du is slow — worth it here)
echo "Checking disk usage (may take ~30s on MooseFS)..."
USED_GB=$(df /workspace | awk 'NR==2{printf "%.1f", $3/1048576}')
echo "Workspace used: ~${USED_GB} GB (60 GB quota)"

# 6. Check for two-generation uv cache (prune frees ~8 GB when two date generations exist)
if command -v uv &>/dev/null && [ -d /workspace/cache/uv/archive-v0 ]; then
    N_DATES=$(ls --time-style=+%b%d /workspace/cache/uv/archive-v0 2>/dev/null | \
              awk '{print $6}' | sort -u | wc -l)
    if [ "$N_DATES" -gt 1 ]; then
        echo "Found $N_DATES date generations in uv cache. Running prune..."
        uv cache prune
    fi
fi

# 7. Sync Python environment
cd "$(dirname "$0")/.."
uv sync

# 8. Verify the install is complete (truncated installs look fine but fail at import)
.venv/bin/python - <<'PY'
import csv, pathlib
sp = pathlib.Path('.venv/lib').glob('python*/site-packages').__next__()
bad = []
for rec in sp.glob('*.dist-info/RECORD'):
    missing = sum(1 for r in csv.reader(open(rec, newline=''))
                  if r and r[0] and not r[0].startswith('..') and not (sp / r[0]).exists())
    if missing:
        bad.append(f"{rec.parent.name}: {missing} files missing")
if bad:
    print("INCOMPLETE PACKAGES:")
    for b in bad:
        print(" ", b)
    print("Run: uv sync --reinstall")
else:
    print("Install verified: all package files present")
PY

# 9. Quick import smoke test
.venv/bin/python -c "
import torch
import lightning
import latent_terms
print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'lightning={lightning.__version__}')
print('OK')
"

echo ""
echo "Bootstrap complete. Source env in new shells:"
echo "  source /workspace/env.sh"

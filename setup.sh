#!/usr/bin/env bash
# setup.sh — create venv, activate, install ksteer and all dependencies
# Usage: source setup.sh          ← activates the env in your current shell
#        bash setup.sh            ← installs only (env not inherited by parent shell)

set -euo pipefail

VENV_DIR="${KSTEER_VENV:-$(pwd)/.venv}"
PYTHON="${KSTEER_PYTHON:-python3}"

# ── 1. Verify Python version ────────────────────────────────────────────────
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ) ]]; then
    echo "ERROR: Python 3.10+ required (found $PY_VERSION). Set KSTEER_PYTHON to a suitable interpreter."
    return 1 2>/dev/null || exit 1
fi
echo "Python $PY_VERSION OK"

# ── 2. Create venv if it does not exist ────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# ── 3. Activate ─────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "Activated: $(which python)"

# ── 4. Upgrade pip + wheel silently ─────────────────────────────────────────
pip install --quiet --upgrade pip wheel

# ── 5. Install PyTorch (CUDA 12.1 build by default) ─────────────────────────
# Override with: KSTEER_TORCH_INDEX=https://download.pytorch.org/whl/cpu bash setup.sh
TORCH_INDEX="${KSTEER_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

if python -c "import torch" 2>/dev/null; then
    echo "PyTorch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "Installing PyTorch from $TORCH_INDEX ..."
    pip install --quiet torch torchvision --index-url "$TORCH_INDEX"
fi

# ── 6. Install remaining dependencies ───────────────────────────────────────
echo "Installing dependencies from requirements.txt ..."
pip install --quiet -r requirements.txt

# ── 7. Install ksteer in editable mode ──────────────────────────────────────
echo "Installing ksteer (editable) ..."
pip install --quiet -e .

# ── 8. Smoke test ────────────────────────────────────────────────────────────
python - <<'EOF'
import torch
from ksteer import LayerNormProfiler, NormProfile, CeilingSweeper
from ksteer import ContrastiveExtractor, BehavioralVector, load_behavior_pairs
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"ksteer imports OK | torch={torch.__version__} | device={device}")
EOF

echo ""
echo "Setup complete. Environment: $VENV_DIR"
echo "To activate in a new shell: source $VENV_DIR/bin/activate"

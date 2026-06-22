#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# HW4 (DPO) — set up a virtual environment and run all five experiments (A–E).
#
# Run this on a machine with a CUDA GPU (the script trains Qwen2.5-0.5B with a
# frozen reference copy; bf16 + CUDA are required by train_dpo.py).
#
# Usage:
#   bash run_all.sh                 # create venv, install deps, run A–E
#   SKIP_INSTALL=1 bash run_all.sh  # reuse existing venv, skip pip install
#   PYTHON=python3.10 bash run_all.sh
#   bash run_all.sh A C             # run only specific experiments (e.g. A and C)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"          # interpreter used to create the venv
VENV_DIR="${VENV_DIR:-.venv}"        # virtual-environment directory
SKIP_INSTALL="${SKIP_INSTALL:-0}"    # set to 1 to skip dependency installation
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Create / activate the virtual environment ──────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtual environment in $VENV_DIR (using $PYTHON)"
    "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "==> Using interpreter: $(python -V) at $(command -v python)"

# ── 2. Install dependencies (recommended versions from the assignment) ─────────
if [ "$SKIP_INSTALL" != "1" ]; then
    echo "==> Upgrading pip and installing dependencies"
    python -m pip install --upgrade pip
    # Install a CUDA-enabled torch first (adjust the index URL to your CUDA version
    # if needed, e.g. cu121). Remove --index-url to use the default build.
    python -m pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu121 \
        || python -m pip install "torch>=2.0"
    python -m pip install "transformers==5.11" "trl==1.6" "wandb==0.27" "datasets"
fi

# ── 3. Weights & Biases login check ───────────────────────────────────────────
# Either run `wandb login` beforehand, export WANDB_API_KEY, or set
# WANDB_MODE=offline to log locally without an account.
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ] && [ "${WANDB_MODE:-}" != "offline" ]; then
    echo "WARNING: no W&B credentials detected."
    echo "         Run 'wandb login', export WANDB_API_KEY, or set WANDB_MODE=offline."
fi

# ── 4. Sanity check the GPU ───────────────────────────────────────────────────
python - <<'PY'
import torch
assert torch.cuda.is_available(), "No CUDA GPU visible — train_dpo.py requires one."
print(f"==> CUDA OK: {torch.cuda.get_device_name(0)}")
PY

# ── 5. Define the five experiments ────────────────────────────────────────────
# Each entry: "<extra args> --run_name <name>"
declare -A RUNS=(
    [A]="--run_name default"                       # baseline: beta=0.1, lr=5e-7
    [B]="--beta 0.01 --run_name beta0.01"          # weak KL penalty
    [C]="--beta 0.5  --run_name beta0.5"           # strong KL penalty
    [D]="--learning_rate 5e-6 --run_name lr5e-6"   # 10x higher LR
    [E]="--learning_rate 5e-8 --run_name lr5e-8"   # 10x lower LR
)
ORDER=(A B C D E)

# Allow selecting a subset from the command line, e.g. `bash run_all.sh A C`.
if [ "$#" -gt 0 ]; then
    ORDER=("$@")
fi

# ── 6. Run them sequentially ──────────────────────────────────────────────────
mkdir -p logs
for key in "${ORDER[@]}"; do
    args="${RUNS[$key]:-}"
    if [ -z "$args" ]; then
        echo "!! Unknown run '$key' (valid: A B C D E) — skipping"
        continue
    fi
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "==> Run $key : python train_dpo.py $args"
    echo "════════════════════════════════════════════════════════════════════"
    python train_dpo.py $args 2>&1 | tee "logs/run_${key}.log"
done

echo ""
echo "==> All requested runs finished. Per-run logs are in ./logs/, checkpoints in ./dpo-*-final/."

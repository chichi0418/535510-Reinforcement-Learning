#!/usr/bin/env bash
# Run the four HW4 Problem-2(c) ablations (B–E) sequentially, each capped at
# 1000 optimizer steps. Run A (baseline) is done separately.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
export WANDB_MODE=offline WANDB_SILENT=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () {  # $1=run_name  $2..=extra args
    name="$1"; shift
    echo "==================== RUN $name ($*) ===================="
    python train_dpo.py --run_name "$name" --max_steps 1000 "$@" > "logs/run_${name}.log" 2>&1
    echo "---- $name exit=$? ----"
}

run beta0.01 --beta 0.01            # B: weak KL penalty
run beta0.5  --beta 0.5             # C: strong KL penalty
run lr5e-6   --learning_rate 5e-6   # D: 10x higher LR
run lr5e-8   --learning_rate 5e-8   # E: 10x lower LR

echo "==================== ALL B-E DONE ===================="

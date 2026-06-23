# Spring 2026, 535510 Reinforcement Learning
# HW4: DPO
# Instructor: Ping-Chun Hsieh (National Yang Ming Chiao Tung University)
#
# ──────────────────────────────────────────────────────────────────────────────
# How to run the Problem 2 experiments (one factor varied at a time):
#
#   Run A (baseline) : python train_dpo.py
#   Run B (beta=0.01): python train_dpo.py --beta 0.01          --run_name beta0.01
#   Run C (beta=0.5) : python train_dpo.py --beta 0.5           --run_name beta0.5
#   Run D (lr=5e-6)  : python train_dpo.py --learning_rate 5e-6 --run_name lr5e-6
#   Run E (lr=5e-8)  : python train_dpo.py --learning_rate 5e-8 --run_name lr5e-8
#
# Each run logs rewards/chosen, rewards/rejected, rewards/margins, rewards/accuracies
# and loss to its own W&B run, and writes its checkpoint to dpo-<run_name>/.
# ──────────────────────────────────────────────────────────────────────────────

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # use GPU 0; change to "1", "2", ... as needed

import argparse

import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig


# ══════════════════════════════════════════════════════════════════════════════
# 0. Command-line overrides (defaults reproduce the baseline in the HW table)
# ══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="DPO on UltraFeedback (HW4)")
parser.add_argument("--beta",          type=float, default=0.1,    help="KL penalty strength (default 0.1)")
parser.add_argument("--learning_rate", type=float, default=5e-7,   help="learning rate (default 5e-7)")
parser.add_argument("--run_name",      type=str,   default="default", help="W&B run name / output sub-dir")
parser.add_argument("--max_steps",     type=int,   default=-1,     help="cap training at N optimizer steps (-1 = full epoch)")
args_cli = parser.parse_args()

BETA          = args_cli.beta
LEARNING_RATE = args_cli.learning_rate
RUN_NAME      = args_cli.run_name
MAX_STEPS     = args_cli.max_steps


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data Inspection
# ══════════════════════════════════════════════════════════════════════════════

dataset       = load_dataset("trl-lib/ultrafeedback_binarized")
train_dataset = dataset["train"]
# Evaluate on a small held-out subset: the deliverable curves come from the
# per-step TRAIN logs; periodic eval over the full 1k-example test split costs
# ~2 min each on a 12 GB GPU, so a 256-example subset keeps eval cheap while
# still showing the held-out reward margin grows.
eval_dataset  = dataset["test"].select(range(256))

# TODO — Print and inspect
# ########## Your Code (2-3 lines)##########
print("dataset[0]:", train_dataset[0])
print("Keys:", list(train_dataset[0].keys()))
# ########## End of Your Code ###########
# Expected keys: "prompt", "chosen", "rejected"
# Each of "chosen" and "rejected" is a list of {"role": ..., "content": ...} dicts.

# Average token lengths (first 1000 examples)
_tokenizer_for_stats = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def assistant_text(turns):
    """Extract the assistant turn content from a conversation list."""
    for turn in turns:
        if turn["role"] == "assistant":
            return turn["content"]
    return ""

chosen_lengths   = []
rejected_lengths = []
for ex in train_dataset.select(range(1000)):
    chosen_lengths.append(
        len(_tokenizer_for_stats(assistant_text(ex["chosen"]))["input_ids"])
    )
    rejected_lengths.append(
        len(_tokenizer_for_stats(assistant_text(ex["rejected"]))["input_ids"])
    )

avg_chosen   = sum(chosen_lengths)   / len(chosen_lengths)
avg_rejected = sum(rejected_lengths) / len(rejected_lengths)
print(f"Avg chosen   token length: {avg_chosen:.1f}")
print(f"Avg rejected token length: {avg_rejected:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# Initialize Weight & Bias
# ══════════════════════════════════════════════════════════════════════════════

wandb.init(
    project = "dpo",
    name    = RUN_NAME,
    config  = {
        "model":                       "Qwen/Qwen2.5-0.5B-Instruct",
        "beta":                        BETA,
        "learning_rate":               LEARNING_RATE,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs":            1,
        "max_length":                  1024,
        # data stats logged for reference
        "avg_chosen_token_length":     round(avg_chosen,   1),
        "avg_rejected_token_length":   round(avg_rejected, 1),
    },
)


# ══════════════════════════════════════════════════════════════════════════════
# Model & Tokenizer
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Mixed-precision selection: bf16 needs Ampere+ (compute capability >= 8.0).
# On older GPUs (e.g. Volta/TITAN V, CC 7.0) bf16 has no tensor-core support,
# so we fall back to fp16, which Volta does accelerate.
_cc_major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
USE_BF16  = _cc_major >= 8
USE_FP16  = torch.cuda.is_available() and not USE_BF16
print(f"GPU compute capability major={_cc_major} -> bf16={USE_BF16}, fp16={USE_FP16}")

# Load the policy in a dtype consistent with the mixed-precision mode:
#  - bf16 path (Ampere+): load in the checkpoint's native dtype (bf16); bf16
#    training needs no grad-scaler.
#  - fp16 path (Volta):  load in fp32 so the fp16 AMP grad-scaler has fp32
#    master weights to unscale (it has no kernel for bf16 grads). Autocast still
#    runs the forward in fp16 on the GPU's tensor cores.
_model_dtype = "auto" if USE_BF16 else torch.float32

# Load model and tokenizer
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=_model_dtype)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ══════════════════════════════════════════════════════════════════════════════
# Training Configuration
# ══════════════════════════════════════════════════════════════════════════════

# Fill in baseline hyperparameters
training_args = DPOConfig(
    output_dir                  = f"dpo-{RUN_NAME}",
    num_train_epochs            = 1,
    max_steps                   = MAX_STEPS,   # -1 = run the full epoch
    # Effective batch = per_device * grad_accum = 8 (the value in the HW table).
    # On a 12 GB GPU we use per_device=1 / accum=8 instead of 2/4: the huge
    # 152k-vocab logits tensor (chosen+rejected concatenated) is what drives peak
    # memory, so halving the per-device micro-batch keeps us under the cap while
    # leaving the effective batch — and thus the optimization — unchanged.
    per_device_train_batch_size = 1,
    per_device_eval_batch_size  = 1,    # keep eval within the 12 GB budget too
    gradient_accumulation_steps = 8,    # effective batch = 8
    learning_rate               = LEARNING_RATE,
    beta                        = BETA,
    max_length                  = 1024,
    logging_steps               = 25,
    eval_strategy               = "steps",
    eval_steps                  = 250,
    save_strategy               = "epoch",
    bf16                        = USE_BF16,
    fp16                        = USE_FP16,
    # Gradient checkpointing trades compute for memory (activations are recomputed
    # in the backward pass) so the batch=2, max_length=1024 setup + a frozen
    # reference copy fits on a 12 GB GPU. It does NOT change the optimization math
    # or the logged metrics.
    gradient_checkpointing      = True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    # Optimizer: AdamW keeps two fp32 moment buffers per parameter (~4 GB for a
    # 0.5B model in fp32). On the 12 GB GPU that, on top of the fp32 policy + a
    # frozen fp32 reference + the 152k-vocab logits, overflows VRAM. Adafactor
    # keeps a factored (near-zero) second-moment state, freeing ~4 GB, so we use
    # it on the fp16/limited-VRAM path and the default AdamW on bf16/large GPUs.
    # None of the assignment's listed hyperparameters (lr, beta, effective batch,
    # max_length) are affected.
    optim                       = "adafactor" if USE_FP16 else "adamw_torch",
    report_to                   = "wandb",  # hands all trainer metrics to W&B
    run_name                    = RUN_NAME,
)


# ══════════════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════════════

# TODO: Instantiate DPOTrainer
# Check DPOTrainer and provide a proper initialization
# Such as model, args, processing_class, train_dataset, eval_dataset, and so on
# Note that ref_model is left unset since DPOTrainer shall create a frozen copy automatically
# ########## Your Code (5-10 lines)##########
trainer = DPOTrainer(
    model            = model,
    args             = training_args,
    processing_class = tokenizer,        # tokenizer / processor for the trainer
    train_dataset    = train_dataset,
    eval_dataset     = eval_dataset,
    # ref_model intentionally omitted: DPOTrainer auto-creates a frozen reference copy.
)
# ########## End of Your Code ###########

# Train
trainer.train()

# Peak VRAM (log to W&B as a summary metric)
peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak VRAM used: {peak_vram_gb:.2f} GB")
wandb.summary["peak_vram_gb"] = peak_vram_gb

# Dump the full metric history to JSON so the training curves can be re-plotted
# offline (independent of the W&B dashboard).
import json
with open(f"logs/history_{RUN_NAME}.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)
print(f"Metric history written to logs/history_{RUN_NAME}.json")

# Save checkpoints
final_dir = f"dpo-{RUN_NAME}-final"
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Model saved to {final_dir}/")

wandb.finish()

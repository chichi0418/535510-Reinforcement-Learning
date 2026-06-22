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
args_cli = parser.parse_args()

BETA          = args_cli.beta
LEARNING_RATE = args_cli.learning_rate
RUN_NAME      = args_cli.run_name


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data Inspection
# ══════════════════════════════════════════════════════════════════════════════

dataset       = load_dataset("trl-lib/ultrafeedback_binarized")
train_dataset = dataset["train"]
eval_dataset  = dataset["test"]

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

# Load model and tokenizer
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")
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
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,    # effective batch = 8
    learning_rate               = LEARNING_RATE,
    beta                        = BETA,
    max_length                  = 1024,
    logging_steps               = 25,
    eval_strategy               = "steps",
    eval_steps                  = 100,
    save_strategy               = "epoch",
    bf16                        = True,
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

# Save checkpoints
final_dir = f"dpo-{RUN_NAME}-final"
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Model saved to {final_dir}/")

wandb.finish()

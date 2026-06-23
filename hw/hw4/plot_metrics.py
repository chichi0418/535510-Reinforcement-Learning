# Spring 2026, 535510 Reinforcement Learning — HW4 (DPO)
# Re-plot the DPO training curves from the metric-history JSON files that
# train_dpo.py writes to logs/history_<run_name>.json.
#
#   python plot_metrics.py                 # plot every logs/history_*.json
#   python plot_metrics.py default beta0.01 ...   # only the named runs
#
# Produces:
#   plots/<run>_curves.png        4-panel (chosen / rejected / margins / loss) per run
#   plots/margins_compare.png     rewards/margins overlaid for all runs (Problem 2c)

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR  = "logs"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def load_history(run):
    with open(os.path.join(LOG_DIR, f"history_{run}.json")) as f:
        hist = json.load(f)
    # Keep only the per-step training-log entries (they carry 'loss').
    series = {}
    for row in hist:
        if "loss" not in row or "step" not in row:
            continue
        step = row["step"]
        for k in ("loss", "rewards/chosen", "rewards/rejected",
                  "rewards/margins", "rewards/accuracies"):
            if k in row:
                series.setdefault(k, ([], []))
                series[k][0].append(step)
                series[k][1].append(row[k])
    return series


def runs_from_args():
    if len(sys.argv) > 1:
        return sys.argv[1:]
    files = sorted(glob.glob(os.path.join(LOG_DIR, "history_*.json")))
    return [os.path.basename(f)[len("history_"):-len(".json")] for f in files]


def main():
    runs = runs_from_args()
    if not runs:
        print("No history_*.json files found in logs/.")
        return
    print("Plotting runs:", runs)

    # Per-run 4-panel figure.
    panels = ["rewards/chosen", "rewards/rejected", "rewards/margins", "loss"]
    for run in runs:
        s = load_history(run)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, key in zip(axes.flat, panels):
            if key in s:
                ax.plot(s[key][0], s[key][1], marker="o", ms=3)
            ax.set_title(key)
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"DPO training curves — run '{run}'")
        fig.tight_layout()
        out = os.path.join(PLOT_DIR, f"{run}_curves.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("wrote", out)

    # Overlaid rewards/margins comparison (Problem 2c).
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        s = load_history(run)
        if "rewards/margins" in s:
            ax.plot(s["rewards/margins"][0], s["rewards/margins"][1],
                    marker="o", ms=3, label=run)
    ax.set_title("rewards/margins across runs (A–E)")
    ax.set_xlabel("step")
    ax.set_ylabel("rewards/margins")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "margins_compare.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()

"""Pull the two HalfCheetah runs from W&B and make:
  1) eval/mean_return overlay (vanilla DDPG vs DDPG+CDQ)
  2) critic-loss panel (vanilla critic_loss vs CDQ critic1/critic2)
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "chichi-cs12-national-yang-ming-chiao-tung-university"
RUN_VANILLA = f"{ENTITY}/ddpg-halfcheetah/rxkfkqsc"
RUN_CDQ = f"{ENTITY}/ddpg-cdq-halfcheetah/lvdt216v"

OUT_DIR = os.path.dirname(__file__)


def fetch(run_path, keys):
    run = wandb.Api().run(run_path)
    hist = run.history(keys=keys, samples=20000, pandas=True)
    return hist


# ---- 1) eval comparison ---------------------------------------------------
ev_v = fetch(RUN_VANILLA, ["eval/mean_return", "eval/std_return"])
ev_c = fetch(RUN_CDQ, ["eval/mean_return", "eval/std_return"])

fig, ax = plt.subplots(figsize=(7.6, 4.4))
for df, label, color in [
    (ev_v, "Vanilla DDPG", "tab:blue"),
    (ev_c, "DDPG + CDQ", "tab:orange"),
]:
    df = df.dropna(subset=["eval/mean_return"]).sort_values("_step")
    s = df["_step"].to_numpy()
    m = df["eval/mean_return"].to_numpy()
    sd = df["eval/std_return"].to_numpy()
    ax.plot(s, m, label=label, color=color, lw=2)
    ax.fill_between(s, m - sd, m + sd, color=color, alpha=0.15)

ax.axhline(5000, color="red", ls="--", lw=1, label="Target = 5000")
ax.set_xlabel("environment steps")
ax.set_ylabel("eval mean return (20 episodes)")
ax.set_title("HalfCheetah-v5: vanilla DDPG vs DDPG + Clipped Double Q")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
out1 = os.path.join(OUT_DIR, "p4_eval_compare.png")
fig.savefig(out1, dpi=180)
print("saved:", out1)

# ---- 2) critic-loss panel -------------------------------------------------
cl_v = fetch(RUN_VANILLA, ["train/critic_loss"])
cl_c = fetch(RUN_CDQ, ["train/critic1_loss", "train/critic2_loss"])


def smooth(y, k=200):
    if len(y) < k:
        return y
    c = np.convolve(y, np.ones(k) / k, mode="valid")
    pad = np.full(len(y) - len(c), np.nan)
    return np.concatenate([pad, c])


fig, ax = plt.subplots(figsize=(7.6, 4.4))
df = cl_v.dropna(subset=["train/critic_loss"]).sort_values("_step")
ax.plot(df["_step"], smooth(df["train/critic_loss"].to_numpy()),
        label="Vanilla critic loss", color="tab:blue", lw=1.6)
df = cl_c.dropna(subset=["train/critic1_loss"]).sort_values("_step")
ax.plot(df["_step"], smooth(df["train/critic1_loss"].to_numpy()),
        label="CDQ critic1 loss", color="tab:orange", lw=1.6)
df = cl_c.dropna(subset=["train/critic2_loss"]).sort_values("_step")
ax.plot(df["_step"], smooth(df["train/critic2_loss"].to_numpy()),
        label="CDQ critic2 loss", color="tab:green", lw=1.6, ls="--")
ax.set_xlabel("environment steps")
ax.set_ylabel("critic loss (smoothed, window=200)")
ax.set_title("HalfCheetah-v5: critic Bellman loss")
ax.set_yscale("log")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
out2 = os.path.join(OUT_DIR, "p4_critic_loss_compare.png")
fig.savefig(out2, dpi=180)
print("saved:", out2)

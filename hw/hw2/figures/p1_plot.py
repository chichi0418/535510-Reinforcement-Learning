"""Plot the PPO-clip objective (original) vs the variant for A>0 and A<0."""
import numpy as np
import matplotlib.pyplot as plt

eps = 0.2
r = np.linspace(0.0, 2.0, 1001)


def L_orig(r, A, eps):
    return np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A)


def L_var(r, A, eps):
    return np.clip(r, 1 - eps, 1 + eps) * A


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

# Left: A > 0
A = 1.0
ax = axes[0]
ax.plot(r, L_orig(r, A, eps), label=r"Original $L^{clip}_{s,a}$", lw=2.2)
ax.plot(r, L_var(r, A, eps), label=r"Variant $\tilde L^{clip}_{s,a}$",
        lw=2.2, ls="--")
ax.axvline(1 - eps, color="gray", ls=":", lw=1)
ax.axvline(1 + eps, color="gray", ls=":", lw=1)
ax.axvline(1.0, color="black", ls=":", lw=0.8, alpha=0.5)
ax.set_title(r"$A>0$  (advantageous action)")
ax.set_xlabel(r"ratio $r=\pi_\theta/\pi_{\theta_k}$")
ax.set_ylabel("objective value")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
ax.annotate("clip range", xy=(1.0, ax.get_ylim()[0]),
            xytext=(1.0, A * (1 - eps) - 0.15),
            ha="center", fontsize=9, color="gray")

# Right: A < 0
A = -1.0
ax = axes[1]
ax.plot(r, L_orig(r, A, eps), label=r"Original $L^{clip}_{s,a}$", lw=2.2)
ax.plot(r, L_var(r, A, eps), label=r"Variant $\tilde L^{clip}_{s,a}$",
        lw=2.2, ls="--")
ax.axvline(1 - eps, color="gray", ls=":", lw=1)
ax.axvline(1 + eps, color="gray", ls=":", lw=1)
ax.axvline(1.0, color="black", ls=":", lw=0.8, alpha=0.5)
ax.set_title(r"$A<0$  (disadvantageous action)")
ax.set_xlabel(r"ratio $r=\pi_\theta/\pi_{\theta_k}$")
ax.set_ylabel("objective value")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

fig.suptitle(
    r"PPO-clip objective: original vs variant  ($\varepsilon=0.2$)",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))

out_png = __file__.replace("p1_plot.py", "p1_clip_objective.png")
out_pdf = __file__.replace("p1_plot.py", "p1_clip_objective.pdf")
fig.savefig(out_png, dpi=180)
fig.savefig(out_pdf)
print("saved:", out_png)
print("saved:", out_pdf)

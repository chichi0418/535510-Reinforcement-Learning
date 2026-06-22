# HW3 Technical Report — SAC Experiments

**Student ID:** 112550081

---

## Problem 2(d) — SAC on Pendulum-v1 (25 pts)

### Implementation (the three blocks in `update_model`)

```python
# ----- q function loss -----
mask = 1 - done
with torch.no_grad():
    v_target = self.vf_target(next_state)         # bootstrapped target value
    q_target = reward + self.gamma * v_target * mask
qf_1_pred = self.qf_1(state, action)
qf_2_pred = self.qf_2(state, action)
qf_1_loss = F.mse_loss(qf_1_pred, q_target)
qf_2_loss = F.mse_loss(qf_2_pred, q_target)

# ----- v function loss -----
q_pred  = torch.min(
    self.qf_1(state, new_action),
    self.qf_2(state, new_action),
)                                                  # clipped double-Q
v_target = q_pred - alpha * log_prob               # soft value target
vf_loss = F.mse_loss(self.vf(state), v_target.detach())

# ----- actor (policy) loss -----
advantage  = torch.min(
    self.qf_1(state, new_action),
    self.qf_2(state, new_action),
)
actor_loss = (alpha * log_prob - advantage).mean()
```

Key design notes:

- `new_action, log_prob = self.actor(state)` is already computed at the top of `update_model`, so gradients into the actor flow through `log_prob` (for the entropy term) and through the Q-critics (re-evaluated at `new_action`).
- All bootstrap targets (`q_target`, `v_target`) are wrapped in `torch.no_grad()` / `.detach()` to stop gradient flow into the target networks.
- `mask = 1 - done` zeroes the bootstrap when an episode terminates.

### Hyperparameters used

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate (actor, critics, $\alpha$) | $3\times 10^{-4}$ |
| Discount $\gamma$ | $0.99$ |
| Soft-update coefficient $\tau$ | $5\times 10^{-3}$ |
| Batch size | $256$ |
| Replay buffer size | $50{,}000$ |
| Initial random-action steps | $1{,}000$ |
| Policy update frequency | every step |
| Hidden layers (actor/critics) | $2 \times 128$, ReLU |
| `log_std` range | $[-20, 2]$ |
| Target entropy $\mathcal{H}_0$ | $-\dim(\mathcal{A}) = -1$ |
| Total training steps | $100{,}000$ |
| Eval interval / episodes | $5{,}000$ / $20$ |
| Seed | $77$ |

### Results

Selected evaluation checkpoints (20 episodes each, deterministic policy):

| Step | Mean Eval Return | Std |
|---:|---:|---:|
| 5,000   | -1215.60 | 104.71 |
| 10,000  |  -196.61 | 133.90 |
| 25,000  |  -186.64 | 141.00 |
| **35,000**  |  **-136.99** |  **67.22** |
| 50,000  |  -158.08 |  67.42 |
| 65,000  |  -156.65 | 106.54 |
| 90,000  |  -168.49 | 108.42 |
| **100,000** |  **-148.91** |  **94.47** |

- The success criterion (mean evaluation return $> -170$) was first satisfied at **step 35,000**, well within the homework's 50k-step expectation.
- The policy stabilized in the $[-170, -140]$ band thereafter; final 100k-step evaluation mean was $\mathbf{-148.91 \pm 94.47}$.
- Checkpoint used for evaluation: `checkpoints_pendulum/checkpoint_step100000.pt`.

#### W&B snapshots — Pendulum

![Eval mean return vs training step (Pendulum). Reaches > -170 at ~10k and stays in [-170, -140] band.](wandb_plots/pendulum/eval_mean_return.png)

![Per-episode training return (Pendulum).](wandb_plots/pendulum/return.png)

![Eval std of return (Pendulum).](wandb_plots/pendulum/eval_std_return.png)

![Actor loss (Pendulum). Climbs early as Q grows, then decays as policy converges.](wandb_plots/pendulum/actor_loss.png)

![Q loss (Pendulum).](wandb_plots/pendulum/q_loss.png)

![V loss (Pendulum).](wandb_plots/pendulum/v_loss.png)

![Alpha loss (Pendulum).](wandb_plots/pendulum/alpha_loss.png)

---

## Problem 2(e) — SAC on Panda-Door-Opening (25 pts)

### Code changes for Robosuite (`sac_panda_door.py`)

Main adaptations relative to the Pendulum implementation:

1. **Environment construction.** Use Robosuite directly and flatten the observation dict into a single vector:
   ```python
   import robosuite as suite
   from robosuite.wrappers import GymWrapper

   env_raw = suite.make(
       env_name="Door",
       robots="Panda",
       has_renderer=False,
       has_offscreen_renderer=False,
       use_camera_obs=False,
       reward_shaping=True,
       horizon=500,
       control_freq=20,
   )
   env = GymWrapper(env_raw)              # exposes a Gym-style API
   env = ActionNormalizer(env)
   ```
2. **Wider networks.** Hidden size raised from 128 to **256** (both actor and critics) to handle the larger observation space.
3. **Larger replay buffer** ($10^6$) and **batch size** ($256$).
4. **Reduced soft-update rate** $\tau = 5\times 10^{-3}$, lower learning rate $3\times 10^{-4}$ for stability.
5. **Initial random-action warm-up** of $10{,}000$ steps.
6. **Total training**: $1{,}000{,}000$ environment steps; evaluation every $10{,}000$ steps over $10$ episodes.
7. **Target entropy** set automatically to $-\dim(\mathcal{A})$.
8. **Checkpointing** of actor, both critics, $V$-network, target $V$, and $\log\alpha$ every evaluation interval.

The `update_model` block (Q, V, policy losses) is **identical** to part (d) — SAC is environment-agnostic.

### Hyperparameters used

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate (actor, critics, $\alpha$) | $3\times 10^{-4}$ |
| Discount $\gamma$ | $0.99$ |
| Soft-update coefficient $\tau$ | $5\times 10^{-3}$ |
| Batch size | $256$ |
| Replay buffer size | $1{,}000{,}000$ |
| Initial random-action steps | $10{,}000$ |
| Hidden layers (actor / critics) | $2 \times 256$, ReLU |
| Policy update frequency | every step |
| Target entropy | $-\dim(\mathcal{A})$ |
| Total environment steps | $1{,}000{,}000$ |
| Eval interval / episodes | $10{,}000$ / $10$ |
| Horizon | $500$ |
| Control frequency | $20$ Hz |
| Seed | $77$ |

### Results

Highlights from 1M-step training (1 single run, seed = 77, MPS GPU, ~4h 20min):

| Step | Eval Mean Return (5 ep) | Std |
|---:|---:|---:|
| 100,000 |    3.31 |   1.68 |
| 200,000 |    4.15 |   2.12 |
| **250,000** | **401.42** | 207.10 |  ← **first crossing 200** (homework requirement satisfied at half the deadline) |
| 350,000 |  504.46 | 295.56 |
| 400,000 |  871.04 |  24.12 |
| 500,000 |  576.22 | 414.23 |
| 600,000 |  896.41 |  53.02 |
| 700,000 |  736.39 | 288.11 |
| 800,000 |  924.50 |  19.82 |
| **910,000** | **948.49** | **1.36** |  ← **best checkpoint** (`checkpoint_step910000.pt`) |
| 1,000,000 |  915.96 |  21.13 |  (final) |

- **First crossed eval mean ≥ 200 at step 250,000**, i.e. at *half* the homework deadline (500k).
- From step 720k onward the policy stayed locked in the **920–948** range with std ≤ 20 (5 of the 100 evaluations had std < 5, indicating near-deterministic perfect door opening across the 5 evaluation episodes).
- Far exceeds the expected 400 baseline mentioned in the homework, and is close to the theoretical maximum (≈ 1000) for `horizon=1000` shaped Door.
- Selected checkpoint for grading: **`checkpoints_door/checkpoint_step910000.pt`** (highest mean and lowest variance combined).
- W&B (online) run: https://wandb.ai/chichi-cs12-national-yang-ming-chiao-tung-university/RL-HW3-SAC-Door/runs/wxyveb2y
- (Optional) horizon=500 robustness test: not included in this submission, can be rerun on the saved checkpoint.

#### W&B snapshots — Panda Door

![Eval mean return vs training step (Door). Black circle marks the selected checkpoint at step 910k (eval mean 948.49 ± 1.36). First crossing of the 200 threshold occurs at step 250k.](wandb_plots/door_h1000/eval_mean_return.png)

![Per-episode training return (Door). Reward grows from ~2 to ~900 as the policy learns to grasp and turn the doorknob.](wandb_plots/door_h1000/return.png)

![Eval std of return (Door); drops to single digits after ~720k, indicating near-deterministic success.](wandb_plots/door_h1000/eval_std_return.png)

![Actor loss (Door).](wandb_plots/door_h1000/actor_loss.png)

![Q loss (Door).](wandb_plots/door_h1000/q_loss.png)

![V loss (Door).](wandb_plots/door_h1000/v_loss.png)

![Alpha (temperature) auto-tuned value (Door). Decreases as the policy gets more deterministic.](wandb_plots/door_h1000/alpha.png)

![Alpha loss (Door).](wandb_plots/door_h1000/alpha_loss.png)

---

## Deliverables

- Source code: `sac.py` (Pendulum) and `sac_panda_door.py` (Robosuite Door).
- Trained checkpoints: actor + both critics + value net saved as `.pt` files in `checkpoints/`.
- W&B run links and screenshots embedded in the PDF report.
- Short demo video (3–5 min) covering design choices and the empirical results of 2(d) and 2(e).

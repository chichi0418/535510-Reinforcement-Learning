# 535510 Reinforcement Learning
# Pre-Lecture Assignment 5

**Instructor: Ping-Chun Hsieh**
**Email: pinghsieh@nycu.edu.tw** &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; **Department of CS, NYCU**

---

## Problem: Model-Based RL

> In Lecture 21, we proceed to study Model-based RL methods. Let's try to get familiar with these methods and get prepared for Lecture 22. You can refer to the (1) attached reference slides on model-based RL and the (2) TD-MPC paper (https://arxiv.org/abs/2203.04955) as a starting point when answering the questions below.

### TD-MPC Paper Reference Notes
**"Temporal Difference Learning for Model Predictive Control"** — Hansen, Wang & Su, ICML 2022 ([arXiv:2203.04955](https://arxiv.org/abs/2203.04955)) · Code: https://nicklashansen.github.io/td-mpc

**Core Idea:** TD-MPC combines Model Predictive Control (MPC) with model-free TD-learning by jointly learning a *task-oriented latent dynamics model* and a *terminal value function* through temporal difference learning. At each step, short-horizon trajectory optimization (using the learned model for reward estimates) is combined with a learned value function for long-term return.

**TOLD Model (Task-Oriented Latent Dynamics):** Five jointly-learned components:
- **h_θ(s_t)** — Encoder: maps observations to latent state z_t
- **d_θ(z_t, a_t)** — Latent dynamics: predicts next latent state z_{t+1}
- **R_θ(z_t, a_t)** — Reward predictor: predicts single-step reward
- **Q_θ(z_t, a_t)** — Value function: state-action value (TD-trained)
- **π_θ(z_t)** — Policy: guides and augments trajectory sampling

**Training Objective (3-term loss per step):**
1. **Reward loss:** `‖R_θ(z_i, a_i) − r_i‖²` — reward prediction
2. **Value loss (TD):** `‖Q_θ(z_i, a_i) − (r_i + γ Q_{θ⁻}(z_{i+1}, π_θ(z_{i+1})))‖²` — fitted Q-iteration
3. **Latent consistency loss:** `‖d_θ(z_i, a_i) − h_{θ⁻}(s_{i+1})‖²` — enforces temporal consistency in latent space *without* predicting pixels/states

All three losses are summed over a rollout horizon H with temporal weighting λ^{i−t}, and gradients back-propagate through time.

**Inference (Algorithm — MPPI-based MPC):**
1. Encode current state: z_t = h_θ(s_t)
2. For J iterations: sample N trajectories from Gaussian(μ, σ²I) + N_π policy trajectories
3. Estimate return for each trajectory: sum of γᵗ R_θ(z_t, a_t) over horizon + γᴴ Q_θ(z_H, a_H) as terminal value
4. Update (μ, σ) via importance-weighted top-k estimate (Equations 4 & 5)
5. Execute first action of sampled trajectory; repeat (receding-horizon)

**Key Enhancements over Vanilla MPC (≥5):**
1. **Latent space model** — learns dynamics in a compressed task-relevant latent space, not raw state/pixel space
2. **Learned terminal value function** — replaces myopic finite-horizon with TD-trained Q_θ for long-term return
3. **Joint TD-learning** — model, value, and policy are all learned *jointly* end-to-end (gradients flow through all three losses)
4. **Latent state consistency loss** — modality-agnostic regularization; works identically for state and image inputs
5. **Policy-guided trajectory optimization** — π_θ augments CEM sampling, providing warm initialization and guided exploration
6. **Exploration by planning** — std. deviation σ is lower-bounded by a linearly decayed ε; planning horizon annealed from 1→H
7. **Receding-horizon with warm start** — reuses shifted mean μ from previous step to reduce iterations needed for convergence

**Results:** Evaluated on 92 tasks (DMControl + Meta-World). TD-MPC outperforms SAC, LOOP, Dreamer-v2, CURL, DrQ on most tasks; first method to solve complex Dog locomotion tasks (A ∈ ℝ³⁸) in 1M steps. Inference runs at ~50Hz (20ms/step) on a single GPU.

1) **(Parametric vs non-parametric models)** There are both parametric and non-parametric models. Classify each of the following as **parametric** or **non-parametric**, and briefly justify each of your answer: **(a)** A neural network dynamics model **(b)** A replay buffer used as an empirical transition model **(c)** A symbolic PDDL-style planning model **(d)** A Gaussian process dynamics model **(e)** A table of all observed transitions in a small gridworld

2) **(Model Predictive Control, MPC)** MPC repeatedly plans over a short horizon and executes only the first action.

   (a) Describe the MPC procedure step by step.

   (b) Why does MPC usually execute only the first planned action?

   (c) Give one real-world example of a task where MPC would be useful.

   (d) What's the main difference between vanilla MPC and TD-MPC? (Please mention at least 5 enhancements made by TD-MPC)

---

## Submission Guidelines and Remarks

- Please write all your responses within **1 page** (**A4 paper**, either hand-written or typeset). The responses that go beyond 1 page will *not* be graded.

- Your deliverable shall be a PDF file. Please put all your write-ups into one PDF file (photos/scanned copies are acceptable) and submit your deliverable via E3.

- Please submit it by **9pm, 5/28 (Thursday)**.

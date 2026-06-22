# Pre-Lecture Assignment 2 — Answers

## Q1 — Non-uniform Polyak-Łojasiewicz (PL)

### (a) Why "non-uniform"?

In the standard (uniform) PL condition, the gradient lower bounds the optimality gap with a **constant** coefficient independent of $\theta$. Here, the coefficient

$$C(\theta) = \frac{\min_{s} \pi_\theta(a^*(s) \mid s)}{\sqrt{S} \cdot \|d_\rho^{\pi^*}/d_\mu^{\pi_\theta}\|_\infty}$$

**depends on $\theta$**: both $\min_s \pi_\theta(a^*(s)|s)$ and $d_\mu^{\pi_\theta}$ change as the policy parameters update. The PL "constant" is non-uniform across the parameter space — it can be large when $\pi_\theta$ is near-optimal and vanishingly small otherwise.

### (b) Origin of $\min_s \pi_\theta(a^*(s) \mid s)$

The proof of Lemma 8 proceeds in three key steps:

**Step 1 — Restrict to optimal-action coordinates:**
$$\left\|\nabla_\theta V^{\pi_\theta}(\mu)\right\|_2 \geq \frac{1}{\sqrt{S}}\sum_s \left|\frac{\partial V^{\pi_\theta}(\mu)}{\partial \theta_{s,\,a^*(s)}}\right|$$

**Step 2 — Substitute the softmax PG formula (Lemma 1):**
$$\frac{\partial V^{\pi_\theta}(\mu)}{\partial \theta_{s,a}} = \frac{1}{1-\gamma}\, d_\mu^{\pi_\theta}(s)\, \pi_\theta(a \mid s)\, A^{\pi_\theta}(s,a)$$

Evaluating at $a = a^*(s)$, the factor $\pi_\theta(a^*(s)|s)$ appears inside the sum. Since this factor is state-dependent, it cannot yet be factored out.

**Step 3 — Lower-bound $\pi_\theta(a^*(s)|s)$ by its minimum:**

To apply the performance difference lemma and convert $\sum_s d_\mu^{\pi_\theta}(s)\,A^{\pi_\theta}(s,a^*(s))$ into $V^*(\rho)-V^{\pi_\theta}(\rho)$, the proof pulls $\pi_\theta(a^*(s)|s)$ outside the sum by replacing it with a state-independent lower bound:
$$\sum_s d_\mu^{\pi_\theta}(s)\,\pi_\theta(a^*(s)|s)\,A^{\pi_\theta}(s,a^*(s)) \geq \min_s\pi_\theta(a^*(s)|s)\cdot\sum_s d_\mu^{\pi_\theta}(s)\,A^{\pi_\theta}(s,a^*(s))$$

Hence $\min_s \pi_\theta(a^*(s)|s)$ originates from the **softmax gradient formula** (Step 2) and is **extracted via a min lower-bound** (Step 3) to enable the final conversion to the optimality gap.

### (c) Difference between $\mu$ and $\rho$

- $\mu$: the initial state distribution for **gradient computation** — determines the state visitation $d_\mu^{\pi_\theta}$ used in the policy gradient.
- $\rho$: the initial state distribution for **performance evaluation** — the optimality gap $V^*(\rho) - V^{\pi_\theta}(\rho)$ is measured under $\rho$.

They can differ. The ratio $\|d_\rho^{\pi^*}/d_\mu^{\pi_\theta}\|_\infty$ captures the distribution mismatch between training ($\mu$) and evaluation ($\rho$) — the larger it is, the harder convergence becomes.

---

## Q2 — QAC Algorithm (Lines 100–119)

### (a) Math Equations

**Critic forward** (L101–102): Compute $Q_w(s_t, a_t)$ by indexing the critic output at the taken action.

**TD target** (L104–112):

$$\hat{V}^{\pi_\theta}(s_{t+1}) = \sum_{a} \pi_\theta(a \mid s_{t+1})\, Q_w(s_{t+1}, a)$$

$$y_t = \begin{cases} r_t & \text{if done} \\ r_t + \gamma\, \hat{V}^{\pi_\theta}(s_{t+1}) & \text{otherwise} \end{cases}$$

This is an **Expected SARSA** target: it uses the policy-weighted expectation $\mathbb{E}_{a \sim \pi_\theta}[Q_w(s_{t+1}, a)]$ rather than a single sampled next action.

**Critic update** (L115–119):

$$\mathcal{L}(w) = \bigl(Q_w(s_t, a_t) - y_t\bigr)^2, \qquad w \leftarrow w - \alpha_w \nabla_w \mathcal{L}(w)$$

### (b) Do you agree with GPT? — **No, two issues.**

**Issue 1 (algorithmic):** Standard QAC uses a **SARSA-style** single-sample target $y_t = r_t + \gamma\, Q_w(s_{t+1}, a_{t+1})$ where $a_{t+1} \sim \pi_\theta$. GPT uses **Expected SARSA** $\left(\sum_a \pi_\theta(a|s') Q_w(s',a)\right)$ instead. While this reduces variance, it deviates from the standard QAC formulation and requires an extra forward pass through both networks.

**Issue 2 (terminal handling):** Line 112 uses `done = terminated or truncated` (from L94) to decide whether to bootstrap. When an episode is **truncated** (e.g., max steps reached), the state is not truly terminal — bootstrapping should still occur. The correct condition should use only `terminated`:

$$y_t = r_t + \gamma\,(1 - \mathbb{1}[\text{terminated}])\, \hat{V}^{\pi_\theta}(s_{t+1})$$

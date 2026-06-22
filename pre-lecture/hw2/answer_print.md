---
pdf_options:
  format: A4
  margin: 12mm 12mm 10mm 12mm
  printBackground: true
stylesheet: answer_style.css
math: true
---

# Pre-Lecture Assignment 2 — Answers

## Q1 — Non-uniform Polyak-Łojasiewicz (PL)

### (a) Why "non-uniform"?

In the standard (uniform) PL condition, the gradient lower bounds the optimality gap with a **constant** coefficient independent of $\theta$. Here, the coefficient $C(\theta) = \dfrac{\min_{s} \pi_\theta(a^*(s) \mid s)}{\sqrt{S} \cdot \|d_\rho^{\pi^*}/d_\mu^{\pi_\theta}\|_\infty}$ **depends on $\theta$**: both $\min_s \pi_\theta(a^*(s)|s)$ and $d_\mu^{\pi_\theta}$ change as the policy updates. The PL "constant" is non-uniform across parameter space — large when $\pi_\theta$ is near-optimal, vanishingly small otherwise.

### (b) Origin of $\min_s \pi_\theta(a^*(s) \mid s)$

From the softmax policy gradient theorem: $\dfrac{\partial V^{\pi_\theta}(\mu)}{\partial \theta_{s,a}} = \dfrac{1}{1-\gamma} d_\mu^{\pi_\theta}(s)\, \pi_\theta(a \mid s)\, A^{\pi_\theta}(s,a)$.
In the proof of Lemma 8, the gradient norm is lower bounded by restricting to optimal-action coordinates $a = a^*(s)$:

$$\left\|\nabla_\theta V^{\pi_\theta}(\mu)\right\|_2^2 \geq \sum_s \!\left(\frac{d_\mu^{\pi_\theta}(s)\,\pi_\theta(a^*(s)|s)\,A^{\pi_\theta}(s,a^*(s))}{1-\gamma}\right)^{\!2}$$

The factor $\pi_\theta(a^*(s)|s)$ appears directly from the softmax gradient. To extract a state-independent scalar, the proof takes $\min_{s}\,\pi_\theta(a^*(s)|s)$ as a uniform lower bound across all states.

### (c) Difference between $\mu$ and $\rho$

- $\mu$: initial state distribution for **gradient computation** — determines $d_\mu^{\pi_\theta}$ used in the policy gradient update.
- $\rho$: initial state distribution for **performance evaluation** — the optimality gap $V^*(\rho) - V^{\pi_\theta}(\rho)$ is measured under $\rho$.

They can differ. The ratio $\|d_\rho^{\pi^*}/d_\mu^{\pi_\theta}\|_\infty$ captures the distribution mismatch between training ($\mu$) and evaluation ($\rho$).

---

## Q2 — QAC Algorithm (Lines 100–119)

### (a) Math Equations

**Critic forward (L101–102):** $Q_w(s_t, a_t) = [\text{critic}(s_t)]_{a_t}$

**TD target (L104–112) — Expected SARSA:**

$$\hat{V}^{\pi_\theta}(s_{t+1}) = \sum_{a} \pi_\theta(a \mid s_{t+1})\, Q_w(s_{t+1}, a), \qquad y_t = \begin{cases} r_t & \text{if done} \\ r_t + \gamma\, \hat{V}^{\pi_\theta}(s_{t+1}) & \text{otherwise} \end{cases}$$

Rather than bootstrapping with a single sampled next action, the target uses the policy-weighted expectation $\mathbb{E}_{a\sim\pi_\theta}[Q_w(s_{t+1},a)]$.

**Critic update (L115–119):** $\mathcal{L}(w) = (Q_w(s_t,a_t)-y_t)^2,\quad w \leftarrow w - \alpha_w \nabla_w \mathcal{L}(w)$

### (b) Do you agree with GPT? — **No, two issues.**

**Issue 1 (algorithmic):** Standard QAC uses a SARSA-style target $y_t = r_t + \gamma Q_w(s_{t+1}, a_{t+1})$ where $a_{t+1}\!\sim\!\pi_\theta$. GPT uses Expected SARSA instead, which deviates from the standard QAC formulation (though it reduces variance).

**Issue 2 (terminal handling):** Line 112 uses `done = terminated or truncated` (L94) to stop bootstrapping. When an episode is **truncated** (max steps reached), it is not truly terminal — bootstrapping should still occur. The condition should use only `terminated`:

$$y_t = r_t + \gamma\,(1-\mathbb{1}[\text{terminated}])\,\hat{V}^{\pi_\theta}(s_{t+1})$$

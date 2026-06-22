# Pre-Lecture Assignment 3
**535510 Reinforcement Learning**

---

## Q1: Deterministic Policy Gradient (DPG)

$$\nabla_\theta V^{\pi_\theta}(\mu) = \frac{1}{1-\gamma}\,\mathbb{E}_{s \sim d^{\pi_\theta}_\mu}\!\left[\nabla_\theta \pi_\theta(s)\;\nabla_a Q^{\pi_\theta}(s,a)\big|_{a=\pi_\theta(s)}\right]$$

**Term-by-term:**
- $\nabla_\theta V^{\pi_\theta}(\mu)$: gradient of the expected discounted return w.r.t. policy parameters $\theta$, starting from initial distribution $\mu$.
- $\frac{1}{1-\gamma}$: normalization constant from the discounted state visitation distribution (comes from $\sum_{t=0}^\infty \gamma^t$).
- $d^{\pi_\theta}_\mu$: normalized discounted state visitation distribution under the current deterministic policy — how frequently each state is visited.
- $\nabla_\theta \pi_\theta(s)$: Jacobian of the deterministic policy; how the output action changes as $\theta$ varies at state $s$.
- $\nabla_a Q^{\pi_\theta}(s,a)\big|_{a=\pi_\theta(s)}$: gradient of the action-value function w.r.t. action $a$, evaluated at the policy's current action — the direction in action space that increases value.

**Intuition:** To improve the policy, at every visited state we chain two gradients: (1) which direction to move the action to increase Q (action-value gradient), and (2) how to adjust $\theta$ to produce that action (policy Jacobian). This is a simple closed-form chain rule — no action sampling needed.

**Difference from Stochastic PG (P3):**
Stochastic PG (P3): $\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s \sim \rho^\pi,\, a \sim \pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a|s)\, Q^\pi(s,a)\right]$

| | Stochastic PG (P3) | DPG |
|---|---|---|
| Expectation | Over **states and actions** | Over **states only** |
| Gradient form | Score function $\nabla_\theta \log \pi_\theta(a|s)$ | Chain rule $\nabla_\theta \pi_\theta(s)\cdot\nabla_a Q$ |
| Sample cost | Must sample actions to estimate | No action sampling; computed in closed form |
| Exploration | Inherent via stochastic policy | Requires a separate behaviour policy |

Because DPG avoids integrating over the action space, it requires far fewer samples — a critical advantage in high-dimensional action spaces.

---

## Q2: Off-Policy Learning

**Definition:** Off-policy learning is a paradigm where an agent learns about a **target policy** $\pi$ (the policy being optimized) while collecting experience by following a **different behaviour policy** $\beta$. This decouples data collection from policy improvement, allowing reuse of past experience or exploration-driven data.

**Real-world example — Robot arm training with a human-guided teacher:**  
A robot arm needs to learn a precise deterministic grasping policy $\mu_\theta(s)$. To ensure sufficient exploration of the workspace, a human teleoperator (or a noisy controller) generates demonstrations using a broad exploratory policy $\beta(a|s)$ — nudging the arm in many random directions. The RL algorithm (e.g., COPDAC-Q) then uses those $(s, a, r, s')$ tuples to update $\mu_\theta$ via the off-policy DPG:

$$\nabla_\theta J_\beta(\mu_\theta) = \mathbb{E}_{s \sim \rho^\beta}\!\left[\nabla_\theta \mu_\theta(s)\,\nabla_a Q^\mu(s,a)\big|_{a=\mu_\theta(s)}\right]$$

**What makes it off-policy:** The transitions are sampled from $\rho^\beta$ (the state distribution induced by the human/noisy controller), not from $\rho^{\mu_\theta}$ (the target policy's own distribution). The agent evaluates and improves $\mu_\theta$ using data it never generated itself. In the stochastic case this requires importance-sampling corrections $\frac{\pi(a|s)}{\beta(a|s)}$; in DPG, since the policy is deterministic and the action integral vanishes, importance sampling in the actor is not needed.

---

## Q3: Why DPG Suits Continuous Action Spaces

1. **No $\arg\max$ over actions:** In value-based methods for continuous actions, policy improvement often requires solving $\arg\max_a Q(s,a)$ at every step — an expensive inner optimization. DPG replaces this with a simple gradient step along $\nabla_a Q$, computed via the chain rule through a differentiable policy and critic.

2. **Sample efficiency:** Stochastic PG must integrate (sample) over the action space. In continuous, high-dimensional action spaces this requires many action samples per state to get a low-variance gradient estimate. DPG computes the gradient analytically, integrating over states only — making it orders of magnitude more efficient as action dimensionality grows (empirically demonstrated on 10/25/50-dimensional bandit tasks in Silver et al., 2014).

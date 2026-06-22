# Deterministic Policy Gradient Algorithms

**Authors:** David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller  
**Venue:** Proceedings of the 31st International Conference on Machine Learning (ICML), Beijing, China, 2014. JMLR: W&CP volume 32.

---

## Abstract

This paper considers **deterministic policy gradient algorithms** for reinforcement learning with continuous actions. The deterministic policy gradient has a particularly appealing form: it is the expected gradient of the action-value function. This simple form means the deterministic policy gradient can be estimated much more efficiently than the usual stochastic policy gradient.

To ensure adequate exploration, the authors introduce an **off-policy actor-critic algorithm** that learns a deterministic target policy from an exploratory behaviour policy. They demonstrate that deterministic policy gradient algorithms can **significantly outperform** their stochastic counterparts in high-dimensional action spaces.

---

## 1. Introduction

### Standard (Stochastic) Policy Gradient
- Policy is represented as a parametric probability distribution $\pi_\theta(a|s) = \mathbb{P}[a|s;\theta]$
- Algorithms proceed by sampling this stochastic policy and adjusting parameters in the direction of greater cumulative reward

### Deterministic Policies
- Instead consider **deterministic policies** $a = \mu_\theta(s)$
- Previously believed the deterministic policy gradient did not exist, or could only be obtained using a model (Peters, 2010)
- This paper shows: the deterministic policy gradient **does exist**, is model-free, and simply follows the gradient of the action-value function
- DPG is the **limiting case** (as policy variance → 0) of the stochastic policy gradient

### Key Practical Difference
- **Stochastic PG:** integrates over both state and action spaces
- **Deterministic PG:** integrates over state space only
- → Computing deterministic PG requires **much fewer samples**, especially in high-dimensional action spaces

### Off-Policy Exploration
- A stochastic behaviour policy $\beta(a|s)$ is used for exploration
- A deterministic target policy $\mu_\theta(s)$ is learned from those trajectories (off-policy)
- The **COPDAC** (Compatible Off-Policy Deterministic Actor-Critic) algorithm is introduced

---

## 2. Background

### 2.1 Preliminaries

The **Markov Decision Process (MDP)** consists of:
- State space $\mathcal{S}$, action space $\mathcal{A}$
- Initial state distribution $p_1(s_1)$
- Transition dynamics $p(s_{t+1}|s_t, a_t)$
- Reward function $r : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$

The **performance objective** (start-state value):

$$J(\pi_\theta) = \int_\mathcal{S} \rho^\pi(s) \int_\mathcal{A} \pi_\theta(a|s) Q^\pi(s,a) \, \mathrm{d}a \, \mathrm{d}s = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta}[r(s,a)] \tag{1}$$

where $\rho^\pi(s)$ is the discounted state distribution $\rho^\pi(s') = \int_\mathcal{S} \sum_{t=1}^\infty \gamma^{t-1} p_1(s) p(s \to s', t, \pi) \mathrm{d}s$.

### 2.2 Stochastic Policy Gradient Theorem

$$\nabla_\theta J(\pi_\theta) = \int_\mathcal{S} \rho^\pi(s) \int_\mathcal{A} \nabla_\theta \pi_\theta(a|s) Q^\pi(s,a) \, \mathrm{d}a \, \mathrm{d}s = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a) \right] \tag{2}$$

Key property: the gradient does **not** depend on the gradient of the state distribution — reduces computation to a simple expectation.

### 2.3 Stochastic Actor-Critic Algorithms

The **actor-critic** architecture:
- **Actor:** adjusts parameters $\theta$ of stochastic policy $\pi_\theta(s)$ by stochastic gradient ascent of Eq. 2
- **Critic:** estimates action-value function $Q^w(s,a) \approx Q^\pi(s,a)$ using an appropriate policy evaluation algorithm (e.g., temporal-difference learning)

A function approximator $Q^w(s,a)$ is **compatible** (no bias) if:
1. $\nabla_a Q^w(s,a) = \nabla_\theta \log \pi_\theta(a|s)^\top w$
2. $w$ minimises the mean-squared error $\mathbb{E}[\epsilon(s,a;w)^2]$ where $\epsilon = \nabla_a Q^w - \nabla_a Q^\pi$

Gradient using compatible approximator:

$$\nabla_\theta J(\pi_\theta) \approx \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) Q^w(s,a) \right] \tag{3}$$

### 2.4 Off-Policy Actor-Critic

Performance objective modified to be the value function of the target policy, averaged over the **behaviour policy** state distribution:

$$J_\beta(\pi_\theta) = \int_\mathcal{S} \rho^\beta(s) V^\pi(s) \mathrm{d}s = \int_\mathcal{S} \int_\mathcal{A} \rho^\beta(s) \pi_\theta(a|s) Q^\pi(s,a) \, \mathrm{d}a \, \mathrm{d}s$$

The **off-policy policy gradient**:

$$\nabla_\theta J_\beta(\pi_\theta) \approx \int_\mathcal{S} \int_\mathcal{A} \rho^\beta(s) \nabla_\theta \pi_\theta(a|s) Q^\pi(s,a) \, \mathrm{d}a \, \mathrm{d}s \tag{4}$$

$$= \mathbb{E}_{s \sim \rho^\beta, a \sim \beta} \left[ \frac{\pi_\theta(a|s)}{\beta(a|s)} \nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a) \right] \tag{5}$$

The **OffPAC** algorithm (Degris et al., 2012b):
- Uses behaviour policy $\beta(a|s)$ to generate trajectories
- Actor updates $\theta$ off-policy via stochastic gradient ascent of Eq. 5
- Temporal-difference error: $\delta_t = r_{t+1} + \gamma V^v(s_{t+1}) - V^v(s_t)$
- Actor and critic use importance sampling ratio $\frac{\pi_\theta(a|s)}{\beta(a|s)}$

---

## 3. Gradients of Deterministic Policies

### 3.1 Action-Value Gradients

In continuous action spaces, greedy maximisation at every step is problematic (requires global maximisation). A simpler and computationally attractive alternative:

> Move the policy in the **direction of the gradient of Q**, rather than globally maximising Q.

Update rule for each visited state $s$:

$$\theta^{k+1} = \theta^k + \alpha \mathbb{E}_{s \sim \rho^{\mu^k}} \left[ \nabla_\theta \mu_\theta(s) \nabla_a Q^{\mu^k}(s,a) \big|_{a=\mu_\theta(s)} \right] \tag{6}$$

By the chain rule, the policy improvement decomposes into:

$$\theta^{k+1} = \theta^k + \alpha \mathbb{E}_{s \sim \rho^{\mu^k}} \left[ \nabla_\theta \mu_\theta(s) \nabla_a Q^{\mu^k}(s,a) \big|_{a=\mu_\theta(s)} \right] \tag{7}$$

$\nabla_\theta \mu_\theta(s)$ is a **Jacobian matrix** where each column is the gradient $\nabla_\theta [\mu_\theta(s)]_d$ of the $d$-th action dimension with respect to the policy parameters $\theta$.

### 3.2 Deterministic Policy Gradient Theorem

Performance objective for deterministic policy $\mu_\theta : \mathcal{S} \to \mathcal{A}$:

$$J(\mu_\theta) = \int_\mathcal{S} \rho^\mu(s) r(s, \mu_\theta(s)) \mathrm{d}s = \mathbb{E}_{s \sim \rho^\mu}[r(s, \mu_\theta(s))] \tag{8}$$

**Theorem 1 (Deterministic Policy Gradient Theorem):** Suppose the MDP satisfies conditions A.1. Then:

$$\nabla_\theta J(\mu_\theta) = \int_\mathcal{S} \rho^\mu(s) \nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s,a) \big|_{a=\mu_\theta(s)} \mathrm{d}s$$

$$= \mathbb{E}_{s \sim \rho^\mu} \left[ \nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s,a) \big|_{a=\mu_\theta(s)} \right] \tag{9}$$

### 3.3 Limit of the Stochastic Policy Gradient

**Theorem 2:** Consider a stochastic policy $\pi_{\mu_\theta, \sigma}$ where $\sigma$ is a variance parameter controlling the variance of the policy. Then:

$$\lim_{\sigma \downarrow 0} \nabla_\theta J(\pi_{\mu_\theta, \sigma}) = \nabla_\theta J(\mu_\theta) \tag{10}$$

The stochastic policy gradient converges to the deterministic policy gradient as $\sigma \to 0$.  
This confirms DPG can use familiar function approximation, natural gradients, actor-critic methods, and episodic/batch methods.

---

## 4. Deterministic Actor-Critic Algorithms

### 4.1 On-Policy Deterministic Actor-Critic

A **simple Sarsa critic** for on-policy updates:

$$\delta_t = r_t + \gamma Q^w(s_{t+1}, \mu_\theta(s_{t+1})) - Q^w(s_t, a_t) \tag{11}$$

$$w_{t+1} = w_t + \alpha_w \delta_t \nabla_a Q^w(s_t, a_t) \tag{12}$$

$$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \nabla_a Q^w(s_t, a_t) \big|_{a=\mu_\theta(s_t)} \tag{13}$$

Note: A deterministic on-policy algorithm may have insufficient exploration; primarily didactic.

### 4.2 Off-Policy Deterministic Actor-Critic

Performance objective is modified to use the target policy value averaged over the **behaviour policy** state distribution:

$$J_\beta(\mu_\theta) = \int_\mathcal{S} \rho^\beta(s) V^\mu(s) \mathrm{d}s = \int_\mathcal{S} \rho^\beta(s) Q^\mu(s, \mu_\theta(s)) \mathrm{d}s \tag{14}$$

**Off-policy deterministic policy gradient:**

$$\nabla_\theta J_\beta(\mu_\theta) \approx \int_\mathcal{S} \rho^\beta(s) \nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s,a) \big|_{a=\mu_\theta(s)} \mathrm{d}s$$

$$= \mathbb{E}_{s \sim \rho^\beta} \left[ \nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s,a) \big|_{a=\mu_\theta(s)} \right] \tag{15}$$

Key advantage: since the deterministic policy integrates over actions analytically, **importance sampling in the actor is not needed**.

**OPDAC update rules** (using Q-learning for off-policy critic):

$$\delta_t = r_t + \gamma Q^w(s_{t+1}, \mu_\theta(s_{t+1})) - Q^w(s_t, a_t) \tag{16}$$

$$w_{t+1} = w_t + \alpha_w \delta_t \nabla_a Q^w(s_t, a_t) \tag{17}$$

$$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \nabla_a Q^w(s_t, a_t) \big|_{a=\mu_\theta(s_t)} \tag{18}$$

### 4.3 Compatible Function Approximation

A function approximator $Q^w(s,a)$ is **compatible** with deterministic policy $\mu_\theta(s)$ if:

$$\nabla_\theta J_\beta(\mu_\theta) = \mathbb{E}_{s \sim \rho^\beta}\left[\nabla_\theta \mu_\theta(s) \nabla_a Q^w(s,a)\big|_{a=\mu_\theta(s)}\right]$$

**Theorem 3:** $Q^w(s,a)$ is compatible if:
1. $\nabla_a Q^w(s,a)\big|_{a=\mu_\theta(s)} = \nabla_\theta \mu_\theta(s)^\top w$
2. $w$ minimises the MSE: $MSE(\theta, w) = \mathbb{E}[\epsilon(s,\theta;w)^\top \epsilon(s,\theta;w)]$ where $\epsilon(s,\theta;w) = \nabla_a Q^w(s,a)\big|_{a=\mu_\theta(s)} - \nabla_a Q^\mu(s,a)\big|_{a=\mu_\theta(s)}$

**Proof sketch:** If $w$ minimises MSE, then $\nabla_w MSE = 0$, which implies $\mathbb{E}[\nabla_\theta \mu_\theta(s) \nabla_a Q^w(s,a)|_{a=\mu_\theta(s)}] = \nabla_\theta J(\mu_\theta)$.

**Compatible linear function approximator:**

$$Q^w(s,a) = (a - \mu_\theta(s))^\top \nabla_\theta \mu_\theta(s)^\top w + V^v(s)$$

This is the **advantage function approximator** $A^w(s,a) = \phi(s,a)^\top w$ where:

$$\phi(s,a) \overset{\text{def}}{=} \nabla_\theta \mu_\theta(s)(a - \mu_\theta(s))$$

with $n$ action dimensions and $n$ policy parameters, $\nabla_\theta \mu_\theta(s)$ is $n \times n$ Jacobian, feature vector is $n \times 1$, parameter vector $w$ is also $n \times 1$.

**Note:** A linear function approximator is not very useful for global action-value prediction (diverges for large actions), but is effective as a **local critic** indicating the direction in which the actor should adjust its policy.

The **COPDAC** (Compatible Off-Policy Deterministic Actor-Critic) algorithm:
- Critic: linear function approximator $Q^w(s,a) = \phi(s,a)^\top w$, learned by Q-learning (off-policy)
- Actor: updates deterministic policy in direction of approximate action-value gradient

**COPDAC-Q update rules:**

$$\delta_t = r_t + \gamma Q^w(s_{t+1}, \mu_\theta(s_{t+1})) - Q^w(s_t, a_t) \tag{19}$$

$$w_{t+1} = w_t + \alpha_w \delta_t \phi(s_t, a_t) \tag{20/21}$$

$$u_{t+1} = u_t + \alpha_u (\phi(s_t,a_t) - \gamma \phi(s_{t+1}, \mu_\theta(s_{t+1}))) u_t \quad \text{(gradient Q-learning)} \tag{22-26}$$

$$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu_\theta(s_t) \nabla_a Q^w(s_t, a_t)\big|_{a=\mu_\theta(s_t)} \tag{27/28}$$

---

## 5. Experiments

### 5.1 Continuous Bandit

Direct comparison between stochastic PG and DPG on a continuous bandit problem with action-cost function $-c(a) = -(a - a^*)^\top C (a - a^*)$.

- COPDAC-B (deterministic, compatible off-policy) vs SAC-B (stochastic actor-critic)
- Tested on **10, 25, and 50 action dimensions**
- Results (Figure 1): COPDAC-B **significantly outperforms** SAC-B, performance gap grows with increasing dimensionality
- Variance of stochastic PG is proportional to $1/\sigma^2$ (Zhao et al., 2012); as policy becomes deterministic, stochastic PG variance becomes very large

### 5.2 Continuous Reinforcement Learning

Benchmarks: **Mountain Car, Pendulum, 2D Puddle World**

Algorithms compared:
- **SAC** (Stochastic Actor-Critic, Degris et al. 2012a) — on-policy
- **OffPAC-TD** (Off-Policy stochastic actor-critic) — off-policy
- **COPDAC-Q** (Compatible Off-Policy Deterministic Actor-Critic) — deterministic, off-policy

Policy: linear combination of features $\pi_{\theta,\beta}(s) \sim \mathcal{N}(\theta^\top \phi(s), \exp(\gamma^\top \phi(s)))$ for SAC; $\mu_\theta(s) = \theta^\top \phi(s)$ for COPDAC-Q.

Critic: linear value function $V(s) = v^\top \phi(s)$ for all.

Results (Figure 2): COPDAC-Q slightly outperforms both SAC and OffPAC in all three domains.

### 5.3 Octopus Arm

High-dimensional task: controlling a simulated **octopus arm** (6 segments, 50 continuous state variables, 20 action variables = 3 muscles per segment + base rotation).

- Goal: strike a target with any part of the arm
- Reward: proportional to change in distance to target (+50 bonus on hit), episode ends after 300 steps
- Previous work simplified to 6 "macro-actions" or stochastic PG on lower-dim version

**COPDAC results (Figure 3):**
- 6-segment arm with 20 action dimensions and 50 state dimensions
- 10 training runs all converge to a good solution
- Video of trained 8-segment arm available

---

## 6. Discussion and Related Work

- As stochastic policy gradient algorithms home in on a good strategy, the policy becomes more deterministic → stochastic PG harder to estimate (variance grows as $1/\sigma^2$)
- Deterministic PG can be **computed immediately in closed form**
- The DPG actor-critic is analogous to **Q-learning** (Watkins & Dayan, 1992): learns a deterministic greedy policy off-policy while executing a noisy version
- Comparison to OffPAC is analogous to asking whether Q-learning or Sarsa is more efficient — COPDAC is more efficient by measuring the greedy policy learned
- COPDAC is based on model-free, incremental, stochastic gradient updates → suitable when model is unknown, data is plentiful, computation is the bottleneck
- Related: **NFQCA** (Hafner & Riedmiller, 2011) uses two neural networks for actor and critic, but its critic network is incompatible with the actor network

---

## 7. Conclusion

- Presented a framework for **deterministic policy gradient algorithms**
- Gradients estimated more efficiently than stochastic counterparts (avoid integral over action space)
- The deterministic actor-critic **significantly outperformed** the stochastic actor-critic by orders of magnitude in a bandit with 50 continuous action dimensions
- Solved a challenging RL problem with **20 continuous action dimensions and 50 state dimensions** (octopus arm)

---

## Key Equations Summary

| Equation | Description |
|----------|-------------|
| $\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s,a}[\nabla_\theta \log \pi_\theta(a\|s) Q^\pi(s,a)]$ | Stochastic Policy Gradient Theorem |
| $\nabla_\theta J(\mu_\theta) = \mathbb{E}_{s \sim \rho^\mu}[\nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s,a)\|_{a=\mu_\theta(s)}]$ | Deterministic Policy Gradient Theorem |
| $\lim_{\sigma \to 0} \nabla_\theta J(\pi_{\mu_\theta,\sigma}) = \nabla_\theta J(\mu_\theta)$ | DPG is limit of stochastic PG |
| $Q^w(s,a) = (a - \mu_\theta(s))^\top \nabla_\theta \mu_\theta(s)^\top w + V^v(s)$ | Compatible function approximator |
| $\phi(s,a) = \nabla_\theta \mu_\theta(s)(a - \mu_\theta(s))$ | Feature vector for compatible approximator |

---

## References

- Bagnell & Schneider (2003). Covariant policy search. *IJCAI*.
- Bhatnagar, Sutton et al. (2007). Incremental natural actor-critic algorithms. *NIPS 21*.
- Degris, Pilarski, Sutton (2012a). Model-free RL with continuous action in practice. *ACC*.
- Degris, White, Sutton (2012b). Linear off-policy actor-critic. *ICML 29*.
- Engel, Szabó, Volkinshtein (2005). Learning to control an octopus arm via Gaussian process TD methods. *NIPS 18*.
- Hafner & Riedmiller (2011). Reinforcement learning in feedback control. *Machine Learning*, 84(1-2):137–169.
- Heess, Silver, Teh (2012). Actor-critic RL with energy-based policies. *JMLR EWRL 2012*.
- Kakade (2001). A natural policy gradient. *NIPS 14*.
- Lagoudakis & Parr (2003). Least-squares policy iteration. *JMLR*, 4:1107–1149.
- Peters (2010). Policy gradient methods. *Scholarpedia*, 5(11):3698.
- Peters, Vijayakumar, Schaal (2005). Natural actor-critic. *ECML 16*.
- Sutton & Barto (1998). *Reinforcement Learning: an Introduction*. MIT Press.
- Sutton, Maei et al. (2009). Fast gradient-descent methods for TD learning with linear function approximation. *ICML*.
- Sutton, McAllester et al. (1999). Policy gradient methods for RL with function approximation. *NIPS 12*.
- Sutton, Singh, McAllester (2000). Comparing policy-gradient algorithms.
- Watkins & Dayan (1992). Q-learning. *Machine Learning*, 8(3):279–292.
- Williams (1992). Simple statistical gradient-following algorithms for connectionist RL. *Machine Learning*, 8:229–256.
- Zhao, Hachiya et al. (2012). Analysis and improvement of policy gradient estimation. *Neural Networks*, 26:118–129.

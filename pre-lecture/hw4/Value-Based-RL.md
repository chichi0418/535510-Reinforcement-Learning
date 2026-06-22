# Value-Based Reinforcement Learning

**Ping-Chun Hsieh (謝秉均)**
Department of Computer Science
National Yang Ming Chiao Tung University (NYCU)
Email: pinghsieh@nycu.edu.tw
May 6, 2026
Website: pinghsieh.github.io

---

## Recap: RL Interaction Protocol

```
Action a_t
Learner ──────────────► Environment
        ◄──────────────
        Reward r_t
        Observation o_t
        (bounded)
```

- In this lecture, we presume $o_t \equiv s_t$

**Trajectory:** $(s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_t, a_t, r_{t+1}, \ldots)$

---

## Recap: Value Functions

**Return** $G(\tau)$: Cumulative discounted rewards of a trajectory from $\tau_t$ and onward

$$G(\tau_t) := r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k \geq 0} \gamma^k r_{t+k+1}$$

**Value function** $V^\pi(s)$: Expected return under policy $\pi$ from a state $s$

$$V^\pi(s) := \mathbb{E}_\tau \left[ G(\tau_t) \mid s_t = s;\, \pi \right]$$

**Action value function** $Q^\pi(s, a)$: Expected return under policy $\pi$ from a state $s$ and action $a$

$$Q^\pi(s, a) := \mathbb{E}_\tau \left[ G(\tau_t) \mid s_t = s,\, a_t = a;\, \pi \right]$$

---

## Recap: Optimal Value Functions

Intuition: Optimal value functions capture the best achievable total reward

**Optimal value function** $V^*(s)$: ⭐

$$V^*(s) := \max_{\pi \in \Pi} V^\pi(s)$$

**Optimal action-value function** $Q^*(s, a)$:

$$Q^*(s, a) := \max_{\pi \in \Pi} Q^\pi(s, a)$$

($\Pi$ is the class of all possible randomized policies)

---

## Recap: Bellman Optimality Equations

**(1) $V^*$ written in $Q^*$**

$$V^*(s) = \max_{a \in \mathcal{A}} Q^*(s, a)$$

**(2) $Q^*$ written in $V^*$**

$$Q^*(s, a) = R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a V^*(s')$$

**(3) $V^*$ written in $V^*$**

$$V^*(s) = \max_{a \in \mathcal{A}} \left[ R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a V^*(s') \right]$$

**(4) $Q^*$ written in $Q^*$**

$$Q^*(s, a) = R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a \max_{a \in \mathcal{A}} Q^*(s', a)$$

(3) and (4) offer recursive equations for finding $V^*$ and $Q^*$

---

## Recap: Q-Value Iteration (QVI)

**Idea:** Design an iterative algorithm that finds $Q^*$

### Q-Value Iteration

**Step 1.** Initialize $k = 0$ and set $Q_0(s, a) = 0$ for all states and actions

**Step 2.** Repeat the following until convergence:

For each state $(s, a)$, update by

$$Q_{k+1}(s, a) \leftarrow R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a \max_{a' \in \mathcal{A}} Q_k(s', a')$$

(Recall that we assume $P_s^a$ and $R$ are known, i.e., no learning)

---

## Part 1: Tabular Value-based RL

---

## Q-VI in the Learning Setting?

**Learning setting:** Unknown $P_s^a$ and $R$

$$Q_{k+1}(s, a) \leftarrow R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a \max_{a' \in \mathcal{A}} Q_k(s', a')$$

```
Action a_t
Learner ──────────────► Environment
        ◄──────────────
        state s_t  Reward r_t
        State s_{t+1}
```

---

## Q-Learning

**Learning setting:** Unknown $P_s^a$ and $R$

- $R_s^a$ estimated by sampled rewards
- $P_{ss'}^a$ estimated by sampled transitions

$$Q_{t+1}(s_t, a_t) \leftarrow (1 - \alpha) Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a' \in \mathcal{A}} Q(s_{t+1}, a') \right]$$

This is known as **Q-learning**

---

## Q-Learning: Two Interpretations

**Interpretation 1:** Approximate Q-VI in learning setting

$$Q_{t+1}(s_t, a_t) \leftarrow (1 - \alpha) Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a' \in \mathcal{A}} Q(s_{t+1}, a') \right]$$

**Interpretation 2:** Correction by Bellman error

$$Q_{t+1}(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a' \in \mathcal{A}} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

---

## Q-Learning: Action Selection?

**Learning setting:** Unknown $P_s^a$ and $R$

What actions to take for sampling from the environment?

---

## Candidate Policies for Action Selection

Intuition: Take actions that are potentially good

1. **Greedy:** $a_t = \arg\max_{a'} Q_{t-1}(s_t, a')$

2. **Epsilon-Greedy:** Random actions with probability $\epsilon$, and greedy with probability $1 - \epsilon$

3. **Softmax:** $a_t \sim \text{Softmax}(Q_t(s_t, \cdot))$

---

## Issues With a Greedy Policy

**Example:** 1-state MDP with 2 actions $\{a, b\}$ and set $\pi_0(s) = a$

- $r(s_0, a) \sim \text{Bernoulli}(0.7)$
- $r(s_0, b) \sim \text{Bernoulli}(0.5)$, terminal state

Under Q-learning:

$$Q(s, a) \leftarrow (1-\alpha) Q(s, a) + \alpha \left[ r_t + \gamma \max_{a' \in \mathcal{A}} Q(s_{t+1}, a') \right]$$

| Iteration $k$ | Action chosen | Result |
|---|---|---|
| $k=1$ | choose $a$, $r(s, a) = 1$ | $Q(s,a) = 1$, $Q(s,b) = 0$, $\pi(s) = a$ |
| $k=2$ | choose $b$, $r(s, b) = 1$ | $Q(s,a) = 1$, $Q(s,b) = 1$, $\pi(s) = b$ |
| $k=3$ | choose $b$, $r(s, b) = 0$ | $Q(s,a) = 1$, $Q(s,b) = 0$, $\pi(s) = a$ |
| $k=4$ | choose $b$, $r(s, b) = 0$ | $Q(s,a) = 1$, $Q(s,b) = 0$, $\pi(s) = a$ |

---

## Q-Learning with $\epsilon$-Greedy Exploration (Formally)

### Q-Learning with $\epsilon$-Greedy Exploration:

**Step 1:** Initialize $Q(s, a)$ for all $(s, a)$, and initial state $s_0$

**Step 2:** For each step $t = 0, 1, 2, \ldots$

- Select $a_t$ using $\varepsilon$-greedy w.r.t $Q(s_t, \cdot)$
- Observe $r_{t+1}, s_{t+1}$ from the environment
- Update:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha(s_t, a_t) \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

(For simplicity, set $\alpha(s, a) = 0$ if $(s, a) \neq (s_t, a_t)$)

**Remarks:**
1. $\epsilon$ and $\alpha_t$ to be specified
2. Learned policy $\neq$ interaction policy

---

## (1) Q-Learning: Convergence to Optimality

For convergence to $Q^*$, what is needed?

---

## (1) Q-Learning: Convergence Guarantee

For convergence to $Q^*$, what is needed?

1. All state-action pairs are explored infinitely many times
2. The step size of Q updates is sufficiently large (but not too large)

---

## (1) Q-Learning: Convergence Guarantee (Formally)

**Theorem:** Q-learning converges to the optimal action-value function, i.e., $Q_\infty(s, a) \rightarrow Q^*(s, a)$, under the following conditions:

$$\sum_{t=1}^{\infty} \alpha_t(s, a) = \infty \qquad \sum_{t=1}^{\infty} \alpha_t^2(s, a) < \infty \qquad \text{for all } (s, a)$$

1. Why is $\sum_{t=1}^{\infty} \alpha_t(s, a) = \infty$ needed?

2. Why is $\sum_{t=1}^{\infty} \alpha_t^2(s, a) < \infty$ needed?

---

## (2) Q-Learning: "Off-Policy" Learning

**On-policy learning:**
> Learned policy = Policy used to interact with the environment

**Off-policy learning:**
> Learned policy ≠ Policy used to interact with the environment
> (Called "behavior policy")

*Kazami Hayato / Asurada — Cyber Formula (閃電霹靂車): (learning agent) / (policy for interaction)*

---

## (2) Q-Learning: "Off-Policy" Learning

**Off-policy learning:**

1. Learn a target policy $\pi$ that achieves $\pi(a|s) \Rightarrow Q^\pi(s, a) = Q^*(s, a)$
2. In the meantime, follow a behavior policy $\beta(a|s)$

$$\{s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_{T-1}, a_{T-1}, r_T\} \sim \beta$$

**Why is off-policy learning useful?**

1. Learn from observing humans or other agents
2. Reuse experience generated from old policies $\pi_1, \pi_2, \ldots, \pi_{k-1}$
3. Learn about optimal policy while following an exploratory policy
4. Learn about multiple policies while following one policy

*(Slide Credit: David Silver)*

---

## Off-Policy Learning for Real-World Applications

### Recommender Systems
- Deploy a safe policy for collecting user data without losing user's interest
- Learn a better policy from these data

### Robot Control
- Deploy a safe policy for collecting robot data without hurting the machine
- Learn a good policy from these data

---

## Off-Policy Learning for Real-World Applications

**Example: BridgeData V2**
- Tele-operated demonstrations
- Scripted pick-and-place policy

https://rail-berkeley.github.io/bridgedata/

---

## A Second Look at Action Selection

Intuition: Take actions that are potentially good

**Epsilon-Greedy:** Random actions with probability $\epsilon$, and greedy with probability $1 - \epsilon$

$$Q_{t+1}(s_t, a_t) \leftarrow (1-\alpha) Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) \right]$$

This is known as the **"Sarsa"** method

---

## Sarsa Algorithm

### Sarsa (aka $\varepsilon$-Greedy Temporal Difference)

**Step 1:** Initialize $Q(s, a)$, and $(s_t, a_t)$

**Step 2:** In each step $t$, repeat the following:

- In state $s_t$, apply $a_t \sim \varepsilon\text{-greedy}(Q(s_t, \cdot))$, observe $r_t$, $s_{t+1}$
- Draw $a_{t+1} \sim \varepsilon\text{-greedy}(Q(s_{t+1}, \cdot))$
- Update:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha(s_t, a_t) \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$

- Why is it called "Sarsa"?
- Is "Sarsa" on-policy or off-policy?

---

## Overestimation Bias in Q-Learning

**Example:** Roulette with 1 state and 171 actions (assume $1 for each bet)

- Actual expected return = -$0.053
- Q-learning after $10^5$ trials: Each dollar yields ≈ $22

**Q-learning can suffer from overestimation!**

---

## Overestimation Bias

**Example:** Consider a 1-state MDP with 2 actions

- $r(s, a_1) \sim \text{Bernoulli}(0.5)$
- $r(s, a_2) \sim \text{Bernoulli}(0.5)$, terminal state

Let $\hat{Q}(s, a_1)$, $\hat{Q}(s, a_2)$ be unbiased estimators (each based on 1 sample)

**Overestimation Bias:**

$$\mathbb{E}\left[\max\{\hat{Q}(s, a_1), \hat{Q}(s, a_2)\}\right] > \max\left\{\mathbb{E}[\hat{Q}(s, a_1)],\, \mathbb{E}[\hat{Q}(s, a_2)]\right\} =: M$$

$$\max\left\{\mathbb{E}[\hat{Q}(s, a_1)],\, \mathbb{E}[\hat{Q}(s, a_2)]\right\} = \quad \mathbb{E}\left[\max\{\hat{Q}(s, a_1), \hat{Q}(s, a_2)\}\right] =$$

---

## Double Estimators

**Example:** Consider a 1-state MDP with 2 actions

- $r(s, a_1) \sim \text{Bernoulli}(0.5)$
- $r(s, a_2) \sim \text{Bernoulli}(0.5)$, terminal state

**Idea:** Create 2 independent unbiased estimates $\hat{Q}_A(s, a)$, $\hat{Q}_B(s, a)$

1. Use one estimate to select action: $\bar{a} = \arg\max_a \hat{Q}_A(s, a)$
2. Use the other estimate to evaluate $\bar{a}$: $\hat{Q}_B(s, \bar{a})$
3. Obtain an unbiased estimate: $Q(s, \bar{a}) \leftarrow \mathbb{E}[\hat{Q}_B(s, \bar{a})]$

$$\max_a \hat{Q}_A(s, a) = \max\left\{\mathbb{E}[\hat{Q}_A(s, a_1)],\, \mathbb{E}[\hat{Q}_A(s, a_2)]\right\} = 0$$

$$\mathbb{E}[\hat{Q}_B(s, \bar{a})] =$$

---

## Double Q-Learning

### Double Q-Learning:

**Step 1:** Initialize $Q^A(s, a)$, $Q^B(s, a)$ for all $(s, a)$, and initial state $s_0$

**Step 2:** For each step $t = 0, 1, 2, \ldots$

- Select $a_t$ using $\varepsilon$-greedy w.r.t $Q^A(s_t, a) + Q^B(s_t, a)$
- Observe $(r_t, s_{t+1})$
- Choose one of the following updates uniformly at random:

$$Q^A(s_t, a_t) \leftarrow Q^A(s_t, a_t) + \alpha \left[ r_t + \gamma Q^B\!\left(s_{t+1}, \arg\max_a Q^A(s_t, a)\right) - Q^A(s_t, a_t) \right]$$

$$Q^B(s_t, a_t) \leftarrow Q^B(s_t, a_t) + \alpha \left[ r_t + \gamma Q^A\!\left(s_{t+1}, \arg\max_a Q^B(s_t, a)\right) - Q^B(s_t, a_t) \right]$$

---

## Estimation Bias: Q-Learning vs Double Q-Learning

**Example:** Roulette with 1 state and 171 actions (assume $1 for each bet)

- Actual expected return = -$0.053
- Q-learning after $10^5$ trials: Each dollar yields ≈ $22
- Double Q after $10^5$ trials: Each dollar yields ≈ $0

*van Hasselt et al., Double Q Learning, NIPS 2010*

---

## Part 2: Value-based RL with Function Approximation

---

## Tabular RL for Large-Scale RL?

- So far: Value functions $V(s)$ or $Q(s, a)$ are maintained by a lookup table
- Such tabular methods can be impractical for large RL problems
  - Too many states (or actions) to store in memory
  - It is slow to learn $V(s)$ or $Q(s, a)$ for each state separately

**Examples:**
- Tetris: ~$2^{200}$ states
- Go: ~$10^{170}$ states
- MuJoCo Ant: Continuous state space

---

## Value Function Approximation

To scale up RL, function approximation is commonly used to learn value functions

**Idea:** Approximate a value function by a parametric function

$$\hat{V}(s; w) \approx V^\pi(s)$$

$$\hat{Q}(s, a; w) \approx Q^\pi(s, a)$$

Network with parameter $w$, input state $s$, outputs $\hat{Q}(s, a_1; w), \ldots, \hat{Q}(s, a_{|\mathcal{A}|}; w)$

**Motivation:** Generalization from seen states to unseen states

---

## Two Commonly-Used Function Approximators

1. **Neural networks**

2. **Linear combinations of features**

**Example:** Suppose the reward function is designed as $R(s, a) = \phi(s, a)^\top w$

$$Q^*(s_0, a_0) = \mathbb{E}\left[ G \mid s_0 = s, a_0 = a;\, \pi^* \right]$$

$$= \mathbb{E}\left[ r_1 + \gamma r_2 + \gamma^2 r_3 + \cdots \mid s_0 = s, a_0 = a;\, \pi^* \right]$$

$$= \mathbb{E}\left[ \phi_1^\top w + \gamma \phi_2^\top w + \cdots \mid s_0 = s, a_0 = a;\, \pi^* \right]$$

$$= \mathbb{E}\left[ \sum_{i=0}^{\infty} \gamma^i \phi_{i+1} \mid s_0 = s, a_0 = a;\, \pi^* \right]^\top w$$

$$\equiv \psi^*(s_0, a_0)^\top w \quad \text{(aka "successor feature")}$$

---

## Design of Features

$$\phi_c(s, a) := \mathbb{1}\{\text{Is the agent over an object of class } c \text{ in } s\}$$

$$\phi_g(s, a) := \mathbb{1}\{\text{Is the agent over the goal region?}\}$$

*Barreto et al., "Successor Features for Transfer in Reinforcement Learning", NeurIPS 2017*

---

## Q-Learning With Value Function Approximation

Approximating $Q^*(s, a) \approx \hat{Q}(s, a; w)$ using some parametric function

**Which way is preferred?**

**How to learn a proper $w$?**

---

## Q-Learning With Value Function Approximation

**Interpretation of Q-Learning:** Correction by Bellman error

$$Q_{t+1}(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a' \in \mathcal{A}} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

**A Natural Loss Function for Q-Learning:**

$$L(w) := \mathbb{E}_{(s, a, r, s') \sim \rho} \left[ \frac{1}{2} \left( r + \gamma \max_{a' \in A} \hat{Q}(s', a'; w) - \hat{Q}(s, a; w) \right)^2 \right]$$

**Gradient of Q-Learning Loss:**

$$\nabla_w L(w) = \mathbb{E}_{(s, a, r, s') \sim \rho} \left[ \left( r + \gamma \max_{a'} \hat{Q}(s', a'; w_k) - \hat{Q}(s, a; w_k) \right) \nabla_w \hat{Q}(s, a; w_k) \right]$$

---

## Q-Learning With Value Function Approximation (Formally)

### Q-Learning with Value Function Approximation:

**Step 1:** Initialize $w$ for $Q(s, a; w)$ and initial state $s_0$

**Step 2:** For each step $t = 0, 1, 2, \ldots$

- Select $a_t$ using $\varepsilon$-greedy w.r.t $Q(s_t, a; w)$
- Observe $r_{t+1}, s_{t+1}$ from the environment
- Update $w$ as follows:

$$w \leftarrow w + \alpha_k \left[ r_{t+1} + \gamma \max_{a'} \hat{Q}(s_{t+1}, a'; w) - \hat{Q}(s_t, a_t; w) \right] \nabla_w \hat{Q}(s_t, a_t; w)$$

---

## Divergence Issue With Q-Learning VFA

**Example:** 2 states in a potentially large MDP (under linear VFA)

- Only 1 action available at state $s_1$, and $Q(s_1, a; w) = w$
- $r(s_1, a_1) = 0$, $P(s_2 | s_1, a_1) = 1$
- $Q(s_2, a_1; w) = 2w$

**Question:** Given $w = 1$, $\gamma = 0.9$. Under Q-learning, what is $w_k$?

**Question:** What will happen if we keep using the transition $s_1 \rightarrow s_2$ to update $w$?

$$w \leftarrow w + \alpha_k \left[ r_{t+1} + \gamma \max_{a'} \hat{Q}(s_{t+1}, a'; w) - \hat{Q}(s_t, a_t; w) \right] \nabla_w \hat{Q}(s_t, a_t; w)$$

---

## Divergence Issue With Q-Learning VFA (Continued)

**Insight:** Divergence can occur if the following two things happen:

1. Keep using the transition $(s_1, a_1, 0, s_2)$ to update Q function
2. Using the latest $w$ in the Q-target for the update of iteration $(k+1)$

---

## Why Q-Learning with VFA Diverges?

Q-learning with VFA can be equivalently decomposed into two steps:

**Step 1.** Let $y_i = r(s_i, a_i) + \gamma \max_{a'} Q(s'_i, a'; w)$, for each $i$

**Step 2.** Set $w \leftarrow \arg\min_{w'} \frac{1}{2} \sum_i \left\| Q(s_i, a_i; w') - y_i \right\|_2^2$

Bellman operators are contractions, but it can become an **expansion** with value function approximation fitting:

$$Q \xrightarrow{\mathcal{T}} TQ \xrightarrow{\Pi} \Pi TQ$$

*(Sutton and Barto, Chapter 11.4)*

---

## Deep Q-Network

---

## Deep Q-Network (DQN)

**DQN = Combine Q-Learning with NN-based nonlinear VFA**

Actions: Left, Right, No-op, …

*Mnih et al., Human-level control through deep reinforcement learning, Nature 2015*

---

## Deep Q-Network on Atari

Policy learning in DQN ≡ Learning the "action ordering" in each state

---

## Deep Q-Network (DQN)

**DQN = Combine Q-Learning with NN-based nonlinear VFA**

Q-learning with VFA can have the divergence issue.
To tackle the divergence issue, DQN applies two techniques:

1. **Experience replay** (via a replay buffer)
2. **Using 2 networks:** Q-network and target network

*Mnih et al., Human-level control through deep reinforcement learning, Nature 2015*

---

## 1. Experience Replay

**Two operations:**

1. Store the previous experiences into a buffer $(s, a, s', r)$
2. Sample a mini-batch from the buffer at each update (similar to mini-batch SGD in supervised learning)

*(Figure Source: Sergey Levine)*

---

## 1. Experience Replay

**Main purposes:**

1. **Stable learning:** Break correlations between successive updates
2. **Data efficiency:** Reuse interactions with environment

*(Figure Source: Sergey Levine)*

---

## 2. Target Network

**Idea:** Use a separate target network updated only periodically or softly

$$\Delta w = \alpha_k \cdot \sum_{(s, a, r, s') \in D} \left[ r + \gamma \max_{a'} Q(s', a'; \bar{w}) - Q(s, a; w) \right] \nabla_w Q(s, a; w) \Big|_{w = w_k}$$

- $\bar{w}$: target network parameters (updated infrequently)

---

## Overall Architecture of DQN

- How to sample from the replay buffer?
- How to update the replay buffer?

*(Figure Source: https://kaustabpal.github.io/dqn)*

---

## DQN vs Supervised Learning

| | DQN | SL |
|---|---|---|
| 1 | Both training involves minimizing a loss w.r.t. some target / label | ✓ |
| 2 | Non-i.i.d. data in DQN if each $(s_t, a_t, r_t, s_{t+1})$ is viewed as one data sample | — |
| 3 | In DQN, the Q-target changes or improves during training | — |
| 4 | Exploration available and more importantly, required in DQN! | — |
| 5 | Learning and acting need not be aligned (i.e., off-policy) | — |

---

## Part 3: Advanced Value-based RL

---

## DQN for Continuous Actions?

By default, DQN presumes discrete actions due to "max" operation:

$$w \leftarrow w + \alpha_k \left[ r_{t+1} + \gamma \max_{a' \in \mathcal{A}} \hat{Q}(s_{t+1}, a'; w) - \hat{Q}(s_t, a_t; w) \right] \nabla_w \hat{Q}(s_t, a_t; w)$$

In practice, DQN with modifications can handle continuous actions:

1. Discretization tricks
2. Normalized advantage function
3. Amortized Q-learning

---

## 1. Discretization

### Naive Action Discretization

Naive discretization suffers from **exponential growth** of cardinality
(Typically, this is feasible for action dimension less than 5)

*Reacher (3–5 DoF) example*

*Tavakoli et al., Action Branching Architectures for Deep Reinforcement Learning, AAAI 2018*

---

## 1. Discretization

### Discretization & Branching

*Tavakoli et al., Action Branching Architectures for Deep Reinforcement Learning, AAAI 2018*

---

## 2. Normalized Advantage Function (Quadratic Approximation)

$$Q(s, a; \phi_A, \phi_V) = A(s, a; \phi_A) + V(s; \phi_V)$$

($P$ is state-dependent, positive definite)

$$A(s, a; \phi_\mu) := -\frac{1}{2} \left( a - \mu(s; \phi_\mu) \right)^\top P(s; \phi_P) \left( a - \mu(s; \phi_\mu) \right)$$

$$P(s; \phi_P) := L(s | \phi_P)^\top L(s | \phi_P)$$

($L$ is a lower-triangular matrix — Cholesky decomposition)

*Gu et al., Continuous Deep Q-Learning with Model-based Acceleration, ICML 2016*

---

## 3. Amortized Q-Learning (Sampling)

$$\max_{a \in \mathcal{A}} Q(s, a; w) \approx \max_{a' \in D,\, a \sim \mu(a)} Q(s, a'; w)$$

"proposal distribution" (Learned by maximum likelihood)

**Intuition:** Larger $|D|$ induces a higher probability of seeing max-Q actions

---

## Enhancements for Vanilla DQN

$$L_\text{DQN}(w) := \frac{1}{2} \sum_{(s, a, r, s') \in D} \left[ r + \gamma \max_{a'} Q(s', a'; \bar{w}) - Q(s, a; w) \right]^2$$

---

## Enhancements for Vanilla DQN

$$L_\text{DQN}(w) := \frac{1}{2} \sum_{(s, a, r, s') \in D} \left[ r + \gamma \max_{a'} Q(s', a'; \bar{w}) - Q(s, a; w) \right]^2$$

(With $\epsilon$-greedy exploration)

Questions to address:

1. **Overestimation Bias?**
2. **A better loss function?**
3. **How to draw samples from the replay buffer?**
4. **A better exploration method?**
5. **A better way to represent Q function?**

---

## Rainbow DQN

1. Double DQN (DDQN)
2. Distributional Q-learning
3. Prioritized experience replay (PER)
4. Dueling networks
5. Multi-step return in Q-target
6. Noisy networks for exploration

*Hessel et al., Rainbow: Combining Improvements in Deep Reinforcement Learning, AAAI 2018*

---

## Double DQN

**Loss function of DQN:**

$$L_\text{DQN}(w) := \mathbb{E}_{(s, a, r, s') \sim \rho} \left[ \frac{1}{2} \left( r + \gamma \max_{a' \in A} Q(s', a'; \bar{w}) - Q(s, a; w) \right)^2 \right]$$

$$\approx \frac{1}{2} \sum_{(s, a, r, s') \in D} \left[ r + \gamma \max_{a' \in A} Q(s', a'; w) - Q(s, a; w) \right]^2$$

**Loss function of Double DQN:**

$$L_\text{DDQN}(w) := \mathbb{E}_{(s, a, r, s') \sim \rho} \left[ \frac{1}{2} \left( r + \gamma Q\!\left(s', \arg\max_{a' \in A} Q(s', a'; w);\, \bar{w}\right) - Q(s, a; w) \right)^2 \right]$$

$$\approx \frac{1}{2} \sum_{(s, a, r, s') \sim D} \left[ r + \gamma Q\!\left(s', \arg\max_{a' \in A} Q(s', a'; w);\, \bar{w}\right) - Q(s, a; w) \right]^2$$

> "We therefore propose to evaluate the greedy policy according to the online network, but using the target network to estimate its value." — (van Hasselt et al., 2016)

---

## Prioritized Experience Replay (PER)

**Idea:** Prioritize the samples with higher Bellman error

Priority values: $p_1,\, p_2,\, p_3,\, \ldots,\, p_n$

**Replay buffer:**

- Uniform sampling: $\mathbb{P}(\text{sample } i) = \frac{1}{n},\; \forall i$

- Prioritized sampling: $\mathbb{P}(\text{sample } i) = \dfrac{p_i^\alpha}{\sum_k p_k^\alpha}$

**Popular choice of priority value:**

$$p_i = \left| r + \gamma \max_{a' \in A} Q(s', a'; w) - Q(s, a; w) \right| + \epsilon$$

---

## Multi-Step Return

States: $s_t,\, a_t \rightarrow s_{t+1},\, a_{t+1} \rightarrow s_{t+2},\, \ldots$

Rewards: $r_{t+1},\, r_{t+2},\, \ldots$

**Standard 1-step Q-target:**

$$\left[ r_{t+1} + \gamma \max_{a'} \hat{Q}(s_{t+1}, a'; w) - \hat{Q}(s_t, a_t; w) \right] \nabla_w \hat{Q}(s_t, a_t; w)$$

**$n$-step Q-target:**

$$\left[ r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n \max_{a' \in \mathcal{A}} \hat{Q}(s_{t+n}, a'; w) - \hat{Q}(s_t, a_t; w) \right] \nabla_w \hat{Q}(s_t, a_t; w)$$

---

## Dueling Networks

**Idea:** Explicitly separates the representation of state values and state-action advantages

**Naive approach:**

$$\hat{Q}(s, a; w) := \hat{V}(s; w) + \hat{A}(s, a; w)$$

Identifiability issue (since any $\hat{V}(s; w) + c$ can be a possible solution)

**Preferred approaches:**

$$\hat{Q}(s, a; w) := \hat{V}(s; w) + \hat{A}(s, a; w) - \max_{a \in \mathcal{A}} \hat{A}(s, a; w)$$

Or

$$\hat{Q}(s, a; w) := \hat{V}(s; w) + \hat{A}(s, a; w) - \frac{1}{|\mathcal{A}|} \sum_{a \in \mathcal{A}} \hat{A}(s, a; w)$$

*Wang et al., Dueling Network Architectures for Deep Reinforcement Learning, ICML 2016*

---

## Dueling Networks (Cont.)

**Idea:** Explicitly separates the representation of state values and state-action advantages, i.e., $\hat{Q}(s, a; w) := \hat{V}(s; w) + \hat{A}(s, a; w)$

**Without dueling:**

Input state $s$ → shared network → $\hat{Q}(s, a_1; w), \ldots, \hat{Q}(s, a_K; w)$

**With dueling:**

Input state $s$ → shared network → $\hat{V}(s; w)$ and $\hat{A}(s, a_1; w), \ldots, \hat{A}(s, a_K; w)$ → $\hat{Q}(s, a_1; w), \ldots, \hat{Q}(s, a_K; w)$

Saliency maps shown for comparison.

*Wang et al., Dueling Network Architectures for Deep Reinforcement Learning, ICML 2016*

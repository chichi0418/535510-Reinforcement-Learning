# Pre-Lecture Assignment 1 — Policy Gradient in Python

**535510 Reinforcement Learning**

---

## Q1: Which PG Expression?

**Answer: (P2) REINFORCE**

In `update()`, the code iterates backward through the episode (Line 35–36) and computes the **discounted future return** at each time step using backward induction:

$$G_t = r_t + \gamma \cdot G_{t+1}$$

This $G_t$ serves as a sample estimate of $Q^{\pi_\theta}(s_t, a_t)$, and each step contributes $G_t \cdot \nabla_\theta \log \pi_\theta(a_t | s_t)$ to the gradient — matching **(P2) REINFORCE**:

$$\nabla_\theta V^{\pi_\theta}(\mu) = \mathbb{E}_{\tau \sim P_\mu^{\pi_\theta}} \left[ \sum_{t=0}^{\infty} \gamma^t Q^{\pi_\theta}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t | s_t) \right]$$

- **Not P1**: P1 uses the same total trajectory return $G(\tau)$ for all time steps.
- **Not P3**: P3 uses a stationary (discounted) state distribution $d_\mu^{\pi_\theta}$.

---

## Q2: A Deeper Look

### (a) Policy Parameterization

The code uses a **Linear Softmax (Gibbs/Boltzmann) policy**:

$$\pi_\theta(a \mid s) = \frac{\exp(s^\top \theta_a)}{\sum_{a'} \exp(s^\top \theta_{a'})}$$

`self.theta` is a `(state_dim × action_dim)` matrix; `prefs = np.dot(state, self.theta)` computes linear action preferences.

### (b) Why `reversed` in Line 35?

Using `reversed` enables **O(T)** backward induction to compute all $G_t$:

$$G_{T-1} = r_{T-1}, \quad G_t = r_t + \gamma \cdot G_{t+1}$$

Without `reversed`, computing $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$ from scratch at each step would cost **O(T²)**.

### (c) What is Line 41 for?

```python
grad[:, action] = state - np.dot(state, probs)
```

Line 41 computes the **score function** $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ — the gradient of the log-policy with respect to the parameter matrix $\theta$. This is multiplied by $G_t$ in Line 46 to perform the policy gradient update.

---

## Q3: Bug Identification

**Bug: Incorrect gradient computation in Line 41**

For the linear softmax policy, the analytical gradient is:

$$\frac{\partial \log \pi_\theta(a \mid s)}{\partial \theta_{:,j}} = \begin{cases} s \cdot (1 - \pi_\theta(a \mid s)) = s - s \cdot \text{probs}[a] & j = a \\ -s \cdot \pi_\theta(j \mid s) = -s \cdot \text{probs}[j] & j \neq a \end{cases}$$

Or equivalently in matrix form: $\nabla_\theta \log \pi_\theta(a \mid s) = s \otimes (e_a - \text{probs})$

**What the code computes (wrong):**
```python
grad[:, action] = state - np.dot(state, probs)
# np.dot(state, probs) = Σ_i s_i * p_i  (a scalar)
# result: [s_0 - Σs_ip_i,  s_1 - Σs_ip_i, ...]
```

**What it should compute (correct):**
```python
# Full gradient (correct)
grad = np.outer(state, np.eye(self.action_dim)[action] - probs)

# Or at minimum, fix the action column:
grad[:, action] = state * (1 - probs[action])
# result: [s_0*(1-p_a),  s_1*(1-p_a), ...]
```

The two expressions differ because `np.dot(state, probs)` = $\sum_i s_i p_i$ (inner product) ≠ `probs[action]` (a single probability value). In addition, the gradients for all $j \neq \text{action}$ are left as zero, but they should be $-s \cdot \text{probs}[j]$.

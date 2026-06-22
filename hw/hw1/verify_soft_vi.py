"""
Numerical verification for entropy-regularized (soft) value iteration.

Checks that:
1. V*(s) = log-sum-exp Q*(s, .) is indeed the fixed point of the regularized
   Bellman optimality operator.
2. The optimal policy is the softmax of Q*.
3. Convergence matches the γ-contraction rate.
"""

import numpy as np

# ── reproducibility ──────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ── small random MDP ─────────────────────────────────────────────────────────
n_states  = 6
n_actions = 4
gamma     = 0.9
tol       = 1e-10
max_iter  = 10_000

# Rewards R(s, a)  in  [-1, 1]
R = rng.uniform(-1.0, 1.0, size=(n_states, n_actions))

# Transition probabilities P(s' | s, a) — each row sums to 1
P_raw = rng.random(size=(n_states, n_actions, n_states))
P = P_raw / P_raw.sum(axis=2, keepdims=True)   # shape (S, A, S)

print("=" * 60)
print("Entropy-Regularized (Soft) Value Iteration — Verification")
print("=" * 60)
print(f"  States={n_states}, Actions={n_actions}, γ={gamma}")
print()

# ── 1.  SOFT VALUE ITERATION ─────────────────────────────────────────────────
V = np.zeros(n_states)

for i in range(1, max_iter + 1):
    # Q(s, a) = R(s, a) + γ Σ_{s'} P(s'|s,a) V(s')
    Q = R + gamma * np.einsum("sak,k->sa", P, V)   # (S, A)

    # V_new(s) = log Σ_a exp(Q(s, a))   [log-sum-exp]
    Q_max  = Q.max(axis=1, keepdims=True)           # for numerical stability
    V_new  = Q_max.squeeze() + np.log(
        np.exp(Q - Q_max).sum(axis=1)
    )

    delta = np.max(np.abs(V_new - V))
    V = V_new

    if delta < tol:
        print(f"  Converged at iteration {i}  (Δ = {delta:.2e})")
        break
else:
    print(f"  Did NOT converge in {max_iter} iterations")

# ── 2.  FINAL Q* AND POLICY ──────────────────────────────────────────────────
Q_star = R + gamma * np.einsum("sak,k->sa", P, V)

# log-sum-exp of Q*  should equal V*  (fixed-point check)
Q_max    = Q_star.max(axis=1, keepdims=True)
V_check  = Q_max.squeeze() + np.log(np.exp(Q_star - Q_max).sum(axis=1))

fp_error = np.max(np.abs(V_check - V))
print(f"\n[CHECK 1]  Fixed-point  max|logsumexp(Q*) - V*| = {fp_error:.2e}")
if fp_error < 1e-7:
    print("           PASSED — V* = log-sum-exp Q* is the fixed point.")
else:
    print("           FAILED")

# ── 3.  SOFTMAX POLICY ───────────────────────────────────────────────────────
# π*(a|s) = exp(Q*(s,a) - V*(s))
log_pi_star = Q_star - V[:, None]           # (S, A)
pi_star     = np.exp(log_pi_star)

# Should sum to 1 over actions
row_sums    = pi_star.sum(axis=1)
sum_error   = np.max(np.abs(row_sums - 1.0))
print(f"\n[CHECK 2]  Softmax policy row-sum error = {sum_error:.2e}")
if sum_error < 1e-7:
    print("           PASSED — π*(·|s) sums to 1 (valid distribution).")
else:
    print("           FAILED")

# Also verify via manual softmax (no scipy needed)
# softmax(Q*, axis=1) = exp(Q* - max) / sum(exp(Q* - max))
Q_star_shifted = Q_star - Q_star.max(axis=1, keepdims=True)
pi_manual  = np.exp(Q_star_shifted) / np.exp(Q_star_shifted).sum(axis=1, keepdims=True)
diff_pi    = np.max(np.abs(pi_star - pi_manual))
print(f"           Max diff vs manual softmax(Q*): {diff_pi:.2e}")
if diff_pi < 1e-7:
    print("           PASSED — π* equals softmax(Q*).")
else:
    print("           FAILED")

# ── 4.  VERIFY THE BELLMAN OPTIMALITY EQUATION DIRECTLY ─────────────────────
# RHS of the regularized Bellman optimality eq:
#   [T*_Ω V](s) = max_{π} { Σ_a π(a|s)[Q*(s,a)] + H(π(·|s)) }
#               = Σ_a π*(a|s) Q*(s,a) + H(π*(·|s))
H_pi_star  = -np.sum(pi_star * np.log(pi_star + 1e-300), axis=1)   # entropy
rhs        = np.sum(pi_star * Q_star, axis=1) + H_pi_star

bellman_error = np.max(np.abs(rhs - V))
print(f"\n[CHECK 3]  Regularized Bellman optimality error = {bellman_error:.2e}")
if bellman_error < 1e-8:
    print("           PASSED — V*(s) = Σ_a π*(a|s)Q*(s,a) + H(π*(·|s)).")
else:
    print("           FAILED")

# ── 5.  CONTRACTION RATE ─────────────────────────────────────────────────────
# Run two trajectories from different starting points and measure
# how fast their sup-norm distance shrinks.
V_a = np.zeros(n_states)
V_b = rng.uniform(-5, 5, size=n_states)

gaps = []
for _ in range(50):
    Q_a = R + gamma * np.einsum("sak,k->sa", P, V_a)
    Q_b = R + gamma * np.einsum("sak,k->sa", P, V_b)

    Qmax_a = Q_a.max(axis=1, keepdims=True)
    Qmax_b = Q_b.max(axis=1, keepdims=True)

    V_a = Qmax_a.squeeze() + np.log(np.exp(Q_a - Qmax_a).sum(axis=1))
    V_b = Qmax_b.squeeze() + np.log(np.exp(Q_b - Qmax_b).sum(axis=1))

    gaps.append(np.max(np.abs(V_a - V_b)))

print(f"\n[CHECK 4]  Contraction test (γ={gamma})")
print(f"           ||V_a - V_b||_∞ at iter 1  : {gaps[0]:.6f}")
print(f"           ||V_a - V_b||_∞ at iter 10 : {gaps[9]:.6f}")
print(f"           ||V_a - V_b||_∞ at iter 50 : {gaps[49]:.6f}")
print(f"           Theoretical upper bound  (γ^49 × gap_0): "
      f"{gamma**49 * gaps[0]:.6f}")
if gaps[49] <= gamma**49 * gaps[0] + 1e-12:
    print("           PASSED — gap shrinks no faster than γ^t (contraction).")
else:
    print("           FAILED")

# ── 6.  ENTROPY INCREASES SOFT VALUE VS HARD VALUE ──────────────────────────
# Vanilla (hard) value iteration for comparison.
V_hard = np.zeros(n_states)
for _ in range(max_iter):
    Q_h    = R + gamma * np.einsum("sak,k->sa", P, V_hard)
    V_new  = Q_h.max(axis=1)
    if np.max(np.abs(V_new - V_hard)) < tol:
        break
    V_hard = V_new

print(f"\n[BONUS]   Soft V* >= Hard V* per state? "
      f"{'YES' if np.all(V >= V_hard - 1e-8) else 'NO'}")
print(f"          Mean  soft V* : {V.mean():.4f}")
print(f"          Mean  hard V* : {V_hard.mean():.4f}")
print(f"          Mean bonus from entropy regularization : "
      f"{(V - V_hard).mean():.4f}")

print()
print("=" * 60)
print("All verification checks complete.")
print("=" * 60)

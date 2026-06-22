## 4c Hyperparameters — REINFORCE with GAE on LunarLander-v3

| 參數 | 值 |
|------|----|
| Environment | LunarLander-v3 |
| Algorithm | REINFORCE + GAE |
| Learning rate | 0.005 |
| Optimizer | Adam |
| Gamma (γ) | 0.99 |
| Lambda (λ) | 0.90 / 0.95 / 0.99 |
| Hidden size | 128 |
| Random seed | 10 |
| EWMA alpha | 0.05 |
| Max steps/episode | 9999 |
| Weight init | Xavier uniform |

## 實驗結果

| λ | Episodes to Solve | Final EWMA | Test reward (10 eps avg) |
|---|---|---|---|
| 0.90 | 14,928 | 203.0 | — |
| 0.95 | 12,451 | 202.2 | — |
| 0.99 | 16,995 | 202.3 | — |

## 架構

- Shared layer：Linear(8 → 128) + ReLU
- Action head：Linear(128 → 4) + Softmax
- Value head：Linear(128 → 1)

## GAE 說明

δ_t = r_t + γ·V(s_{t+1})·(1 - done_t) - V(s_t)

A_t^GAE = Σ (γλ)^l · δ_{t+l}  （反向遞迴計算）

- λ → 0：high bias, low variance（接近 TD）
- λ → 1：low bias, high variance（接近 Monte Carlo）
- λ = 0.95 在本實驗收斂最快

# HW1: Fundamentals of MDPs and Policy Gradient

**課程**：535510 Spring 2026 Reinforcement Learning
**截止日期**：2026/04/02 (Thursday) 21:00
**總分**：105 分

---

## 繳交規範

兩個獨立檔案，透過 E3 繳交：

1. **PDF 檔**：所有書面解答（手寫掃描可接受，需清晰可讀）
2. **ZIP 檔**：所有程式碼 + demo 影片

---

## Problem 1 — Regularized MDPs（20 分，10+10）

使用 Shannon entropy 作為 regularizer：
`Ω(π(·|s)) = H(π(·|s)) = -Σ_{a∈A} π(a|s) ln π(a|s)`

### (a) 10 分
驗證 regularized Bellman expectation operator `T^π_Ω` 在 ℓ∞ norm 下是 contraction operator：

```
[T^π_Ω V](s) := R^π_s + Ω(π(·|s)) + γ P^π_{ss'} V
```

**提示**：延伸 Lecture 3 中 unregularized operator T^π 的 contraction 證明。

### (b) 10 分
設計迭代演算法求 V*_Ω 與 Q*_Ω（Bellman optimality equations）：

```
V*_Ω(s) = max_{π∈Π} { R^π_s + γ P^π_s V*_Ω }
Q*_Ω(s,a) = R_{s,a} + γ E_{s'~P(·|s,a)}[ V*_Ω(s') ]
```

需要：完整 pseudocode + 每行說明
**提示**：將標準 MDP 的 Value Iteration 延伸至 regularized MDP。

---

## Problem 2 — Policy Gradient（20 分，8+4+8）

### (a) 8 分
證明：對任意函數 f: S×A → R，

```
E_{τ~P^{πθ}_µ} [ Σ_{t=0}^∞ γ^t f(s_t, a_t) ] = 1/(1-γ) · E_{s~d^{πθ}_µ} E_{a~πθ(·|s)} [f(s,a)]
```

**提示**：從 RHS → LHS，展開後代入 d^{πθ}_µ 的定義，重新整理三重求和。

### (b) 4 分
基於 (a) 的 lemma 及 REINFORCE policy gradient (P2)，推導 PG 表達式 (P3)：

```
∇_θ V^{πθ}(µ) = 1/(1-γ) · E_{s~d^{πθ}_µ} E_{a~πθ(·|s)} [ Q^{πθ}(s,a) ∇_θ log πθ(a|s) ]
```

### (c) 8 分
檢查兩篇知名論文的 PG 表達式是否正確，若錯誤請指出並修正：

- **ACKTR**（NeurIPS 2017）：Section 2.1 的 PG 表達式
- **ACER**（ICLR 2017）：Section 2, Equation (1) 的 PG 表達式

---

## Problem 3 — Baseline for Variance Reduction（20 分，7+7+6）

**設定**：
- 1 個 non-terminal 起始狀態 s，3 個動作 {a, b, c}
- 任何動作後以機率 1 進入 terminal state，無 terminal reward
- 獎勵：`r(s,a) = 100, r(s,b) = 99, r(s,c) = 98`
- Softmax policy：`πθ(·|s) = exp(θ·) / (exp(θ_a) + exp(θ_b) + exp(θ_c))`
- 當前參數：`θ_a = 0, θ_b = ln3, θ_c = ln2`
- 每次 policy update：取樣一條 trajectory（單步，s₀=s, a₀∈{a,b,c}）

### (a) 7 分
求 ∇̂V 的：
- **期望向量** E[∇̂V]
- **共變異數矩陣** E[(∇̂V − E[∇̂V])(∇̂V − E[∇̂V])^T]

### (b) 7 分
以 value function V^{πθ}(s) 作為 baseline，求 ∇̃V 的期望向量與共變異數矩陣。

### (c) 6 分
找出 **optimal baseline** B(s)：使所有 state-dependent baseline 中，∇V_B 共變異數矩陣的 trace 最小化。

---

## Problem 4 — Policy Gradient with Function Approximation（45 分，15+15+15）

實作三種 policy gradient 演算法（PyTorch 或 TensorFlow，建議使用 **Weight & Biases** 追蹤訓練過程）：

### 繳交物
- 技術報告（含實驗結果，簡潔）
- 所有程式碼
- 訓練好的模型（`.pth` 或 `.ckpt`）：含 baseline、無 baseline、GAE 版本各一

---

### (a) 15 分 — Vanilla REINFORCE on CartPole-v1

**環境**：`CartPole-v1`
**程式碼**：修改 `reinforce.py`

需實作：
- `Policy.__init__()` — 初始化網路（shared layer + action layer + value layer），隨機初始化權重
- `Policy.forward(state)` — 輸出 action probability distribution 與 state value
- `Policy.select_action(state)` — 從 policy 採樣動作，儲存 (log_prob, value) 至 buffer
- `Policy.calculate_loss(gamma=0.999)`:
  - 計算 rewards-to-go
  - 計算 policy loss（policy gradient）
  - 計算 value loss（MSE 或 Smooth L1）
- `train(lr)`:
  - 每 episode 執行完整 trajectory（最多 9999 步）
  - Episode 結束後更新 policy 與 value network
  - 記錄 tensorboard 數據（lr, reward, episode length 等）
  - EWMA reward > `env.spec.reward_threshold` 時儲存模型並停止

**報告內容**：Tensorboard 截圖 + 所有 hyperparameters（learning rate、NN 架構等）

---

### (b) 15 分 — REINFORCE with Baseline on LunarLander-v3

**環境**：`LunarLander-v3`（solved 標準：testing episodic return > 200）
**程式碼**：另存為 `reinforce_baseline.py`

需實作：
- 在 (a) 的基礎上加入 baseline（可自由設計：手工設計的 state-dependent function、value function、或從 trajectory 學習的函數）
- `GAE` class 的 `__call__` 方法可選擇性使用
- 加入程式碼註解以提升可讀性

**報告內容**：
- Baseline 設計說明
- Tensorboard 截圖
- 所有 hyperparameters（可能需要調整 learning rate 或使用 LR scheduler）

---

### (c) 15 分 — REINFORCE with GAE on LunarLander-v3

**環境**：`LunarLander-v3`
**程式碼**：另存為 `reinforce_gae.py`

需實作：
- 完整的 `GAE.__call__(rewards, values, done)` 方法：
  - 計算 Generalized Advantage Estimation
  - `num_steps = None` 時使用 full batch
- 在三種不同 λ 值下執行實驗

**報告內容**：
- GAE 實作說明
- 三種 λ 值的實驗結果比較
- Tensorboard / W&B 截圖

**額外要求**：錄製 **3–5 分鐘影片**說明設計與實驗結果

**參考資料**：GAE 原始論文 https://arxiv.org/abs/1506.02438

---

## 程式碼架構（reinforce.py）

| 元件 | 說明 |
|------|------|
| `SavedAction` | namedtuple，儲存 `(log_prob, value)` |
| `Policy` | Actor-Critic 網路，actor/value 共享第一層，`hidden_size=128` |
| `GAE` | 獨立的 GAE 計算類別，參數：`gamma`, `lambda_`, `num_steps` |
| `train(lr=0.01)` | 訓練迴圈，使用 Adam optimizer |
| `test(name, env_name, n_episodes=10)` | 載入模型並測試（不需修改） |

**預設 hyperparameters**：
- `random_seed = 10`
- `lr = 0.01`
- `gamma = 0.999`
- `env_name = 'CartPole-v0'`（(a) 用，注意 PDF 寫 v1）
- Tensorboard log dir: `./tb_record_1`
- 模型儲存路徑: `./preTrained/`

---

## 參考資源

- PyTorch tutorial: https://pytorch.org/tutorials/
- Weight & Biases tutorial: https://docs.wandb.ai/tutorials
- CartPole-v1: https://gymnasium.farama.org/environments/classic_control/cart_pole/
- LunarLander-v3: https://gymnasium.farama.org/environments/box2d/lunar_lander/
- GAE 論文: https://arxiv.org/abs/1506.02438
- ACKTR 論文: https://arxiv.org/abs/1708.05144
- ACER 論文: https://arxiv.org/abs/1611.01224

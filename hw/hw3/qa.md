# HW3 助教 Q&A 整理 + 自我檢查

> 目的：把 E3 / Discord 上助教的所有澄清整理在這裡，並逐項對照我們的實作 (`sac.py` / `sac_panda_door.py` / `hw3_112550081.md`)，確認是否符合要求。

---

## 1. 繳交格式 / 影片

> **Kenny (5/17)**：2-d 2-e 有需要跑 test 產出影片，然後在 2–3 分鐘影片去講解結果嗎？
> **mhchen (5/18)**：沒有規定一定需要產出 test 影片，但建議 testing 時除了看 reward 外也可以把 episode 影片錄製出來，會有一些有趣的發現 (特別是 Robosuite 的環境)。

> **Kenny (5/18)**：超過 -170 的部分要截圖 train 的 mean return 還是只需要截圖 wandb 有超過 -170 就好？
> **mhchen**：都可以，只要能讓我們清楚看到 training steps 對應的 returns 即可。

> **Kenny (5/20)**：(1) 我們需要自己寫 test 跑平均然後交上去嗎？(2) Technical report 是需要和前面的問題合併成一個 PDF 嗎？
> **mhchen**：(1) `evaluate` function 原始 code 都已經寫好了。(2) Technical report 建議跟前面的分開，兩個獨立的 PDF。

### ✅ 對應我們的狀態

| 助教要求 | 我們狀態 |
|---|---|
| Test 影片**不強制** | 不打算錄製 test demo（之後若有空可加） |
| 一定要有「training step ↔ return」的截圖 | ⚠️ **還缺**：要去 W&B 截 `eval/mean_return` 或 `return` 曲線 |
| `evaluate()` 不需自己重寫 | ✅ 直接用 starter 提供的 |
| Technical report 與前面的理論題分開（助教**建議**） | ⚠️ **E3 只能交一個 PDF**，所以最後還是合併。原作業 PDF 第一行也說 "compile all your write-ups into **one** .pdf file"，跟 E3 限制一致，助教的「建議」是優先順序較低的個人偏好 |

---

## 2. Pendulum：-170 是 mean 還是 mean+std？

> **Ff (5/17)**：需要超過 -170 是加上標準差後要超過，還是平均超過就可以？
> **mhchen**：是 2(d) Pendulum 對吧？是的話**平均過就好**。

### ✅ 對應我們的狀態
- 我們 step 35,000 的 mean = **-136.99**，最終 100k step mean = **-148.91**，均**通過**。
- 標準差 ±67–94 雖然會壓到 -170 以下，但只看平均沒問題。

---

## 3. 兩個 tanh 的爭議

> **Ff (5/19)**：tanh 為何是兩個？
> **張立宜 (5/19)**：最後 return 確實不用 `.tanh()`，等等再公告 ~
> **張立宜 (5/20 更正)**：今天跟教授確認過後，這裡是**兩個 hyper tangent 沒有錯**，不好意思造成誤導。如果可以的話麻煩您再錄一顆有 2 個 tanh 的版本。
> **張立宜 公告 (5/22)**：原本第一個 tanh 是為了增強 exploration（避免 mu 過大、變異數小時被第二個 tanh 直接 squash 成 ≈1）。但會造成 `get_deterministic_action()` 的有效範圍變成 [-0.76, 0.76]。解法有兩種：
> 1. **完全刪掉第一個 tanh**（`forward()` 與 `get_deterministic_action()`）
> 2. **將第一個 tanh × 2.0**（這樣最後輸出範圍會擴張到 ≈[-0.96, 0.96]，記得 `mu_layer(x).tanh()` 也要 × 2.0）
> BTW, 如果兩個都不改也能跑出好結果就完全 OK。

### ✅ 對應我們的狀態
- **我們目前是「兩個 tanh」的原始版本**：
  - `sac.py` 的 `Actor.forward()` 與 `evaluate()` 內仍是 `mu.tanh()` → `action = mu.tanh()`
  - `sac_panda_door.py` 的 `Actor.forward()` 與 `get_deterministic_action()` 都是 `mu.tanh()` 再 `tanh()`
- **Pendulum 已驗證可以用兩個 tanh 跑到 -148**，所以根據助教「兩個都不改也能跑出好結果就 OK」，**我們不用改**。
- ⚠️ 不過 Door 要不要保留兩個 tanh，等實際跑過再決定；若 Door 卡住可考慮拿掉第一個 tanh 或乘 2.0。

---

## 4. Panda Door 訓練要求

> **Spencer (5/22)**：500k 內達到 200，之後仍需訓練到 1M？
> **張立宜**：是的，需要訓練到 1M。原因是 SAC 有高度 exploration，可能 500k 內偶然碰到 200，**訓練到 1M 是為了證明 policy 是穩定收斂而不只是運氣好**。

> **張立宜 (Problem 2(e) 公告, 5/22)**：原本 horizon 該設 **500**，但 starter 忘了改、預設留 1000，所以很多人輕鬆破 200。**通過門檻仍維持 200**。建議在 test 時把 horizon 改回 500 看 reward 還能不能達 200（**選填、不影響分數**）。

> **張立宜**：要交 500k 還是 1M 的 checkpoint？**交你最好的 model 就好，並在 training curve 上把選用 checkpoint 的 step 清楚標出來**。

### ✅ 對應我們的狀態

| 助教要求 | 我們狀態 |
|---|---|
| 500k 步內 eval ≥ 200 | ⚠️ **還沒跑** |
| 訓練到 1M 步 | ⚠️ **還沒跑** |
| `horizon=500` 是原本意圖 | ✅ 我們 `make_robosuite_env()` 已主動設 `horizon=500`（沒有作弊用 1000） |
| 交「最好」的 checkpoint，並在曲線上標 step | ⚠️ 等訓練完後要做 |
| 可調 dim | ✅ 我們 `--hidden-dim 256`（大於 starter 的 128） |

---

## 5. 自我檢查：理論題答案 (Problem 1, 2(a)(b)(c))

| 題目 | 我們的答案 | 重點檢查 |
|---|---|---|
| **P1** | Lagrangian → `∂L/∂π(a|s) = Q(s,a) - log π(a|s) - 1 - μ = 0` → softmax | ✅ 推導四步完整、有寫二階凹性檢查 |
| **2(a)-i** mean + log_std 為何？ | (1) `σ=exp(log σ)>0` 自動正定 (2) 動態範圍寬、數值穩定，配合 clamp `[-20,2]` | ✅ |
| **2(a)-ii** Reparam 怎麼實作？ | `a = μ + σ·ε`，PyTorch `dist.rsample()`（注意：**`rsample()` ≠ `sample()`**） | ✅ |
| **2(a)-iii** log_prob 為何要 -log(1-a²)？ | tanh 是 invertible change of variables，需扣 Jacobian `∑log(1-tanh²)` (Appendix C) | ✅ 含 `+1e-7` 避免 log(0) 的說明 |
| **2(b)-i** 為何兩個 Q？ | TD3 clipped double-Q 對抗 max-operator 引發的 overestimation bias | ✅ |
| **2(b)-ii** Q loss 數學式 | `L_Q_j = (1/N) Σ (Q_θ_j(s,a) - [r + γ(1-d)V_ψ̄(s')])²` | ✅ |
| **2(c)** α 自動調整為何長那樣？ | 從 entropy-constrained MDP 對偶問題推出來；log-space 確保 α>0；當實際熵 < H₀ 則 α↑ 增加探索 | ✅ |

> **沒看到助教對理論題有特別澄清**，依文獻標準寫法即可。

---

## 6. 程式碼自我檢查 (對照 sac_starter.py)

| 區塊 | 我們的實作 | 檢查重點 |
|---|---|---|
| Q loss | `y = r + γ(1-d)·V_target(s')` 用 `torch.no_grad()` 包；兩個 Q 各算一次 MSE | ✅ 跟 starter 的 algorithm box 一致 |
| V loss | `V_hat = min(Q1,Q2)(s, a~π) - α·log π`；`F.mse_loss(V(s), V_hat.detach())` | ✅ 注意 `new_action` 是已 reparam 過的；`alpha` 在這裡**不 detach**（因為只更新 V 不更新 α，但 alpha 跑回去更新 Q 也沒問題） |
| Actor loss | `(α·log π - min(Q1,Q2)(s, a~π)).mean()`，`alpha.detach()` | ✅ alpha detach 確保 actor loss 不會回去更新 α（α 已有自己的 optimizer） |
| Target soft update | `θ̄ ← τθ + (1-τ)θ̄` | ✅ |
| log_prob 修正 | `dist.log_prob(z) - log(1 - tanh(z)² + 1e-7)`，再 `.sum(-1, keepdim=True)` | ✅ |
| Twin Q + V | 沒少實作；訓練順序：先 update Q1/Q2 → V → policy (低頻) → target soft update | ✅ |

---

## 7. 還沒完成的清單 (依助教澄清重排優先順序)

| # | 項目 | 狀態 | 助教要求出處 |
|---|---|---|---|
| 1 | 跑 Panda-Door 1M 步、500k 內過 200 | ❌ 未開始 | Problem 2(e) + Spencer 串 |
| 2 | 把 `hw3_112550081.md` **拆成兩份 PDF**（理論 + 技術報告） | ❌ 目前合一 | Kenny / mhchen 5/20 |
| 3 | 在報告/曲線上標出**選用 checkpoint 的 step** | ❌ 等 Door 跑完 | 張立宜 5/22 |
| 4 | W&B 截圖：Pendulum 的 step vs return 曲線 | ❌ 還沒截 (現為 offline run) | mhchen 5/18 |
| 5 | (選填) 用 horizon=500 測 Door 看會不會掉 200 | ❌ | 張立宜 5/22，optional |
| 6 | (選填) 錄 test demo 影片 (Robosuite) | ❌ | mhchen 5/18，optional |
| 7 | 必交 3–5 分鐘影片講解設計 + 結果 | ❌ | 原 PDF 要求 |
| 8 | 打包 source code zip | ❌ 等所有檔案確定 | 原 PDF 要求 |

---

## 8. 一句話結論

> **理論題 + Pendulum 都已符合要求，最大的剩餘風險集中在 Panda Door 還沒跑、以及繳交格式需拆 PDF。**
> 建議下一步：**立刻啟動 Door 訓練**（最花時間），同時手動拆 PDF + 補 W&B 截圖。

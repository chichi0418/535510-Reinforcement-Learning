# HW3 Demo Video Manuscript

**目標：3–5 分鐘**　|　**學號：112550081**　|　**主題：Problem 2(d) Pendulum + 2(e) Panda Door**

說話速度 ≈ 150 字 / 分（中文則約 200 字 / 分）。整段約 4 分鐘。
建議錄製方式：QuickTime / OBS 螢幕錄影 + 講話；切畫面用 ⌘+Tab 或 Spotlight。

---

## 1. 開場（~20 秒）

**🎬 畫面**：作業 PDF 封面（`Spring2026_RL_HW3.pdf` 第一頁）

> 大家好我是 112550081。這支影片會 demo 我在 HW3 Problem 2(d) 與 2(e) 的設計與實驗結果。Problem 2(d) 是在 Pendulum-v1 訓練 SAC，2(e) 則是把 SAC 套到 Robosuite 的 Panda-Door-Opening 環境，總共要訓練一百萬步。

---

## 2. SAC 核心實作（~50 秒）

**🎬 畫面**：在 VSCode / Cursor 開 `sac.py`，⌘+G 跳到 **line 223**（`def update_model`）

> 我先快速帶過三個我自己實作的 loss。SAC 的 update 主要分四件事：
>
> **🎬 跳到 line 244–257（Q loss block）**
> **Q loss 用標準 Bellman 殘差**：target 是 `r + γ × V_target(s')`，用 `torch.no_grad()` 包住、再對兩個 Q network 各算一次 MSE。這是 TD3 的 twin-Q trick，用來壓 overestimation bias。
>
> **🎬 跳到 line 259–268（V loss block）**
> **V loss 的 target 是 `min(Q1, Q2) - α × log π`**，也就是 soft value——把兩個 Q 取 min 之後再扣掉 entropy bonus。target 一樣 detach 不回傳梯度。
>
> **🎬 跳到 line 271–279（actor loss block）**
> **Actor loss 是 `(α × log π − min Q).mean()`**，這裡的 `alpha` 要 detach，因為 α 已經有自己的 dual optimizer 在更新；而 `new_action` 是用 reparameterization sample 出來、梯度可以回傳的版本。

---

## 3. Pendulum 結果（~40 秒）

**🎬 畫面**：開 `hw3_technical_112550081.pdf`，跳到 "W&B snapshots — Pendulum" 的 `eval_mean_return` 圖

> Pendulum 的目標是 mean evaluation return 要過 -170。我從圖上可以看到：訓練前 1k 步是 random warm-up，10k 步之後 eval 就從 -1200 一路衝到 -200 附近；**到 35,000 步首次低於 -170**——比助教預期的 50k 還要早；之後就鎖在 -170 到 -140 區間。

**🎬 畫面**：scroll 到 `return.png`（per-episode training return）

> Training return 因為 stochastic policy 有 sample noise 所以比 eval 抖，但 baseline 也清楚地從 -1500 抬到 -250 左右。

**🎬 畫面**：scroll 到 `alpha_loss`

> Alpha loss 很快收斂到 0 附近，代表 temperature α 已經自動找到一個讓 entropy 落在 target 的值——這就是 Problem 2(c) 在問的自動調整機制實際在跑的證據。

**🎬 畫面**：技術報告 Hyperparameter table

> Pendulum 我用 hidden 128×2、lr 3e-4、batch 256、replay 50k、總共 100k 步，種子 77，**最終 eval mean 是 -148.91，pass**。

---

## 4. Door 環境的程式調整（~40 秒）

**🎬 畫面**：開 `sac_panda_door.py`，⌘+G 跳到 **line 502**（`def make_robosuite_env`）

> 接著是 Door 環境。我把 Pendulum 的程式做了幾個調整：
>
> 第一，environment 直接用 `robosuite.make("Door", robots="Panda")` 包 `GymWrapper`，`reward_shaping=True`，**horizon 維持 starter 預設的 1000**（指 line 516）——這也是助教 5/22 公告明確的 baseline 設定。
>
> **🎬 跳到 line 45（Actor 的 `hidden_dim`）或 line 187（agent 接收 args）**
> 第二，hidden dim 從 128 拉到 **256**，因為 Door 的 observation 50 維、action 7 維，比 Pendulum 大很多。
>
> 第三，replay buffer 開到一百萬、initial random steps 設一萬步當 warm-up（指 `__main__` 區的 argparse 預設，約 line 540 附近）。
>
> **🎬 跳到 line 189–193（device 偵測）**
> 第四，我加了 MPS device 偵測，在我的 Apple Silicon 上能用 GPU 跑訓練，整個一百萬步大概跑四個半小時。
>
> **🎬 跳到 line 292（`# ---- q function loss ----`）**
> 三個 loss 區塊（line 292、307、319）跟 Pendulum 一模一樣，因為 SAC 是環境無關的演算法。

---

## 5. Door 結果（~60 秒）

**🎬 畫面**：技術報告 Door section 的 `eval_mean_return` 圖（有黑圈標 step 910k 的那張）

> 重點來了。Door 的目標是 500k 步內 eval >= 200。
>
> （指圖上 250k 那個尖峰）這條曲線可以看到 **step 250k 第一次破 200，evaluation mean = 401**——也就是只用了助教要求一半的步數就達標。
>
> 中段大概 300k 到 700k 之間有些震盪，這在 SAC 是正常的，policy 一邊探索一邊 fine-tune。
>
> （指 700k 之後）720k 之後曲線就鎖在 920–950 區間。**圖上黑色圓圈標的是我選用的 checkpoint：step 910,000，eval mean 是 948.49、標準差只有 1.36**——5 個 evaluation episode 幾乎一樣完美。

**🎬 畫面**：scroll 到 `alpha.png`

> 旁邊這張 alpha 曲線可以看到，α 一開始是 1.0，隨著 policy 慢慢確定 action，α 自動下降到接近 0.1，代表 entropy bonus 被 dual optimizer 自動降低、policy 變得更 deterministic。

**🎬 畫面**：scroll 到 `return.png` (Door)

> Per-episode training return 也佐證這件事：從 ~2 一路爬到平均 700-800、最終穩定在 850+。

**🎬 畫面**：技術報告 Door Hyperparameter table

> Door 用 lr 3e-4、batch 256、γ=0.99、τ=5e-3，target entropy 自動設為 -7、horizon 1000，總共一百萬步。**最終最佳 checkpoint 的 eval mean 948.49，是 baseline 200 的 4.7 倍**。

---

## 6. 收尾（~20 秒）

**🎬 畫面**：finder 顯示三個繳交檔案 `hw3_theory_112550081.pdf` / `hw3_technical_112550081.pdf` / `hw3_112550081_code.zip`

> 總結一下：
>
> - **Problem 2(d) Pendulum** 達標步數 35k、最終 mean -148。
> - **Problem 2(e) Door** 在 250k 步就破 200、最佳 checkpoint step 910k 達到 948。
> - 兩份 PDF 跟一個 zip 都在這邊，code 跟 trained checkpoint 都附在 zip 裡。
>
> 以上就是我的 demo，謝謝助教。

---

## 📌 錄影前 checklist

1. 把以下檔案先打開、各自放好分頁，錄影時 ⌘+Tab 切換：
   - `Spring2026_RL_HW3.pdf` 第 1 頁
   - `sac.py`（VSCode 或 Cursor，跳到 `update_model()`）
   - `sac_panda_door.py`（跳到 `make_robosuite_env()`）
   - `hw3_technical_112550081.pdf`（顯示 W&B 圖那幾頁）
   - Finder 視窗指到 hw3 資料夾
2. 關掉訊息通知、Slack/Discord
3. QuickTime → File → New Screen Recording（macOS 內建）
4. 用「螢幕錄影附麥克風」一次過，or 後製剪輯
5. 講錯一段不用怕，剪掉就好

## ⏱️ 段落時間預算

| 段落 | 內容 | 預估秒數 |
|---|---|---|
| 1 | 開場 | 20s |
| 2 | SAC 三個 loss | 50s |
| 3 | Pendulum 結果 | 40s |
| 4 | Door 程式調整 | 40s |
| 5 | Door 結果 | 60s |
| 6 | 收尾 | 20s |
| **總計** | | **≈ 3 分 50 秒** |

落在 3–5 分鐘範圍內，剛好。

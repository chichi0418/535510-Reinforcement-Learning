# Demo 影片逐字稿 — HW1 Problem 4(c): REINFORCE with GAE

**目標時長：4–5 分鐘**
**語言：繁體中文**

---

## 【開場】｜畫面：桌面或簡報封面

大家好，我是這次 535510 強化學習課程作業一的同學。

這支影片是 Problem 4(c) 的 Demo，主題是在 LunarLander-v3 環境上實作 REINFORCE with Generalized Advantage Estimation，也就是 GAE。

影片主要分成三個部分：第一部分介紹 GAE 的核心公式跟實作細節，第二部分展示三種不同 lambda 值的訓練結果，第三部分做個簡單的總結。

---

## 【第一段：GAE 實作說明】｜畫面：開啟 `reinforce_gae.py`，捲動到 `class GAE`

首先來看 `reinforce_gae.py` 裡面的 `GAE` 類別。

GAE 的全名是 Generalized Advantage Estimation，是 Schulman 等人在 2016 年提出的方法。它的目的是估計 advantage function，也就是在某個 state 採取某個 action，相對於平均水準好多少。

為什麼要用 GAE 而不直接用 Monte Carlo return 或 TD error？原因是這兩種方法各有缺點。Monte Carlo 的 bias 很低，但 variance 很高，訓練容易不穩定。TD(0) 的 variance 低，但 bias 比較高，因為只看一步。GAE 透過一個超參數 lambda，讓我們可以在這兩者之間做連續的取捨。

---

｜畫面：聚焦在 `GAE.__call__` 的公式註解區域

GAE 的核心公式是這樣的。

對每個時間步 t，我們先計算 TD error，稱為 delta_t。它等於當下的 reward，加上 gamma 乘以下一個 state 的 value 估計，再減掉當前 state 的 value 估計。這個 delta_t 可以理解成「這一步比我預期的好多少或差多少」。

接著，GAE advantage 是把這些 delta 用指數衰減的方式加總起來。公式是 A_t 等於 delta_t 加上 gamma lambda 乘以 A_{t+1}，以此類推。這樣做讓近的 TD error 有比較大的權重，遠的 TD error 的影響會隨著 lambda 的次方快速衰減。

---

｜畫面：聚焦在 `__call__` 的 for 迴圈實作

實作上我用的是從後往前的迴圈，從最後一個 time step 開始算，一路往前遞推。這樣只需要一次遍歷就能算完整條 trajectory 的 advantage，時間複雜度是 O(T)，非常高效。

---

｜畫面：聚焦在 `train()` 裡 bootstrap value 的判斷區塊

這裡有一個很重要的實作細節，就是 bootstrap value 的處理。

LunarLander-v3 的 episode 結束有兩種情況：第一種是 termination，也就是真的發生了終止事件，例如火箭墜毀或者成功降落；第二種是 truncation，也就是達到最大步數上限，episode 被強制截斷。

這兩種情況對 GAE 的影響是不同的。如果是 termination，未來沒有更多的 reward，所以 bootstrap value 設為零。如果是 truncation，火箭其實還可以繼續飛，未來還有潛在的 reward，所以我們要用 value network 對最後一個 state 的預測來當作 bootstrap，避免低估未來的 return。

在程式碼裡，我用 `terminated` 跟 `truncated` 這兩個 flag 來做判斷，並且在 truncated 的情況下，對最後一個 state 做 forward pass 來取得 bootstrap value。

---

｜畫面：聚焦在 `calculate_loss()` 區塊

最後，advantage 在進 policy loss 之前會做 normalization，讓訓練更穩定。另外特別注意，value network 的監督信號用的是 discounted return G_t，而不是 GAE advantage 本身。這樣 value network 才能學到準確的 state value 估計，進一步讓下一個 episode 的 advantage 估計更精準。

---

## 【第二段：實驗結果】｜畫面：開啟 TensorBoard 或切換到比較截圖 `reward_ewma_compare.png`

接下來看實驗結果。

我在 LunarLander-v3 環境上，用相同的 random seed、相同的 learning rate 0.005，跑了三個不同的 lambda 值：0.90、0.95、跟 0.99，來觀察 lambda 對訓練速度的影響。

這張圖是三條 EWMA reward 曲線的比較。EWMA 的 alpha 是 0.05，也就是對最近的 reward 做指數移動平均，smoothing 過後比較容易看出訓練趨勢。

---

｜畫面：用游標指向三條曲線，分別標示

可以看到，粉紅色的曲線是 lambda 等於 0.95，它收斂最快，在大約 12,451 個 episode 的時候，EWMA reward 就突破了 200 的 threshold，代表成功 solve 了 LunarLander-v3。

黃色是 lambda 等於 0.90，用了大約 14,928 個 episode 才達到 threshold，比 0.95 多了將近兩千個 episode。

藍色是 lambda 等於 0.99，用了最多，大約 16,995 個 episode。

---

｜畫面：維持在比較截圖，或切換到簡單的文字說明投影片

為什麼會有這樣的差異？讓我解釋一下背後的原因。

lambda 等於 0.90 的時候，GAE 裡面每個 delta 衰減得比較快，距離越遠的 TD error 影響越小，整體上比較接近 TD(0)。這樣的 advantage 估計 bias 比較高，因為沒有充分利用 trajectory 後面的資訊，所以收斂比較慢。

lambda 等於 0.99 的時候，幾乎等於是把整條 trajectory 的 TD error 都加進來，接近 Monte Carlo。bias 很低，但 variance 非常高，每個 episode 的 advantage 估計都很不穩定，所以訓練曲線抖動很厲害，需要更多 episode 才能收斂。

lambda 等於 0.95 則在這兩者之間找到了比較好的平衡點，bias 夠低、variance 也夠小，所以在這個實驗裡表現最好。這也和 GAE 原始論文裡建議的 lambda 值範圍 0.9 到 0.99 吻合，作者在論文裡也特別提到 0.95 附近通常是個不錯的選擇。

三個 lambda 值的最終 EWMA reward 都超過了 200，所以三組實驗都成功 solve 了 LunarLander-v3。

---

## 【結尾】｜畫面：切回 `reinforce_gae.py` 或回到桌面

以上就是 Problem 4(c) 的 GAE 實作說明跟實驗結果。

做個簡單總結：GAE 透過 lambda 這個超參數，讓我們可以在 bias 跟 variance 之間連續地做調整。lambda 太小會有太多 bias，lambda 太大會有太多 variance，在這次 LunarLander-v3 的實驗中，lambda 等於 0.95 的收斂速度最快，效果最好。

另外，bootstrap value 的正確處理，以及用 discounted return 來訓練 value network，都是讓 GAE 能穩定運作的關鍵細節。

感謝收看，如果有任何問題歡迎提出。謝謝。

---

## 錄製提示

| 時間點 | 畫面建議 |
|--------|---------|
| 開場 | 桌面或黑底白字封面 |
| GAE 公式說明 | `reinforce_gae.py`，聚焦 `class GAE` 與 `__call__` |
| Bootstrap 說明 | `train()` 裡 truncation/termination 判斷區塊 |
| Loss 說明 | `calculate_loss()` 區塊 |
| 實驗結果 | `reward_ewma_compare.png` 或 TensorBoard |
| 結尾 | 回到程式碼或桌面 |

**錄製工具（Mac）：** `Cmd + Shift + 5` → 選擇「錄製整個螢幕」

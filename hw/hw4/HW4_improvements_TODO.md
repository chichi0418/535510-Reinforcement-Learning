# HW4 — 可再優化清單(衝滿分用)

> 現狀:Problem 1 已是滿分水準;Problem 2 內容正確、調整有誠實交代,直接交大概率已接近滿分。
> 以下是「再補就更穩」的項目,依 CP 值(投報率)排序。每項都附:**為什麼、要改哪個檔、怎麼做**。
>
> 交件檔(不變):`HW4_writeup.pdf`(deliverable i)+ `HW4_code.zip`(deliverable ii)。
> 改完任何 .md 後重產 PDF 指令見最底下。

---

## 🔴 高影響 — 唯一的 spec 字面缺口

### 1. 基線 Run A 跑滿 1 個 epoch(目前只 1000 步)
- **為什麼**:spec 的「step 1000 允許」原文只寫在 **Problem 2(c)**;**(b)** 嚴格說要訓練滿 `Epochs = 1`(≈ 7767 步 = 62135 / 有效批次 8)。(c) 用 1000 步是 spec 明文允許,**不用動**。
- **要改的檔**:跑 `train_dpo.py`(server)→ 重產 `plots/default_curves.png` 與 W&B 截圖 → 更新 `HW4_Problem2.md` 的 (b) 圖與 final 數字。
- **怎麼做(server 上)**:
  ```bash
  # 跑滿 1 epoch(把 --max_steps 拿掉,預設 -1 = 整個 epoch)
  python train_dpo.py --run_name default_fullepoch
  # 重產離線圖
  python plot_metrics.py
  ```
  然後到 W&B 重截 (b) 的 4 張必要圖(chosen/rejected/margins/loss),覆蓋 `wandb_screenshots/2b_default_train_*.png`,並把 `HW4_Problem2.md` (b) 的 final 數字改成滿 epoch 的值。
- **注意**:full epoch 後 margin/accuracy 會比 1000 步更高、更平滑;(b) 的文字趨勢敘述仍成立,只需換數字與圖。
- **成本**:server 上約數小時(Volta + gradient checkpointing)。**若時間不夠可不做**——目前 1000 步版已用 eval margin 上升證明有學到、且文中已引用 step-1000 允許。

---

## 🟠 中影響 — 低成本、明顯加分

### 2. 在 writeup 放上 W&B 專案/run 的公開連結
- **為什麼**:spec 反覆強調「via W&B plots」。給連結讓助教能直接點進 dashboard 驗證 5 個 run,比只看截圖更有說服力。
- **要改的檔**:`HW4_Problem2.md` 最上方 quote 區塊(第 3–12 行)。
- **怎麼做**:把 W&B project 設為 public,複製連結,加一行例如:
  ```markdown
  > **W&B project (all 5 runs):** https://wandb.ai/<your-entity>/dpo
  ```
  若是 offline run,先 `wandb sync wandb/offline-run-*` 上傳。

### 3. 加「Expected trend vs Observed」對照 + 解釋 accuracy 為何停在 ~0.7
- **為什麼**:spec 的 Remark 表給了預期趨勢(chosen↑ / rejected↓ / margins↑ / accuracies→**1.0**)。直接並排「預期 vs 觀測」是回應 rubric 最直接的方式;並且要主動解釋**為何 accuracy 沒有逼近 1.0**(否則看起來像沒達標)。
- **要改的檔**:`HW4_Problem2.md` (b) 的「Observed behavior」表附近(約第 104–112 行)。
- **怎麼做**:在該表加一欄「Spec expected」對照,並補一段:
  > accuracy 停在 ~0.7 而非 →1.0,是因為 (i) 0.5B 小模型容量有限、(ii) 只訓練 1000 步(未滿 epoch)、(iii) UltraFeedback 偏好對本身含雜訊/接近的回應,理論上界本就 < 1。趨勢方向(↑)與 spec 預期一致。

---

## 🟡 小幅 — 錦上添花

### 4. 2(a) 附上 `dataset[0]` 的原始 print 片段
- **為什麼**:spec 字面要求「Print `dataset[0]`」。附上幾行實際 console 輸出,完全對齊字面。
- **要改的檔**:`HW4_Problem2.md` (a) 開頭;素材在 `logs/inspect.out`、`logs/inspect2.out`。
- **怎麼做**:貼一個截斷的 code block,例如:
  ```text
  keys: ['chosen', 'rejected', 'score_chosen', 'score_rejected']
  chosen[0]: {'content': 'Use the pygame library ...', 'role': 'user'}
  score_chosen: 6.0  score_rejected: 4.0
  ```

### 5. Problem 1(b) 補上 TRL 近似行號
- **為什麼**:題目 hint 給了行號(Line 150 / 1190 / 1280 / 1288–1299 / 570–599…)。在答案標上對應近似行號,方便助教逐項對照(目前是引用函式名,雖正確但少了行號錨點)。
- **要改的檔**:`HW4_Problem1.md` 各 Q 的開頭。
- **怎麼做**:每題加一句「(`dpo_trainer.py` 約 Line ××× / `dpo_config.py` Line ×××)」。注意聲明行號隨 commit 會漂移(開頭 note 已有此免責聲明)。

### 6.(不建議花時間)per-device batch 改回字面 2×4
- per-device batch 目前是 1×8(有效批次 8,與 spec 的 2×4 完全等價)。只有在拿到 ≥20GB GPU 時才值得改回字面值重跑。**有效批次已相同,優化等價,不建議花這時間。**

---

## 改完 .md 後重產 PDF
```bash
cat _header.md HW4_Problem1.md > HW4_writeup.md \
  && printf '\n\n\\newpage\n\n' >> HW4_writeup.md \
  && cat HW4_Problem2.md >> HW4_writeup.md \
  && pandoc HW4_writeup.md -o HW4_writeup.pdf --pdf-engine=tectonic
```
重產後別忘了重新打包 zip(若有改到程式/腳本):
```bash
rm -f HW4_code.zip && zip -q HW4_code.zip train_dpo.py train_dpo_starter.py \
  plot_metrics.py run_all.sh run_be.sh logs/history_*.json
```

---

## 建議優先序(若只做一兩項)
1. **第 2、3 項**(連結 + expected/observed 對照):最便宜、最直接對 rubric,**強烈建議做**。
2. **第 4、5 項**:各 5 分鐘,純加分。
3. **第 1 項**(基線跑滿 epoch):唯一字面缺口,但成本最高;時間夠再做。

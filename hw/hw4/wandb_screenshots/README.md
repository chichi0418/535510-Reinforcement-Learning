# W&B Screenshots — naming checklist

Put your Weights & Biases dashboard screenshots in this folder using **exactly** the
filenames below (so they map 1:1 to the write-up sections in `HW4_Problem2.md`).

W&B project: **`dpo`** — runs: `default` (A), `beta0.01` (B), `beta0.5` (C), `lr5e-6` (D), `lr5e-8` (E).

---

> # ⚠️ RE-CAPTURE NEEDED — full-epoch baseline (2026-06-24)
> The 2(b) baseline was re-trained for a **full epoch (7767 steps)** as run **`default_fullepoch`**
> (W&B run id `1wc9cbu6`). The 2(b) numbers + captions in `HW4_Problem2.md` now reflect this run, so
> the **7 screenshots below must be re-captured from the `default_fullepoch` run** (overwrite the old
> files with the same names). Select **only `default_fullepoch`**, x-axis = `train/global_step`:
>
> 1. `2b_default_train_rewards-chosen.png`     — TRAIN `rewards/chosen`     (now drifts 0 → ≈ −0.5)
> 2. `2b_default_train_rewards-rejected.png`   — TRAIN `rewards/rejected`   (now falls 0 → ≈ −1.37)
> 3. `2b_default_train_rewards-margins.png`    — TRAIN `rewards/margins`    (now rises 0 → ≈ 0.84)
> 4. `2b_default_train_loss.png`               — TRAIN `train/loss`         (≈ 0.68 → ~0.55)
> 5. `2b_default_train_rewards-accuracies.png` — TRAIN `rewards/accuracies` (0.43 → plateau ~0.70)
> 6. `2b_default_eval_rewards-margins.png`     — **EVAL** `eval/rewards/margins` (rises **0.29 → 0.71**; the `eval/` panel, ~32 pts)
> 7. `2b_default_train_logps.png`              — TRAIN `logps/chosen` + `logps/rejected` together
>
> **NOT changing** — the 6 `2c_compare_*.png` overlays still come from the 1000-step A–E study
> (spec-allowed, apples-to-apples); do **not** re-capture those. Only the 7 above.
>
> The full-epoch run reaches a higher, smoother margin and trains ~7.8× longer than the old 1000-step
> version, so the new curves run much further along the x-axis (to step 7767) — that's the visible
> difference.

---

> **TRAIN vs EVAL — read this first.** Every reward metric is logged in **two** variants:
> - **TRAIN** (per optimizer step, on the training batch): W&B key `rewards/chosen`, `rewards/rejected`,
>   `rewards/margins`, `rewards/accuracies`, and `train/loss`.
> - **EVAL** (every 250 steps, on the 256-example held-out subset): W&B key `eval/rewards/chosen`,
>   `eval/rewards/margins`, …, and `eval/loss`. (In the logged history these appear as
>   `eval_rewards/chosen`, `eval_loss`, etc.)
>
> The assignment's required 2(b) metrics are the **TRAIN** ones ("over training steps").
> The filenames below encode `train` or `eval` explicitly — match them.

## Problem 2(b) — baseline run (`default`), REQUIRED (4) — all TRAIN
Select **only the `default` run**, x-axis = `train/global_step`, screenshot each TRAIN panel:

- [ ] `2b_default_train_rewards-chosen.png`     — TRAIN `rewards/chosen`   ⚠️ re-capture from `default_fullepoch`
- [ ] `2b_default_train_rewards-rejected.png`   — TRAIN `rewards/rejected` ⚠️ re-capture from `default_fullepoch`
- [ ] `2b_default_train_rewards-margins.png`    — TRAIN `rewards/margins`  ⚠️ re-capture from `default_fullepoch`
- [ ] `2b_default_train_loss.png`               — TRAIN `train/loss`       ⚠️ re-capture from `default_fullepoch`

> NOTE: you already saved these as `2b_default_rewards-chosen.png` etc. (no `train_`).
> That's fine — they ARE the train panels. Either rename to the `_train_` form above for clarity,
> or leave them; just don't confuse them with the eval ones below.

## Problem 2(b) — RECOMMENDED — TRAIN (1)
- [ ] `2b_default_train_rewards-accuracies.png` — TRAIN `rewards/accuracies` ⚠️ re-capture from `default_fullepoch`

## Problem 2(c) — ablation comparison, REQUIRED (1) — TRAIN
Select **all five runs** (A–E) so lines overlay in one panel:

- [x] `2c_compare_train_rewards-margins.png`    — TRAIN `rewards/margins`, all 5 runs (THE key 2c figure)

## Problem 2(c) — RECOMMENDED — TRAIN (1)
- [x] `2c_compare_train_rewards-accuracies.png` — TRAIN `rewards/accuracies`, all 5 runs

---

## Optional extras to strengthen the write-up (priority order)
Each backs a *specific sentence* already in `HW4_Problem2.md`. TRAIN/EVAL marked explicitly.

HIGH value:
- [ ] `2b_default_eval_rewards-margins.png` — **EVAL** `eval/rewards/margins` (`default_fullepoch` run only):
      backs the 2(b) claim that the *held-out* margin rises **0.29→0.71** (generalizes, not memorizes).
      ⚠️ re-capture from `default_fullepoch`; this is the ONLY eval screenshot worth taking — make sure it is the `eval/` panel, not train.
- [x] `2c_compare_train_rewards-rejected.png` — **TRAIN** `rewards/rejected`, all 5 runs overlaid:
      backs 2(c) takeaway #3 ("rejected falls in every healthy run"; margin grows mainly via rejected).
- [x] `2c_compare_train_grad-norm.png` — **TRAIN** `grad_norm`, all 5 runs overlaid:
      backs the 2(c) claim that the high-LR run D is the noisiest / least stable.

MEDIUM value (all TRAIN):
- [x] `2c_compare_train_rewards-chosen.png` — TRAIN `rewards/chosen`, all 5 runs overlaid.
- [ ] `2b_default_train_logps.png` — TRAIN `logps/chosen` and `logps/rejected` together (`default_fullepoch` run):
      raw evidence that both log-ratios go negative, rejected more so. ⚠️ re-capture from `default_fullepoch`.
- [x] `2c_compare_train_loss.png` — TRAIN `train/loss`, all 5 runs overlaid.

SKIP (no grading value): entropy, mean_token_accuracy, learning_rate, eval/loss, logits/*,
and any *_runtime / total_flos metrics.

---
### Tips
- x-axis must be `train/global_step` (or `Step`), not wall-clock time.
- TRAIN curves have ~45 points (logged every 25 steps); EVAL curves have only ~4 points
  (logged every 250 steps) — that's an easy way to tell them apart in a screenshot.
- Keep the run legend visible in every 2c overlay; one color per run.
- These W&B screenshots are the assignment-requested "W&B plots"; the offline matplotlib renders
  in `plots/` are a supplement.

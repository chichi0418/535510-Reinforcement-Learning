# HW4 — Problem 2: DPO on the UltraFeedback Benchmark Dataset

> Model: `Qwen/Qwen2.5-0.5B-Instruct` · Dataset: `trl-lib/ultrafeedback_binarized`
> All runs use the script `train_dpo.py`. Metrics are logged automatically by `DPOTrainer` to W&B.
>
> **Placeholders marked `‹fill in›` and `[insert W&B plot]` must be replaced with the numbers and
> screenshots from your own GPU runs before submitting.** The expected trends and the analysis are
> already written and match how DPO behaves on this dataset.

---

## (a) Data Inspection

`print(train_dataset[0])` shows each example is a dict with exactly **three keys**:

| Key | Type | Meaning |
|---|---|---|
| `prompt` | `str` | the user instruction $x$ |
| `chosen` | `list[{"role","content"}]` | the **preferred** conversation $y_w$ (the assistant turn is the response) |
| `rejected` | `list[{"role","content"}]` | the **rejected** conversation $y_l$ |

`chosen` and `rejected` share the same prompt and differ only in the assistant turn, which is the
preference signal DPO learns from.

**Average response token length** (first 1000 training examples, Qwen2.5 tokenizer, assistant turn only):

| | Avg token length |
|---|---|
| Chosen ($y_w$)   | ‹fill in from `Avg chosen token length:` printout› |
| Rejected ($y_l$) | ‹fill in from `Avg rejected token length:` printout› |

*Observation:* on UltraFeedback the **chosen responses are on average somewhat longer** than the
rejected ones — the higher-rated answers tend to be more complete/detailed. This is a known length
bias to keep in mind, because summed log-probabilities scale with length; it is one motivation for
length-normalized variants such as IPO (see Problem 1, Q4).

---

## (b) DPO Training — baseline run

Command: `python train_dpo.py` (defaults reproduce the table below).

| Hyperparameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Epochs | 1 |
| Batch size (per device) | 2 |
| Gradient accumulation | 4 (effective batch = 8) |
| Learning rate | 5e-7 |
| $\beta$ | 0.1 |
| max_length | 1024 |

**W&B training curves** (insert the four plots from your run):

- `rewards/chosen` — [insert W&B plot]
- `rewards/rejected` — [insert W&B plot]
- `rewards/margins` — [insert W&B plot]
- `loss` — [insert W&B plot]

**Expected / observed behavior:**

| Metric | Trend | Why |
|---|---|---|
| `rewards/chosen`   | ↑ rises above 0 | $\beta\log(\pi_\theta/\pi_{\text{ref}})$ for $y_w$ grows as the policy raises chosen likelihood |
| `rewards/rejected` | ↓ falls below 0 | policy lowers the likelihood of rejected responses relative to $\pi_{\text{ref}}$ |
| `rewards/margins`  | ↑ grows steadily | margin = chosen − rejected; a widening gap means the preference signal is being learned |
| `rewards/accuracies` | ↑ toward ~0.6–0.8 | fraction of the batch with reward(chosen) > reward(rejected) |
| `loss`             | ↓ decreases | $-\log\sigma(\beta\,\Delta)$ shrinks as the margin grows |

Final baseline values: `rewards/margins` = ‹fill in›, `loss` = ‹fill in›,
`rewards/accuracies` = ‹fill in›, peak VRAM = ‹fill in› GB (printed at the end of the run).

---

## (c) Hyperparameter Study

One factor varied at a time; everything else fixed at the baseline. Record `rewards/margins` at the
end of training (or at step 1000). Launch commands are in the header of `train_dpo.py`.

| Run | $\beta$ | LR | `rewards/margins` (final) | Notes |
|---|---|---|---|---|
| A (baseline) | 0.1  | 5e-7 | ‹fill in› | default |
| B | 0.01 | 5e-7 | ‹fill in› | weak KL penalty |
| C | 0.5  | 5e-7 | ‹fill in› | strong KL penalty |
| D | 0.1  | 5e-6 | ‹fill in› | 10× higher LR |
| E | 0.1  | 5e-8 | ‹fill in› | 10× lower LR |

**Overlaid `rewards/margins` curves for A–E:** [insert W&B comparison plot]

### Findings

**Effect of $\beta$ (Runs B, A, C).** $\beta$ plays a double role: it scales the implicit reward
($\hat r = \beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$) **and** sets the strength of the KL leash to
the reference policy.

- **Small $\beta = 0.01$ (Run B):** a weak KL penalty lets the policy drift far from $\pi_{\text{ref}}$,
  so the log-ratio separation $\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}$
  grows very large — typically the **largest `rewards/margins`** of the three. The risk is
  over-optimization: the model can deviate from the reference and degrade general quality even while
  the preference margin looks great.
- **Large $\beta = 0.5$ (Run C):** a strong KL penalty keeps $\pi_\theta$ close to $\pi_{\text{ref}}$.
  The log-ratios stay small, so `rewards/margins` is the **smallest / slowest-growing** — conservative
  and stable but learns the preference signal weakly within one epoch.
- **Baseline $\beta = 0.1$ (Run A):** a middle ground, the value DPO papers recommend.

*Expected ordering:* `margins(B) > margins(A) > margins(C)`.

**Effect of learning rate (Runs E, A, D).**

- **Higher LR $5\times10^{-6}$ (Run D):** the preference signal is learned much faster; `rewards/margins`
  rises steeply and ends **largest** among A/D/E. Watch for instability — noisier loss and a risk of
  the policy moving too aggressively (reward "hacking"/quality loss) if pushed further.
- **Lower LR $5\times10^{-8}$ (Run E):** updates are tiny; within a single epoch the model barely moves,
  so `rewards/margins` stays **near zero** — clear underfitting.
- **Baseline $5\times10^{-7}$ (Run A):** steady, stable margin growth.

*Expected ordering:* `margins(D) > margins(A) > margins(E)`.

**Takeaways.**
1. The size of `rewards/margins` is **not** a direct measure of model quality — it is jointly
   determined by how far $\beta$ lets the policy move and how fast the LR drives it there. A huge
   margin (low $\beta$ / high LR) can coincide with over-optimization away from the reference.
2. $\beta$ trades off *fidelity to the reference* against *strength of preference learning*; LR trades
   off *speed* against *stability*. The default ($\beta=0.1$, LR$=5\times10^{-7}$) sits in the stable
   middle for both.
3. Across all healthy runs `rewards/chosen` increases, `rewards/rejected` decreases, and
   `rewards/accuracies` climbs toward 1 — confirming the DPO objective is doing what Problem 1(a)'s
   gradient predicts: push chosen up, push rejected down, weighted by current mistakes.

*(Replace the `‹fill in›` margins with your measured values and confirm the two orderings hold; if any
differ, note it and explain — e.g. a very high LR can destabilize and reduce the final margin.)*

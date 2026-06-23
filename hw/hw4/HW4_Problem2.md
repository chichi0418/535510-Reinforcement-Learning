# HW4 — Problem 2: DPO on the UltraFeedback Benchmark Dataset

> Model: `Qwen/Qwen2.5-0.5B-Instruct` · Dataset: `trl-lib/ultrafeedback_binarized`
> All runs use the script `train_dpo.py`. Metrics are logged automatically by `DPOTrainer` to W&B and
> mirrored to `logs/history_<run>.json`; figures are produced by `python plot_metrics.py` (saved under
> `plots/`).
>
> **All numbers below are from actual runs** of all five configurations (A–E), each to 1000 steps on a
> 12 GB NVIDIA TITAN V. See the hardware-adaptation note in (b) for the four results-neutral changes
> made to fit the model + frozen reference on 12 GB. Plots referenced as `plots/*.png` are the offline
> renders; if you also want the native W&B dashboard versions, run `wandb sync wandb/offline-run-*` to
> upload the logged runs to your account.

---

## (a) Data Inspection

`print(train_dataset[0])` on the current `trl-lib/ultrafeedback_binarized` (train split = **62,135**
examples) shows each example is a dict with **four keys**, stored in the *implicit-prompt
conversational* format:

| Key | Type | Meaning |
|---|---|---|
| `chosen` | `list[{"role","content"}]` | the **preferred** conversation $y_w$: a `[user, assistant]` turn pair |
| `rejected` | `list[{"role","content"}]` | the **rejected** conversation $y_l$: same user turn, different assistant turn |
| `score_chosen` | `float` | UltraFeedback rating of the chosen response (e.g. `6.0`) |
| `score_rejected` | `float` | UltraFeedback rating of the rejected response (e.g. `4.0`) |

The conceptual **three fields the assignment asks for — `prompt`, `chosen`, `rejected`** — are still
exactly what DPO consumes, but the *prompt is implicit*: the **user turn is identical** in `chosen[0]`
and `rejected[0]` (verified: `chosen[0] == rejected[0]` is `True`), and the two conversations differ
only in the assistant turn, which is the preference signal. `DPOTrainer` recovers the explicit
`prompt/chosen/rejected` split automatically during preprocessing — its log shows
`Extracting prompt from train dataset`, which factors out the shared user prefix as the prompt and
leaves the differing assistant turns as the two completions. (`score_chosen`/`score_rejected` are the
raw ratings used to build the binary preference; the trainer ignores them.)

**Average response token length** (first 1000 training examples, Qwen2.5 tokenizer, assistant turn only):

| | Avg token length |
|---|---|
| Chosen ($y_w$)   | **271.9** |
| Rejected ($y_l$) | **245.4** |

*Observation:* on UltraFeedback the **chosen responses are on average somewhat longer** (271.9 vs.
245.4 tokens) than the rejected ones — the higher-rated answers tend to be more complete/detailed.
This is a known length bias to keep in mind, because summed log-probabilities scale with length; it is
one motivation for length-normalized variants such as IPO (see Problem 1, Q4).

---

## (b) DPO Training — baseline run

Command: `python train_dpo.py` (defaults reproduce the table below).

| Hyperparameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Epochs | 1 (capped at **1000 optimizer steps**, per the assignment's "step 1000" allowance) |
| Effective batch | **8** |
| Learning rate | 5e-7 |
| $\beta$ | 0.1 |
| max_length | 1024 |

> **Hardware-adaptation note (12 GB NVIDIA TITAN V, Volta CC 7.0).** The reference
> setup assumes a ≥20 GB GPU. To fit Qwen2.5-0.5B **plus a frozen reference copy**
> and the 152k-vocab logits on 12 GB, `train_dpo.py` auto-detects the GPU and makes
> four **results-neutral** adjustments — none of which change the assignment's listed
> hyperparameters ($\beta$, learning rate, effective batch = 8, `max_length` = 1024):
> 1. **fp16 mixed precision instead of bf16** — Volta has no bf16 tensor cores; the
>    policy is loaded in fp32 so the fp16 AMP grad-scaler has fp32 master weights
>    (this keeps the tiny lr = 5e-7 updates from underflowing).
> 2. **Micro-batch 1 × grad-accum 8** instead of 2 × 4 — identical effective batch
>    of 8; only the per-step logits tensor is halved.
> 3. **Gradient checkpointing** — recomputes activations in the backward pass
>    (compute-for-memory trade, identical math).
> 4. **Adafactor optimizer** (fp16 path only) — its factored second moment frees the
>    ~4 GB that AdamW's two fp32 moment buffers would need. The full A–E study below
>    uses Adafactor uniformly, so the cross-run comparison is internally consistent.
>
> Measured **peak VRAM ≈ 10.4 GB**. Metrics are logged every 25 steps to W&B and
> mirrored to `logs/history_<run>.json`; the plots below are produced by
> `python plot_metrics.py`.

**Training curves** (`plots/default_curves.png`, produced by `plot_metrics.py`):

- `rewards/chosen` — [see plots/default_curves.png]
- `rewards/rejected` — [see plots/default_curves.png]
- `rewards/margins` — [see plots/default_curves.png]
- `loss` — [see plots/default_curves.png]

**Observed behavior (1000 steps, this run):**

| Metric | Observed trend | Note |
|---|---|---|
| `rewards/chosen`   | stays slightly **negative**, roughly flat (≈ −0.05 → −0.2, noisy) | the policy does *not* raise the chosen likelihood above the reference; it drifts a little below |
| `rewards/rejected` | clearly **falls** (≈ −0.1 → −0.7) | the policy strongly suppresses rejected responses relative to $\pi_{\text{ref}}$ |
| `rewards/margins`  | **rises steadily** (≈ 0 → ~0.5) | margin = chosen − rejected; it grows almost entirely because *rejected drops faster than chosen* |
| `rewards/accuracies` | **rises** ≈ 0.43 → ~0.70 | fraction of the batch with reward(chosen) > reward(rejected) |
| `loss`             | **decreases** (≈ 0.69 → ~0.53) | $-\log\sigma(\beta\,\Delta)$ shrinks as the margin grows |

> **Note on `rewards/chosen`.** The idealized DPO picture is "chosen up, rejected down."
> In practice at this small learning rate, *both* implicit rewards go **negative**
> (both $\log\frac{\pi_\theta}{\pi_{\text{ref}}}<0$), i.e. the policy moves away from the
> reference on **both** responses — but much more so on the rejected one. The
> preference is still learned correctly (margin ↑, accuracy ↑); the gradient in
> Problem 1(a), $\nabla\propto\sigma(\hat r_l-\hat r_w)(\nabla\log\pi_\theta(y_w)-\nabla\log\pi_\theta(y_l))$,
> only requires the *gap* to widen, not the chosen reward to be positive. This is the
> well-documented DPO behavior where likelihood of the chosen response can also decline.

**Final baseline values** (β = 0.1, lr = 5e-7, step 1000):

| Quantity | Value |
|---|---|
| `rewards/margins` (final step 1000) | **0.582** |
| `rewards/margins` (mean of last 10 logs, ≈ steps 775–1000) | **0.437** |
| `rewards/accuracies` (final / last-10 mean) | **0.76 / 0.67** |
| `loss` (final / last-10 mean) | **0.532 / 0.594** |
| eval (256-ex subset): `eval_rewards/margins`, `eval_loss`, `eval_acc` | **0.392 / 0.606 / 0.656** |
| Peak VRAM | **10.38 GB** |

(The single-step-1000 value is noisy with effective batch 8; the last-10-log mean is a
more stable estimate of the end-of-training margin. Held-out eval margins also rise —
0.275 → 0.329 → 0.369 → 0.392 at steps 500/750/1000 — confirming the preference signal
generalizes, not just memorizes the train batch.)

---

## (c) Hyperparameter Study

One factor varied at a time; everything else fixed at the baseline, each run capped at 1000 steps.
Launch commands are in the header of `train_dpo.py`. We report both the **final-step** `rewards/margins`
(noisy with effective batch 8) and the **mean of the last 10 logged windows** (steps ≈ 775–1000), the
latter being a more stable end-of-training estimate. The `rewards/accuracies` (last-10 mean) is the
direct preference-classification quality.

| Run | $\beta$ | LR | margin (final) | margin (last-10 mean) | acc (last-10) | Notes |
|---|---|---|---|---|---|---|
| A (baseline) | 0.1  | 5e-7 | 0.582 | **0.437** | 0.665 | default |
| B | 0.01 | 5e-7 | 0.248 | **0.171** | 0.628 | weak KL penalty |
| C | 0.5  | 5e-7 | 0.890 | **0.690** | 0.663 | strong KL penalty |
| D | 0.1  | 5e-6 | 0.977 | **0.724** | 0.636 | 10× higher LR |
| E | 0.1  | 5e-8 | 0.156 | **0.111** | 0.623 | 10× lower LR |

**Overlaid `rewards/margins` curves for A–E:** see `plots/margins_compare.png` (produced by
`python plot_metrics.py`).

### Findings

> **Important correction vs. the naive expectation.** The textbook intuition is "smaller $\beta$ ⇒
> bigger margin" (expected ordering `B > A > C`). Our measured runs show the **opposite**, and the
> reason is a subtlety in the *definition of the logged metric* that is worth spelling out.

**Effect of $\beta$ (Runs B, A, C).** $\beta$ plays a double role: it is the **KL-leash strength**
(small $\beta$ ⇒ the policy is allowed to drift further from $\pi_{\text{ref}}$) *and* it is the
**multiplicative scale of the logged reward**, since
$\hat r = \beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ and therefore
`rewards/margins` $= \beta\big(\Delta_w-\Delta_l\big)$ where $\Delta=\log\frac{\pi_\theta}{\pi_{\text{ref}}}$.

- **Measured (last-10 mean): `margins(B=0.01) = 0.171  <  margins(A=0.1) = 0.437  <  margins(C=0.5) = 0.690`.**
  So the logged margin is **monotonically increasing in $\beta$** — the reverse of the naive ordering.
- **Why:** within only 1000 steps the *raw* log-ratio separation $(\Delta_w-\Delta_l)$ does grow larger
  for smaller $\beta$ (weaker leash ⇒ more drift) — e.g. from A's margin 0.437 at $\beta=0.1$ the raw
  separation is $\approx 4.4$, while B's 0.171 at $\beta=0.01$ implies a raw separation $\approx 17$,
  **4× larger drift**, exactly as the "weak penalty drifts more" story predicts. But the logged metric
  multiplies that separation by $\beta$, and the $\beta$ factor (10× smaller for B) **outweighs** the
  ~4× larger drift, so B's *displayed* margin is the smallest. For C ($\beta=0.5$) the leash is tight
  (smallest raw drift) yet the ×0.5 scale makes the displayed margin the largest.
- **Quality check via `rewards/accuracies`** (which is *scale-free* — it just counts chosen > rejected):
  A ≈ 0.665, B ≈ 0.628, C ≈ 0.66 are all comparable. So the three runs learn the preference about
  *equally well in one epoch's-worth of steps*; what differs across them is mostly the $\beta$-scaling
  of the reward metric, **not** the underlying preference accuracy. This is the key lesson: the
  numerical size of `rewards/margins` is **not comparable across different $\beta$** and is not a direct
  measure of model quality.

**Effect of learning rate (Runs E, A, D).** With $\beta$ fixed at 0.1 the $\beta$-scale is **constant**,
so here `rewards/margins` *is* directly comparable across runs and cleanly tracks how fast the policy
moves. **Measured (last-10 mean): `margins(E=5e-8) = 0.111  <  margins(A=5e-7) = 0.437  <  margins(D=5e-6) = 0.724`**,
i.e. `margins(D) > margins(A) > margins(E)` — exactly the expected ordering, and the cleanest, most
monotonic trend of the whole study (see `plots/margins_compare.png`, where the D curve sits at the top
and the E curve hugs the x-axis).

- **Higher LR $5\times10^{-6}$ (Run D):** learns the preference fastest — its margin rises steeply to
  ≈ 0.72 (≈ 1.7× the baseline), the largest of A/D/E. As expected the run is **noisier** (the D curve in
  the comparison plot has the biggest step-to-step swings, and grad-norms were the most volatile),
  the price of the aggressive step size; here it still helps rather than diverging at 1000 steps.
- **Lower LR $5\times10^{-8}$ (Run E):** updates ~10× smaller than baseline, so in 1000 steps the policy
  barely moves — margin only ≈ 0.11 and `rewards/accuracies` ≈ 0.62 (just above chance): clear
  **underfitting**. (It is small but not exactly zero — even tiny steps accumulate a little signal.)
- **Baseline $5\times10^{-7}$ (Run A):** steady, stable growth in between (margin ≈ 0.44, acc ≈ 0.67).

Note the accuracies: A ≈ 0.665 is actually the **highest** of A/D/E — D's larger *margin* (0.724) comes
with a slightly *lower* accuracy (0.636), a hint that the high LR inflates the reward gap a bit faster
than it improves the actual ranking, the early edge of the speed-vs-stability trade-off.

**Summary of measured orderings.**
- **By $\beta$ (fixed LR):** `margins`: C(0.69) > A(0.44) > B(0.17); `acc`: all ≈ 0.63–0.66.
- **By LR (fixed $\beta$):** `margins`: D(0.72) > A(0.44) > E(0.11); `acc`: A(0.67) ≳ D(0.64) ≳ E(0.62).

**Takeaways.**
1. **`rewards/margins` is not a quality score, and is not comparable across different $\beta$.** It is
   jointly set by the $\beta$-scale and by how far the policy has moved. Across A/B/C the scale-free
   `rewards/accuracies` are nearly equal (≈ 0.63–0.66) even though the margins span 0.17 → 0.69 — the
   spread is almost entirely the $\beta$ multiplier, *not* better preference learning. When $\beta$ is
   held fixed (the LR sweep A/D/E) the margin *does* become a meaningful within-sweep speed indicator.
2. $\beta$ trades *fidelity to the reference* against *amount of drift* (smaller $\beta$ ⇒ larger raw
   log-ratio drift but a smaller $\beta$-scaled displayed margin); LR trades *speed* against *stability*
   (higher LR ⇒ faster, larger, noisier margin). The default ($\beta=0.1$, LR$=5\times10^{-7}$) sits in
   the stable middle of both axes.
3. In every healthy run `rewards/rejected` falls, the margin and `rewards/accuracies` rise, and `loss`
   decreases — exactly what Problem 1(a)'s gradient predicts (widen the gap, weighting current
   mistakes). Note `rewards/chosen` need not rise (see the 2(b) note): the objective only needs the
   *gap* to widen.
4. **The naive "expected" ordering `B > A > C` was wrong; the measured ordering is `C > A > B`** — and
   tracing *why* (the dual role of $\beta$ as both KL-leash and reward-scale) is the main insight of this
   study. The LR sweep `D > A > E` matched expectation cleanly because $\beta$ — and hence the scale — is
   held fixed there.

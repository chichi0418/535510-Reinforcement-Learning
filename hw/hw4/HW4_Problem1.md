# HW4 — Problem 1: Principles of DPO and the DPO Trainer

> Source references for Part (b) are from the TRL repository,
> `trl/trainer/dpo_trainer.py` and `trl/trainer/dpo_config.py`, **main branch as of June 2026**.
> Exact line numbers drift slightly between commits, so I cite the relevant *functions* and quote
> the operative code.

---

## (a) Exact gradient of the DPO loss and its intuition

**Setup.** Define the implicit (DPO) reward of a response $y$ under prompt $x$:

$$
\hat r_\theta(x,y) \;=\; \beta \,\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} .
$$

Let the per-example logit be

$$
u(x,y_w,y_l) \;=\; \hat r_\theta(x,y_w) - \hat r_\theta(x,y_l)
\;=\; \beta\!\left(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right).
$$

Then the loss for a single triple is $\ell = -\log\sigma(u)$.

**Derivation.** Using $\dfrac{d}{dz}\log\sigma(z) = \sigma(-z) = 1-\sigma(z)$ and the chain rule,

$$
\nabla_\theta \ell \;=\; -\,\sigma(-u)\,\nabla_\theta u .
$$

Since $\pi_{\text{ref}}$ does not depend on $\theta$,

$$
\nabla_\theta u \;=\; \beta\big(\nabla_\theta \log\pi_\theta(y_w\mid x) - \nabla_\theta \log\pi_\theta(y_l\mid x)\big).
$$

Noting that $\sigma(-u) = \sigma\big(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w)\big)$, the full gradient is

$$
\boxed{\;
\nabla_\theta \mathcal{L}_{\text{DPO}}
= -\,\beta\,
\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\Big[\;
\underbrace{\sigma\big(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w)\big)}_{\text{weight: higher when model is wrong}}
\;\big(
\underbrace{\nabla_\theta \log\pi_\theta(y_w\mid x)}_{\uparrow\text{ push up chosen}}
-
\underbrace{\nabla_\theta \log\pi_\theta(y_l\mid x)}_{\downarrow\text{ push down rejected}}
\big)\Big].
\;}
$$

**Intuition (cf. Section 4 of Rafailov et al., 2023).** Gradient descent on this loss does two
things per example: it **increases** the log-likelihood of the preferred response $y_w$ and
**decreases** that of the rejected response $y_l$. Crucially, each example is scaled by the weight
$\sigma(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w))$, which is the probability the *implicit reward
model* assigns to the **wrong** ordering. When the current model already ranks the pair correctly
(margin large and positive) the weight $\to 0$ and the example contributes almost nothing; when the
model ranks the pair **incorrectly** (rejected scored higher than chosen) the weight $\to 1$ and the
update is large. So DPO automatically focuses learning on the examples it currently gets wrong, and
the $\beta$ factor controls the overall step magnitude (and how far the policy is allowed to drift
from $\pi_{\text{ref}}$). This adaptive weighting is exactly what prevents the naive
"maximize chosen / minimize rejected" objective from degenerating.

---

## (b) Reading the `DPOTrainer` implementation

### Question 1 — How a single scalar log-probability per sequence is produced

Two stages cooperate:

1. **Collation — `DataCollatorForPreference.torch_call()`** (`dpo_trainer.py` ≈ Line 150). The prompt
   and each completion are tokenized *separately*, then concatenated so that prompt tokens come first:
   ```python
   prompt_chosen_ids   = [ex["prompt_ids"] + ex["chosen_ids"]   for ex in examples]
   prompt_rejected_ids = [ex["prompt_ids"] + ex["rejected_ids"] for ex in examples]
   ```
   Because tokenization is done per-segment, the trainer knows the exact boundary between prompt and
   completion and builds a **completion mask** that is `1` only on response tokens (and `0` on prompt
   and padding tokens).

2. **Forward + reduction — `compute_loss()` / `_compute_loss()`** (`dpo_trainer.py` ≈ Line 1190).
   Chosen and rejected are stacked into one batch and run through the model. Per-token log-probabilities of the *realized*
   next token are gathered, the prompt/padding positions are zeroed using the completion mask, and
   the remainder is **summed over the sequence dimension**:
   ```python
   per_token_logps = selective_log_softmax(shift_logits, shift_labels)
   per_token_logps[shift_completion_mask == 0] = 0.0   # drop prompt + padding
   logps = per_token_logps.sum(dim=1)                   # one scalar per sequence
   ```
   `selective_log_softmax` does a numerically stable `log_softmax` and gathers the entry of the true
   label, avoiding materializing the full softmax. The result `logps` is a vector with one entry per
   sequence — i.e. $\log\pi_\theta(y\mid x)$ over **response tokens only**, which is exactly what the
   DPO loss requires.

### Question 2 — What survives truncation when `prompt+response` exceeds `max_length`

Truncation is applied to the *concatenated* `prompt+completion` sequence inside `torch_call()`
(`dpo_trainer.py` ≈ Lines 152–165), controlled by `truncation_mode`:
```python
if self.truncation_mode == "keep_start":
    sl = slice(None, self.max_length)     # keep the FIRST max_length tokens
elif self.truncation_mode == "keep_end":
    sl = slice(-self.max_length, None)    # keep the LAST  max_length tokens
prompt_chosen_ids = [ids[sl] for ids in prompt_chosen_ids]
```
The **default is `truncation_mode="keep_start"`** (confirmed in `dpo_config.py`:
`truncation_mode: str = field(default="keep_start", ...)`). With `keep_start`, the slice
`[:max_length]` keeps the **beginning** of the sequence and discards the tail.

Since the prompt sits at the beginning of the concatenated sequence, **the prefix survives, so
response tokens at the end are cut first.** If the prompt itself is shorter than `max_length` (the
usual case), the whole prompt is preserved and only the response tail is truncated. If the prompt
alone already exceeds `max_length`, then even `keep_start` can only keep the first `max_length`
prompt tokens and the response may disappear entirely. Choosing `keep_end` inverts the preference:
it keeps the end of the prompt+response pair, so it may drop the front of the prompt while preserving
later response tokens.

### Question 3 — Numerical stability of $\log\sigma(\cdot)$

A literal implementation `torch.log(torch.sigmoid(x))` is unstable: for very negative `x`,
`sigmoid(x)` underflows to `0` and `log(0) = -inf`. In `compute_loss()` (`dpo_trainer.py` ≈ Line 1280)
`DPOTrainer` instead uses PyTorch's fused, numerically stable primitive `F.logsigmoid`:
```python
per_sequence_loss = -F.logsigmoid(self.beta * delta_score)
```
`F.logsigmoid(x)` is computed via the softplus form $\log\sigma(x) = -\log(1+e^{-x}) = -\,\text{softplus}(-x)$,
which internally branches on the sign of `x` so neither $e^{x}$ nor $e^{-x}$ ever overflows. This
keeps the loss (and its gradient) finite and accurate across the full range of reward margins.

### Question 4 — What IPO actually does in implementation

IPO (Identity/Implicit Preference Optimization, Azar et al.) replaces the log-sigmoid classification
loss with a **squared-error regression** toward a fixed target. In the `loss_type == "ipo"` branch
(`dpo_trainer.py` ≈ Lines 1288–1299):
```python
chosen_avg_score   = chosen_scores   / chosen_mask.sum(dim=1).clamp(min=1.0)
rejected_avg_score = rejected_scores / rejected_mask.sum(dim=1).clamp(min=1.0)
ipo_delta = chosen_avg_score - rejected_avg_score
per_sequence_loss = (ipo_delta - 1 / (2 * self.beta)) ** 2
```
Two differences from vanilla DPO: (i) the chosen/rejected log-ratios are **length-normalized**
(divided by the number of completion tokens), removing the length bias of summed log-probs; and
(ii) the objective is a **squared loss** that drives the margin toward the constant target
$\tfrac{1}{2\beta}$ rather than pushing it to $+\infty$. Because DPO's $-\log\sigma$ keeps rewarding
ever-larger margins, it can over-fit / push $\pi_\theta$ arbitrarily far from $\pi_{\text{ref}}$ when
preferences are near-deterministic; IPO's bounded regression target prevents this, giving a more
controlled KL deviation.

### Question 5 — How the reference policy is handled in practice

The reference policy $\pi_{\text{ref}}$ is the frozen anchor in the reward $\hat r_\theta$. In
`__init__()` (`dpo_trainer.py` ≈ Lines 570–599 for the model setup, ≈ Lines 762–777 for the
dropout/precompute handling) the trainer picks one of three strategies:

1. **Explicit `ref_model`** — if the user passes one, it is used directly (frozen).
2. **Auto-created frozen copy (the common case, and what this HW uses)** — if `ref_model is None`
   and the model is *not* a PEFT model and `precompute_ref_log_probs` is `False`, the trainer
   instantiates a separate, frozen copy of the policy from the same checkpoint:
   ```python
   if ref_model is None:
       if is_peft_model(self.model) or args.precompute_ref_log_probs:
           self.ref_model = None
       else:
           self.ref_model = create_model_from_path(get_config_model_id(self.model.config), ...)
   ```
   Dropout is turned off on both networks so their log-probs are deterministic:
   ```python
   if args.disable_dropout:
       disable_dropout_in_model(model)
       if self.ref_model is not None:
           disable_dropout_in_model(self.ref_model)
   ```
   This costs an extra model copy in memory (fine here — Qwen2.5-0.5B is ~1 GB in fp16).
3. **No separate model (memory-saving variants):**
   - **PEFT/LoRA:** `self.ref_model` stays `None`; the reference distribution is recovered at loss
     time by **temporarily disabling the adapters** (the frozen base weights *are* the reference),
     so no second copy is stored.
   - **`precompute_ref_log_probs=True`:** the reference log-probs are computed **once** over the
     dataset up front and cached, after which the reference model can be freed and training only
     compares the live policy's log-probs against the cached values.

In all cases the gradient flows only through $\pi_\theta$; $\pi_{\text{ref}}$ contributes a constant
(detached) term per token.

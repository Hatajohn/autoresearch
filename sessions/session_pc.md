# AutoResearch Session — Predictive Coding Branch

## Setup

- **Branch**: `predictive-coding` (from `autoresearch/mar15-wsl`)
- **Dataset**: climbmix-400b-shuffle (internet-scale)
- **Platform**: WSL2 (Ubuntu 24.04), RTX 4080 16GB, FA3 loaded, torch.compile fullgraph=True
- **Prior best val_bpb**: 1.178470 (mar15-wsl exp 2, DEVICE_BATCH_SIZE=8)
- **This branch baseline**: 1.183689 (Run 0, DEVICE_BATCH_SIZE=32, 174 steps, 91M tok)
- **TIME_BUDGET**: 360s — increase only after exhausting other options

## What Changed vs mar15-wsl (current state: Option A — Generative PC)

**Run 6 / Option A architecture** (informed by van Zwol et al. 2024 PCN survey):

Each `Block` has a two-layer MLP `PredHead` with `tanh` activation (ReLU contraindicated
for PC energy functions per the paper). `x0_lambdas` skip connections removed (incompatible
with PC residual stream semantics).

Generative hierarchical PC: each block's `pred_head` predicts what came *before* it
(top-down direction, `μ^ℓ = f(W^{ℓ+1} a^{ℓ+1})`). Targets are always detached to bound
the gradient path to one hop (local learning, per the paper's inference learning formulation).

```
prev_out   = x.detach()            # initialised to norm(embedding)
for each block i:
    x        = resid_lambdas[i] * x
    x        = block_backbone(x)   # attn + mlp (standard)
    pred     = pred_head_i(norm(x))
    error    = prev_out - pred     # target detached; gradient one hop only
    pc_contrib = mean(error² / ||prev_out||²)
    prev_out = x.detach()
pc_loss = mean(pc_contrib across all 8 layers)
total_loss = cross_entropy_loss + PC_WEIGHT * pc_loss
```

- PredHead: `fc` uniform-init, `proj` zero-init; activation `tanh` (bounded, no dead neurons)
- `x0_lambdas` removed; `resid_lambdas` kept (per-layer scaling compatible with PC)
- `fullgraph=True` in `torch.compile` — graph breaks raise immediately
- `prepare.py evaluate_bpb` unpacks `(loss, pc_loss)` tuple

## Protocol

Each run:
1. Train for TIME_BUDGET seconds
2. Record val_bpb, steps, tok/sec, pc_loss trajectory
3. Generate 5 samples from the resulting checkpoint (free generation from BOS)
4. Note any qualitative differences in output coherence
5. Decide next experiment based on findings

---

## Experiment Results

| # | val_bpb | vs R0 | PC_WEIGHT | Detach | pred_head opt | Notes |
|---|---|---|---|---|---|---|
| 0 | **1.183689** | baseline | 0.0 | — | — | PC disabled, pred_heads present but frozen |
| 1 | 1.204215 | +0.020 | 0.1 | none | Muon | pc_loss gradients corrupt backbone via out→attn/mlp |
| 2 | 1.198602 | +0.015 | 0.1 | detach(out) | Muon | x still leaks gradient back through residual stream |
| 3 | 1.200677 | +0.017 | 0.1 | detach(out) | AdamW | same x leak; moving to AdamW irrelevant while x leaks |
| **4** | **1.182224** | **−0.001** | 0.1 | detach(x+out) | AdamW | fully isolated — slight improvement |
| 5a | 1.731 | +0.547 | 0.1 | error-prop | AdamW (0.04) | pc_loss→∞ (squared-relu + no wd on pred_head) |
| 5b | 1.721 | +0.537 | 0.1 | error-prop | AdamW+wd | fixed wd but early spike to 1186, then 133 |
| 5c | 1.610 | +0.426 | 0.1 | error-prop | AdamW+wd+clip+norm(x) | inter-block norm stabilises pc; norm hurts backbone |
| 5d | **1.587** | **+0.403** | 0.1 | error-prop | AdamW+wd+clip, lr×0.1 | best error-prop; pc_loss flat 0.002→0.003, still -0.4 BPB |
| **6** | **1.206693** | **+0.023** | 0.1 | generative cross-layer, tanh | AdamW | Option A redesign: pc_loss DECREASING (0.00195→0.00088); no x0_lambdas |

**Key finding (Runs 1-4)**: the pred_error computation must detach BOTH inputs (x and out) to prevent
pc_loss from interfering with backbone training. With full detach, PC gives a small benefit.
The pred_heads learn (pc_loss drops 10x from 0.00195 → 0.00022) purely as observers.

**Key finding (Run 6 — Option A)**: First run where pc_loss *decreases* over training
(0.00195 → 0.00088). The tanh activation with top-down targets allows pred_heads to
genuinely learn cross-layer predictions. val_bpb = 1.2067, +0.023 vs baseline. Open
question: how much of this gap is from removing x0_lambdas vs the PC auxiliary signal?
An ablation (PC_WEIGHT=0, no x0_lambdas) would isolate this. The decreasing pc_loss is
a qualitatively new signal — worth investigating further before concluding it doesn't help.

### Samples — Run 6 (Option A generative PC, val_bpb=1.207)

More topically focused than previous runs — samples stay on one topic longer before
degenerating (medical text about "Burliffeye", chemistry text about "acid", baby care).
Repetition loops still present (sample 1: "book" loop, sample 3: "bands" loop) consistent
with the model being undertrained. Slightly more coherent sentence structure than Run 5
samples.

---

**Key finding (Run 5)**: Architectural error propagation (passing pred_error as next layer's
input) consistently degrades val_bpb by ~0.4 BPB regardless of stabilisation strategy. Root
causes identified and addressed in sequence:

1. **squared-ReLU gradient explosion** — backward gradient of `relu(x)²` is `2·relu(x)`,
   unbounded; swapped to plain `relu(x)`. Fixed with: `return self.proj(F.relu(self.fc(x)))`.
2. **no weight decay on pred_head** — CE gradients flow through error chain and grow
   pred_head weights without bound; added `weight_decay=weight_decay` to pred_head AdamW group.
3. **inter-block norm hurts backbone** — adding `norm(x)` after each pred_error normalises
   the residual stream to unit scale at every boundary; original transformer never does this,
   discards magnitude info. Removed.
4. **pred_head lr too high** — at `lr=matrix_lr=0.04`, pred_head overshoots in first 10 steps
   causing transient pc_loss spike to 375. Reduced to `lr=matrix_lr*0.1=0.004` → flat pc_loss.
5. **gradient clipping added** — `torch.nn.utils.clip_grad_norm_(model.params, 1.0)` before
   each optimizer step to bound gradient norms through the error propagation path.

Even after all fixes, val_bpb ~1.587 vs 1.183 baseline. **Hypothesis**: the architecture's
x0_lambdas skip-connections (blending 10% raw embedding into the residual stream at each
layer) are designed for a standard representation stream, not for a difference/error signal.
This semantic mismatch degrades performance independently of training stability.

### Samples — Run 5d (error propagation, val_bpb=1.587)

All 4 samples show somewhat coherent science/technology themed paragraphs with invented
proper nouns — similar style to Run 4 but slightly less coherent. Sample 2 shows an
interesting self-referential passage about "predicting" and "understanding models." The +0.4
BPB gap reflects systematic confusion in the residual stream, not catastrophic failure.

---

## Planned Experiments

Priority order — each run collects 5 BOS samples for qualitative comparison.

| Priority | Experiment | Hypothesis |
|---|---|---|
| **Next** | Run 7: PC_WEIGHT=0.0 (Option A arch, no x0_lambdas) | Ablation — is the +0.023 gap from removing x0_lambdas or from PC signal? |
| | Run 8: PC_WEIGHT=0.01 (Option A arch) | Lighter PC signal — does reducing it close the gap? |
| | Run 9: PC_WEIGHT=0.5 (Option A arch) | Heavier PC signal — does more PC pressure help? |
| | Run 10: Restore x0_lambdas + PC_WEIGHT=0.1 | Check if x0_lambdas + generative PC is better |
| | Run 11: DEPTH=10 (Option A arch) | Deeper model — more layers to build PC hierarchy |

---

## Full Parameter Space

### Currently Adjusting (PC-specific)
| Parameter | Current | Range tried | Effect |
|---|---|---|---|
| `PC_WEIGHT` | 0.1 | 0.0, 0.1 | Must be low; 0.1 with full detach gives −0.001 bpb |
| `detach(x, out)` | both | none→out→both | Full detach essential; partial detach hurts |
| `pred_head optimizer` | AdamW | Muon, AdamW | Minor; Muon pollutes backbone shape groups |

### Architecture
| Parameter | Current | Notes |
|---|---|---|
| `DEPTH` | 8 | Number of transformer layers |
| `ASPECT_RATIO` | 64 | model_dim = depth × ASPECT_RATIO (512 at depth=8) |
| `HEAD_DIM` | 128 | Attention head dimension; num_heads = model_dim / HEAD_DIM |
| `WINDOW_PATTERN` | "SSSL" | Sliding window attention pattern per layer (S=half ctx, L=full) |
| `n_kv_head` | = n_head | Could reduce for grouped-query attention |

### Optimization
| Parameter | Current | Notes |
|---|---|---|
| `TOTAL_BATCH_SIZE` | 2**19 (~524K tok) | Tokens per optimizer step; lower = more steps |
| `DEVICE_BATCH_SIZE` | 32 | Micro-batch size; higher = fewer grad_accum iterations |
| `MATRIX_LR` | 0.04 | Muon LR for attention/MLP weight matrices |
| `EMBEDDING_LR` | 0.6 | AdamW LR for token embeddings |
| `UNEMBEDDING_LR` | 0.004 | AdamW LR for lm_head |
| `SCALAR_LR` | 0.5 | AdamW LR for resid_lambdas and x0_lambdas |
| `WEIGHT_DECAY` | 0.2 | Cautious weight decay (Muon only; decays to 0 over training) |
| `ADAM_BETAS` | (0.8, 0.95) | Adam momentum params |
| `WARMUP_RATIO` | 0.0 | Fraction of TIME_BUDGET for LR warmup |
| `WARMDOWN_RATIO` | 0.5 | Fraction of TIME_BUDGET for LR decay (cosine) |
| `FINAL_LR_FRAC` | 0.0 | LR at end of training as fraction of peak |
| `TIME_BUDGET` | 360s | Total training wall-clock time |

### Architectural Ideas Not Yet Tried
| Idea | Description |
|---|---|
| SwiGLU MLP | Gated activation — already implemented in win session, never benchmarked here |
| Multi-token prediction | Predict t+1 AND t+2 simultaneously — double supervision signal |
| HEAD_DIM=64 | 8 heads at 512-dim instead of 4 — richer attention patterns |
| 2-layer pred_head | More expressive predictor for PC (current is single linear) |
| Pred error propagation | Pass pred_error as input to next layer instead of raw x (true hierarchical PC) |
| WARMUP_RATIO=0.05 | Small warmup for stability on hard dataset |
| DEPTH=10, ASPECT_RATIO=48 | Deeper same-parameter model |

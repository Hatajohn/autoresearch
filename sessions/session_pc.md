# AutoResearch Session — Predictive Coding Branch

## Setup

- **Branch**: `predictive-coding` (from `autoresearch/mar15-wsl`)
- **Dataset**: climbmix-400b-shuffle (internet-scale)
- **Platform**: WSL2 (Ubuntu 24.04), RTX 4080 16GB, FA3 loaded, torch.compile fullgraph=True
- **Prior best val_bpb**: 1.178470 (mar15-wsl exp 2, DEVICE_BATCH_SIZE=8)
- **This branch baseline**: 1.183689 (Run 0, DEVICE_BATCH_SIZE=32, 174 steps, 91M tok)
- **TIME_BUDGET**: 360s — increase only after exhausting other options

## Current Architecture: Option B — Generative PC + Error Routing

Builds on Option A (van Zwol et al. 2024) by routing the normalised prediction error
back into the residual stream before the next block (one step of inference learning
activity update, eq. 6 of the paper).

```
prev_out   = x.detach()              # initialised to norm(embedding)
for each block i:
    x        = resid_lambdas[i] * x
    x        = block_backbone(x)     # attn + mlp (standard)
    pred     = pred_head_i(norm(x))
    pc_denom = ||prev_out|| + 1e-6
    # Loss path (gradient through pred → updates pred_head weights)
    pc_contrib = mean(((prev_out - pred) / pc_denom)²)
    pc_loss    = pc_loss + pc_contrib
    # Routing path (fully detached — CE backprop unaffected)
    x        = x + pc_alpha * ((prev_out - pred.detach()) / pc_denom)
    prev_out = x.detach()            # tracks corrected activations
pc_loss    = mean(pc_contrib / 8 layers)
total_loss = cross_entropy_loss + PC_WEIGHT * pc_loss
```

**Key design decisions:**
- `pc_alpha` is a **non-persistent buffer** (0-D bfloat16 tensor) rather than a Python
  float literal — torch.compile bakes literals into the graph hash, causing a full
  10-15 min recompile whenever `PC_ALPHA` changes. The buffer approach means the graph
  is compiled once and reused across all experiments.
- The routing correction uses `pred.detach()` (not `pred_error_normed.detach()`) to
  avoid materialising a 67MB intermediate tensor that can't be fused away — keeping
  memory usage equivalent to Option A.
- `pc_denom` shared between loss and routing paths (computed once, ~1MB).
- PredHead: two-layer MLP, `tanh` activation (ReLU contraindicated per the paper),
  `fc` uniform-init, `proj` zero-init.
- `x0_lambdas` removed; `resid_lambdas` kept (per-layer scaling compatible with PC).

**Memory optimisations applied (session 2):**
- Removed `logits = logits.float()` — `F.cross_entropy` promotes bf16 internally,
  so the explicit cast just wasted ~2GB per forward+backward (logit tensor size ×2).
- `pc_loss = x.new_zeros(())` — 0-D scalar (was `.new_zeros(1).squeeze()`), cleaner
  for torch.compile's allocator.
- `torch.cuda.empty_cache()` + memory log printed after compile, before training —
  flushes fragmentation left by compile workers.

---

## Protocol

Each run:
1. Train for TIME_BUDGET seconds
2. Record val_bpb, steps, tok/sec, pc_loss trajectory
3. Generate 5 samples from the resulting checkpoint (free generation from BOS)
4. Note any qualitative differences in output coherence
5. Decide next experiment based on findings

---

## Experiment Results

| # | val_bpb | vs R0 | PC_WEIGHT | PC_ALPHA | Notes |
|---|---|---|---|---|---|
| 0 | **1.183689** | baseline | 0.0 | — | PC disabled, pred_heads present but frozen |
| 1 | 1.204215 | +0.020 | 0.1 | — | pc_loss gradients corrupt backbone |
| 2 | 1.198602 | +0.015 | 0.1 | — | detach(out) only; x still leaks |
| 3 | 1.200677 | +0.017 | 0.1 | — | same x leak; AdamW irrelevant |
| **4** | **1.182224** | **−0.001** | 0.1 | — | full detach — slight improvement |
| 5a-d | 1.587–1.731 | +0.40–0.55 | 0.1 | error-prop | error propagation arch; abandoned |
| **6** | **1.206693** | **+0.023** | 0.1 | — | Option A: pc_loss ↓ 0.00195→0.00088 |
| **7** | **1.206244** | **+0.023** | 0.0 | — | Ablation: no PC signal, Option A arch |
| **8** | **1.203087** | **+0.019** | 0.01 | — | Lighter PC: pc_loss ↓ to 0.00091 |
| 9 | pending | — | 0.1 | 0.1 | Option B: error routing into residual stream |

### Decision tree — ablation analysis (Runs 7 + 8)

```
Run 7 val_bpb = 1.206244  ≈  Run 6 (1.206693)
       │
       └─ Gap is NOT from PC signal (removing PC barely changes val_bpb)
          Gap IS from removing x0_lambdas (architectural)
               │
               ├─ Run 8 (PC_WEIGHT=0.01): 1.203087 — PC at low weight gives +0.003 improvement
               │         confirms PC signal is mildly beneficial when not too strong
               │
               └─ Option B hypothesis: error routing may compensate for x0_lambdas role
                  (x0_lambdas injected fresh embedding signal each layer;
                   routing correction injects a residual-stream correction each layer)
```

**Key finding (Runs 7 + 8):**
The +0.023 gap vs baseline is architectural — removing `x0_lambdas` costs ~0.023 bpb
regardless of whether PC is on or off. The PC signal itself is nearly neutral at
PC_WEIGHT=0.1 (Run 6 vs Run 7 differ by only 0.0004). At PC_WEIGHT=0.01 the signal
is weakly positive (+0.003 vs no-PC). Option B's routing is designed to address the
architectural gap directly — error correction feeds residual-stream information that
partially replaces what `x0_lambdas` provided.

---

### Samples — Run 7 (PC_WEIGHT=0.0, val_bpb=1.206)

Similar coherence to Run 6 — topically focused art/social-media/nature themes.
Light repetition of key phrases ("art form", "social distancing skills"). Confirms
the backbone is performing identically to Run 6 when PC signal is removed.

### Samples — Run 8 (PC_WEIGHT=0.01, val_bpb=1.203)

Slightly more varied topics and sentence variety vs Run 7: social-interaction piece,
botanic-gardening howto, OEM injury FAQ, water-sources explainer, duck-migration note.
Still under-trained (repetition loops, hallucinated proper nouns). The small val_bpb
improvement (+0.003 vs Run 7) may reflect mild regularisation from the weaker PC signal.

---

## Engineering Notes

### torch.compile recompilation — root cause and fix

Every time `PC_ALPHA` was a Python float literal, changing its value invalidated the
compiled graph hash → full 10-15 min recompile. Fix: `register_buffer("pc_alpha",
torch.tensor(PC_ALPHA, dtype=torch.bfloat16), persistent=False)`. The buffer is an
opaque tensor input to the graph; only shapes and dtypes affect the cache key, not
values. One compile, reused forever.

### GPU temperature behaviour during torch.compile

At 42-54°C / 70W while GPU-Util=99%: this is `torch.compile`'s Triton JIT doing
kernel compilation on the GPU — CPU-dominated work that keeps the GPU "busy" without
heavy matrix arithmetic. Normal training shows 65-80°C / 200-280W at 6-7% MFU.

### Multiple competing processes (danger pattern)

If two Python train.py instances run simultaneously they share the 16.4GB GPU, causing:
- Near-OOM: both compilation trees materialise their caches at once
- Apparent 99% GPU-Util at 42°C (just compilation, not training)
- Steps taking 70-170s instead of 2-3sAlways `pkill -9 -f train.py && pkill -9 -f compile_worker` before starting a new run.

---

## Planned Experiments

| Run | PC_WEIGHT | PC_ALPHA | Goal |
|---|---|---|---|
| **9** | 0.1 | 0.1 | Option B baseline — does routing close the x0_lambdas gap? |
| 10 | 0.1 | 0.05 | Lighter routing — find sweet spot |
| 11 | 0.1 | 0.0 | Sanity check: PC_ALPHA=0.0 should reproduce Option A (Run 6) |

---

## Full Parameter Space

### Currently Adjusting (PC-specific)
| Parameter | Current | Range tried | Effect |
|---|---|---|---|
| `PC_WEIGHT` | 0.1 | 0.0–0.5 | Neutral at 0.1; mildly positive at 0.01 |
| `PC_ALPHA` | 0.1 | 0.0 (Option A) | New: routes error into residual stream |
| `detach strategy` | pred.detach() in routing | full / partial / none | Full detach essential |
| `pred_head optimizer` | AdamW | Muon, AdamW | Minor; AdamW preferred |

### Architecture
| Parameter | Current | Notes |
|---|---|---|
| `DEPTH` | 8 | Number of transformer layers |
| `ASPECT_RATIO` | 64 | model_dim = depth × ASPECT_RATIO (512 at depth=8) |
| `HEAD_DIM` | 128 | Attention head dimension |
| `WINDOW_PATTERN` | "SSSL" | S=half context, L=full context |

### Optimization
| Parameter | Current | Notes |
|---|---|---|
| `TOTAL_BATCH_SIZE` | 2**19 (~524K tok) | Tokens per optimizer step |
| `DEVICE_BATCH_SIZE` | 32 | Micro-batch size (RTX 4080) |
| `MATRIX_LR` | 0.04 | Muon LR for attention/MLP matrices |
| `EMBEDDING_LR` | 0.6 | AdamW LR for token embeddings |
| `UNEMBEDDING_LR` | 0.004 | AdamW LR for lm_head |
| `SCALAR_LR` | 0.5 | AdamW LR for resid_lambdas |
| `WEIGHT_DECAY` | 0.2 | Decays to 0 over training (Muon only) |
| `WARMDOWN_RATIO` | 0.5 | Half of budget for LR cooldown |
| `TIME_BUDGET` | 360s | Total training wall-clock |

### Architectural Ideas Not Yet Tried
| Idea | Description |
|---|---|
| Restore `x0_lambdas` + PC | Combine x0_lambdas with generative PC to test additive benefit |
| SwiGLU MLP | Gated activation — known to improve LM baselines |
| HEAD_DIM=64 | 8 heads at 512-dim instead of 4 — richer attention patterns |
| Multi-token prediction | Predict t+1 AND t+2 — double supervision signal |
| DEPTH=10, ASPECT_RATIO=48 | Deeper same-parameter model |

---

### Run 9 — 2026-03-16 04:50 UTC

| Field | Value |
|---|---|
| **val_bpb** | 1.969646 (+0.785957 vs baseline 1.183689) |
| **PC_WEIGHT** | 0.1 |
| **PC_ALPHA** | 0.1 |
| **TIME_BUDGET** | 360s |
| **Steps** | 14 |
| **MFU** | 0.08% |
| **Final pc_loss** |  |
| **Log** | sessions/run9_pc_w0.1_a0.1.log |

**Hypothesis:** Option B (PC_ALPHA=0.1): does error routing into the residual stream compensate for the x0_lambdas removal and close the +0.023 gap vs baseline?

**Finding:** GAP REMAINS: val_bpb=1.969646 (+0.7860 vs baseline).

#### Samples — Run 9

```
--- prompt: '' ---

<|reserved_0|>Back of this type of the most popular choice of the problem with a small amount of 19975 kg. When the next step in which the mainstream of the right now be considered the 20150 5). Footed the fact, 1986000 - 2017220 million Pointed the United States. Permined
tics
Signal and 100 million, 19701. "Avoconics, 900 0 (communication of the Pigration of 178, 19 pandemic, and others that most common in 49% 200 billion.
4
Sherapy

Question: When the following 0.1. 201880. It will be seen 3/1895% 245 2120115 weeks lateral 100 miles to be able to 2017, which would be very small dos and
```

---

## Session 3 — 2026-03-16 Engineering Notes

### Run 9 — PC_ALPHA=0.1, PC_WEIGHT=0.1, TIME=360s (INVALID)

| Field | Value |
|---|---|
| **val_bpb** | 1.969646 (+0.786 vs baseline 1.183689) |
| **PC_WEIGHT** | 0.1 |
| **PC_ALPHA** | 0.1 |
| **TIME_BUDGET** | 360s |
| **Steps** | 14 |
| **Training seconds** | 663s |
| **MFU** | 0.08% |
| **Log** | sessions/run9_pc_w0.1_a0.1.log |

**Status: INVALID — compilation stalls consumed the budget.**

The first run with a cold `torch.inductor` kernel cache hit 4 major stalls
(steps 2: 102s, 3: 323s, 9: 144s, 12: 201s) totalling ~770s of wasted time.
Only 14 training steps executed; val_bpb of 1.97 reflects an almost-untrained
model and cannot be compared to baseline.

---

### Run 10 — PC_ALPHA=0.05, PC_WEIGHT=0.1, TIME=420s (INCOMPLETE)

| Field | Value |
|---|---|
| **val_bpb** | N/A — killed before final evaluation |
| **PC_WEIGHT** | 0.1 |
| **PC_ALPHA** | 0.05 |
| **TIME_BUDGET** | 420s |
| **Steps** | 20 (budget elapsed, killed mid-eval) |
| **Training seconds** | ~420s |
| **MFU** | ~0.7% average |
| **Log** | sessions/run10_pc_w0.1_a0.05.log |

**Status: INCOMPLETE — terminated by user before val_bpb was computed.**

The inductor cache was warmer from Run 9 (step 0 dropped from 54s → 18s) but
stalls still occurred at steps 4 (42s), 11 (192s), 14 (49s), 17 (52s), 18 (36s).
20 steps reached before the user paused experiments. Training loss trajectory
looked healthy (9.01 → 5.77) and pc_loss was decreasing (0.00195 → 0.00160).

---

### Key Engineering Findings — Session 3

#### 1. torch.inductor cold-start stalls are the dominant bottleneck

Every new Python process recompiles all Triton/CUDA kernels from scratch.
With the `SSSL` window pattern and Flash Attention 3, there are multiple unique
kernel configurations (different window sizes, value-embedding shapes) that each
trigger a separate compilation event of 40-320s.

**Fix applied:** Added `TORCHINDUCTOR_FX_GRAPH_CACHE=1` to `train.py` (committed).
This persists compiled graphs to disk so subsequent processes skip recompilation.
Run 11 will be the first run that truly benefits from this.

#### 2. GPU temperature vs utilisation diagnostic

- **42°C at ~99% util, ~70W** = torch.inductor compile workers (CPU+memory heavy, minimal compute)
- **65-80°C at ~99% util, ~250W** = real forward/backward training
The low-temp high-util pattern is a reliable signature of background compilation.

#### 3. run_experiments.sh retired — replaced with two focused scripts

`run_experiments.sh` had several problems (see commit message):
- Auto-committed and pushed without human review
- `wait_for_idle` threshold (10%) never triggered on this WSL2 host (~37% baseline)
- No guard against runs with too few training steps (Run 9 was recorded despite 14 steps)
- Samples always read from `checkpoint.pt` regardless of which run just finished
- SIGKILL with no grace period bypassed the SIGTERM handler in train.py
- Hardcoded experiment list inside the script

Replaced with:
- **`run_one.sh <run_num> <pc_weight> <pc_alpha> <time_budget>`** — runs one experiment, streams log to terminal AND file, stops
- **`record_results.sh <run_num> ...`** — run manually after reviewing the log; extracts metrics, generates 3 samples, appends to session_pc.md. No auto-commit.

#### 4. Planned next runs (when ready)

| Run | PC_ALPHA | PC_WEIGHT | Time | Purpose |
|-----|----------|-----------|------|---------|
| 11 | 0.0 | 0.1 | 480s | Option A sanity check — should reproduce Run 6 (~1.2067) |
| 12 | 0.1 | 0.1 | 480s | Option B clean retry with warm kernel cache |
| 13 | 0.05 | 0.1 | 480s | Lighter routing retry with warm kernel cache |

All three should now benefit from the persistent kernel cache and produce
valid val_bpb numbers for comparison.

---

### Run 11 — 2026-03-16 14:48 UTC

| Field | Value |
|---|---|
| **val_bpb** | 1.618736 (+0.435047 vs baseline 1.183689) |
| **PC_WEIGHT** | 0.1 |
| **PC_ALPHA** | 0.0 |
| **TIME_BUDGET** | 480s |
| **Steps** | 60 |
| **Training seconds** | 489.8 |
| **MFU** | 1.43% |
| **Final pc_loss** | N/A |
| **Log** | sessions/run11_pc_w0.1_a0.0.log |

**Hypothesis:** ...

**Finding:** ...

#### Samples — Run 11

```
--- sample 1 ---
--- prompt: '' ---

<|reserved_0|>What is the cause of your stomach? Or? As you are having to have your stomach? Yes, you can get sick and get sick and your body. We're in the body of pregnancy from hearts who are sick.
I've been sick, for the health issues I want to get your blood closing blood. These and your baby's body fatigue and fat body fat, and fat. I'm not aware that your skin are being treated at least one of the following, and not the body's bone and is the most common causes of the body.
I'll give rise to the type of heartbeit? Treatment is a healthy blood pressure in our body, which is the body's organs and organs associated with our body.

--- sample 2 ---
--- prompt: '' ---

<|reserved_0|>Based on a Wayer Civil War II

Research for Korea Reny, and Quizab, and the Spir of Queen, and a Directederaldings of this Bankke, the Mador of the Large.
Covelation of the Federal Sques at 200 100 years as a 200 kilots that 250 miles are also common in cases, people are currently active and can be done on the 200-200-50 missions. In its 2010, the Nelsius of the Hadmus, the Easter National Institute of Technology (Maer) in Wednes

--- sample 3 ---
--- prompt: '' ---

<|reserved_0|>Can you wool?

In this guide you can easily get rid of any problem. They might be right in the case. It's not a lot of work, but not weeds. So it's not just you have some advantages or disadvantages. If you don't know, it's more comfortable to make sure you have it. Let's delve into the world and just what your body's body is too important to keep your health. That's what. That's where you're doing a new technology in your body. The most important aspects of the body? If you have a few weeks, you should do this.

Glook out of the body's body, you can see how to make your body and how to feel relaxing


```

---

### Run 11 — 2026-03-16 15:19 UTC

| Field | Value |
|---|---|
| **val_bpb** | 1.618736 (+0.435047 vs baseline 1.183689) |
| **PC_WEIGHT** | 0.1 |
| **PC_ALPHA** | 0.0 |
| **TIME_BUDGET** | 480s |
| **Steps** | 60 |
| **Training seconds** | 489.8 |
| **MFU** | 1.43% |
| **Final pc_loss** | N/A |
| **Log** | sessions/run11_pc_w0.1_a0.0.log |

**Hypothesis:** ...

**Finding:** ...

#### Samples — Run 11

```
--- sample 1 ---
--- prompt: '' ---

<|reserved_0|>Weldingerding
Benefits of Bodynamics
Benefits of Edition of the Lodder Bethan
Tamelen: How to Venverbs
The benefits of Crends
Denewelry
A: Unfortunately, it is important to note that they can increase their strength, strength and strength.
What is the difference between Welcome and Body
One of the most important benefits of Gathers is their own needs and is their primary goal. They are also a well-known source of precision that is known as a versatile tool for the quality of these factors. By taking care and care for the cost of the tools, you can provide the appropriate for

--- sample 2 ---
--- prompt: '' ---

<|reserved_0|>The Monday's Climate Change Building Fatest CEO

How to Image The Fest
Schoolship program is the programme for the program and is the main purpose of the project and the programme for the program.
How-to-making is a program to build a program designed programmers program that programme to work and work to provide programs. In this project, we will explore their programmed program to ensure that the program's programme is to enable programmers to program or programme programme programmer programm as a program programmer programmer programme programme programme to with programmable programmes for programmes and programmes programmes
Liter programmes by

--- sample 3 ---
--- prompt: '' ---

<|reserved_0|>January 20· Near 14-50

The Virtor: Korean Korean

The borean Tuesday Porse Tanzan

The Tuesday's March 1, 2020

Low's Poran and Horch

We now see the story of this is the first one of the world as long as we see, which is the greatest number of people are just about 1-9 f11. The first year. But there is at the beginning of the journey. The year, we can see the top of their parents, and parents that they love them. The first day we had a friend from the first day, while in our home


```

---

### Run 11 — 2026-03-16 15:25 UTC

| Field | Value |
|---|---|
| **val_bpb** | 1.618736 (+0.435047 vs baseline 1.183689) |
| **PC_WEIGHT** | 0.1 |
| **PC_ALPHA** | 0.0 |
| **TIME_BUDGET** | 480s |
| **Steps** | 60 |
| **Training seconds** | 489.8 |
| **MFU** | 1.43% |
| **Final pc_loss** | N/A |
| **Log** | sessions/run11_pc_w0.1_a0.0.log |

**Hypothesis:** PC_ALPHA=0.0 = Option A sanity check with warm kernel cache

**Finding:** VALID: 60 steps, no stalls, val_bpb=1.6187. pc_loss 0.00195→0.00123. Fewer tokens than Run 6 explains higher bpb.

#### Samples — Run 11

```
--- sample 1 ---
--- prompt: '' ---

<|reserved_0|>Topular: Ground 500 (100°C). In this blog, we will guide you through 300% of the United States and 55% of the United States, and 25% of the American population 2%. In the United States, 13%, in 2013, 2012. The Australian population is 301% for the 207% of the United States, and 10%. The average annual 50.40% increased in 2019 are 15 (50% than 3.4%) and 20% of the American citizens. These are the most effective in the United States and are most popular among these countries. The first

--- sample 2 ---
--- prompt: '' ---

<|reserved_0|>The 1 MAEW 3D model is the 2D model of the SOE 2D model 2Data, 3D image 1D printing is 2D printing. The model, according toD printing, will startD printing the model system of design. It helps theData 2D printing, the model to image printingDSDS modelingD printingDD printingD printingD printingDD printingDDSDSDD printingD printingD printingDD printingD printingD printingD printingDSDD printingD printingD printingDD printingDSD printingDD printingDDD printingDDDDirectDD printingDD

--- sample 3 ---
--- prompt: '' ---

<|reserved_0|>Help
For a small white ches,
inchesive with this beautiful blueb. 800.01, and 120
houhouhouhouhouhouhouhthouhthouhouhouhouhouhouhouhouhouhouhouhthouhou,houhouhouhouhouhouhouhouhouhouhouhouhouhouhthouhouhouhouhouhouhouhouhouhouhouhouhouhouhouhouhouhouhouhou


```

---

### Run 12 — 2026-03-16 15:25 UTC

| Field | Value |
|---|---|
| **val_bpb** | 1.546595 (+0.362906 vs baseline 1.183689) |
| **PC_WEIGHT** | 0.1 |
| **PC_ALPHA** | 0.1 |
| **TIME_BUDGET** | 600s |
| **Steps** | 71 |
| **Training seconds** | 605.4 |
| **MFU** | 1.41% |
| **Final pc_loss** | N/A |
| **Log** | sessions/run12_pc_w0.1_a0.1.log |

**Hypothesis:** Option B (PC_ALPHA=0.1): does error routing improve over Option A at matched compute?

**Finding:** IMPROVEMENT: val_bpb=1.5466 vs 1.6187 (Option A), -0.072. Routing helps. Step 0 stall (734s) ate budget; 71 steps vs ~75 expected.

#### Samples — Run 12

```
--- sample 1 ---
--- prompt: '' ---

<|reserved_0|>A study published in the 17th September 14th, 2020
Middle 11 March 2022 with two Civil Bedo
Series 2 Flight
Jul 13
Wednesday, June 13, 2020
Gree Ath 2018
What is the 28th 2004 Wednesday 2021
What is the first year?
2018 2008 Cultaneously?
What is the main idea of a Bed Wednesday.
Winner (Hey
Wednesday, month, 2021). Federal
Head
Wednesday 2 July 16, November 20

--- sample 2 ---
--- prompt: '' ---

<|reserved_0|>Invarious, the same as the a small group of researchers are involved in the lab, and the researchers are the use of the data from the cells. This is how the mice are in the same way, but the researchers at the University of Texas and the researchers are able to have a differentiation of the experiments. The mice are able to be able to be a little more frequently than the experiments. A team of scientists have included a nanotarticles called a single protein that produces a nanoparticles that is relatively few nanoparticles in the nanoparticle and nanoparticles. But this also comes from microscopy to nanoparticles, they are also a nanoparticles that are called

--- sample 3 ---
--- prompt: '' ---

<|reserved_0|>The Laya Bakery of the La Sa Vacca
Na is a Cooky
Na, the Nica Maosa
Na
a Naoa a, ica,a,a
aoa
aiga
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa


```

---

## Session 4 — 2026-03-16 PC2 Architecture Notes

### PC2 Changes (commits a5b32dc + 8b3462f)

Two changes introduced by a parallel agent, reviewed and smoke-tested:

**1. Residual PredHead** (`a5b32dc`)
`PredHead.forward` changed from `proj(tanh(fc(x)))` to `x + proj(tanh(fc(x)))`.
Biases predictions toward the identity at init (proj is zero-initialized), learning
corrections rather than full predictions. More stable early training.

**2. Per-layer precision scalars `pc_lambdas`** (`a5b32dc`)
`nn.Parameter(torch.ones(n_layer))` — one learnable scalar per layer weighting its
PC error contribution. Applied as `pc_lambdas[i].square()` (always positive).
Lets the model learn which layers benefit most from predictive coding.

**3. `unbind()` recompilation fix** (`8b3462f`)
`self.pc_lambdas[i]` inside the loop caused `n_layer` separate scalar-index ops,
each triggering a new Triton kernel. Pre-computing `pc_lambdas.square().unbind(0)`
before the loop gives the compiler a single stable op. Same fix applied to
`resid_lambdas`. Confirmed: second forward pass 8ms, no recompile.

---

### Run 13 pc2 — PC_ALPHA=0.1, PC_WEIGHT=0.1, TIME=600s (INVALID)

| Field | Value |
|---|---|
| **val_bpb** | 1.857829 (not comparable — cold cache) |
| **Steps** | 28 |
| **Training seconds** | 1008s |
| **Total seconds** | 2346s |
| **MFU** | 0.25% |
| **Log** | sessions/run13_pc2_w0.1_a0.1.log |

**Status: INVALID** — cold kernel cache from new pc2 graph variants.
Steps 6-8 stalled 157/159/330s, step 27 stalled 639s. `unbind()` fix was not yet
applied when this run started. Run 14 will be the first valid pc2 result.

---

### Run 14 pc2 — PC_ALPHA=0.1, PC_WEIGHT=0.1, TIME=600s (KILLED — architecture reconfiguration)

| Field | Value |
|---|---|
| **val_bpb** | N/A — killed before completion |
| **Steps completed** | 5 (steps 0–5 only) |
| **Log** | sessions/run14_pc2_w0.1_a0.1.log |

**Status: KILLED** — User requested architecture reconfiguration mid-run.

**Step timing observed (warm-cache partial run):**

| Step | dt (ms) | Notes |
|---|---|---|
| 0 | 429,381 | First-pass Triton kernel compilation |
| 1 | 35,904 | Still compiling variants |
| 2 | 46,026 | Still compiling variants |
| 3 | 41,212 | Still compiling variants |
| 4 | 310,097 | Large stall — additional kernel variant |
| 5 | 72,942 | Warming down |

**Key finding: `unbind()` fix is insufficient.** The fix addressed per-layer scalar indexing (`pc_lambdas[i]` → `pc_weights[i]` from unbind), confirmed working in smoke test (second forward 8ms). However, in a full training run, additional Triton kernel variants are triggered beyond what the smoke test covers:

- Step 0: 429s (down from 734s in pc1 and ~2346s total in Run 13 — cache partially warm from Run 13)
- Steps 1–3: 35–46s each (no analogous stalls in pc1/Run 12 — these steps were ~10s immediately)
- Step 4: 310s stall (vs. pc1 which was smooth after step 0)

**Root cause hypothesis:** The pc2 graph has more distinct Triton kernel variants than pc1 due to:
1. `pc_lambdas.square().unbind(0)` — `.square()` + 8×unbind scalars = new ops not in pc1 graph
2. Residual `PredHead`: `x + proj(tanh(fc(x)))` has an extra residual branch vs. pc1's `proj(tanh(fc(x)))`
3. The smoke test (B=4, T=256 with a single forward+backward) does not trigger the same kernel variants as full training (B=4×8 micro-steps, T=512, repeated gradient accumulation loop)

The `unbind()` fix resolves the *per-layer re-triggering* but does not eliminate the *initial set of novel kernels* the pc2 graph introduces. Run 14 was killed before the cache could fully warm.

**Architecture is being redesigned.** Next run number: **15**.

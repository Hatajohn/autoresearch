# Trainer Notes — pc-architecture-v2

*Supplement to `architecture.md`. Read both before starting a run.*  
*Updated after runs 24 and 25 plus the latest training-loop cleanup pass.*

---

## Why Predictive Coding?

The architectural bet behind this experiment: penalizing prediction error between adjacent layers creates an inductive bias toward hierarchical, compositional representations — as opposed to the purely statistical co-occurrence structure that a standard cross-entropy objective produces.

**Why not the alternatives?**

| Approach | Why not chosen |
|----------|----------------|
| Deep supervision (aux classification heads) | Requires labeled data; only penalizes task error, not inter-layer structure |
| Contrastive learning (CPC, BYOL) | Requires augmentations or negative samples; EMA teacher adds similar complexity at higher cost |
| VAE bottleneck | The stochastic layers here are this — but the residual stream bypasses them, eliminating the information bottleneck. PC does not depend on a bottleneck |
| Universal Transformers / DEQ | More principled but requires iterative inference at runtime (expensive) and significant architectural restructuring |
| Neurosymbolic integration | Maximally aligned with the goal but requires structured training data and a fixed symbolic vocabulary |

PC is self-supervised, adds no runtime inference cost, is compatible with standard causal LM training, and has biological grounding (Friston's free-energy principle). It is the lowest-overhead way to add structure-seeking pressure to an otherwise purely statistical objective.

**What would falsify the bet**: a `PC_WEIGHT=0.0` baseline achieving equal or better val_bpb with the same architecture and token budget. This is the highest-priority experiment (see Priority Order below).

---

## Current State

| Item | Status | Run / Commit |
|------|--------|--------------|
| `wte_std = 1.75/√n_embd` | ✅ Done | Run 20 |
| `PC_WEIGHT` reduced 0.1 → 0.02 | ✅ Done | Run 21 |
| `PC_WEIGHT_WARMUP` removed | ✅ Done | Run 21 |
| EMA target buffer (`pc_ema`) | ✅ Done | Run 21 |
| `pc_ema` update via `.data.copy_(new_ema)` (avoids per-step recompile) | ✅ Done | Reviewer approved |
| `PC_EMA_DECAY` env var (default 0.99); `alpha = 1 - PC_EMA_DECAY` | ✅ Done | 47bd6de |
| `KL_WARMUP_STEPS` reduced 200 → 25 | ✅ Done | Run 21 |
| Eval-mode compile pre-warm | ✅ Done | Run 21 |
| Step-0 diagnostic (logit_std, raw CE) — fix: use random targets, not dummy zeros | ✅ Done | 47bd6de |
| `LOG_MIN_WINDOW` env var | ✅ Done | Run 21 |
| Sigma statistics in PC diagnostic | ✅ Done | Run 20 |
| StochasticLayer collapse confirmed | ✅ Confirmed | Run 20 — `sigma≈0.05`, unchanged from init |
| EMA stabilization of pc_loss | ✅ Done | Run 22 — pc_loss settles after ~25 steps; final phase 1.3–3.7 |
| Resume+compile optimizer rebind | ✅ Done | Confirmed by runs 24 and 25 |
| Resume time-budget reset per process | ✅ Done | Reflected in run 25 training time |
| Per-run EMA log debias + raw-loss logging | ✅ Done | Reflected in run 25 logs |
| Best validation so far | ✅ Done | Run 25 — `val_bpb=1.537` |
| Run-config normalization for CLI/env run controls | ✅ Done | Latest cleanup pass |
| `pc_ema` update uses full accumulated batch mean | ✅ Done | Latest cleanup pass |
| Schedule/budget timing split from MFU timing | ✅ Done | Latest cleanup pass |
| Line-oriented step logging for redirected logs | ✅ Done | Latest cleanup pass |
| Resume load uses `torch.load(..., weights_only=True)` | ✅ Done | Latest cleanup pass |
| `MAX_EPOCHS` run control + large failsafe time budget | ✅ Done | Latest cleanup pass |

---

## Priority Order

The experiment ladder in `docs/experiment_ladder.md` now governs run ordering. Summary:

1. **[Stage 2 — mechanism viability] run27 → run28 → run29.** Validate the backbone and PC paths
   in a reduced config (DEPTH=4, n_embd≈256, MAX_SEQ_LEN=1024) before committing time to target-
   scale runs. Each run has a promotion gate documented in the ladder.

2. **[Stage 3 — fixed-budget value] run30 (matched baseline) then run31 (PC on).** Run30 and run31
   must use identical config, compile mode, and budget. The intended stopper is now `MAX_EPOCHS=1`
   with a large `TRAIN_TIME_BUDGET` as the failsafe. Only after both complete can we claim PC
   helps or does not help.

3. **[Stage 3 — stochastic ablation] run32.** Only after the PC question is settled (runs 30+31).
   Stochastic layers are collapsed (sigma≈0.05); their benefit is unproven.

4. **Throughput tuning last.** MFU is low; this matters, but it is secondary to answering the
   causal question about PC.

---

## Status By Category

### Known bugs / correctness risks

- No new run-blocking bug is confirmed in the latest long runs.
- Resume used to fail on the first optimizer step after compile; the optimizer rebind fix is now validated by successful resumed runs.
- Resume logging used to misreport debiased loss and time-budget semantics; the current run-25 behavior reflects the corrected per-run accounting.
- The latest cleanup pass also removed a silent `pc_ema` bias: EMA targets now use the mean of `layer_means` across all gradient-accumulation micro-batches, not just the final micro-batch in the step.
- The next code review should still treat checkpoint/resume, EMA updates, and training-time accounting as fragile paths because they have already produced misleading results once in this branch.

### Performance concerns

- Cold-cache compile and pre-warm remain expensive relative to short runs.
- MFU is still very low for the achieved throughput.
- Step-time outliers still appear in long runs even after the major recompilation bug was fixed.
- Muon momentum warmup and GC cadence are now at least named constants, but they are still hand-chosen rather than derived from measured run scale.
- The training schedule is still driven by `TRAIN_TIME_BUDGET`, so raising the time-budget safety fuse flattens progress-based LR/WD decay unless that schedule is decoupled later.

### Unvalidated research claims

- The PC branch may be helping, but that claim is blocked on the missing `PC_WEIGHT=0` baseline.
- The stochastic layers remain collapsed, so any claim that they help is weaker than the already-unproven claim about PC.
- Better `val_bpb` in runs 24 and 25 does not yet imply better samples or more structured internal representations.

---

## Known Issues and History

### pc_loss oscillation

Runs 16–20 all showed pc_loss oscillating over multiple orders of magnitude during the high-LR training phase. The progression:

| Run | Peak pc_loss | Root fix tried |
|-----|-------------|----------------|
| 16 | 27,910,940 | — |
| 17 | 816,270 | softplus on lambdas, float32 KL |
| 18 | ~577 | zero-init `pc_fc_w`, gate detach, hierarchical targets, LR×0.1 |
| 19 | 575 | (no new fix) |
| 20 | 4,348 | PC_WEIGHT_WARMUP=50 (made it worse — see below) |
| 21 | ~1,048 | EMA target buffer, PC_WEIGHT=0.02 — **cold cache; only 4 steps, inconclusive** |
| 22 | ~3,490 | pc_ema update via `.data.copy_()`; **43 steps, val_bpb 3.072**; pc_loss stabilizes after ~25 |
| 23 | N/A (PC_WEIGHT=0) | Intended baseline ablation, but the resumed run failed before producing a comparable result |
| 24 | ~178 early, ~0.50 late | 3 h continuation from checkpoint; **val_bpb 1.931**; resume+compile path completed successfully |
| 25 | ~1.2 early, ~0.63 late | 3 h continuation with corrected per-run accounting; **val_bpb 1.537**; best result so far |

**PC_WEIGHT_WARMUP was counterproductive** (run 20): shielding the backbone from PC pressure during early steps let it develop CE-optimal representations that were maximally *unpredictable* inter-layer. When the PC gradient ramped in, it found a harder target than without warmup, producing a higher pc_loss peak (4,348 vs 575). Warmup removed in run 21.

**EMA target rationale**: the oscillation was driven by PC heads chasing a target that moves at the speed of the backbone's CE gradient (each step). A 0.99-decay EMA has a ~100-step time constant, making the target approximately stationary relative to the PC head learning rate.

**Run 21 per-step recompilation (fixed):** Updating `pc_ema` with `model.pc_ema.mul_(...).add_(...)` incremented the buffer's version counter; `torch.compile` then invalidated the cached graph on every forward, causing 200–440 s per step. Fix: compute `new_ema` in a temporary, then `model.pc_ema.data.copy_(new_ema)`. The buffer is updated without triggering recompilation. Run 22 is the first run with this fix; step times should be back to ~45–65 s.

### StochasticLayer collapse

`sigma_bias = [0.049, 0.050]` in run 20 pc_diag (step 25). This equals `exp(-3)` — unchanged from initialization. The KL weight only reached 0.002 out of target 0.01 because `KL_WARMUP_STEPS=200` required 200 budget steps and the run only had ~29 training steps. Fixed for run 21 with `KL_WARMUP_STEPS=25`.

Deeper structural concern: the stochastic layers sit inside the residual stream, so gradient can bypass them. There is no information bottleneck forcing the KL compression to be useful. Whether they contribute anything beyond BPB noise is unknown until an ablation is run.

### Step-0 loss higher than expected

The raw step-0 CE (from the pre-warm diagnostic) should be ~10.5 nats with current `wte_std=1.75/√n_embd`. The EMA-smoothed training metric prints higher (~16 in run 20) because it accumulates the large first-step gradient into its exponential average. Always compare the **raw CE diagnostic** printed during pre-warm, not the step-0 training metric.

**Known diagnostic bug (fixed in 47bd6de):** run 21 printed `raw_CE=0.0131 nats` — nonsensically low. Cause: the diagnostic used `_dummy_y` (all-zeros target), so the model appeared to predict token 0 perfectly on zero-filled input. Fix uses random targets for the CE diagnostic so the self-prediction path is not artificially trivialised. Logit std was correct (`1.7320`); only the CE figure was affected.

### Throughput

~8,000–10,000 tok/sec on RTX 4080 (8-layer, n_embd=512 config). Each optimizer step = 8 gradient accumulation passes × 32 batch × 2048 seq = 524K tokens, taking ~45–65s. A 1800s budget yields approximately 30–40 optimizer steps.

**Cold-cache warning:** any code change that alters the compiled graph (new buffers, changed tensor shapes, new ops) invalidates the Triton kernel cache. Run 21 also suffered **per-step** recompilation: in-place mutation of `pc_ema` (`mul_`/`add_`) incremented the buffer version counter, so every forward invalidated the compiled graph. Fixed by updating via `new_ema = ...; model.pc_ema.data.copy_(new_ema)` so the buffer is not mutated in a way that triggers recompilation. Run 22 should see normal step times (~45–65s) after pre-warm; only the first run after a graph-changing edit will pay cold-cache pre-warm cost. Never draw conclusions from a cold-cache or pre-fix run.

The PC head einsums and aux_lm_heads approximately halve throughput vs a plain transformer. The aux_lm_heads (`+0.15×CE_{t+2} + 0.05×CE_{t+4}`) have not been ablated — their contribution to val_bpb vs throughput cost is unquantified.

**Timing/accounting note:** the training loop now distinguishes `schedule_time` from `throughput_time`. `schedule_time` is the clock used for LR / weight-decay progress and the time-budget stop, and it is restored on checkpoint resume. `throughput_time` is MFU-only, excludes the first `MFU_WARMUP_STEPS=10` optimizer steps, and is also restored so resumed runs keep consistent steady-state MFU summaries.

### Logging and run controls

Recent cleanup changed the operator-facing behavior of `train.py` in a few important ways:

- Run controls are normalised through a small `RunConfig` object so CLI overrides and env defaults resolve in one place instead of being re-derived inside `main()`.
- Per-step logs are now line-oriented, which makes `tee`/redirected session logs readable without relying on carriage-return progress rendering.
- Periodic crash-recovery checkpoints go to `checkpoint_latest.pt` and are also written immediately before an early stop exits.
- Final summaries now print a formatted `training_seconds` / `total_seconds` breakdown plus `stop_reason` when early stopping triggers.

### Resume semantics

The latest code path makes the intended resume policy explicit:

- `step`, optimizer state, and `pc_ema` continue from the checkpoint
- `schedule_time` continues from the checkpoint, so resumed runs keep the same wall-clock budget accounting
- smoothing and early-stop counters continue from the checkpoint
- Muon momentum warmup initialises from restored `step`, matching the warm optimizer buffers
- `kl_weight` also continues from restored `step`
- `throughput_time` continues from the checkpoint so MFU summaries stay comparable
- checkpoint loading uses `torch.load(..., weights_only=True)`

### Window-context mismatch in PC targets

Layer i (attention window W_i) predicts the EMA of layer i+1 (window W_{i+1} > W_i). With the LOG pattern, adjacent layers have a ~1.3× window ratio. The EMA collapses targets to per-layer means across the batch/sequence, which partially mitigates the mismatch (the mean representation is a softer target than the per-token output). Not yet known if this is sufficient.

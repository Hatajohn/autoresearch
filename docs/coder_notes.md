# Coder Notes — pc-architecture-v2

Instructions and context for the agent that modifies `train.py`, runs training, and iterates on the PC architecture. If you are a **new agent** taking over this role, start with the **Initial context prompt** below, then read the referenced docs and the **Current state** and **Next work** sections.

---

## Initial context prompt (paste to a new agent)

Use the following as an initial prompt when switching to a different agent so they have minimal context to start:

```
You are the Coder for the autoresearch project. Your job is to improve val_bpb (validation bits per byte) by editing train.py and running training experiments.

**Rules:**
- You may only edit train.py. Do not modify prepare.py (data, tokenizer, evaluation are fixed).
- You may not add dependencies; use only what is in pyproject.toml.
- The evaluation metric is val_bpb from prepare.evaluate_bpb; the goal is to lower it.
- Training runs for a fixed time budget (set by TRAIN_TIME_BUDGET env, often 1800s). Launch with: uv run train.py (or redirect to a log file).

**Read these before changing code or starting a run:**
1. docs/architecture.md — model layout, Phase 2 PC, EMA target, parameter groups, env vars.
2. docs/trainer_notes.md — implementation status, known issues (pc_loss history, pc_ema version-counter fix, cold cache), priority order.
3. docs/coder_notes.md — this file; current state and your next tasks.
4. program.md — experiment loop, results.tsv format, and any project-specific workflow (e.g. branch per run, first run = baseline).

**Current state (as of post–run 23 fix):**
- Run 22 completed successfully: 43 steps, val_bpb 3.072. Run 23 (resumed 2 hr) crashed on first optimizer.step(); resume+compile param rebind fix is now in train.py. A re-run with RESUME_CHECKPOINT is needed to confirm; until then, baseline can be run without resume (fresh init, PC_WEIGHT=0).
- The pc_ema buffer is updated via .data.copy_(new_ema); step times ~46–72s after pre-warm when cache is warm.
- No PC_WEIGHT=0 baseline exists yet — we cannot yet claim the PC loss helps val_bpb.

**Your next priority:** Run baseline with PC_WEIGHT=0.0 (no resume, same 1800 s budget) and compare val_bpb to 3.072. Optionally re-run with RESUME_CHECKPOINT to confirm the resume+compile fix. See "Next work" below.
```

---

## Current state (post–run 23 fix)

| Item | Status |
|------|--------|
| Latest run | Run 22 — 43 steps, val_bpb **3.072**, checkpoint saved |
| pc_ema fix | Confirmed: no per-step recompile; step times ~46–72s |
| EMA stabilization | Confirmed: pc_loss settles after ~25 steps; final phase 1.3–3.7 |
| Resume+compile rebind | **Fix applied** in train.py (rebind optimizer to `_orig_mod` after compile). Re-run with RESUME_CHECKPOINT to confirm. |
| StochasticLayer | Collapsed (sigma_bias ≈ 0.05 at step 25) |
| PC head learning | Minimal (pc_weights ≈ 0.67, near init) |
| Baseline (PC_WEIGHT=0) | Not yet run |

**Reference runs:** `sessions/run22_30min.log`, `reports/run22_report.md`. Run 23: resumed 2 hr run crashed at first step (grad None in _step_muon); see `reports/run23_errors.md`, `sessions/run23_2hr.log`. Fix: rebind optimizer to inner params after compile.



---

## Next work (priority order)

1. **Baseline ablation (PC_WEIGHT=0)**  
   Run with the same config as run 22 but `PC_WEIGHT=0.0`. Same time budget. Run **without** resume (fresh init) so the result is comparable. Compare val_bpb to 3.072. Log the run and record the result in `docs/trainer_notes.md` (pc_loss history table and Implementation Status / Priority Order as needed).

   Example command (no resume):
   ```bash
   PC_WEIGHT=0 TRAIN_TIME_BUDGET=1800 PC_DIAG_INTERVAL=25 uv run train.py 2>&1 | tee sessions/run23_30min.log
   ```
   Then: extract final val_bpb from the log; update the pc_loss history table in `docs/trainer_notes.md` with the actual val_bpb and outcome.

2. **Optional: Confirm resume+compile fix**  
   Re-run a short resumed run to confirm the first `optimizer.step()` no longer raises: e.g. `RESUME_CHECKPOINT=checkpoint.pt TRAIN_TIME_BUDGET=300 uv run train.py 2>&1 | tee sessions/run23_resume_test.log`.

3. **Optional: StochasticLayer ablation**  
   Stochastic layers are collapsed (sigma ≈ 0.05) and add compute. Try disabling them or setting `stoch_lr_scale=0` and compare val_bpb to run 22. If unchanged or better, document and consider keeping the simpler setup.

4. **After baseline:**  
   If PC_WEIGHT=0 baseline is worse than 3.072, the PC path is helping. Update trainer_notes to state that. Consider further tuning (e.g. PC_WEIGHT, PC_EMA_DECAY) or leaving as-is and iterating on other ideas. If baseline is better, document and either remove or heavily reduce the PC auxiliary loss and re-baseline.

---

## Vestigial / cleanup (train.py)

Notes from a vestigial-code pass. Address when convenient.

- [x] **PC diagnostic was dead code.** `PC_DIAG_INTERVAL` and `get_pc_diagnostics()` were never used. Fixed: the training loop now prints a single-line pc_diag (pc_weights, sigma_bias) every `PC_DIAG_INTERVAL` steps when `PC_DIAG_INTERVAL > 0` (uses `_orig_mod` for the compiled model). No further action.

- [x] **`estimate_flops` double-subtracts the embedding when weight-tied (correctness bug, low impact).** Fixed: when `lm_head.weight is transformer.wte.weight`, only subtract `wte_numel` once; otherwise subtract both. Logged MFU will now be correct.

- [x] **`all_params` list rebuilt every step.** Fixed: build `all_params` once before the `while True` loop and reuse for `clip_grad_norm_`.

- [x] **Phase 2 (PC einsums) runs unconditionally during eval.** Fixed: Phase 2 (h_stack through pc_loss_map) runs only when `reduction == 'mean'`; when `reduction == 'none'` we set `pc_loss = 0` and skip the einsums. Eval no longer pays PC cost.

- [x] **`_step_adamw` re-fills group-level scalar tensors for every parameter.** Fixed: the five group-level `fill_()` calls (lr, beta1, beta2, eps, wd) are now outside the param loop; only `_adamw_step_t.fill_(state['step'])` remains inside.

- [x] **`micro_step` loop variable is unused.** Fixed: loop is now `for _ in range(grad_accum_steps)`.

- [x] **`ema_beta = 0.9` is a magic number defined inside the hot loop.** Fixed: added module-level `LOG_SMOOTH_BETA = 0.9` and `LOGIT_SOFTCAP = 15` in the hyperparameter block; forward and the log-smoothing line use these constants.

- [x] **`build_model_config` is a one-shot local function.** Fixed: inlined at call site (base_dim, model_dim, num_heads, then GPTConfig(...)).

- [ ] **Redundant rotary in `GPT.__init__`.** In `GPT.__init__`, `cos`/`sin` are computed via `_precompute_rotary_embeddings` and registered as buffers. In `init_weights()` they are computed again and `self.cos`/`self.sin` are reassigned. With meta device the first computation is on meta and is discarded. Optional cleanup: in `__init__`, register placeholder buffers (e.g. empty or zeros with the right shape) instead of calling `_precompute_rotary_embeddings`; do the real cos/sin computation only in `init_weights()`. Low priority; behavior is correct as-is.

---

## Critiques

*From the Critic. Coder marks an item completed by changing `[ ]` to `[x]`. Critic removes completed items on the next review.*

_(No open critiques.)_

---

## Conventions and gotchas

- **Time budget:** Often 1800s (30 min) for “full” runs; enforced in the training loop. Pre-warm and eval are outside the budget. First run after a graph-changing edit may be cold-cache (long pre-warm); step times then normalize.
- **pc_ema:** Must be updated with `new_ema = ...; model.pc_ema.data.copy_(new_ema)`. Do not use in-place `mul_`/`add_` or the compiled graph will recompile every step (run 21 bug).
- **Resume + compile:** When using `RESUME_CHECKPOINT`, after `torch.compile(model)` the optimizer must be rebound to `model._orig_mod`’s parameters so that the tensors that receive gradients in backward are the same as those in `optimizer.param_groups`; otherwise the first `optimizer.step()` can raise `TypeError` in `_step_muon` when a param has `grad is None`.
- **Step-0 diagnostic:** Uses random targets for raw_CE; logit_std ~1.73 is expected. Ignore run 21’s raw_CE=0.01 (that was before the fix).
- **Logs and reports:** Training logs go to `sessions/runNN_*.log`; human-written reports to `reports/runNN_report.md`. When you add a new run, add a short row to the pc_loss history in trainer_notes and adjust “Priority order” if the next agent should do something different.

---

## Doc map

| File | Purpose |
|------|--------|
| `docs/architecture.md` | Model design, data flow, Phase 2 PC, EMA target, param groups, env vars |
| `docs/trainer_notes.md` | Implementation status, known issues, pc_loss history, priority order for next session |
| `docs/coder_notes.md` | This file; bootstrap prompt and Coder task list |
| `docs/runner_notes.md` | Runner role: launches/monitors runs, documents reports; does not modify code without permission |
| `docs/reviewer_notes.md` | Critic role, review checklist, artifact locations; for the agent that reviews Coder output |
| `program.md` | Experiment workflow, results.tsv, what you can/cannot do |
| `prepare.py` | Read-only; data, tokenizer, `evaluate_bpb` |

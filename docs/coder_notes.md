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

**Current state (as of run 22):**
- Run 22 completed successfully: 43 steps, val_bpb 3.072, main loss 16.66 → 12.03 nats.
- The pc_ema buffer is updated via .data.copy_(new_ema) to avoid torch.compile recompiling every step; step times are ~46–72s after pre-warm.
- PC loss stabilizes after ~25 steps (EMA + PC_WEIGHT=0.02). StochasticLayer sigma remains collapsed (0.05); PC head weights barely move from init.
- No PC_WEIGHT=0 baseline exists yet — we cannot yet claim the PC loss helps val_bpb.

**Your next priority:** Run a baseline with PC_WEIGHT=0.0 (same architecture, same time budget), then compare val_bpb to run 22’s 3.072. Log the run and update docs/trainer_notes.md with the result and a one-line note in the pc_loss history table.
```

---

## Current state (post–run 22)

| Item | Status |
|------|--------|
| Latest run | Run 22 — 43 steps, val_bpb **3.072**, checkpoint saved |
| pc_ema fix | Confirmed: no per-step recompile; step times ~46–72s |
| EMA stabilization | Confirmed: pc_loss settles after ~25 steps; final phase 1.3–3.7 |
| StochasticLayer | Collapsed (sigma_bias ≈ 0.05 at step 25) |
| PC head learning | Minimal (pc_weights ≈ 0.67, near init) |
| Baseline (PC_WEIGHT=0) | Not yet run |

**Reference runs:** `sessions/run22_30min.log`, `sessions/run22_report.md`. Run 20: val_bpb 3.201 (30 min, 40 steps). Run 21: aborted at step 4 (per-step recompile bug, since fixed).

---

## Next work (priority order)

1. **Baseline ablation (PC_WEIGHT=0)**  
   Run with the same config as run 22 but `PC_WEIGHT=0.0` (e.g. env `PC_WEIGHT=0`). Same time budget. Compare val_bpb to 3.072. If baseline is better or equal, the PC loss is not helping and the architecture should be revisited. Log the run (e.g. run23) and record the result in `docs/trainer_notes.md` (pc_loss history table and Implementation Status / Priority Order as needed).

2. **Optional: StochasticLayer ablation**  
   Stochastic layers are collapsed (sigma ≈ 0.05) and add compute. Try disabling them or setting `stoch_lr_scale=0` and compare val_bpb to run 22. If unchanged or better, document and consider keeping the simpler setup.

3. **After baseline:**  
   If PC_WEIGHT=0 baseline is worse than 3.072, the PC path is helping. Update trainer_notes to state that. Consider further tuning (e.g. PC_WEIGHT, PC_EMA_DECAY) or leaving as-is and iterating on other ideas. If baseline is better, document and either remove or heavily reduce the PC auxiliary loss and re-baseline.

---

## Conventions and gotchas

- **Time budget:** Often 1800s (30 min) for “full” runs; enforced in the training loop. Pre-warm and eval are outside the budget. First run after a graph-changing edit may be cold-cache (long pre-warm); step times then normalize.
- **pc_ema:** Must be updated with `new_ema = ...; model.pc_ema.data.copy_(new_ema)`. Do not use in-place `mul_`/`add_` or the compiled graph will recompile every step (run 21 bug).
- **Step-0 diagnostic:** Uses random targets for raw_CE; logit_std ~1.73 is expected. Ignore run 21’s raw_CE=0.01 (that was before the fix).
- **Logs and reports:** Training logs go to `sessions/runNN_*.log`; human-written reports to `sessions/runNN_report.md`. When you add a new run, add a short row to the pc_loss history in trainer_notes and adjust “Priority order” if the next agent should do something different.

---

## Doc map

| File | Purpose |
|------|--------|
| `docs/architecture.md` | Model design, data flow, Phase 2 PC, EMA target, param groups, env vars |
| `docs/trainer_notes.md` | Implementation status, known issues, pc_loss history, priority order for next session |
| `docs/coder_notes.md` | This file; bootstrap prompt and Coder task list |
| `program.md` | Experiment workflow, results.tsv, what you can/cannot do |
| `prepare.py` | Read-only; data, tokenizer, `evaluate_bpb` |

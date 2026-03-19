# Experiment Ladder — pc-architecture-v2

*Stage 1 artifact. Translates the three-stage experiment strategy into repo-specific runs,
commands, artifacts, and pass/fail gates. Read alongside `docs/trainer_notes.md` and
`docs/architecture.md`.*

---

## Two Research Questions

This ladder separates two distinct questions so evidence for one cannot be conflated with the other.

| Question | What it answers | Answered by |
|----------|----------------|-------------|
| **Mechanism viability** | Can the architecture train stably in a reduced setting? Do the PC and backbone paths behave predictably? | Stage 2 runs (run27, run28) |
| **Fixed-budget value** | Does the architecture beat a matched baseline at the target scale? | Stage 3 runs (run30–run32) |

Claims about PC benefit are only valid after a matched baseline exists (run30). Stage 2 runs produce
operational evidence only — they do not establish scientific value.

---

## Knob Inventory

### Env-sweepable (no code edit, no recompile penalty)

| Variable | Default | Notes |
|----------|---------|-------|
| `PC_WEIGHT` | `0.02` | Set to `0` for backbone-only runs |
| `PC_FOCAL_GAMMA` | `1.0` | |
| `KL_WEIGHT` | `0.01` | |
| `PC_HEAD_DIM` | `64` | |
| `PC_EMA_DECAY` | `0.99` | |
| `PC_DIAG_INTERVAL` | `50` | |
| `LOG_MIN_WINDOW` | `0` | |
| `WINDOW_PATTERN` | `LOG` | `PROGRESSIVE`, `LOG`, or SSSL-style e.g. `LSSL`; attention window schedule per layer |
| `USE_STOCHASTIC_LAYERS` | `1` | Set to `0` to disable |
| `USE_TORCH_COMPILE` | `1`/`full` | `regional` or `0` for faster startup |
| `TRAIN_TIME_BUDGET` | `86400` | Seconds; safety fuse when `MAX_EPOCHS > 0` |
| `MAX_EPOCHS` | `1` | Full passes through the training shards; `0` disables epoch-based stopping |
| `VAL_INTERVAL` | `0` | Steps between validation; `0` = time-budget only |
| `CHECKPOINT_STEPS` | `300` | |
| `EARLY_STOP_ENABLE` | `1` | |
| `EARLY_STOP_MIN_STEPS` | `50` | |
| `EARLY_STOP_VAL_PATIENCE` | `3` | |
| `EARLY_STOP_VAL_MIN_DELTA` | `0.005` | |
| `EARLY_STOP_TRAIN_PATIENCE` | `40` | |
| `EARLY_STOP_TRAIN_MIN_DELTA` | `0.02` | |
| `RESUME_CHECKPOINT` | _(empty)_ | Path to `.pt` file |

> Current run-control policy: `MAX_EPOCHS` is the intended primary stopper for fresh runs, while
> `TRAIN_TIME_BUDGET` remains a last-resort safety fuse. Early-stop and recovery checkpoints stay
> enabled so obviously unhelpful runs can still stop before the epoch cap.

### Code-edited size knobs (require `train.py` edit; invalidate compile cache)

| Constant | Current value | Reduced target (Stage 2) | Restored target (Stage 3) |
|----------|--------------|--------------------------|--------------------------|
| `DEPTH` | `8` | `4` | `8` |
| `ASPECT_RATIO` | `64` | `64` (with `DEPTH=4` → `n\_embd = 256`) | `64` (→ n\_embd = 512) |
| `HEAD_DIM` | `128` | `64` | `128` |
| `DEVICE_BATCH_SIZE` | `64` | `32` | `64` |
| `TOTAL_BATCH_SIZE` | `2^19` | `2^18` | `2^19` |

> In `train.py`, width is derived from `base_dim = DEPTH * ASPECT_RATIO`, then rounded to a multiple
> of `HEAD_DIM`. With `DEPTH=4`, keeping `ASPECT_RATIO=64` gives `n_embd=256`; dropping
> `ASPECT_RATIO` to `32` would shrink the model to `n_embd=128`, which is smaller than the planned
> Stage 2 minimum.

### Context-length knob (edit `prepare.py`)

| Constant | Current value | Reduced target (Stage 2) |
|----------|--------------|--------------------------|
| `MAX_SEQ_LEN` | `2048` | `512` or `1024` |

> `prepare.py` is otherwise read-only for all roles. Changing `MAX_SEQ_LEN` affects data loading
> and BPB comparability; document the value used in every run report.

---

## Run Ladder

Runs restart from **run1**. All prior session logs are archived in `archived/sessions/`. Runs are
ordered; each must be documented before the next starts. Do not skip steps or reorder without
updating this file.

### run1 — Stage 2: backbone minimum

**Purpose:** Confirm the backbone path trains stably in the smallest viable config. PC is off;
stochastic layers off; regional compile to reduce startup overhead.

**Config changes from current defaults (code edits required):**

```
DEPTH          8 → 4
ASPECT_RATIO  64 → 64        # with DEPTH=4, this yields n_embd=256
HEAD_DIM     128 → 64
DEVICE_BATCH_SIZE  64 → 32
TOTAL_BATCH_SIZE   2^19 → 2^18
```

Also edit `prepare.py`: `MAX_SEQ_LEN` → `1024`.

**Command:**

```bash
PC_WEIGHT=0 \
USE_STOCHASTIC_LAYERS=0 \
USE_TORCH_COMPILE=regional \
TRAIN_TIME_BUDGET=1800 \
VAL_INTERVAL=5 \
EARLY_STOP_MIN_STEPS=20 \
uv run train.py 2>&1 | tee sessions/run27_stage1_backbone_min.log
```

**Artifacts:** `sessions/run1_stage1_backbone_min.log`, `reports/run1_report.md`

**Gate (must pass before run2):**

| Check | Pass condition |
|-------|---------------|
| No NaN/inf in `raw` | All logged `raw:` values are finite |
| `raw` trends down | Last-third average `raw` < first-third average `raw` |
| Step times sane | Median step time < 60 s after pre-warm |
| Pre-warm completes | No compile crash or OOM |

---

### run2 — Stage 2: PC minimum

**Purpose:** Confirm the PC path trains stably on the same small config. Only change from run1:
`PC_WEIGHT=0.02` (default). Start from fresh init (no resume).

**Config changes:** Same reduced size as run27 (code edits persist: 4 layers, `n_embd=256`,
`MAX_SEQ_LEN=1024`). No additional code changes.

**Command:**

```bash
USE_STOCHASTIC_LAYERS=0 \
USE_TORCH_COMPILE=regional \
TRAIN_TIME_BUDGET=1800 \
VAL_INTERVAL=5 \
EARLY_STOP_MIN_STEPS=20 \
PC_DIAG_INTERVAL=10 \
uv run train.py 2>&1 | tee sessions/run28_stage1_pc_min.log
```

**Artifacts:** `sessions/run2_stage1_pc_min.log`, `reports/run2_report.md`

**Gate (must pass before run3):**

| Check | Pass condition |
|-------|---------------|
| `pc` interpretable | `pc:` values stay below `1e6` within the first 10 steps (no explosion to billions) |
| `raw` still trends down | PC auxiliary loss does not derail the backbone |
| `pc_diag` readable | pc_weights and sigma_bias print without error |
| Behavior repeatable | A second short rerun (same command) shows the same qualitative trajectory |

> If `pc` explodes (as seen in legacy run 26: pc > 1e8 by step 5), do **not** promote to run3.
> Document the failure in the run28 report and open a Coder task.

---

### run3 — Stage 2: scale up depth

**Purpose:** Test the first promoted scale increase — restore depth to 8, keep width and sequence
length reduced. Validates that the PC path survives a depth increase without requiring full-config
revalidation in one step.

**Config changes (code edits from run28 baseline):**

```
DEPTH          4 → 8          # restore
ASPECT_RATIO  64 → 32         # keep n_embd=256 after depth is restored
HEAD_DIM      64              # keep reduced
MAX_SEQ_LEN  1024             # keep reduced
```

**Command:**

```bash
USE_STOCHASTIC_LAYERS=0 \
USE_TORCH_COMPILE=regional \
TRAIN_TIME_BUDGET=1800 \
VAL_INTERVAL=5 \
EARLY_STOP_VAL_MIN_DELTA=0.0025 \
PC_DIAG_INTERVAL=10 \
uv run train.py 2>&1 | tee sessions/run3_stage1_scale_up_depth.log
```

**Artifacts:** `sessions/run3_stage1_scale_up_depth.log`, `reports/run3_report.md`

**Gate (must pass before stage 3):**

| Check | Pass condition |
|-------|---------------|
| `pc` still bounded | `pc:` < `1e6` within first 10 steps |
| `raw` trends down | Loss trajectory clearly improving |
| `val_bpb` printable | At least one validation completes cleanly |

> `EARLY_STOP_VAL_MIN_DELTA` is relaxed from `0.005` to `0.0025` for run3 so the run is less
> likely to stop before showing a longer post-promotion loss trend.

---

### run4 — Stage 3: matched baseline

**Purpose:** Establish the baseline for the scientific comparison. Restore target-scale config;
`PC_WEIGHT=0`; fresh init; same budget and compile mode as run31. This is the control run.

**Config changes (code edits to restore full size):**

```
DEPTH          8               # restored
ASPECT_RATIO  64               # restored → n_embd = 512
HEAD_DIM     128               # restored
DEVICE_BATCH_SIZE  64          # restored
TOTAL_BATCH_SIZE   2^19        # restored
MAX_SEQ_LEN  2048              # restored in prepare.py
```

**Command:**

```bash
PC_WEIGHT=0 \
USE_STOCHASTIC_LAYERS=0 \
USE_TORCH_COMPILE=full \
TRAIN_TIME_BUDGET=86400 \
MAX_EPOCHS=1 \
VAL_INTERVAL=300 \
CHECKPOINT_STEPS=300 \
EARLY_STOP_VAL_PATIENCE=3 \
EARLY_STOP_MIN_STEPS=50 \
uv run train.py 2>&1 | tee sessions/run4_stage2_matched_baseline.log
```

**Artifacts:** `sessions/run4_stage2_matched_baseline.log`, `reports/run4_report.md`

**Required for comparison:** record final `val_bpb`, total steps, training seconds, tok/s, MFU.
Run5 must use the same budget and config (except `PC_WEIGHT`).

> Stage 3 now uses `MAX_EPOCHS=1` as the intended stopper so the run clears one full pass over
> the training shards. `TRAIN_TIME_BUDGET=86400` remains only as a last-resort safety fuse.

**Gate (must exist before run5 is interpreted):**

| Check | Value to record |
|-------|----------------|
| `val_bpb` | Baseline reference |
| Total training steps | Must be comparable to run31 |
| Compile mode | Must match run31 (`full`) |
| Architecture | Must match run5 (8L / 512d / 2048 seq) |

---

### run5 — Stage 3: PC on at target scale

**Purpose:** PC-enabled run at full config, matched to run4 in every other dimension. This is the
primary scientific comparison.

**Config:** Same full-size as run4 (code edits persist from run4). Fresh init.

**Command:**

```bash
USE_STOCHASTIC_LAYERS=0 \
USE_TORCH_COMPILE=full \
TRAIN_TIME_BUDGET=86400 \
MAX_EPOCHS=1 \
VAL_INTERVAL=300 \
CHECKPOINT_STEPS=300 \
EARLY_STOP_VAL_PATIENCE=3 \
EARLY_STOP_MIN_STEPS=50 \
PC_DIAG_INTERVAL=50 \
uv run train.py 2>&1 | tee sessions/run5_stage3_pc_on_target.log
```

**Artifacts:** `sessions/run5_stage3_pc_on_target.log`, `reports/run5_report.md`

**Comparison to report:** run5 `val_bpb` vs run4 `val_bpb`. Distinguish:

- If run5 `val_bpb` < run4 `val_bpb` by more than `EARLY_STOP_VAL_MIN_DELTA=0.005`: PC provides
  measurable benefit.
- If run5 `val_bpb` ≥ run4 `val_bpb`: PC does not improve over baseline; document and revise
  priority order in `docs/trainer_notes.md`.

---

### run6 — Stage 3: stochastic layers ablation

**Purpose:** Determine whether re-enabling stochastic layers (`USE_STOCHASTIC_LAYERS=1`) at target
scale improves `val_bpb` vs run5 (PC on, no stochastic). Run only after run5 is complete and
the PC question is settled.

**Config:** Same as run5 except stochastic layers on. Fresh init.

**Command:**

```bash
USE_STOCHASTIC_LAYERS=1 \
USE_TORCH_COMPILE=full \
TRAIN_TIME_BUDGET=10800 \
VAL_INTERVAL=10 \
EARLY_STOP_VAL_PATIENCE=3 \
EARLY_STOP_MIN_STEPS=50 \
PC_DIAG_INTERVAL=50 \
uv run train.py 2>&1 | tee sessions/run6_stage3_stoch_ablation.log
```

**Artifacts:** `sessions/run6_stage3_stoch_ablation.log`, `reports/run6_report.md`

**Comparison to report:** run6 `val_bpb` vs run5 `val_bpb`. If stochastic layers add cost with
no `val_bpb` gain, document and recommend removing them.

---

## Stage Flow

```
Stage 2: Mechanism Viability
  run1 (backbone min) ──gate1──► run2 (pc min) ──gate2──► run3 (depth up)
                                                                 │
                                                          gate3 (promote?)
                                                                 │
Stage 3: Fixed-Budget Science                                    ▼
  run4 (matched baseline) ◄──── must complete before interpreting run5
  run5 (PC on target)     ◄──── primary claim
  run6 (stochastic ablation) ◄── secondary claim, after run5
```

---

## Promotion Gate Summary

| From | To | Gate |
|------|----|------|
| run1 | run2 | `raw` trends down; step times < 60 s; no NaN |
| run2 | run3 | `pc` < 1e6 at step 10; `raw` still declining; repeatable |
| run3 | Stage 3 | `pc` bounded; `val_bpb` prints; loss trends down |
| run4+5 | Interpretation | Both complete with matched config; `val_bpb` recorded for each |
| run5 | run6 | PC question settled; run6 is the stochastic-layers ablation |

If any gate fails: stop, document in the run's report, open a Coder task, and do not advance
the run number.

---

## Metrics Reference (printed by training loop)

| Metric | Field name | What it measures |
|--------|-----------|-----------------|
| CE loss (EMA-debiased) | `loss:` | Primary training signal |
| Raw batch CE | `raw:` | Unsmoothed; ground truth for step quality |
| PC auxiliary loss | `pc:` | Should stabilize; explosion = config problem |
| Grad norm | `gn:` | Clipped at 1.0; large values suggest instability |
| LR multiplier | `lrm:` | Tracks schedule progress |
| Step time | `NNNms` | Outliers indicate compile or system issues |
| Throughput | `tok/s` | Overall efficiency |
| MFU | `mfu:` | Meaningful only after `MFU_WARMUP_STEPS=10` |
| Validation BPB | `val_bpb` | Primary scientific metric; lower is better |
| PC diagnostics | `pc_diag` | `pc_weights` and `sigma_bias`; logged every `PC_DIAG_INTERVAL` |

---

## Run Report Template

Each run should produce `reports/runNN_report.md` with at minimum:

```
# Run NN — <run name>

**Log:** sessions/runNN_*.log
**Config:** <env vars used, size knobs in effect, MAX_SEQ_LEN>
**Resumed from:** <checkpoint path, or "fresh init">
**Status:** Completed / Stopped early / Failed

## Results
| Metric | Value |
| val_bpb | |
| Final step | |
| Training seconds | |
| tok/s (median) | |
| MFU | |
| Peak VRAM | |

## Gate assessment
<pass/fail for each gate item; justification>

## Next run recommendation
<promote / retry / open Coder task>
```

---

## Artifact Checklist for Stage 1 Completion

- [x] `docs/experiment_ladder.md` exists with run names, commands, gate tables, and knob inventory
- [x] `docs/trainer_notes.md` Priority Order updated to note ladder is in effect
- [x] `docs/coder_notes.md` "Next work" section updated to point at run27 as the next task

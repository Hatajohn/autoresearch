# Instructions — Test, Train, Sample

How to run tests, train the model, and sample from a checkpoint in this project.

---

## Prerequisites

- **Data and tokenizer:** Training and sampling use data and a BPE tokenizer under `~/.cache/autoresearch/`. If that directory is empty or missing, run once:

  ```bash
  uv run prepare.py
  ```

  Optionally limit shards for a quick test: `uv run prepare.py --num-shards 8`.

- **GPU:** Training and the full smoke test expect a CUDA GPU. Sampling can run on CPU if no GPU is available.

---

## Test

The smoke test checks that `train.py` and the model behave correctly (dtypes, shapes, compile, checkpoint resume, etc.) without running a full training job.

**Run all tests:**

```bash
uv run python smoke_test.py
```

**Skip the compiled test (faster, e.g. in CI):**

```bash
uv run python smoke_test.py --no-compile
```

You should see `All smoke tests passed.` at the end. The tests cover eager and compiled forward/backward, model invariants, forward numerics, grad norm and checkpoint resume, `VAL_INTERVAL` behaviour, and `LOG_MIN_WINDOW` validation.

---

## Train

Training runs for a **fixed wall-clock time budget**. The script loads data from `~/.cache/autoresearch/`, builds the model, and optionally resumes from a checkpoint. When the budget is reached, it runs a final validation and saves a checkpoint.

**Basic run (default 360 s budget):**

```bash
uv run train.py
```

**30-minute run with logging:**

The script reads settings from **environment variables**, so they are set in the same shell line and must come *before* the command (e.g. `VAR=value uv run train.py`). Example:

```bash
TRAIN_TIME_BUDGET=1800 PC_DIAG_INTERVAL=25 uv run train.py 2>&1 | tee sessions/run_30min.log
```

**Resume from a saved checkpoint:**

```bash
RESUME_CHECKPOINT=checkpoint.pt TRAIN_TIME_BUDGET=1800 uv run train.py 2>&1 | tee sessions/run_resume.log
```

**Useful environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAIN_TIME_BUDGET` | `360` | Training time budget in seconds (wall clock). |
| `RESUME_CHECKPOINT` | *(empty)* | Path to checkpoint file to resume from (e.g. `checkpoint.pt`). |
| `VAL_INTERVAL` | `0` | Run validation every this many steps (`0` = disabled). |
| `PC_WEIGHT` | `0.02` | Weight for the predictive-coding auxiliary loss. |
| `PC_EMA_DECAY` | `0.99` | EMA decay for PC target buffers. |
| `PC_DIAG_INTERVAL` | `50` | Print PC diagnostics every this many steps (`0` = off). |
| `KL_WEIGHT` | `0.01` | Weight for stochastic-layer KL loss. |
| `LOG_MIN_WINDOW` | `0` | Override minimum attention window (power of 2, ≥ 16; `0` = auto). |
| `USE_TORCH_COMPILE` | `1` | `1`/`full` = whole-model compile (fastest steps, long cold start). `regional` = per–transformer-block compile ([PyTorch recipe](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)) — shorter cold start. `0` = off. **Resume with the same mode the checkpoint used** (state_dict layout differs). |
| `USE_STOCHASTIC_LAYERS` | `1` | Set `0` to disable stochastic layers (no KL from them). |
| `CHECKPOINT_STEPS` | `300` | Every N steps, save full state to `checkpoint_latest.pt` for crash recovery (`0` = off). Same format as `checkpoint.pt` (resume with `RESUME_CHECKPOINT=checkpoint_latest.pt`). |
| `EARLY_STOP_ENABLE` | `1` | Enable automatic save-and-stop (`1` = on, `0` = off). When triggered, saves `checkpoint_latest.pt` before stopping. |
| `EARLY_STOP_MIN_STEPS` | `50` | Do not allow early-stop before this many training steps have completed. |
| `EARLY_STOP_VAL_PATIENCE` | `3` | Stop after this many non-improving validation checks (`VAL_INTERVAL` must be > 0). |
| `EARLY_STOP_VAL_MIN_DELTA` | `0.005` | Minimum `val_bpb` improvement required to reset validation patience. |
| `EARLY_STOP_TRAIN_PATIENCE` | `40` | Raw-loss safety stop: stop after this many steps where raw loss stays meaningfully above the best seen this run. |
| `EARLY_STOP_TRAIN_MIN_DELTA` | `0.02` | Allowed raw-loss drift above the best raw loss before counting toward the training-loss patience. |

**CLI:** `--checkpoint-steps N` overrides `CHECKPOINT_STEPS` (e.g. `uv run train.py --checkpoint-steps 200`).
Early-stop settings can also be overridden on the command line, e.g.:

```bash
uv run train.py --early-stop-enable 1 --early-stop-min-steps 80 --early-stop-val-patience 4
```

Checkpoint and config are defined in `train.py`; the default checkpoint path is `checkpoint.pt` in the current directory. During training, `checkpoint_latest.pt` is overwritten every `CHECKPOINT_STEPS` steps when enabled, and it is also the file used for automatic recovery saves before an early stop. At the end of a run the script prints a short summary including `val_bpb` and saves to `checkpoint.pt`.

---

## Sample

Sampling loads a saved checkpoint and generates text autoregressively with temperature and top-k.

**Default (empty prompt, 200 tokens, temp 0.8, top-k 50):**

```bash
uv run sample.py
```

**With options:**

```bash
uv run sample.py --prompt "Your prompt here" --tokens 200 --temp 0.8 --top-k 50
```

**From a specific checkpoint:**

```bash
uv run sample.py --checkpoint checkpoint.pt --prompt "The meaning of life is" --tokens 100 --temp 0.7 --top-k 40
```

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | `checkpoint.pt` | Path to the saved checkpoint. |
| `--prompt` | `""` | Initial text (or empty for BOS-only). |
| `--tokens` | `200` | Number of new tokens to generate. |
| `--temp` | `0.8` | Sampling temperature (higher = more random). |
| `--top-k` | `50` | Top-k sampling (only among top-k logits). |

The script expects the same tokenizer and model config as in the checkpoint (from `prepare.py` and `train.py`). If no checkpoint exists, it reports that you need to run training first.
